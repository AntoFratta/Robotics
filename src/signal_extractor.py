# src/signal_extractor.py
"""
Estrazione segnali emotivi e tematici dalle risposte utente.
Usa LLM leggero (qwen2.5:3b) per classificazione veloce e stabile.
"""

from __future__ import annotations

import json
from typing import Dict, List, Any

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


# ===========================
# Categorie
# ===========================

EMOTIONS = [
    "ansia/panico",
    "tristezza",
    "paura",
    "rabbia/frustrazione",
    "sconforto/impotenza",
    "serenità",
    "speranza",
    "neutro/misto"
]

INTENSITY_LEVELS = ["bassa", "media", "alta"]

THEMES = [
    "autonomia/mobilità",
    "dolore/malessere fisico",
    "farmaci/terapie",
    "sonno/fatica",
    "ansia/panico/respirazione",
    "memoria/cognizione",
    "famiglia/visite",
    "solitudine/supporto sociale",
    "routine/attività quotidiane",
    "paure sul futuro",
    "alimentazione/appetito",
    "sicurezza/cadute"
]


# ===========================
# LLM Classifier (separato)
# ===========================

# LLM dedicato per classificazione (temp=0 per stabilità)
_classifier_llm = ChatOllama(model="qwen2.5:3b", temperature=0.0)


# ===========================
# Funzioni
# ===========================

def should_extract_signals(answer: str) -> bool:
    """
    Fast path: determina se estrarre segnali o usare default.
    Skip se risposta troppo corta (< 5 parole).
    
    Args:
        answer: Risposta dell'utente
    
    Returns:
        True se estrarre, False per fast path
    """
    if not answer or not answer.strip():
        return False
    
    words = answer.strip().split()
    return len(words) >= 5


def get_default_signals() -> Dict[str, Any]:
    """
    Output default per risposte corte (fast path).
    Bassa confidenza, neutro/misto.
    
    Returns:
        Dict con emotion, intensity, themes, confidence
    """
    return {
        "emotion": "neutro/misto",
        "intensity": "bassa",
        "themes": [],
        "confidence": 0.3
    }


def extract_signals(answer: str, question: str) -> Dict[str, Any]:
    """
    Estrae segnali emotivi/tematici dalla risposta usando LLM.
    
    Args:
        answer: Risposta dell'utente
        question: Domanda che ha generato la risposta
    
    Returns:
        {
            "emotion": str,
            "intensity": str,
            "themes": List[str],
            "confidence": float
        }
    """
    # System prompt con categorie e istruzioni
    system = SystemMessage(
        content=(
            "Sei un assistente che analizza risposte di pazienti anziani "
            "per estrarre segnali emotivi e tematici.\n\n"
            
            "IMPORTANTE: Rispondi SOLO con un JSON valido, nessun testo extra.\n\n"
            
            "Categorie disponibili:\n"
            f"- Emozioni: {', '.join(EMOTIONS)}\n"
            f"- Intensità: {', '.join(INTENSITY_LEVELS)}\n"
            f"- Temi: {', '.join(THEMES)}\n\n"
            
            "Analizza la risposta e restituisci SOLO questo JSON:\n"
            "{\n"
            '  "emotion": "<emozione predominante>",\n'
            '  "intensity": "<bassa|media|alta>",\n'
            '  "themes": ["<tema1>", "<tema2>"],\n'
            '  "confidence": <0.0-1.0>\n'
            "}\n\n"
            
            "Regole:\n"
            "- Scegli SOLO dalle categorie sopra\n"
            "- themes: lista (max 3 temi più rilevanti)\n"
            "- confidence: stima accuratezza (0.0-1.0)\n"
            "- Se incerto: usa 'neutro/misto' e confidence bassa\n"
        )
    )
    
    human = HumanMessage(
        content=(
            f"Domanda: {question}\n\n"
            f"Risposta: {answer}\n\n"
            "JSON:"
        )
    )
    
    try:
        # Prima chiamata LLM
        result = _classifier_llm.invoke([system, human])
        raw_output = (result.content or "").strip()
        
        # Parse JSON
        signals_data = json.loads(raw_output)
        
        # Validazione base
        if not _validate_signals(signals_data):
            raise ValueError("Invalid signals structure")
        
        return signals_data
        
    except (json.JSONDecodeError, ValueError) as e:
        # Retry con prompt più esplicito
        try:
            strict_system = SystemMessage(
                content=(
                    system.content + 
                    "\n\nATTENZIONE: Devi rispondere ESATTAMENTE con il formato JSON mostrato sopra. "
                    "Niente testo prima o dopo. Solo JSON puro."
                )
            )
            
            result = _classifier_llm.invoke([strict_system, human])
            raw_output = (result.content or "").strip()
            
            # Prova a estrarre JSON se c'è testo extra
            if "{" in raw_output and "}" in raw_output:
                start = raw_output.index("{")
                end = raw_output.rindex("}") + 1
                json_str = raw_output[start:end]
                signals_data = json.loads(json_str)
                
                if _validate_signals(signals_data):
                    return signals_data
        
        except Exception:
            pass
        
        # Fallback: default con nota di errore
        print(f"[WARNING] Signal extraction failed, using default. Error: {e}")
        return get_default_signals()


def _validate_signals(data: Dict[str, Any]) -> bool:
    """
    Valida struttura segnali estratti.
    
    Args:
        data: Dati da validare
    
    Returns:
        True se validi, False altrimenti
    """
    required_keys = {"emotion", "intensity", "themes", "confidence"}
    
    if not isinstance(data, dict):
        return False
    
    if not required_keys.issubset(data.keys()):
        return False
    
    if data["emotion"] not in EMOTIONS:
        return False
    
    if data["intensity"] not in INTENSITY_LEVELS:
        return False
    
    if not isinstance(data["themes"], list):
        return False
    
    # Valida temi (devono essere nella lista o lista vuota)
    for theme in data["themes"]:
        if theme not in THEMES:
            return False
    
    if not isinstance(data["confidence"], (int, float)):
        return False
    
    if not 0.0 <= data["confidence"] <= 1.0:
        return False
    
    return True
