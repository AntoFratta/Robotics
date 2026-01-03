# src/response_classifier.py
"""
Classificatore di risposte basato su keyword matching (deterministico).
Rileva risposte evasive e temi emotivi forti.
"""

from typing import Optional


# --- Keyword sets per ogni emozione (Ekman) ---
EVASIVE_KEYWORDS = [
    "no", "niente", "non ricordo", "non so", "nulla", 
    "non mi viene in mente", "boh", "mah"
]

PAURA_KEYWORDS = [
    "panico", "ansia", "ansios", "paura", "spavent", 
    "affanno", "affann", "respiro", "respir", 
    "batticuore", "battito", "tremore", "tremor",
    "agitato", "agitaz", "nervoso", "nerv",
    "preoccup", "timore", "timor"
]

RABBIA_KEYWORDS = [
    "rabbia", "arrabbiato", "arrabbiat", "furioso", "furios",
    "frustrato", "frustrat", "irritato", "irritat",
    "nervoso", "nerv", "esasperato", "esasperat",
    "stufo", "stuf", "non ne posso più"
]

TRISTEZZA_KEYWORDS = [
    "triste", "tristezza", "tristezz",
    "vuoto", "vuota", "sconforto", "sconfort",
    "scoraggiato", "scoraggiat", "demoralizzato", "demoralizzat",
    "non ce la faccio", "non ce la", "depresso", "depress",
    "piango", "piangere", "lacrime", "lacrim",
    "solo", "sola", "solitudine", "solitudin",
    "nessuno", "abbandona", "lasciato", "isolato"
]

FELICITA_KEYWORDS = [
    "felice", "felicità", "contento", "content",
    "gioia", "gioios", "allegro", "allegr",
    "bene", "beni", "benissimo", "ottimo", "ottim",
    "sereno", "seren", "tranquillo", "tranquill"
]


def _normalize(text: str) -> str:
    """Normalizza il testo per keyword matching (lowercase, strip)"""
    return text.lower().strip()


def _contains_any_keyword(text: str, keywords: list[str]) -> bool:
    """Controlla se il testo contiene almeno una delle keyword"""
    normalized = _normalize(text)
    return any(keyword in normalized for keyword in keywords)


def is_evasive_answer(answer: str) -> bool:
    """
    Rileva risposte evasive (no, niente, non ricordo, etc.)
    
    Logica:
    - Risposta molto corta (max 15 caratteri) + keyword match
    - Oppure risposta che contiene solo keyword evasive
    
    Args:
        answer: La risposta dell'utente
    
    Returns:
        True se la risposta è evasiva, False altrimenti
    """
    normalized = _normalize(answer)
    
    # Se risposta vuota, è evasiva
    if not normalized:
        return True
    
    # Se risposta molto corta e contiene keyword evasive
    if len(normalized) <= 15 and _contains_any_keyword(normalized, EVASIVE_KEYWORDS):
        return True
    
    # Se risposta è esattamente una keyword evasiva (es: solo "niente")
    if normalized in EVASIVE_KEYWORDS:
        return True
    
    return False


def detect_emotional_theme(answer: str) -> Optional[str]:
    """
    Rileva emozione forte dalla risposta usando keyword matching (Ekman).
    
    Priorità (se match multipli):
    1. Paura
    2. Rabbia
    3. Tristezza
    4. Felicità
    
    Args:
        answer: La risposta dell'utente
    
    Returns:
        - "Paura" se rileva paura/ansia/panico
        - "Rabbia" se rileva rabbia/frustrazione
        - "Tristezza" se rileva tristezza/solitudine
        - "Felicità" se rileva felicità/gioia
        - None se nessuna emozione forte rilevata
    """
    normalized = _normalize(answer)
    
    # Se risposta troppo corta (< 3 caratteri), skip detection
    if len(normalized) < 3:
        return None
    
    # Check in ordine di priorità
    if _contains_any_keyword(normalized, PAURA_KEYWORDS):
        return "Paura"
    
    if _contains_any_keyword(normalized, RABBIA_KEYWORDS):
        return "Rabbia"
    
    if _contains_any_keyword(normalized, TRISTEZZA_KEYWORDS):
        return "Tristezza"
    
    if _contains_any_keyword(normalized, FELICITA_KEYWORDS):
        return "Felicità"
    
    return None


def get_theme_display_name(theme: str) -> str:
    """
    Ottiene il nome human-readable del tema/emozione (per debug/logging)
    
    Args:
        theme: Il codice del tema/emozione
    
    Returns:
        Nome leggibile
    """
    theme_names = {
        "evasive": "Risposta evasiva",
        "Paura": "Paura/Ansia",
        "Rabbia": "Rabbia/Frustrazione",
        "Tristezza": "Tristezza/Solitudine",
        "Felicità": "Felicità/Gioia",
    }
    return theme_names.get(theme, theme)
