# src/response_classifier.py
"""
Classificatore di risposte basato su keyword matching (deterministico).
Rileva risposte evasive e temi emotivi forti.
"""

from typing import Optional


# --- Keyword sets per ogni tema ---
EVASIVE_KEYWORDS = [
    "no", "niente", "non ricordo", "non so", "nulla", 
    "non mi viene in mente", "boh", "mah"
]

ANSIA_PANICO_KEYWORDS = [
    "panico", "ansia", "ansios", "paura", "spavent", 
    "affanno", "affann", "respiro", "respir", 
    "batticuore", "battito", "tremore", "tremor",
    "agitato", "agitaz", "nervoso", "nerv"
]

DOLORE_FISICO_KEYWORDS = [
    "dolore", "male", "fa male", "mi fa male",
    "rigidità", "rigid", "tremori", "tremor",
    "difficoltà a muover", "non riesco a muover",
    "stanchezza fisica", "spossato", "spossa",
    "debol", "affaticato", "affatic"
]

SOLITUDINE_KEYWORDS = [
    "solo", "sola", "solitudine", "solitudin",
    "nessuno", "abbandona", "lasciato",
    "figli non vengono", "figli non", "non mi cerca",
    "isolato", "isola", "nessuno mi"
]

TRISTEZZA_KEYWORDS = [
    "triste", "tristezza", "tristezz",
    "vuoto", "vuota", "sconforto", "sconfort",
    "scoraggiato", "scoraggiat", "demoralizzato", "demoralizzat",
    "non ce la faccio", "non ce la", "depresso", "depress",
    "piango", "piangere", "lacrime", "lacrim"
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
    Rileva tema emotivo forte dalla risposta usando keyword matching.
    
    Priorità (se match multipli):
    1. ansia_panico
    2. dolore_fisico
    3. solitudine
    4. tristezza
    
    Args:
        answer: La risposta dell'utente
    
    Returns:
        - "ansia_panico" se rileva ansia/panico
        - "dolore_fisico" se rileva dolore/malessere
        - "solitudine" se rileva solitudine/abbandono
        - "tristezza" se rileva tristezza/sconforto
        - None se nessun tema forte rilevato
    """
    normalized = _normalize(answer)
    
    # Se risposta troppo corta (< 3 caratteri), skip detection temi
    if len(normalized) < 3:
        return None
    
    # Check in ordine di priorità
    if _contains_any_keyword(normalized, ANSIA_PANICO_KEYWORDS):
        return "ansia_panico"
    
    if _contains_any_keyword(normalized, DOLORE_FISICO_KEYWORDS):
        return "dolore_fisico"
    
    if _contains_any_keyword(normalized, SOLITUDINE_KEYWORDS):
        return "solitudine"
    
    if _contains_any_keyword(normalized, TRISTEZZA_KEYWORDS):
        return "tristezza"
    
    return None


def get_theme_display_name(theme: str) -> str:
    """
    Ottiene il nome human-readable del tema (per debug/logging)
    
    Args:
        theme: Il codice del tema (es: "ansia_panico")
    
    Returns:
        Nome leggibile del tema
    """
    theme_names = {
        "evasive": "Risposta evasiva",
        "ansia_panico": "Ansia/Panico",
        "dolore_fisico": "Dolore fisico",
        "solitudine": "Solitudine",
        "tristezza": "Tristezza/Sconforto",
    }
    return theme_names.get(theme, theme)
