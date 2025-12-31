# src/text_utils.py
"""
Utility per manipolazione testo.
Include: gestione genere, pulizia testo, validazione formalità.
"""
from __future__ import annotations

import re


# =============================================================================
# GENERE - Gestione Accordi Grammaticali
# =============================================================================

def gender_label(gender: str) -> str:
    """
    Normalizza genere in MASCHILE, FEMMINILE, o NON_SPECIFICATO.
    """
    g = (gender or "").strip().upper()
    if g in {"M", "MALE", "UOMO", "MASCHIO", "MASCHILE"}:
        return "MASCHILE"
    if g in {"F", "FEMALE", "DONNA", "FEMMINA", "FEMMINILE"}:
        return "FEMMINILE"
    return "NON_SPECIFICATO"


def coerce_gender(text: str, gender_label: str) -> str:
    """
    Adatta gli accordi grammaticali nel testo in base al genere.
    """
    if gender_label not in {"MASCHILE", "FEMMINILE"}:
        return text

    if gender_label == "MASCHILE":
        repl = {
            "preoccupata": "preoccupato",
            "stressata": "stressato",
            "determinata": "determinato",
            "legata": "legato",
            "stata": "stato",
            "serena": "sereno",
            "tranquilla": "tranquillo",
            "angosciata": "angosciato",
            "spaventata": "spaventato",
            "stanca": "stanco",
        }
    else:
        repl = {
            "preoccupato": "preoccupata",
            "stressato": "stressata",
            "determinato": "determinata",
            "legato": "legata",
            "stato": "stata",
            "sereno": "serena",
            "tranquillo": "tranquilla",
            "angosciato": "angosciata",
            "spaventato": "spaventata",
            "stanco": "stanca",
        }

    out = text
    for k, v in repl.items():
        out = re.sub(rf"\b{k}\b", v, out, flags=re.IGNORECASE)
    return out


def format_question_for_gender(question_text: str, gender_label: str) -> str:
    """
    Adatta la domanda al genere del paziente.
    """
    if gender_label == "FEMMINILE":
        replacements = {
            "si è sentito": "si è sentita",
            "si è sentita": "si è sentita",  
            "è riuscito": "è riuscita",
            "è riuscita": "è riuscita",  
            "particolarmente preoccupato": "particolarmente preoccupata",
            "pensa di essersi sentito": "pensa di essersi sentita",
            "si è sentito in difficoltà": "si è sentita in difficoltà",
        }
    else:
        return question_text
    
    result = question_text
    for old, new in replacements.items():
        result = result.replace(old, new)
    
    return result


# =============================================================================
# PULIZIA TESTO
# =============================================================================

def strip_questions(text: str) -> str:
    """
    Rimuove righe che sono domande (iniziano con parole interrogative o finiscono con ?).
    Ma NON rimuove righe che contengono '?' nel mezzo come parte di una frase descrittiva.
    """
    lines = []
    for ln in text.splitlines():
        ln_stripped = ln.strip()
        # Salta righe vuote
        if not ln_stripped:
            continue
        # Rimuovi solo se è CHIARAMENTE una domanda
        if re.match(r'^\s*(come|cosa|quando|dove|perché|perchè|chi|quale|quanto)\b', ln_stripped.lower()):
            continue
        if ln_stripped.endswith('?'):
            continue
        lines.append(ln_stripped)
    return '\n'.join(lines).strip()


def strip_labels(text: str) -> str:
    """
    Rimuove etichette tipo "Riflesso:", "Validazione:", ecc.
    """
    out = text
    out = re.sub(r"(?im)^\s*(riflesso|validazione|valido|valida)\s*:\s*", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def is_formal_ok(text: str) -> bool:
    """
    Verifica che il testo non contenga forme informali (tu/ti/te).
    """
    low = text.lower()
    forbidden = [
        r"\btu\b", r"\bti\b", r"\bte\b", r"\btua\b", r"\btuo\b",
        r"\bstai\b", r"\bsei\b", r"\bper te\b"
    ]
    return not any(re.search(p, low) for p in forbidden)


def trim_to_max_sentences(text: str, max_sentences: int = 3) -> str:
    """
    Limita il testo a un massimo di N frasi.
    """
    s = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", s)
    return " ".join(parts[:max_sentences]).strip()
