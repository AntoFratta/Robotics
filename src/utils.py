# src/utils.py
"""
Utility functions per il chatbot empatico.
Contiene tutte le funzioni helper per profili, genere, e pulizia testo.
"""
from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


# =============================================================================
# PROFILO - Gestione e Selezione
# =============================================================================

def generate_patient_id(profile_path: Path) -> str:
    """
    Genera ID paziente univoco ma anonimo basato su hash del path.
    Es: demo_profile_01.json -> P_a3f2b1c4
    """
    path_str = str(profile_path.stem)  # nome file senza estensione
    hash_obj = hashlib.md5(path_str.encode())
    hash_short = hash_obj.hexdigest()[:8]
    return f"P_{hash_short}"


def get_safe_field(profile: Dict[str, Any], field: str, default: str = "NON_SPECIFICATO") -> str:
    """
    Recupera campo profilo con fallback se mancante o vuoto.
    """
    value = profile.get(field)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip()


def load_profile_safe(profile_path: Path) -> Dict[str, Any]:
    """
    Carica profilo JSON con gestione errori.
    """
    try:
        return json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️ Errore caricamento profilo {profile_path}: {e}")
        return {}


def list_available_profiles(profiles_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    """
    Ritorna lista di (path, profile_data) per tutti i profili in profiles_dir.
    """
    profiles = []
    for profile_path in sorted(profiles_dir.glob("*.json")):
        profile_data = load_profile_safe(profile_path)
        if profile_data:  # Solo se caricamento OK
            profiles.append((profile_path, profile_data))
    return profiles


def save_last_used_profile(profile_path: Path, config_dir: Path):
    """
    Salva il profilo usato l'ultima volta in .last_profile.
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    last_profile_file = config_dir / ".last_profile"
    last_profile_file.write_text(str(profile_path), encoding="utf-8")


def load_last_used_profile(config_dir: Path) -> Optional[Path]:
    """
    Carica il profilo usato l'ultima volta, se esiste.
    """
    last_profile_file = config_dir / ".last_profile"
    if last_profile_file.exists():
        try:
            path_str = last_profile_file.read_text(encoding="utf-8").strip()
            path = Path(path_str)
            if path.exists():
                return path
        except Exception:
            pass
    return None


def display_profile_menu(profiles: List[Tuple[Path, Dict[str, Any]]], last_used: Optional[Path] = None) -> int:
    """
    Mostra menu CLI e ritorna l'indice del profilo scelto.
    """
    print("\n" + "=" * 50)
    print("  CHATBOT EMPATICO - SELEZIONE PROFILO")
    print("=" * 50 + "\n")
    
    if not profiles:
        print("⚠️ Nessun profilo trovato in data/profiles/")
        return -1
    
    print("Profili disponibili:\n")
    
    for idx, (path, profile) in enumerate(profiles, start=1):
        name = get_safe_field(profile, "name", "Sconosciuto")
        age = get_safe_field(profile, "age", "N/A")
        gender = get_safe_field(profile, "gender", "N/A")
        condition = get_safe_field(profile, "main_condition", "N/A")
        
        default_marker = ""
        if last_used and path == last_used:
            default_marker = " [ULTIMO USATO]"
        
        print(f"  {idx}. {name} ({age} anni, {gender}) - {condition}{default_marker}")
    
    print()
    
    # Trova default index
    default_idx = 1
    if last_used:
        for idx, (path, _) in enumerate(profiles, start=1):
            if path == last_used:
                default_idx = idx
                break
    
    while True:
        try:
            choice = input(f"Scegli profilo (1-{len(profiles)}) o INVIO per default [{default_idx}]: ").strip()
            
            if choice == "":
                return default_idx - 1  # 0-indexed
            
            choice_int = int(choice)
            if 1 <= choice_int <= len(profiles):
                return choice_int - 1  # 0-indexed
            else:
                print(f"⚠️ Scelta non valida. Inserisci un numero tra 1 e {len(profiles)}.")
        except ValueError:
            print("⚠️ Input non valido. Inserisci un numero.")
        except KeyboardInterrupt:
            print("\n\nUscita...")
            return -1


def select_profile_interactive(profiles_dir: Path, config_dir: Path) -> Optional[Tuple[Path, Dict[str, Any], str]]:
    """
    Mostra menu interattivo e ritorna (profile_path, profile_data, patient_id).
    """
    profiles = list_available_profiles(profiles_dir)
    
    if not profiles:
        return None
    
    last_used = load_last_used_profile(config_dir)
    
    selected_idx = display_profile_menu(profiles, last_used)
    
    if selected_idx == -1:
        return None
    
    profile_path, profile_data = profiles[selected_idx]
    patient_id = generate_patient_id(profile_path)
    
    # Salva come ultimo usato
    save_last_used_profile(profile_path, config_dir)
    
    return profile_path, profile_data, patient_id


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
