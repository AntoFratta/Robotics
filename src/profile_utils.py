# src/profile_utils.py
"""
Utility per gestione profili utente.
Include: caricamento, selezione interattiva, generazione ID paziente.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


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
        print(f"Errore caricamento profilo {profile_path}: {e}")
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
        print("Nessun profilo trovato in data/profiles/")
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
                print(f"Scelta non valida. Inserisci un numero tra 1 e {len(profiles)}.")
        except ValueError:
            print("Input non valido. Inserisci un numero.")
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
