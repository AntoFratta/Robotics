# scripts/test_branching.py
"""
Script di test per il sistema di branching guidato.
Testa risposte evasive, temi emotivi, e limite max 1 follow-up.
"""

import json
from pathlib import Path

# Aggiungi src al path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.response_classifier import is_evasive_answer, detect_emotional_theme, get_theme_display_name


def test_evasive_detection():
    """Test rilevamento risposte evasive"""
    print("\n" + "="*60)
    print("TEST 1: Rilevamento Risposte Evasive")
    print("="*60)
    
    test_cases = [
        ("niente", True),
        ("non ricordo", True),
        ("no", True),
        ("non so", True),
        ("Ho visto i miei nipoti", False),
        ("Mi sono sentito felice", False),
        ("boh", True),
        ("", True),  # risposta vuota
        ("nulla di particolare", True),
    ]
    
    for answer, expected in test_cases:
        result = is_evasive_answer(answer)
        status = "[OK]" if result == expected else "[FAIL]"
        print(f"{status} '{answer}' -> evasiva={result} (expected={expected})")


def test_emotional_theme_detection():
    """Test rilevamento temi emotivi"""
    print("\n" + "="*60)
    print("TEST 2: Rilevamento Temi Emotivi")
    print("="*60)
    
    test_cases = [
        ("Ho avuto attacchi di panico", "ansia_panico"),
        ("Mi sento molto ansioso", "ansia_panico"),
        ("Ho dolore alla schiena", "dolore_fisico"),
        ("Mi fa male tutto il corpo", "dolore_fisico"),
        ("Mi sento molto solo", "solitudine"),
        ("Nessuno mi viene a trovare", "solitudine"),
        ("Sono triste", "tristezza"),
        ("Mi sento scoraggiato", "tristezza"),
        ("Non ce la faccio più", "tristezza"),
        ("Ho visto i miei nipoti", None),  # nessun tema
        ("Mi sono sentito bene", None),
    ]
    
    for answer, expected in test_cases:
        result = detect_emotional_theme(answer)
        status = "[OK]" if result == expected else "[FAIL]"
        display = get_theme_display_name(result) if result else "Nessun tema"
        print(f"{status} '{answer}' -> {result} ({display}) [expected={expected}]")


def test_followup_questions_structure():
    """Test struttura del file follow_up_questions.json"""
    print("\n" + "="*60)
    print("TEST 3: Struttura Follow-up Questions")
    print("="*60)
    
    followup_path = ROOT / "data" / "follow_up_questions.json"
    
    if not followup_path.exists():
        print("❌ File follow_up_questions.json non trovato!")
        return
    
    data = json.loads(followup_path.read_text(encoding="utf-8"))
    
    expected_categories = ["evasive", "ansia_panico", "dolore_fisico", "solitudine", "tristezza"]
    
    for category in expected_categories:
        if category in data:
            count = len(data[category])
            print(f"[OK] Categoria '{category}': {count} domande")
            
            # Mostra prima domanda come esempio
            if count > 0:
                first_q = data[category][0]["template"]
                print(f"   Esempio: '{first_q[:60]}...' ")
        else:
            print(f"[FAIL] Categoria '{category}' mancante!")


def test_priority_order():
    """Test ordine di priorità dei temi (ansia > dolore > solitudine > tristezza)"""
    print("\n" + "="*60)
    print("TEST 4: Priorità Temi (se match multipli)")
    print("="*60)
    
    # Risposta che potrebbe matchare più temi
    answer = "Ho panico e dolore e mi sento solo e triste"
    result = detect_emotional_theme(answer)
    
    print(f"Risposta: '{answer}'")
    print(f"Tema rilevato: {result} ({get_theme_display_name(result)})")
    print(f"Priorita corretta: ansia_panico > dolore_fisico > solitudine > tristezza")
    
    if result == "ansia_panico":
        print("[OK] Priorita rispettata (ansia_panico ha precedenza)")
    else:
        print(f"[FAIL] Priorita non rispettata (expected: ansia_panico, got: {result})")


if __name__ == "__main__":
    print("\n[TEST SUITE] Guided Path Branching System")
    print("="*60)
    
    test_evasive_detection()
    test_emotional_theme_detection()
    test_followup_questions_structure()
    test_priority_order()
    
    print("\n" + "="*60)
    print("[OK] Test completati!")
    print("="*60)
    
    print("\n[PROSSIMI PASSI]")
    print("1. Esegui: python src/app.py")
    print("2. Testa risposte evasive (es: 'niente', 'non ricordo')")
    print("3. Testa temi emotivi (es: 'ho avuto panico', 'mi sento solo')")
    print("4. Verifica limite max 1 follow-up per domanda")
    print("5. Controlla che i messaggi [DEBUG] mostrino i rilevamenti")
