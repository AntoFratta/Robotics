# src/state.py
from typing_extensions import TypedDict
from typing import List, Dict, Any, Optional


class DialogueState(TypedDict):
    # Domande del diario (caricate da JSON)
    diary_questions: List[Dict[str, Any]]

    # Indice domanda corrente
    current_index: int

    # Testo della domanda corrente
    current_question: Optional[str]

    # Ultima risposta dell'utente
    last_user_answer: Optional[str]

    # Storico Q/A
    qa_history: List[Dict[str, str]]

    # Flag fine
    done: bool
