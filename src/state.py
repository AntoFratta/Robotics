# src/state.py
from typing_extensions import TypedDict
from typing import List, Dict, Any, Optional


class DialogueState(TypedDict):
    # Profilo
    profile_path: str
    profile_context: str

    # Domande del diario (caricate da JSON)
    diary_questions: List[Dict[str, Any]]

    # Indice domanda corrente
    current_index: int

    # Testo della domanda corrente
    current_question: Optional[str]

    # Ultima risposta dell'utente (input appena letto)
    last_user_answer: Optional[str]

    # Storico Q/A
    qa_history: List[Dict[str, str]]

    # Flag fine
    done: bool

    # Retriever profilo (Chroma)
    retriever: object

    # LLM (Ollama via LangChain)
    llm: object

    # Ultimo output assistente
    assistant_reply: Optional[str]

    # Se True, non ristampare il numero della domanda perché già mostrata dall'assistente
    skip_question_print: bool

    # Campi per branching (follow-up/deepening)
    question_mode: str  # "MAIN", "FOLLOWUP", "DEEPENING"
    pending_question: Optional[str]  # Override domanda da porre
    branch_count_for_current: int  # Contatore branch per domanda corrente (max 1)

    # Tipo di diramazione attiva (se presente): "evasive", "ansia_panico", etc.
    current_branch_type: Optional[str]

    # Backup dell'indice originale (per debug/tracking)
    original_question_index: int

    # Session logger (per tracking branches)
    session_logger: object

    # Signals estratti dalle risposte (emotion/theme extraction)
    signals: List[Dict[str, Any]]
