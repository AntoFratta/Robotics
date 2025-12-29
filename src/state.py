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

    # Se True, non ristampare "DOMANDA N:" perché la domanda è già stata mostrata dall'assistente
    skip_question_print: bool

    # --- Guided Path: Branching fields ---
    # Modalità della domanda corrente: "MAIN", "FOLLOWUP", "DEEPENING"
    question_mode: str

    # Override della domanda da chiedere (usato per follow-up/deepening)
    pending_question: Optional[str]

    # Contatore di branch per la domanda corrente del diario (max 1)
    branch_count_for_current: int

    # Tipo di diramazione attiva (se presente): "evasive", "ansia_panico", etc.
    current_branch_type: Optional[str]

    # Backup dell'indice originale (per debug/tracking)
    original_question_index: int

    # Session logger (per tracking branches)
    session_logger: object
