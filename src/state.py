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
