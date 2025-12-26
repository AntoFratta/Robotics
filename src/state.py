# src/state.py
from typing_extensions import TypedDict
from typing import List, Dict, Any, Optional


class DialogueState(TypedDict):
    diary_questions: List[Dict[str, Any]]
    current_index: int

    # domanda originale dal diario
    current_question: Optional[str]

    # domanda da mostrare (riscritta dall'LLM)
    display_question: Optional[str]

    last_user_answer: Optional[str]
    qa_history: List[Dict[str, str]]
    done: bool

    # Profilo
    profile_path: str
    profile_context: str

    # Retriever (Chroma + OllamaEmbeddings)
    retriever: object

    # LLM (ChatOllama) creato una sola volta in app.py
    llm: object
