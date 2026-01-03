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

    # Campi per branching (follow-up e approfondimenti emotivi)
    question_mode: str  # "MAIN", "FOLLOWUP", "DEEPENING"
    pending_question: Optional[str]  # Domanda di override da porre
    branch_count_for_current: int  # Contatore branch per domanda corrente (massimo 2 iterazioni)

    # Tipo di diramazione attiva (es: "evasive", "Paura", "Rabbia", "Tristezza")
    current_branch_type: Optional[str]

    # Indice originale della domanda (per tracking)
    original_question_index: int

    # Session logger (per tracking branches)
    session_logger: object

    # Signals estratti dalle risposte (emotion/theme extraction)
    signals: List[Dict[str, Any]]
    
    # Emozione e intensità estratte dall'ultima risposta
    last_emotion: Optional[str]
    last_intensity: Optional[str]
    
    # Emozione iniziale (salvata all'entrata del loop di dialogo libero)
    initial_emotion: Optional[str]
    initial_intensity: Optional[str]
    
    # Contesto salute recuperato via RAG
    health_context: Optional[str]
