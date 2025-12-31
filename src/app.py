# src/app.py
from __future__ import annotations

import json
from pathlib import Path

from langchain_ollama import ChatOllama

from src.graph import build_graph
from src.state import DialogueState
from src.profile_store import ProfileStoreConfig, build_profile_retriever
from src.profile_utils import select_profile_interactive, get_safe_field
from src.session_logger import SessionLogger



def main():
    ROOT = Path(__file__).resolve().parents[1]

    data_dir = ROOT / "data"
    profiles_dir = data_dir / "profiles"
    questions_path = data_dir / "diary_questions.json"
    schema_path = data_dir / "profile_schema.json"
    
    runtime_dir = ROOT / "runtime"
    config_dir = runtime_dir / "config"
    sessions_dir = runtime_dir / "sessions"

    # Selezione profilo interattiva
    selection = select_profile_interactive(profiles_dir, config_dir)
    
    if selection is None:
        print("\nNessun profilo selezionato. Uscita.")
        return
    
    profile_path, profile_data, patient_id = selection
    
    print(f"\nProfilo caricato: {get_safe_field(profile_data, 'name', 'Paziente')}")
    print(f"Patient ID: {patient_id}")
    print()

    # Caricamento domande del diario
    diary_questions = json.loads(questions_path.read_text(encoding="utf-8"))

    # Costruzione retriever per vector store profilo
    db_dir = runtime_dir / "chroma_profile_db" / profile_path.stem
    db_dir.parent.mkdir(parents=True, exist_ok=True)
    
    cfg = ProfileStoreConfig(
        profile_path=profile_path,
        schema_path=schema_path,
        db_dir=db_dir,
        k=3,
    )
    retriever = build_profile_retriever(cfg)

    # Inizializzazione LLM (Ollama)
    llm = ChatOllama(model="qwen2.5:7b", temperature=0.75)

    # Inizializzazione logger di sessione
    session_logger = SessionLogger(patient_id, sessions_dir)

    # Costruzione grafo di dialogo
    graph = build_graph()

    initial_state: DialogueState = {
        "profile_path": str(profile_path),
        "profile_context": "",
        "diary_questions": diary_questions,
        "current_index": 0,
        "current_question": None,
        "last_user_answer": None,
        "qa_history": [],
        "done": False,
        "retriever": retriever,
        "llm": llm,
        "assistant_reply": None,
        "skip_question_print": False,
        "question_mode": "MAIN",
        "pending_question": None,
        "branch_count_for_current": 0,
        "current_branch_type": None,
        "original_question_index": 0,
        "session_logger": session_logger,
        "signals": [],
    }

    final_state = graph.invoke(initial_state, config={"recursion_limit": 400})

    # Salvataggio sessione
    print("\n" + "=" * 50)
    print("  FINE DIALOGO")
    print("=" * 50)
    print(f"Domande risposte: {len(final_state['qa_history'])}\n")
    
    # Logging di ogni coppia domanda-risposta
    for i, qa in enumerate(final_state["qa_history"], start=1):
        q_text = qa["question"]
        a_text = qa["answer"]
        
        # Recupera reply assistente dalla history (se disponibile)
        assistant_reply = qa.get("assistant_reply", "")
        
        session_logger.log_qa(i, q_text, a_text, assistant_reply)
        
        # Mostra anche a schermo
        print(f"{i}) Q: {q_text}")
        print(f"   A: {a_text}\n")
    
    # Salva sessione (CSV + JSON con qa_history, branches, signals)
    session_logger.save(profile_data, final_state.get("signals", []))


if __name__ == "__main__":
    main()
