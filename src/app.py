# src/app.py
from __future__ import annotations

import json
from pathlib import Path

from langchain_ollama import ChatOllama

from src.graph import build_graph
from src.state import DialogueState
from src.profile_store import ProfileStoreConfig, build_profile_retriever


def pick_latest_profile(profiles_dir: Path) -> Path:
    profiles = sorted(
        profiles_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not profiles:
        raise FileNotFoundError(f"Nessun profilo .json trovato in: {profiles_dir}")
    return profiles[0]


def main():
    ROOT = Path(__file__).resolve().parents[1]  # cartella Robotics/

    data_dir = ROOT / "data"
    profiles_dir = data_dir / "profiles"

    questions_path = data_dir / "diary_questions.json"
    schema_path = data_dir / "profile_schema.json"

    # prende AUTOMATICAMENTE il profilo più recente dentro data/profiles/
    profile_path = pick_latest_profile(profiles_dir)

    runtime_dir = ROOT / "runtime"
    db_dir = runtime_dir / "chroma_profile_db" / profile_path.stem
    db_dir.parent.mkdir(parents=True, exist_ok=True)

    # --- Load diary questions ---
    diary_questions = json.loads(questions_path.read_text(encoding="utf-8"))

    # --- Build retriever (profilo -> vector store) ---
    cfg = ProfileStoreConfig(
        profile_path=profile_path,
        schema_path=schema_path,
        db_dir=db_dir,
        k=3,
    )
    retriever = build_profile_retriever(cfg)

    # --- LLM (Ollama) ---
    # Qwen2.5:7b ha migliori capacità di seguire istruzioni e gestire l'italiano
    # Temperatura 0.65 per varietà nelle risposte mantenendo coerenza
    llm = ChatOllama(model="qwen2.5:7b", temperature=0.65)

    # --- Graph ---
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
    }

    print(f"\n[INFO] Profilo in uso: {profile_path}")
    final_state = graph.invoke(initial_state, config={"recursion_limit": 400})

    print("\n=== FINE DIALOGO ===")
    print(f"Domande risposte: {len(final_state['qa_history'])}")
    for i, qa in enumerate(final_state["qa_history"], start=1):
        print(f"\n{i}) Q: {qa['question']}\n   A: {qa['answer']}")


if __name__ == "__main__":
    main()
