# src/app.py
from __future__ import annotations

import json
from pathlib import Path

from src.graph import build_graph
from src.state import DialogueState
from src.profile_store import ProfileStoreConfig, build_profile_retriever


def load_diary_questions(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    # 1) carico domande diario
    questions_path = Path("data/diary_questions.json")
    diary_questions = load_diary_questions(questions_path)

    # 2) creo retriever UNA SOLA VOLTA (ottimizzazione)
    profile_path = Path("data/profiles/demo_profile_01.json")
    cfg = ProfileStoreConfig(
        profile_path=profile_path,
        schema_path=Path("data/profile_schema.json"),
        db_dir=Path("runtime/chroma_profile_db"),
        k=3,
        embed_model="mxbai-embed-large",
    )
    retriever = build_profile_retriever(cfg)

    # 3) costruisco grafo
    graph = build_graph()

    # 4) stato iniziale
    initial_state: DialogueState = {
        "diary_questions": diary_questions,
        "current_index": 0,
        "current_question": None,
        "last_user_answer": None,
        "qa_history": [],
        "done": False,
        "profile_path": str(profile_path),
        "profile_context": "",
        "retriever": retriever,
    }

    # NB: serve perché il grafo fa molti step (14 domande * più nodi)
    final_state = graph.invoke(initial_state, config={"recursion_limit": 200})

    print("\n=== FINE DIALOGO ===")
    print(f"Domande risposte: {len(final_state['qa_history'])}")
    for i, qa in enumerate(final_state["qa_history"], start=1):
        print(f"\n{i}) Q: {qa['question']}\n   A: {qa['answer']}")


if __name__ == "__main__":
    main()
