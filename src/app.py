# src/app.py
from __future__ import annotations

import json
from pathlib import Path

from src.graph import build_graph
from src.state import DialogueState


def load_diary_questions(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    questions_path = Path("data/diary_questions.json")
    diary_questions = load_diary_questions(questions_path)

    graph = build_graph()

    initial_state: DialogueState = {
        "diary_questions": diary_questions,
        "current_index": 0,
        "current_question": None,
        "last_user_answer": None,
        "qa_history": [],
        "done": False,
    }

    final_state = graph.invoke(initial_state, config={"recursion_limit": 200})

    print("\n=== FINE DIALOGO ===")
    print(f"Domande risposte: {len(final_state['qa_history'])}")
    for i, qa in enumerate(final_state["qa_history"], start=1):
        print(f"\n{i}) Q: {qa['question']}\n   A: {qa['answer']}")


if __name__ == "__main__":
    main()
