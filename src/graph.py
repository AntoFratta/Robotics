# src/graph.py
from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from .state import DialogueState


def node_select_question(state: DialogueState) -> DialogueState:
    """Seleziona la prossima domanda dal path (diary_questions)."""
    idx = state["current_index"]
    questions = state["diary_questions"]

    if idx >= len(questions):
        state["current_question"] = None
        state["done"] = True
        return state

    state["current_question"] = questions[idx]["text"]
    return state


def node_ask_and_read(state: DialogueState) -> DialogueState:
    """Stampa la domanda e legge input utente (CLI)."""
    q = state["current_question"]
    if not q:
        state["last_user_answer"] = None
        return state

    print(f"\nDOMANDA {state['current_index'] + 1}: {q}")
    ans = input("Risposta (scrivi Q per uscire): ").strip()

    if ans.lower() == "q":
        state["done"] = True
        state["last_user_answer"] = None
        return state

    state["last_user_answer"] = ans
    return state


def node_save_and_advance(state: DialogueState) -> DialogueState:
    """Salva Q/A e avanza l'indice."""
    q = state["current_question"]
    a = state["last_user_answer"]

    if q and a:
        state["qa_history"].append({"question": q, "answer": a})

    state["current_index"] += 1
    state["last_user_answer"] = None
    return state


def route_after_select(state: DialogueState) -> str:
    """Se finito -> END, altrimenti vai a chiedere."""
    return END if state.get("done") else "ask_and_read"


def route_after_ask(state: DialogueState) -> str:
    """Se l'utente ha chiuso -> END, altrimenti salva e continua."""
    return END if state.get("done") else "save_and_advance"


def build_graph():
    builder = StateGraph(DialogueState)

    builder.add_node("select_question", node_select_question)
    builder.add_node("ask_and_read", node_ask_and_read)
    builder.add_node("save_and_advance", node_save_and_advance)

    # START -> seleziona la prima domanda
    builder.add_edge(START, "select_question")

    # Selezione -> (END oppure ask)
    builder.add_conditional_edges("select_question", route_after_select)

    # Ask -> (END oppure salva)
    builder.add_conditional_edges("ask_and_read", route_after_ask)

    # Salva -> torna a selezionare la prossima domanda
    builder.add_edge("save_and_advance", "select_question")

    return builder.compile()
