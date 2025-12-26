# src/graph.py
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from .state import DialogueState
from .profile_store import retrieve_profile_context


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


def node_profile_context(state: DialogueState) -> DialogueState:
    """
    Recupera dal profilo i campi utili per personalizzare la domanda corrente.
    Retriever già pronto nello state (ottimizzazione).
    """
    retriever = state["retriever"]
    q = state.get("current_question") or "personalizzazione dialogo"
    state["profile_context"] = retrieve_profile_context(retriever, q)
    return state


def node_ask_and_read(state: DialogueState) -> DialogueState:
    """Stampa la domanda e legge input utente (CLI)."""
    q = state["current_question"]
    if not q:
        state["last_user_answer"] = None
        return state

    # Debug: mostra cosa è stato recuperato dal profilo per questa domanda
    print("\n[CONTESTO PROFILO - DEBUG]")
    print(state.get("profile_context", ""))

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
    """Se finito -> END, altrimenti vai al recupero profilo."""
    return END if state.get("done") else "profile_context"


def route_after_ask(state: DialogueState) -> str:
    """Se l'utente ha chiuso -> END, altrimenti salva e continua."""
    return END if state.get("done") else "save_and_advance"


def build_graph():
    builder = StateGraph(DialogueState)

    builder.add_node("select_question", node_select_question)
    builder.add_node("profile_context", node_profile_context)
    builder.add_node("ask_and_read", node_ask_and_read)
    builder.add_node("save_and_advance", node_save_and_advance)

    # START -> seleziona la prima domanda
    builder.add_edge(START, "select_question")

    # select_question -> (END oppure profile_context)
    builder.add_conditional_edges("select_question", route_after_select)

    # profile_context -> ask_and_read
    builder.add_edge("profile_context", "ask_and_read")

    # ask_and_read -> (END oppure save_and_advance)
    builder.add_conditional_edges("ask_and_read", route_after_ask)

    # save_and_advance -> select_question (loop)
    builder.add_edge("save_and_advance", "select_question")

    return builder.compile()
