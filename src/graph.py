# src/graph.py
from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from .state import DialogueState
from .profile_store import retrieve_profile_context


def node_select_question(state: DialogueState) -> DialogueState:
    idx = state["current_index"]
    questions = state["diary_questions"]

    if idx >= len(questions):
        state["current_question"] = None
        state["display_question"] = None
        state["done"] = True
        return state

    state["current_question"] = questions[idx]["text"]
    state["display_question"] = None  # verrà riempita dal rewrite
    return state


def node_profile_context(state: DialogueState) -> DialogueState:
    retriever = state["retriever"]

    question = (state.get("current_question") or "").strip()

    prev_answer = ""
    if state.get("qa_history"):
        prev_answer = (state["qa_history"][-1].get("answer") or "").strip()

    query = f"{question}\nContesto risposta precedente: {prev_answer}".strip()
    if not query:
        query = "personalizzazione dialogo"

    state["profile_context"] = retrieve_profile_context(retriever, query)
    return state


def node_rewrite_question(state: DialogueState) -> DialogueState:
    """
    Usa ChatOllama per riscrivere la domanda in modo più chiaro,
    mantenendo lo stesso significato e adattandola al profilo.
    """
    original = (state.get("current_question") or "").strip()
    if not original:
        state["display_question"] = None
        return state

    profile_ctx = (state.get("profile_context") or "").strip()
    prev_answer = ""
    if state.get("qa_history"):
        prev_answer = (state["qa_history"][-1].get("answer") or "").strip()

    system = SystemMessage(
        content=(
            "Sei un assistente che conduce un'intervista con una persona anziana.\n"
            "Devi RISCRIVERE la domanda mantenendo ESATTAMENTE lo stesso significato e obiettivo.\n"
            "Adatta la formulazione alle esigenze del profilo (es. frasi brevi, una cosa per volta).\n"
            "Non aggiungere nuove domande o nuove informazioni.\n"
            "Mantieni il registro formale (Lei).\n"
            "Output: SOLO la domanda riscritta, senza spiegazioni."
        )
    )

    human = HumanMessage(
        content=(
            f"DOMANDA ORIGINALE:\n{original}\n\n"
            f"PROFILO (contesto recuperato):\n{profile_ctx}\n\n"
            f"ULTIMA RISPOSTA UTENTE (se utile):\n{prev_answer}\n"
        )
    )

    llm = state["llm"]
    result = llm.invoke([system, human])
    rewritten = (result.content or "").strip()

    # fallback se il modello risponde vuoto
    state["display_question"] = rewritten if rewritten else original
    return state


def node_ask_and_read(state: DialogueState) -> DialogueState:
    q_original = state.get("current_question")
    q_display = state.get("display_question") or q_original

    if not q_display:
        state["last_user_answer"] = None
        return state

    # Debug: mostra profilo + originale/riscritta
    print("\n[CONTESTO PROFILO - DEBUG]")
    print(state.get("profile_context", ""))

    if q_original and q_display and q_display.strip() != q_original.strip():
        print("\n[DOMANDA ORIGINALE]")
        print(q_original)
        print("\n[DOMANDA RISCRITTA]")
        print(q_display)
    else:
        print(f"\nDOMANDA {state['current_index'] + 1}: {q_display}")

    ans = input("Risposta (scrivi Q per uscire): ").strip()
    if ans.lower() == "q":
        state["done"] = True
        state["last_user_answer"] = None
        return state

    state["last_user_answer"] = ans
    return state


def node_save_and_advance(state: DialogueState) -> DialogueState:
    q = state.get("current_question")
    a = state.get("last_user_answer")

    if q and a:
        state["qa_history"].append({"question": q, "answer": a})

    state["current_index"] += 1
    state["last_user_answer"] = None
    return state


def route_after_select(state: DialogueState) -> str:
    return END if state.get("done") else "profile_context"


def route_after_ask(state: DialogueState) -> str:
    return END if state.get("done") else "save_and_advance"


def build_graph():
    builder = StateGraph(DialogueState)

    builder.add_node("select_question", node_select_question)
    builder.add_node("profile_context", node_profile_context)
    builder.add_node("rewrite_question", node_rewrite_question)
    builder.add_node("ask_and_read", node_ask_and_read)
    builder.add_node("save_and_advance", node_save_and_advance)

    builder.add_edge(START, "select_question")
    builder.add_conditional_edges("select_question", route_after_select)

    builder.add_edge("profile_context", "rewrite_question")
    builder.add_edge("rewrite_question", "ask_and_read")

    builder.add_conditional_edges("ask_and_read", route_after_ask)
    builder.add_edge("save_and_advance", "select_question")

    return builder.compile()
