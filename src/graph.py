# src/graph.py
from __future__ import annotations

import re
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from .state import DialogueState
from .profile_store import retrieve_profile_context


def _cleanup(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"(?is)^\s*ecco.*?:\s*", "", t).strip()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return "\n".join(lines).strip()


def _is_bad_rewrite(candidate: str, original: str) -> bool:
    """Heuristics anti-deriva: se sembra che l'LLM abbia cambiato significato o stile, fallback."""
    if not candidate:
        return True

    c = candidate.lower()

    # vieta il "tu"
    forbidden_tokens = {"tu", "ti", "te", "tua", "tuo", "tue", "tuoi"}
    if any(tok in c.split() for tok in forbidden_tokens):
        return True

    # vieta nomi propri presi dal profilo (esempio: mario)
    if "mario" in c:
        return True

    # frasi tipiche di “re-interpretazione” che cambiano significato
    bad_phrases = [
        "a causa della",
        "considerando",
        "tenendo conto",
        "in base alla sua",
        "dato che",
        "per via della",
        "le è difficile",
        "ha provato a fare qualcosa",
    ]
    if any(p in c for p in bad_phrases):
        return True

    # errori grammaticali frequenti visti (“Lei ha riuscito…”)
    if "lei ha riuscito" in c:
        return True

    # se NON contiene un punto interrogativo e l'originale sì, sospetto
    if ("?" in original) and ("?" not in candidate):
        return True

    return False


def node_select_question(state: DialogueState) -> DialogueState:
    idx = state["current_index"]
    questions = state["diary_questions"]

    if idx >= len(questions):
        state["current_question"] = None
        state["display_question"] = None
        state["done"] = True
        return state

    state["current_question"] = questions[idx]["text"]
    state["display_question"] = None
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
    NON paraprasare: solo semplificazione e spezzatura in frasi brevi.
    Se l'LLM deraglia -> fallback alla domanda originale.
    """
    original = (state.get("current_question") or "").strip()
    if not original:
        state["display_question"] = None
        return state

    profile_ctx = (state.get("profile_context") or "").strip()

    system = SystemMessage(
        content=(
            "Sei un assistente che fa domande a una persona anziana.\n"
            "Devi SOLO SEMPLIFICARE la domanda, non riscriverla liberamente.\n\n"
            "REGOLE (rigide):\n"
            "1) Mantieni lo STESSO significato parola per parola, senza cambiare concetti.\n"
            "2) Puoi SOLO: spezzare in frasi brevi, rendere più chiari i pronomi, togliere ridondanze.\n"
            "3) NON sostituire parole chiave con sinonimi (es. 'preoccupato' resta 'preoccupato').\n"
            "4) Mantieni SEMPRE 'Lei/Le/Suo/Sua'. Non usare mai 'tu'.\n"
            "5) Non inserire nomi propri o dettagli del profilo.\n"
            "6) Se la domanda ha due parti (es. 'Se sì... Se no...'), mantienile entrambe.\n"
            "7) Output: SOLO la domanda semplificata (può essere su più righe)."
        )
    )

    human = HumanMessage(
        content=(
            f"DOMANDA:\n{original}\n\n"
            f"VINCOLI DI STILE (non da citare):\n{profile_ctx}\n"
        )
    )

    llm = state["llm"]
    result = llm.invoke([system, human])
    candidate = _cleanup(result.content or "")

    if _is_bad_rewrite(candidate, original):
        state["display_question"] = original
    else:
        state["display_question"] = candidate

    return state


def node_ask_and_read(state: DialogueState) -> DialogueState:
    q_original = state.get("current_question")
    q_display = state.get("display_question") or q_original

    if not q_display:
        state["last_user_answer"] = None
        return state

    # Debug
    print("\n[CONTESTO PROFILO - DEBUG]")
    print(state.get("profile_context", ""))

    if q_original and q_display and q_display.strip() != q_original.strip():
        print("\n[DOMANDA ORIGINALE]")
        print(q_original)
        print("\n[DOMANDA SEMPLIFICATA]")
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
