# src/graph.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from .state import DialogueState
from .profile_store import retrieve_profile_context


# -----------------------------
# Helpers: profile fields
# -----------------------------
def _load_profile_json(profile_path: str) -> dict[str, Any]:
    p = Path(profile_path)
    return json.loads(p.read_text(encoding="utf-8"))


def _get_profile_field(state: DialogueState, field: str, default: str = "") -> str:
    try:
        prof = _load_profile_json(state["profile_path"])
        v = prof.get(field, default)
        return str(v).strip() if v is not None else default
    except Exception:
        return default


def _gender_label(gender: str) -> str:
    g = (gender or "").strip().upper()
    if g in {"M", "MALE", "UOMO", "MASCHIO", "MASCHILE"}:
        return "MASCHILE"
    if g in {"F", "FEMALE", "DONNA", "FEMMINA", "FEMMINILE"}:
        return "FEMMINILE"
    return "NON_SPECIFICATO"


def _coerce_gender(text: str, gender_label: str) -> str:
    if gender_label not in {"MASCHILE", "FEMMINILE"}:
        return text

    if gender_label == "MASCHILE":
        repl = {
            "preoccupata": "preoccupato",
            "stressata": "stressato",
            "determinata": "determinato",
            "legata": "legato",
            "stata": "stato",
            "serena": "sereno",
            "tranquilla": "tranquillo",
            "angosciata": "angosciato",
            "spaventata": "spaventato",
            "stanca": "stanco",
        }
    else:
        repl = {
            "preoccupato": "preoccupata",
            "stressato": "stressata",
            "determinato": "determinata",
            "legato": "legata",
            "stato": "stata",
            "sereno": "serena",
            "tranquillo": "tranquilla",
            "angosciato": "angosciata",
            "spaventato": "spaventata",
            "stanco": "stanca",
        }

    out = text
    for k, v in repl.items():
        out = re.sub(rf"\b{k}\b", v, out, flags=re.IGNORECASE)
    return out


# -----------------------------
# Cleanup
# -----------------------------
def _strip_questions(text: str) -> str:
    """
    Rimuove righe che sono domande (iniziano con parole interrogative o finiscono con ?).
    Ma NON rimuove righe che contengono '?' nel mezzo come parte di una frase descrittiva.
    """
    lines = []
    for ln in text.splitlines():
        ln_stripped = ln.strip()
        # Salta righe vuote
        if not ln_stripped:
            continue
        # Rimuovi solo se è CHIARAMENTE una domanda
        # - Inizia con parola interrogativa
        # - OPPURE finisce con '?' (domanda diretta)
        if re.match(r'^\s*(come|cosa|quando|dove|perché|perchè|chi|quale|quanto)\b', ln_stripped.lower()):
            continue
        if ln_stripped.endswith('?'):
            continue
        lines.append(ln_stripped)
    return '\n'.join(lines).strip()


def _strip_labels(text: str) -> str:
    # rimuove etichette tipo "Riflesso:", "Validazione:", ecc.
    out = text
    out = re.sub(r"(?im)^\s*(riflesso|validazione|valido|valida)\s*:\s*", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _is_formal_ok(text: str) -> bool:
    low = text.lower()
    forbidden = [
        r"\btu\b", r"\bti\b", r"\bte\b", r"\btua\b", r"\btuo\b",
        r"\bstai\b", r"\bsei\b", r"\bper te\b"
    ]
    return not any(re.search(p, low) for p in forbidden)


def _trim_to_max_sentences(text: str, max_sentences: int = 3) -> str:
    s = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", s)
    return " ".join(parts[:max_sentences]).strip()


# -----------------------------
# Graph nodes
# -----------------------------
def node_select_question(state: DialogueState) -> DialogueState:
    idx = state["current_index"]
    questions = state["diary_questions"]

    if idx >= len(questions):
        state["current_question"] = None
        state["done"] = True
        return state

    state["current_question"] = questions[idx]["text"]
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


def node_ask_and_read(state: DialogueState) -> DialogueState:
    q = state.get("current_question")
    if not q:
        state["last_user_answer"] = None
        return state

    if not state.get("skip_question_print", False):
        print(f"\nDOMANDA {state['current_index'] + 1}: {q}")

    ans = input("Risposta (scrivi Q per uscire): ").strip()

    if ans.lower() == "q":
        state["done"] = True
        state["last_user_answer"] = None
        return state

    state["last_user_answer"] = ans
    state["skip_question_print"] = False
    return state


def node_save_current_answer(state: DialogueState) -> DialogueState:
    q = state.get("current_question")
    a = state.get("last_user_answer")

    if q and a:
        state["qa_history"].append({"question": q, "answer": a})

    state["last_user_answer"] = None
    return state


def node_empathy_bridge(state: DialogueState) -> DialogueState:
    """
    LLM produce SOLO empatia (1-3 frasi). Poi noi agganciamo la prossima domanda del diario.
    Ora include contesto profilo e prompt migliorato con esempi.
    """
    if state.get("done"):
        state["assistant_reply"] = None
        return state

    idx = state["current_index"]
    questions = state["diary_questions"]

    last_q = (state.get("current_question") or "").strip()
    last_a = ""
    if state.get("qa_history"):
        last_a = (state["qa_history"][-1].get("answer") or "").strip()

    # gender
    gender = _gender_label(_get_profile_field(state, "gender", ""))
    
    # NUOVO: recupera il contesto profilo
    profile_context = state.get("profile_context", "").strip()

    llm = state["llm"]

    # PROMPT MIGLIORATO con varietà e esempi negativi
    system = SystemMessage(
        content=(
            "Sei un assistente empatico per pazienti anziani.\n"
            "Scrivi UNA risposta breve e naturale (1-2 frasi) che mostri comprensione.\n\n"
            
            "COME RISPONDERE:\n"
            "1. Riconosci l'emozione o esperienza della persona\n"
            "2. Riprendi 1 elemento concreto che ha detto (azione, emozione, dettaglio)\n"
            "3. Mostra che è normale/comprensibile quello che sta provando\n\n"
            
            "ESEMPI DI BUONE RISPOSTE (varia lo stile):\n\n"
            
            "Domanda: 'Come si è sentito?'\n"
            "Risposta: 'Ho avuto attacchi di panico'\n"
            "→ Buono 1: 'Gli attacchi di panico possono essere davvero difficili da affrontare. "
            "È comprensibile sentirsi sopraffatti.'\n"
            "→ Buono 2: 'Capisco che gli attacchi di panico Le abbiano causato molto disagio. "
            "Sono esperienze molto intense.'\n\n"
            
            "Domanda: 'Che colore userebbe per il suo umore?'\n"
            "Risposta: 'Grigio'\n"
            "→ Buono: 'Il grigio può riflettere un momento di incertezza. "
            "Mi sembra di capire come si sente.'\n\n"
            
            "ESEMPIO NEGATIVO (da NON fare):\n"
            "❌ 'Comprendo che lei ha riuscito a fare questo' (errore grammaticale: 'ha riuscito' → 'è riuscito/a')\n"
            "❌ 'Mario, capisco che tu...' (NON usare nome, NON usare 'tu')\n"
            "❌ 'È normale. Come si sente ora?' (NON fare domande)\n\n"
            
            "VARIAZIONI per iniziare (non sempre 'Comprendo che'):\n"
            "- 'Capisco che...'\n"
            "- 'Mi rendo conto che...'\n"
            "- '[Elemento concreto] può essere...'\n"
            "- 'È comprensibile che...'\n"
            "- 'Immagino quanto...'\n\n"
            
            "REGOLE CRITICHE:\n"
            f"- Usa SEMPRE e SOLO 'Lei' (mai tu/ti/te/tuo/tua)\n"
            f"- Genere: {gender} - usa accordi corretti ('Lei è riuscito/a', non 'ha riuscito')\n"
            "- NON fare domande\n"
            "- NON usare etichette come 'Validazione:' o 'Riflesso:'\n"
            "- NON dire 'Lei ha detto' o 'Lei ha risposto'\n"
            "- NON usare il nome del paziente\n"
            "- Linguaggio caldo e naturale, non clinico\n"
        )
    )

    # Migliora la gestione di risposte brevi
    short_hint = ""
    if len(last_a.split()) <= 2:
        short_hint = (
            "NOTA: La risposta è molto breve. Sviluppa comunque una risposta empatica "
            "riprendendo il contesto della domanda e validando comunque l'esperienza.\n\n"
        )

    human = HumanMessage(
        content=(
            f"{short_hint}"
            f"Domanda del diario:\n{last_q}\n\n"
            f"Risposta dell'utente:\n{last_a}"
        )
    )

    # Primo tentativo
    result = llm.invoke([system, human])
    raw = (result.content or "").strip()

    empathy = _strip_questions(raw)
    empathy = _strip_labels(empathy)
    empathy = _trim_to_max_sentences(empathy, 3)
    empathy = _coerce_gender(empathy, gender)

    # Se fallisce validazione formale, prova a rigenerare con prompt più esplicito
    if empathy and not _is_formal_ok(empathy):
        # Tentativo 2: rigenera con avvertimento esplicito
        strict_system = SystemMessage(
            content=(
                system.content + 
                "\n\n⚠️ ATTENZIONE CRITICA: Usa SOLO 'Lei', MAI 'tu/ti/te/tuo/tua'. "
                "Controlla attentamente ogni frase prima di rispondere."
            )
        )
        result = llm.invoke([strict_system, human])
        raw = (result.content or "").strip()
        empathy = _strip_questions(raw)
        empathy = _strip_labels(empathy)
        empathy = _trim_to_max_sentences(empathy, 3)
        empathy = _coerce_gender(empathy, gender)

    # Fallback contestuale solo se entrambi i tentativi falliscono
    if (not empathy) or (not _is_formal_ok(empathy)):
        # Fallback più specifico basato sul tipo di risposta
        if len(last_a.split()) <= 2:
            empathy = "Comprendo. La ringrazio per averlo condiviso con me."
        else:
            empathy = "Capisco. La ringrazio per aver condiviso questa esperienza."

    # Bridge alla prossima domanda
    if idx + 1 < len(questions):
        next_q = questions[idx + 1]["text"]
        state["assistant_reply"] = f"{empathy}\n\nPer capire meglio: {next_q}"

        print("\nASSISTENTE:")
        print(state["assistant_reply"])

        # così node_ask_and_read non ristampa DOMANDA 2 ecc.
        state["skip_question_print"] = True
    else:
        state["assistant_reply"] = empathy
        print("\nASSISTENTE:")
        print(state["assistant_reply"])
        state["done"] = True

    return state


def node_advance_to_next_question(state: DialogueState) -> DialogueState:
    state["current_index"] += 1
    idx = state["current_index"]

    if idx < len(state["diary_questions"]):
        state["current_question"] = state["diary_questions"][idx]["text"]
    else:
        state["current_question"] = None
        state["done"] = True

    state["assistant_reply"] = None
    return state


# -----------------------------
# Routers
# -----------------------------
def route_after_select(state: DialogueState) -> str:
    return END if state.get("done") else "profile_context"


def route_after_ask(state: DialogueState) -> str:
    return END if state.get("done") else "save_current_answer"


def route_after_save(state: DialogueState) -> str:
    return END if state.get("done") else "empathy_bridge"


def route_after_bridge(state: DialogueState) -> str:
    return END if state.get("done") else "advance_to_next_question"


def build_graph():
    builder = StateGraph(DialogueState)

    builder.add_node("select_question", node_select_question)
    builder.add_node("profile_context", node_profile_context)
    builder.add_node("ask_and_read", node_ask_and_read)
    builder.add_node("save_current_answer", node_save_current_answer)
    builder.add_node("empathy_bridge", node_empathy_bridge)
    builder.add_node("advance_to_next_question", node_advance_to_next_question)

    builder.add_edge(START, "select_question")
    builder.add_conditional_edges("select_question", route_after_select)

    builder.add_edge("profile_context", "ask_and_read")
    builder.add_conditional_edges("ask_and_read", route_after_ask)

    builder.add_conditional_edges("save_current_answer", route_after_save)
    builder.add_conditional_edges("empathy_bridge", route_after_bridge)

    builder.add_edge("advance_to_next_question", "select_question")

    return builder.compile()
