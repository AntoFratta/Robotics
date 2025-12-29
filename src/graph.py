# src/graph.py
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from .state import DialogueState
from .profile_store import retrieve_profile_context
from .response_classifier import is_evasive_answer, detect_emotional_theme, get_theme_display_name
from .utils import (
    gender_label,
    coerce_gender,
    format_question_for_gender,
    strip_questions,
    strip_labels,
    is_formal_ok,
    trim_to_max_sentences,
)


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


# -----------------------------
# Helpers: follow-up questions
# -----------------------------
_FOLLOWUP_CACHE: Optional[dict] = None


def _load_followup_questions() -> dict:
    """Carica follow_up_questions.json (con cache)"""
    global _FOLLOWUP_CACHE
    if _FOLLOWUP_CACHE is not None:
        return _FOLLOWUP_CACHE
    
    # Trova il file nella stessa directory del profilo (data/)
    current_file = Path(__file__)
    data_dir = current_file.parents[1] / "data"
    followup_path = data_dir / "follow_up_questions.json"
    
    _FOLLOWUP_CACHE = json.loads(followup_path.read_text(encoding="utf-8"))
    return _FOLLOWUP_CACHE



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

    # Ottieni genere per formattare domanda
    gender_lbl = gender_label(_get_profile_field(state, "gender", ""))
    
    raw_question = questions[idx]["text"]
    formatted_question = format_question_for_gender(raw_question, gender_lbl)
    
    state["current_question"] = formatted_question
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
    """
    Input utente - supporta modalità MAIN/FOLLOWUP/DEEPENING.
    Usa pending_question se presente, altrimenti current_question.
    """
    mode = state.get("question_mode", "MAIN")
    
    # Se c'è pending_question, usa quella (follow-up/deepening)
    q = state.get("pending_question")
    if not q:
        q = state.get("current_question")
    
    if not q:
        state["last_user_answer"] = None
        return state
    
    # Stampa diversa in base al mode
    if mode == "MAIN":
        if not state.get("skip_question_print", False):
            print(f"\nDOMANDA {state['current_index'] + 1}: {q}")
    else:
        # Follow-up o deepening: senza numero, come se fosse assistente
        print(f"\n{q}")
    
    ans = input("Risposta (scrivi Q per uscire): ").strip()
    
    # NON resettiamo pending_question qui - serve a save_current_answer
    # Verrà resettata dopo il salvataggio
    
    if ans.lower() == "q":
        state["done"] = True
        state["last_user_answer"] = None
        return state
    
    state["last_user_answer"] = ans
    state["skip_question_print"] = False
    return state


def node_save_current_answer(state: DialogueState) -> DialogueState:
    # Usa pending_question se presente (follow-up), altrimenti current_question
    q = state.get("pending_question")
    if not q:
        q = state.get("current_question")
    
    a = state.get("last_user_answer")

    if q and a:
        # Salva Q&A - assistant_reply verrà aggiunto dopo in empathy_bridge
        state["qa_history"].append({
            "question": q,
            "answer": a,
            "assistant_reply": ""  # Placeholder, verrà riempito da empathy_bridge
        })

    # Reset dopo aver salvato
    state["last_user_answer"] = None
    state["pending_question"] = None
    
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
    gender_lbl = gender_label(_get_profile_field(state, "gender", ""))
    
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
            
            "Risposta: 'Ho visto i miei nipoti'\n"
            "→ BUONO: 'Che bello che abbia potuto vedere i suoi nipoti! Questi momenti sono davvero preziosi.'\n"
            "→ BUONO: 'I momenti con i nipoti possono portare tanta gioia.'\n\n"
            
            "Risposta: 'Ho avuto attacchi di panico'\n"
            "→ BUONO: 'Gli attacchi di panico sono esperienze molto intense. Mi rendo conto di quanto possano essere difficili.'\n"
            "→ BUONO: 'Immagino quanto possa essere stato difficile affrontare quegli attacchi di panico.'\n\n"
            
            "Risposta: 'Felicità'\n"
            "→ BUONO: 'La felicità è un'emozione bellissima. È importante riconoscere quando ci sentiamo così.'\n\n"
            
            "❌ ESEMPI DA EVITARE (NON ripetere sempre lo stesso pattern!):\n"
            "→ EVITA: 'Capisco che... È normale/comprensibile...' [TROPPO RIPETITIVO]\n"
            "→ EVITA: Iniziare SEMPRE con 'Capisco che'\n"
            "→ EVITA: Finire SEMPRE con 'È normale/comprensibile'\n\n"
            
            "⚠️ VARIETÀ OBBLIGATORIA:\n"
            "- Cambia il modo di iniziare: 'Che bello', 'Immagino', 'Mi rendo conto', '[Elemento concreto] può essere...'\n"
            "- Cambia il modo di validare: non sempre 'è normale', usa anche 'è comprensibile', 'è importante', 'sono momenti preziosi'\n"
            "- VARIA LA STRUTTURA: non seguire sempre lo stesso schema\n\n"
            
            "REGOLE CRITICHE:\n"
            f"- Usa SEMPRE e SOLO 'Lei' (mai tu/ti/te/tuo/tua)\n"
            f"- Genere: {gender_lbl} - usa accordi corretti ('Lei è riuscito/a', non 'ha riuscito')\n"
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

    empathy = strip_questions(raw)
    empathy = strip_labels(empathy)
    empathy = trim_to_max_sentences(empathy, 3)
    empathy = coerce_gender(empathy, gender_lbl)

    # Se fallisce validazione formale, prova a rigenerare con prompt più esplicito
    if empathy and not is_formal_ok(empathy):
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
        empathy = strip_questions(raw)
        empathy = strip_labels(empathy)
        empathy = trim_to_max_sentences(empathy, 3)
        empathy = coerce_gender(empathy, gender_lbl)

    # Fallback contestuale solo se entrambi i tentativi falliscono
    if (not empathy) or (not is_formal_ok(empathy)):
        # Fallback più specifico basato sul tipo di risposta
        if len(last_a.split()) <= 2:
            empathy = "Comprendo. La ringrazio per averlo condiviso con me."
        else:
            empathy = "Capisco. La ringrazio per aver condiviso questa esperienza."

    # IMPORTANTE: Aggiorna l'ultimo qa_history con l'assistant_reply generato
    if state.get("qa_history") and len(state["qa_history"]) > 0:
        state["qa_history"][-1]["assistant_reply"] = empathy

    # Bridge alla prossima domanda
    if idx + 1 < len(questions):
        gender_lbl_for_next = gender_label(_get_profile_field(state, "gender", ""))
        
        raw_next_q = questions[idx + 1]["text"]
        next_q = format_question_for_gender(raw_next_q, gender_lbl_for_next)
        
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
    
    # Reset branch counter per la nuova domanda
    state["branch_count_for_current"] = 0
    state["current_branch_type"] = None
    state["question_mode"] = "MAIN"
    
    return state


# -----------------------------
# Guided Path: Branching nodes
# -----------------------------
def node_follow_up_evasive(state: DialogueState) -> DialogueState:
    """
    Prepara follow-up per risposte evasive ("niente", "non ricordo", etc.)
    """
    followup_data = _load_followup_questions()
    templates = followup_data.get("evasive", [])
    
    if not templates:
        # Fallback: nessun follow-up disponibile
        return state
    
    # Scegli random dalla categoria 'evasive'
    chosen = random.choice(templates)
    followup_q = chosen["template"]
    
    # Debug: mostra rilevamento
    print("\n[DEBUG] Rilevata risposta evasiva → follow-up")
    
    # Log branch per tracking
    if state.get("session_logger"):
        state["session_logger"].log_branch(
            branch_type="evasive",
            theme_display="Risposta evasiva",
            followup_question=followup_q
        )
    
    # Imposta nello state
    state["pending_question"] = followup_q
    state["question_mode"] = "FOLLOWUP"
    state["branch_count_for_current"] = state.get("branch_count_for_current", 0) + 1
    state["current_branch_type"] = "evasive"
    
    return state


def node_emotional_deepening(state: DialogueState) -> DialogueState:
    """
    Prepara domanda di approfondimento per temi emotivi forti.
    IMPORTANTE: Rileva il tema dalla risposta utente, perché i router non possono modificare lo state.
    """
    # Rileva il tema dalla risposta dell'utente (ultima risposta in qa_history)
    theme = None
    if state.get("qa_history"):
        last_answer = state["qa_history"][-1].get("answer", "").strip()
        theme = detect_emotional_theme(last_answer)
    
    if not theme:
        # Fallback: nessun tema rilevato, skip
        return state
    
    followup_data = _load_followup_questions()
    templates = followup_data.get(theme, [])
    
    if not templates:
        # Fallback: nessun follow-up disponibile per questo tema
        return state
    
    chosen = random.choice(templates)
    followup_q = chosen["template"]
    
    # Debug: mostra rilevamento
    theme_name = get_theme_display_name(theme)
    print(f"\n[DEBUG] Rilevato tema emotivo: {theme_name} → approfondimento")
    
    # Log branch per tracking
    if state.get("session_logger"):
        state["session_logger"].log_branch(
            branch_type=theme,
            theme_display=theme_name,
            followup_question=followup_q
        )
    
    # Imposta nello state
    state["pending_question"] = followup_q
    state["question_mode"] = "DEEPENING"
    state["branch_count_for_current"] = state.get("branch_count_for_current", 0) + 1
    state["current_branch_type"] = theme  # Salva per tracking
    
    return state


# -----------------------------
# Routers
# -----------------------------
def route_after_select(state: DialogueState) -> str:
    return END if state.get("done") else "profile_context"


def route_after_ask(state: DialogueState) -> str:
    return END if state.get("done") else "save_current_answer"


def route_answer_type(state: DialogueState) -> str:
    """
    Router intelligente: decide se fare follow-up, deepening o procedere normale.
    
    Logica:
    1. Se già in FOLLOWUP/DEEPENING mode → empathy_bridge (skip classification)
    2. Se già fatto 1 follow-up per questa domanda → empathy_bridge
    3. Se risposta evasiva → follow_up_evasive
    4. Se tema emotivo → emotional_deepening
    5. Altrimenti → empathy_bridge
    """
    if state.get("done"):
        return END
    
    # IMPORTANTE: Se siamo già in FOLLOWUP/DEEPENING, non riclassificare
    # La risposta è a un follow-up, quindi procedi sempre a empathy_bridge
    mode = state.get("question_mode", "MAIN")
    if mode in ["FOLLOWUP", "DEEPENING"]:
        # Reset mode a MAIN per la prossima domanda del diario
        state["question_mode"] = "MAIN"
        return "empathy_bridge"
    
    # Limite max 1 follow-up per domanda
    branch_count = state.get("branch_count_for_current", 0)
    if branch_count >= 1:
        return "empathy_bridge"
    
    # Ottieni ultima risposta
    last_answer = ""
    if state.get("qa_history"):
        last_answer = state["qa_history"][-1].get("answer", "").strip()
    
    if not last_answer:
        return "empathy_bridge"
    
    # 1. Check risposta evasiva (priorità alta)
    if is_evasive_answer(last_answer):
        return "follow_up_evasive"
    
    # 2. Check tema emotivo
    theme = detect_emotional_theme(last_answer)
    if theme:
        # Nota: Il tema verrà ri-rilevato in node_emotional_deepening
        # (i router non possono mutare lo state in LangGraph)
        return "emotional_deepening"
    
    # 3. Nessuna diramazione → normale
    return "empathy_bridge"


# DEPRECATO: ora usiamo route_answer_type come router dopo save
# def route_after_save(state: DialogueState) -> str:
#     return END if state.get("done") else "empathy_bridge"


def route_after_bridge(state: DialogueState) -> str:
    return END if state.get("done") else "advance_to_next_question"


def build_graph():
    builder = StateGraph(DialogueState)

    # Nodi esistenti
    builder.add_node("select_question", node_select_question)
    builder.add_node("profile_context", node_profile_context)
    builder.add_node("ask_and_read", node_ask_and_read)
    builder.add_node("save_current_answer", node_save_current_answer)
    builder.add_node("empathy_bridge", node_empathy_bridge)
    builder.add_node("advance_to_next_question", node_advance_to_next_question)
    
    # NUOVI nodi per guided path
    builder.add_node("follow_up_evasive", node_follow_up_evasive)
    builder.add_node("emotional_deepening", node_emotional_deepening)

    # Edges
    builder.add_edge(START, "select_question")
    builder.add_conditional_edges("select_question", route_after_select)

    builder.add_edge("profile_context", "ask_and_read")
    builder.add_conditional_edges("ask_and_read", route_after_ask)

    # NUOVO: router intelligente dopo save_current_answer
    builder.add_conditional_edges("save_current_answer", route_answer_type)
    
    # Follow-up e deepening ritornano ad ask_and_read
    builder.add_edge("follow_up_evasive", "ask_and_read")
    builder.add_edge("emotional_deepening", "ask_and_read")
    
    builder.add_conditional_edges("empathy_bridge", route_after_bridge)
    builder.add_edge("advance_to_next_question", "select_question")

    return builder.compile()
