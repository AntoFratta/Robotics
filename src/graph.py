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
from .signal_extractor import should_extract_signals, extract_signals, get_default_signals
from .text_utils import (
    gender_label,
    coerce_gender,
    format_question_for_gender,
    strip_questions,
    strip_labels,
    is_formal_ok,
    trim_to_max_sentences,
)


# Helpers: campi profilo
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


def _get_empathy_profile_data(state: DialogueState) -> dict[str, str]:
    """
    Recupera i campi del profilo rilevanti per personalizzare l'empatia.
    
    Returns:
        Dict con: communication_needs, living_situation, health_goal, routine_info
    """
    communication_needs = _get_profile_field(state, "communication_needs", "")
    living_situation = _get_profile_field(state, "living_situation", "")
    health_goal = _get_profile_field(state, "health_goal", "")
    
    # Routine: combina wakes_up_at e goes_to_sleep_at
    wakes = _get_profile_field(state, "wakes_up_at", "")
    sleeps = _get_profile_field(state, "goes_to_sleep_at", "")
    routine_info = ""
    if wakes and sleeps:
        routine_info = f"Si sveglia alle {wakes}, va a dormire alle {sleeps}"
    elif wakes:
        routine_info = f"Si sveglia alle {wakes}"
    elif sleeps:
        routine_info = f"Va a dormire alle {sleeps}"
    
    return {
        "communication_needs": communication_needs,
        "living_situation": living_situation,
        "health_goal": health_goal,
        "routine_info": routine_info
    }


def _get_last_signals(state: DialogueState) -> dict[str, Any]:
    """
    Recupera i segnali emotivi/tematici dell'ultima risposta.
    
    Returns:
        Dict con: emotion, intensity, themes, confidence
        (oppure valori default se non disponibili)
    """
    signals_list = state.get("signals", [])
    if not signals_list:
        return {
            "emotion": "Neutralità",
            "intensity": "bassa",
            "themes": [],
            "confidence": 0.3
        }
    
    # Prendi l'ultimo
    last_signal = signals_list[-1]
    return last_signal.get("extracted", {
        "emotion": "Neutralità",
        "intensity": "bassa",
        "themes": [],
        "confidence": 0.3
    })


# Helpers: domande di follow-up
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
        # question_id è current_index + 1 (1-indexed)
        state["qa_history"].append({
            "question_id": state.get("current_index", 0) + 1,
            "question": q,
            "answer": a,
            "assistant_reply": ""  # Placeholder, verrà riempito da empathy_bridge
        })

    # Reset dopo aver salvato
    state["last_user_answer"] = None
    state["pending_question"] = None
    
    return state


def node_extract_emotion(state: DialogueState) -> DialogueState:
    """
    Nodo dedicato: estrae l'emozione dall'ultima risposta.
    Output: emotion, intensity salvati nello state.
    """
    if not state.get("qa_history"):
        state["last_emotion"] = "Neutralità"
        state["last_intensity"] = "bassa"
        return state
    
    last_qa = state["qa_history"][-1]
    answer = last_qa.get("answer", "").strip()
    question = last_qa.get("question", "")
    question_id = last_qa.get("question_id", 0)
    
    # Rileva emozione: bilanciare keyword matching e LLM
    keyword_emotion = detect_emotional_theme(answer)
    
    # Fast path per risposte corte
    if not should_extract_signals(answer):
        signals_data = get_default_signals()
    else:
        signals_data = extract_signals(answer, question)
    
    # Strategia bilanciata:
    # - Se keyword rileva E LLM conferma (stessa emozione o Neutralità) → usa keyword
    # - Se keyword rileva MA LLM rileva emozione diversa CON alta confidence → usa LLM
    # - Se solo keyword → usa keyword
    # - Se solo LLM → usa LLM
    llm_emotion = signals_data.get("emotion", "Neutralità")
    llm_confidence = signals_data.get("confidence", 0.3)
    
    if keyword_emotion and llm_emotion != "Neutralità" and llm_emotion != keyword_emotion:
        # Conflitto: keyword vs LLM
        if llm_confidence > 0.7:
            # LLM molto sicuro, usa LLM
            final_emotion = llm_emotion
            final_intensity = signals_data.get("intensity", "media")
        else:
            # LLM incerto, usa keyword (più affidabile per pattern noti)
            final_emotion = keyword_emotion
            final_intensity = "alta"
    elif keyword_emotion:
        # Solo keyword o LLM conferma
        final_emotion = keyword_emotion
        final_intensity = "alta"
    else:
        # Solo LLM
        final_emotion = llm_emotion
        final_intensity = signals_data.get("intensity", "bassa")
    
    # Salva segnali completi per logging
    signals_data["emotion"] = final_emotion
    signals_data["intensity"] = final_intensity
    state["signals"].append({
        "question_id": question_id,
        "extracted": signals_data
    })
    
    # Estrai solo emotion e intensity nello state (per prompt snello)
    state["last_emotion"] = final_emotion
    state["last_intensity"] = final_intensity
    
    return state


def node_retrieve_health_context(state: DialogueState) -> DialogueState:
    """
    Nodo dedicato: recupera contesto di salute dal profilo tramite RAG.
    Output: contesto rilevante salvato nello state.
    """
    retriever = state["retriever"]
    
    # Query basata su ultima risposta
    last_a = ""
    if state.get("qa_history"):
        last_a = (state["qa_history"][-1].get("answer") or "").strip()
    
    if not last_a:
        state["health_context"] = ""
        return state
    
    # Recupera contesto rilevante
    context = retrieve_profile_context(retriever, last_a)
    state["health_context"] = context
    
    return state


def node_empathy_bridge(state: DialogueState) -> DialogueState:
    """
    Genera risposta empatica usando gli output dei nodi precedenti.
    Prompt SNELLO ottimizzato per Small Language Model (Qwen).
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
    
    # Recupera emotion e intensity (output di node_extract_emotion)
    # Se uscito da dialogo libero (branch_count > 0), usa emozione iniziale
    if state.get("branch_count_for_current", 0) > 0 and state.get("initial_emotion"):
        emotion = state.get("initial_emotion", "Neutralità")
        intensity = state.get("initial_intensity", "bassa")
    else:
        emotion = state.get("last_emotion", "Neutralità")
        intensity = state.get("last_intensity", "bassa")
    
    # Recupera contesto salute (output di node_retrieve_health_context)
    health_context = state.get("health_context", "")
    
    # Ultimi 2 messaggi per contesto (ridotto per prompt snello)
    recent_messages = ""
    if state.get("qa_history") and len(state["qa_history"]) > 1:
        last_few = state["qa_history"][-3:-1]  # Solo ultimi 2 (escludi corrente)
        recent_messages = " | ".join([
            f"{qa.get('answer', '')[:30]}..."
            for qa in last_few
        ])
    
    # Contesto follow-up
    original_context = ""
    branch_count = state.get("branch_count_for_current", 0)
    branch_type = state.get("current_branch_type", None)
    
    if branch_count > 0 and len(state.get("qa_history", [])) >= 2:
        prev_qa = state["qa_history"][-2]
        prev_answer = prev_qa.get("answer", "").strip()
        if prev_answer:
            original_context = f"Contesto precedente: '{prev_answer}'\n\n"
            
            negative_themes = ["anxiety_panic", "loneliness", "fear", "sadness"]
            if branch_type in negative_themes:
                original_context += "ATTENZIONE: Tema NEGATIVO, non inventare eventi positivi.\n\n"

    llm = state["llm"]
    
    # ===== FAST PATH: Fallback per risposte ultra-brevi (1 parola) =====
    if len(last_a.split()) == 1:
        import random
        word = last_a.lower().strip()
        
        if word in ["bene", "beni", "buono", "buona", "ok", "si", "sì"]:
            empathy = random.choice([
                "Che bello sentirlo, mi fa davvero piacere.",
                "Sono contento che vada bene.",
                "Mi rallegra sentire che va bene.",
                "Che bella notizia!",
                "Mi fa piacere saperlo."
            ])
        elif word in ["male", "no", "niente", "nulla"]:
            empathy = random.choice([
                "Capisco, mi dispiace sentirlo.",
                "Ti ascolto, sono qui per te.",
                "Comprendo, grazie per avermelo detto.",
                "Mi dispiace, ti ascolto.",
                "Capisco come ti senti."
            ])
        else:
            empathy = random.choice([
                "Ti ascolto con attenzione.",
                "Capisco, grazie per averlo condiviso.",
                "Ti ascolto.",
                "Comprendo."
            ])
        
        if state.get("qa_history"):
            state["qa_history"][-1]["assistant_reply"] = empathy
        
        # Bridge prossima domanda
        if idx + 1 < len(questions):
            gender_lbl_next = gender_label(_get_profile_field(state, "gender", ""))
            raw_next = questions[idx + 1]["text"]
            next_q = format_question_for_gender(raw_next, gender_lbl_next)
            state["assistant_reply"] = f"{empathy}\n\nPer capire meglio: {next_q}"
            print("\nASSISTENTE:")
            print(state["assistant_reply"])
            state["skip_question_print"] = True
        else:
            state["assistant_reply"] = empathy
            print("\nASSISTENTE:")
            print(state["assistant_reply"])
            state["done"] = True
        return state
    
    # ===== PROMPT ULTRA-SNELLO per Small LLM (Qwen) =====
    
    # Costruisci prompt minimalista
    prompt_parts = [
        f"Sei un assistente per il supporto emotivo delle persone anziane. Rispondi in modo breve e empatico a questo messaggio: \"{last_a}\"",
        f"- Emozione rilevata: {emotion}",
    ]
    
    # Aggiungi solo info essenziali
    if intensity == "alta":
        prompt_parts.append("- Intensità rilevata: alta - Tono da usare: tono supportivo e rassicurante")
    
    if recent_messages:
        prompt_parts.append(f"- Messaggi recenti: {recent_messages}")
    
    # Health context: solo se breve e rilevante
    if health_context and "cammin" in last_a.lower():
        health_summary = health_context[:80] if len(health_context) > 80 else health_context
        prompt_parts.append(f"- Informazioni sulla salute: {health_summary}")
    
    if original_context:
        prompt_parts.insert(2, original_context.strip()[:100])  # Max 100 char
    
    # Regole ultra-compatte
    prompt_parts.append(f"Regole: genere {gender_lbl}, 1-2 frasi naturali e calde")
    
    system_prompt = "\n".join(prompt_parts)
    
    system = SystemMessage(content=system_prompt)
    human = HumanMessage(content=f"Domanda: {last_q}\nRisposta: {last_a}")

    # Invoca LLM
    result = llm.invoke([system, human])
    raw = (result.content or "").strip()
    
    empathy = strip_questions(raw)
    empathy = strip_labels(empathy)
    empathy = trim_to_max_sentences(empathy, 2)
    
    # Fallback vari per evitare ripetizione
    if not empathy:
        import random
        fallbacks = [
            "Ti ascolto con attenzione.",
            "Grazie per averlo condiviso con me.",
            "Capisco, sono qui per te.",
            "Ti ringrazio per la tua sincerità.",
            "Grazie per avermelo detto.",
            "Ti ascolto, continua pure.",
            "Sono qui ad ascoltarti.",
            "Capisco quello che mi dici."
        ]
        empathy = random.choice(fallbacks)
    
    # Aggiorna qa_history
    if state.get("qa_history"):
        state["qa_history"][-1]["assistant_reply"] = empathy
    
    # Bridge prossima domanda
    if idx + 1 < len(questions):
        gender_lbl_next = gender_label(_get_profile_field(state, "gender", ""))
        raw_next = questions[idx + 1]["text"]
        next_q = format_question_for_gender(raw_next, gender_lbl_next)
        state["assistant_reply"] = f"{empathy}\n\nPer capire meglio: {next_q}"
        print("\nASSISTENTE:")
        print(state["assistant_reply"])
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
    state["initial_emotion"] = None
    state["initial_intensity"] = None
    
    return state


# -----------------------------
# Nodo Dialogo Libero (sostituisce follow-up/deepening)
# -----------------------------
def node_free_dialogue(state: DialogueState) -> DialogueState:
    """
    Nodo di dialogo libero con loop su sé stesso.
    Gestisce follow-up evasivi e approfondimenti emotivi in modo unificato.
    
    Logica:
    1. Rileva se serve approfondimento (risposta evasiva O emozione forte)
    2. Genera domanda di approfondimento
    3. Loop: continua finché non si raggiunge profondità sufficiente
    4. Poi ritorna a empathy_bridge per passare alla domanda successiva
    """
    # Ottieni ultima risposta
    last_answer = ""
    if state.get("qa_history"):
        last_answer = state["qa_history"][-1].get("answer", "").strip()
    
    if not last_answer:
        return state
    
    # Rileva tipo di approfondimento necessario
    is_evasive = is_evasive_answer(last_answer)
    emotion_theme = detect_emotional_theme(last_answer) if not is_evasive else None
    
    # Se né evasiva né emotiva, skip (non dovrebbe accadere se router funziona)
    if not is_evasive and not emotion_theme:
        return state
    
    followup_data = _load_followup_questions()
    
    # Scegli categoria
    if is_evasive:
        category = "evasive"
        theme_display = "Risposta evasiva"
    else:
        category = emotion_theme
        theme_display = get_theme_display_name(emotion_theme)
    
    templates = followup_data.get(category, [])
    
    if not templates:
        # Fallback: nessun follow-up disponibile
        return state
    
    # Scegli random dalla categoria
    chosen = random.choice(templates)
    followup_q = chosen["template"]
    
    # Salva emozione iniziale solo alla primissima entrata (branch_count == 0)
    # Non sovrascrivere mai durante il loop, anche se cambia categoria
    branch_count = state.get("branch_count_for_current", 0)
    if branch_count == 0 and not state.get("initial_emotion"):
        state["initial_emotion"] = state.get("last_emotion", "Neutralità")
        state["initial_intensity"] = state.get("last_intensity", "bassa")
    
    # Log branch per tracking
    if state.get("session_logger"):
        state["session_logger"].log_branch(
            branch_type=category,
            theme_display=theme_display,
            followup_question=followup_q
        )
    
    # Imposta pending_question per ask_and_read
    state["pending_question"] = followup_q
    state["question_mode"] = "DEEPENING"  # Modalità dialogo libero
    state["branch_count_for_current"] = state.get("branch_count_for_current", 0) + 1
    state["current_branch_type"] = category
    
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
    Router intelligente: decide se entrare in dialogo libero o procedere normale.
    
    NOTA: Usa detect_emotional_theme (keyword) per decisione router.
    Keyword matching può mancare sinonimi, ma è veloce per decisione routing.
    L'estrazione completa (con LLM) avviene in node_extract_emotion.
    
    Logica:
    1. Se già in DEEPENING mode → loop o uscita da free_dialogue
    2. Se già fatto 2+ interazioni → empathy_bridge (forza uscita)
    3. Se risposta evasiva O emozione forte → free_dialogue
    4. Altrimenti → empathy_bridge
    """
    if state.get("done"):
        return END
    
    # Se già in DEEPENING, controlla se continuare il loop
    mode = state.get("question_mode", "MAIN")
    if mode == "DEEPENING":
        # Controlla contatore branch
        branch_count = state.get("branch_count_for_current", 0)
        
        # Limite max: 2 interazioni di dialogo libero
        if branch_count >= 2:
            # Forza uscita: reset mode e vai a empathy_bridge
            state["question_mode"] = "MAIN"
            return "empathy_bridge"
        
        # Ottieni ultima risposta
        last_answer = ""
        if state.get("qa_history"):
            last_answer = state["qa_history"][-1].get("answer", "").strip()
        
        # Se risposta ancora evasiva O emozione ancora forte, continua loop
        if is_evasive_answer(last_answer):
            return "free_dialogue"
        elif detect_emotional_theme(last_answer):
            return "free_dialogue"
        else:
            # Risposta soddisfacente, esci dal loop
            state["question_mode"] = "MAIN"
            return "empathy_bridge"
    
    # Prima volta: controlla se serve dialogo libero
    branch_count = state.get("branch_count_for_current", 0)
    if branch_count >= 1:  # Max 1 entrata in free_dialogue per domanda
        return "empathy_bridge"
    
    # Ottieni ultima risposta
    last_answer = ""
    if state.get("qa_history"):
        last_answer = state["qa_history"][-1].get("answer", "").strip()
    
    if not last_answer:
        return "empathy_bridge"
    
    # Check risposta evasiva O emozione forte
    if is_evasive_answer(last_answer):
        return "free_dialogue"
    
    emotion = detect_emotional_theme(last_answer)
    if emotion:
        return "free_dialogue"
    
    # Risposta normale
    return "empathy_bridge"


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
    
    # Nodi per separazione task
    builder.add_node("extract_emotion", node_extract_emotion)
    builder.add_node("retrieve_health_context", node_retrieve_health_context)
    
    # Nodo dialogo libero unificato (sostituisce follow_up_evasive e emotional_deepening)
    builder.add_node("free_dialogue", node_free_dialogue)

    # Edges
    builder.add_edge(START, "select_question")
    builder.add_conditional_edges("select_question", route_after_select)

    builder.add_edge("profile_context", "ask_and_read")
    builder.add_conditional_edges("ask_and_read", route_after_ask)

    # Flusso: save → extract_emotion → retrieve_health_context → router
    builder.add_edge("save_current_answer", "extract_emotion")
    builder.add_edge("extract_emotion", "retrieve_health_context")
    builder.add_conditional_edges("retrieve_health_context", route_answer_type)
    
    # Free dialogue: può loopare su sé stesso O andare ad ask_and_read
    builder.add_edge("free_dialogue", "ask_and_read")
    
    builder.add_conditional_edges("empathy_bridge", route_after_bridge)
    builder.add_edge("advance_to_next_question", "select_question")

    return builder.compile()
