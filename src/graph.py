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
            "emotion": "neutro/misto",
            "intensity": "bassa",
            "themes": [],
            "confidence": 0.3
        }
    
    # Prendi l'ultimo
    last_signal = signals_list[-1]
    return last_signal.get("extracted", {
        "emotion": "neutro/misto",
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


def node_extract_signals(state: DialogueState) -> DialogueState:
    """
    Estrae segnali emotivi/tematici dall'ultima risposta.
    Fast path per risposte corte (< 5 parole).
    """
    # Ottieni ultima risposta
    if not state.get("qa_history"):
        return state
    
    last_qa = state["qa_history"][-1]
    answer = last_qa.get("answer", "").strip()
    question = last_qa.get("question", "")
    question_id = last_qa.get("question_id", 0)
    
    # Fast path: skip LLM per risposte corte
    if not should_extract_signals(answer):
        signals_data = get_default_signals()
    else:
        # Estrazione LLM completa
        signals_data = extract_signals(answer, question)
    
    # Aggiungi a state
    state["signals"].append({
        "question_id": question_id,
        "extracted": signals_data
    })
    
    return state


def node_empathy_bridge(state: DialogueState) -> DialogueState:
    """
    LLM produce empatia PERSONALIZZATA guidata da profilo + segnali.
    La risposta è adattata a: communication_needs, living_situation, health_goal, emotion/intensity.
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
    
    # Recupera dati profilo per personalizzazione
    profile_data = _get_empathy_profile_data(state)
    
    # Recupera segnali emotivi/tematici
    signals = _get_last_signals(state)
    
    # IMPORTANTE: Se siamo in un follow-up, recupera il contesto della risposta originale
    # per evitare che il LLM inventi interpretazioni positive per risposte brevi
    original_context = ""
    branch_count = state.get("branch_count_for_current", 0)
    branch_type = state.get("current_branch_type", None)
    
    if branch_count > 0 and len(state.get("qa_history", [])) >= 2:
        # Prendi la risposta precedente (quella che ha triggerato il follow-up)
        prev_qa = state["qa_history"][-2]
        prev_answer = prev_qa.get("answer", "").strip()
        if prev_answer:
            original_context = f"Contesto: L'utente aveva risposto in precedenza '{prev_answer}'\n\n"
            
            # Se follow-up di tema negativo, aggiungi warning esplicito
            negative_themes = ["anxiety_panic", "loneliness", "fear", "sadness"]
            if branch_type in negative_themes:
                original_context += (
                    "IMPORTANTE: Questa è una risposta a follow-up su tema NEGATIVO (ansia/solitudine/paura).\n"
                    "Il contesto originale era NEGATIVO. La risposta corrente NON è un evento positivo.\n\n"
                    "ESEMPI CORRETTI:\n"
                    "- Utente solo dice 'vedere i miei figli' → 'Comprendo quanto Le manchi il contatto con loro.'\n"
                    "- NON dire: 'quanto sia stato bello vederli' (NON li ha visti!)\n"
                    "- NON dire: 'che bello' o 'mi fa piacere' (il contesto è NEGATIVO)\n\n"
                )

    llm = state["llm"]
    
    # ===== FAST PATH 1: Fallback per risposte ultra-brevi (1 parola) =====
    # Per risposte di 1 sola parola, usa template fisso invece di LLM
    # L'LLM tende a inventare interpretazioni anche con istruzioni severe
    if len(last_a.split()) == 1:
        word = last_a.lower().strip()
        
        # Parole positive generiche
        if word in ["bene", "beni", "buono", "buona", "ok", "si", "sì"]:
            empathy = "Mi fa piacere sentirlo."
        # Parole negative generiche
        elif word in ["male", "no", "niente", "nulla"]:
            empathy = "Comprendo."
        # Altro (tempi, nomi, ecc.)
        else:
            empathy = "Capisco."
        
        # Aggiorna qa_history e ritorna senza chiamare LLM
        if state.get("qa_history") and len(state["qa_history"]) > 0:
            state["qa_history"][-1]["assistant_reply"] = empathy
        
        # Bridge alla prossima domanda
        idx = state["current_index"]
        questions = state["diary_questions"]
        if idx + 1 < len(questions):
            gender_lbl_for_next = gender_label(_get_profile_field(state, "gender", ""))
            raw_next_q = questions[idx + 1]["text"]
            next_q = format_question_for_gender(raw_next_q, gender_lbl_for_next)
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
    
    # ===== FAST PATH 2: Template fissi per follow-up su temi negativi =====
    # L'LLM confonde desideri con eventi anche con warning forti
    # Usa template sicuri per follow-up di ansia/solitudine/paura
    branch_count = state.get("branch_count_for_current", 0)
    branch_type = state.get("current_branch_type", None)
    negative_themes = ["anxiety_panic", "loneliness", "fear", "sadness"]
    
    if branch_count > 0 and branch_type in negative_themes:
        # Template basati su pattern comuni
        last_a_lower = last_a.lower()
        
        # Pattern 1: Menzione di persone care
        persone_care = ["figli", "figlio", "figlia", "famiglia", "nipoti", "moglie", "marito", "amici", "parenti", "genitori"]
        ha_persona = any(person in last_a_lower for person in persone_care)
        
        # Pattern 2: Verbi di contatto/desiderio
        verbi_contatto = ["veder", "sentir", "parlar", "chiamar", "visita", "stare con", "parlare con"]
        verbi_desiderio = ["vorrei", "mi piacerebbe", "desider", "spero", "voglio", "ho bisogno"]
        
        ha_verbo_contatto = any(verb in last_a_lower for verb in verbi_contatto)
        ha_verbo_desiderio = any(verb in last_a_lower for verb in verbi_desiderio)
        
        # Pattern 3: Tempi della giornata (risposta a "quando/quale momento")
        tempi_giornata = ["mattina", "mattino", "pomeriggio", "sera", "notte", "giorno", "tutto il giorno"]
        ha_tempo = any(tempo in last_a_lower for tempo in tempi_giornata)
        
        # LOGICA PRIORITARIA:
        # 1. Se risponde con tempo della giornata (risposta diretta alla domanda sul "quando")
        if ha_tempo and len(last_a.split()) <= 8:
            # Risposta breve con tempo = sta rispondendo alla domanda "quando è più difficile"
            empathy = "Capisco quanto possa essere difficile."
        # 2. Se SOLO menziona persone + desiderio (senza tempo, es: "vedere i miei figli")
        elif ha_persona and (ha_verbo_contatto or ha_verbo_desiderio) and not ha_tempo:
            empathy = "Comprendo quanto sia importante per Lei il contatto con i Suoi cari."
        # 3. Se solo verbo senza persona
        elif ha_verbo_contatto or ha_verbo_desiderio:
            empathy = "Capisco quanto possa essere difficile."
        # 4. Risposta breve generica
        elif len(last_a.split()) <= 4:
            empathy = "Capisco quanto possa essere difficile."
        else:
            # Risposta lunga e complessa, lascia al LLM con warning
            empathy = None
        
        # Se abbiamo un template, usalo e ritorna
        if empathy:
            if state.get("qa_history") and len(state["qa_history"]) > 0:
                state["qa_history"][-1]["assistant_reply"] = empathy
            
            idx = state["current_index"]
            questions = state["diary_questions"]
            if idx + 1 < len(questions):
                gender_lbl_for_next = gender_label(_get_profile_field(state, "gender", ""))
                raw_next_q = questions[idx + 1]["text"]
                next_q = format_question_for_gender(raw_next_q, gender_lbl_for_next)
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

    # ===== BUILD PERSONALIZED PROMPT =====
    
    # Base rules (sempre presenti)
    base_rules = (
        "Sei un assistente empatico per pazienti anziani.\n"
        "Scrivi UNA risposta breve e naturale che mostri comprensione.\n\n"
        
        "COME RISPONDERE:\n"
        "1. Riconosci l'emozione o esperienza della persona\n"
        "2. Riprendi 1 elemento concreto che ha detto (azione, emozione, dettaglio)\n"
        "3. Mostra che è normale/comprensibile quello che sta provando\n\n"
        
        "ESEMPI DI BUONE RISPOSTE (varia lo stile):\n"
        "- 'Che bello che abbia potuto vedere i suoi nipoti! Questi momenti sono davvero preziosi.'\n"
        "- 'Gli attacchi di panico sono esperienze molto intense. Mi rendo conto di quanto possano essere difficili.'\n"
        "- 'La felicità è un'emozione bellissima. È importante riconoscere quando ci sentiamo così.'\n\n"
        
        "VARIETÀ OBBLIGATORIA:\n"
        "- Cambia il modo di iniziare: 'Che bello', 'Immagino', 'Mi rendo conto', '[Elemento concreto] può essere...'\n"
        "- NON ripetere sempre 'Capisco che... È normale...'\n"
        "- VARIA LA STRUTTURA: non seguire sempre lo stesso schema\n\n"
        
        "REGOLE CRITICHE:\n"
        f"- Usa SEMPRE e SOLO 'Lei' (mai tu/ti/te/tuo/tua/tuoi/tue)\n"
        f"- Genere: {gender_lbl} - usa accordi corretti\n"
        "- NON fare domande\n"
        "- NON usare etichette come 'Validazione:' o 'Riflesso:'\n"
        "- NON dire 'Lei ha detto' o 'Lei ha risposto'\n"
        "- NON usare il nome del paziente\n"
        "- NON inventare eventi o dettagli non esplicitamente detti dall'utente\n"
        "- Linguaggio caldo e naturale, non clinico\n"
    )
    
    # PERSONALIZZAZIONE 1: Communication needs
    comm_rules = ""
    if profile_data["communication_needs"]:
        comm_lower = profile_data["communication_needs"].lower()
        if "sente poco" in comm_lower or "brevi" in comm_lower:
            comm_rules = (
                "\nCOMUNICAZIONE:\n"
                "- Il paziente ha difficoltà uditive: usa frasi MOLTO BREVI (massimo 1 frase, 10-15 parole)\n"
                "- Struttura semplice: soggetto-verbo-complemento\n"
                "- Evita subordinate complesse\n"
                "- Esempio: 'Immagino quanto sia stato difficile.' NON 'Immagino quanto sia stato difficile per Lei affrontare questa situazione complessa.'\n"
            )
    
    # PERSONALIZZAZIONE 2: Emotional signals (emotion + intensity)
    emotion_rules = ""
    emotion = signals.get("emotion", "neutro/misto")
    intensity = signals.get("intensity", "bassa")
    
    if emotion != "neutro/misto":
        if emotion in ["ansia/panico", "tristezza", "paura", "sconforto/impotenza", "rabbia/frustrazione"]:
            # Emozioni negative: tono più supportivo
            if intensity == "alta":
                emotion_rules = (
                    f"\nEMOZIONE RILEVATA: {emotion} (intensità alta)\n"
                    "- Usa tono calmo e rassicurante\n"
                    "- Valida l'intensità dell'emozione ('molto intenso', 'davvero difficile')\n"
                    "- Mostra comprensione profonda\n"
                    "- NON interpretare risposte brevi come positive\n"
                )
            else:
                emotion_rules = (
                    f"\nEMOZIONE RILEVATA: {emotion}\n"
                    "- Usa tono supportivo e validante\n"
                    "- Riconosci la difficoltà dell'esperienza\n"
                    "- NON interpretare risposte brevi come positive\n"
                )
        elif emotion in ["serenità", "speranza"]:
            # Emozioni positive: tono caldo e incoraggiante
            emotion_rules = (
                f"\nEMOZIONE RILEVATA: {emotion}\n"
                "- Celebra il momento positivo\n"
                "- Usa tono caldo e incoraggiante\n"
                "- Rinforza l'importanza di riconoscere questi momenti\n"
            )
    else:
        # Neutro/misto: attenzione a non inventare
        emotion_rules = (
            "\nRISPOSTA NEUTRA/BREVE:\n"
            "- Empatia sobria e validante\n"
            "- NON inventare dettagli o interpretazioni positive non dette\n"
            "- Se la risposta è solo una parola/tempo, limitati a riconoscere senza elaborare\n"
            "- Esempi CORRETTI per 'bene': 'Mi fa piacere sentirlo.' NON 'quanto sia stato bello'\n"
        )
    
    # PERSONALIZZAZIONE 3: Living situation + Health goal (contestuale, se rilevante)
    context_hint = ""
    living = profile_data["living_situation"]
    goal = profile_data["health_goal"]
    
    if living or goal:
        context_hint = "\nCONTESTO PERSONALE:\n"
        if living:
            context_hint += f"- Situazione abitativa: {living}\n"
        if goal:
            context_hint += f"- Obiettivo salute: {goal}\n"
        context_hint += "→ Usa questi dati SOLO se pertinenti alla risposta (non forzare collegamento)\n"
        
        # Rendi esplicito quando il goal è pertinente
        if goal and last_a:
            last_a_lower = last_a.lower()
            if any(word in last_a_lower for word in ["cammin", "passi", "minuti", "passeggia"]):
                context_hint += f"NOTA: La risposta parla di movimento/camminata. L'obiettivo è: {goal}\n"
    
    # Lunghezza target (adattata a communication_needs)
    length_target = "1-2 frasi"
    if comm_rules:  # Se ha difficoltà uditive
        length_target = "1 frase breve (max 15 parole)"
    
    # Combina prompt
    system = SystemMessage(
        content=(
            base_rules + 
            comm_rules + 
            emotion_rules + 
            context_hint + 
            f"\nLUNGHEZZA TARGET: {length_target}\n"
        )
    )

    # Gestione risposte brevi
    short_hint = ""
    if len(last_a.split()) <= 2:
        short_hint = (
            "NOTA: La risposta è molto breve. Sviluppa comunque una risposta empatica "
            "riprendendo il contesto. NON inventare dettagli non detti.\n\n"
        )

    human = HumanMessage(
        content=(
            f"{short_hint}"
            f"{original_context}"  # Include contesto risposta precedente se follow-up
            f"Domanda:\n{last_q}\n\n"
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
                "\n\nATTENZIONE: Usa SOLO 'Lei', MAI 'tu/ti/te/tuo/tua'. "
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
    
    # NUOVO nodo per signal extraction
    builder.add_node("extract_signals", node_extract_signals)

    # Edges
    builder.add_edge(START, "select_question")
    builder.add_conditional_edges("select_question", route_after_select)

    builder.add_edge("profile_context", "ask_and_read")
    builder.add_conditional_edges("ask_and_read", route_after_ask)

    # NUOVO flusso: save → extract → router
    builder.add_edge("save_current_answer", "extract_signals")
    builder.add_conditional_edges("extract_signals", route_answer_type)
    
    # Follow-up e deepening ritornano ad ask_and_read
    builder.add_edge("follow_up_evasive", "ask_and_read")
    builder.add_edge("emotional_deepening", "ask_and_read")
    
    builder.add_conditional_edges("empathy_bridge", route_after_bridge)
    builder.add_edge("advance_to_next_question", "select_question")

    return builder.compile()
