# generate_evaluation_sessions.py
"""
Genera 20 sessioni per evaluation:
- 10 sessioni FULL (routing_enabled=True)
- 10 sessioni NO-ROUTING (routing_enabled=False)
Alterna le configurazioni per fairness.

Usa user simulator LLM con seed fisso per riproducibilità.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from src.graph import build_graph
from src.state import DialogueState
from src.profile_store import ProfileStoreConfig, build_profile_retriever
from src.profile_utils import get_safe_field
from src.session_logger import SessionLogger


# ===== CONFIGURAZIONE =====
SEED = 42  # Seed fisso per riproducibilità
NUM_SESSIONS_PER_CONFIG = 10
SIMULATOR_MODEL = "qwen2.5:3b"  # LLM leggero per user simulator
SIMULATOR_TEMPERATURE = 0.7  # Variabilità controllata

# Profili da usare (diversificati)
PROFILES_TO_USE = [
    "demo_profile_01",
    "demo_profile_02", 
    "demo_profile_03_minimal",
    "demo_profile_cognitive_mild_impairment",
    "demo_profile_emotion_prone",
    "demo_profile_social_isolation",
    "demo_profile_fragmented_sleep",
    "demo_profile_complex_conditions",
    "demo_profile_evasive_style",
    "demo_profile_high_personalization_signal",
]


# ===== USER SIMULATOR =====
class UserSimulator:
    """
    Simula un paziente anziano rispondendo alle domande del diario.
    Usa LLM con seed fisso per riproducibilità.
    """
    
    def __init__(self, profile_data: dict, seed: int = 42):
        self.profile_data = profile_data
        self.llm = ChatOllama(
            model=SIMULATOR_MODEL,
            temperature=SIMULATOR_TEMPERATURE,
            seed=seed  # Seed per riproducibilità
        )
        
        # Estrai caratteristiche chiave dal profilo
        self.name = get_safe_field(profile_data, "name", "Paziente")
        self.age = get_safe_field(profile_data, "age", "75")
        self.gender = get_safe_field(profile_data, "gender", "M")
        self.main_condition = get_safe_field(profile_data, "main_condition", "")
        self.communication_style = get_safe_field(profile_data, "communication_needs", "")
        
        # Contatore domande (per variare le risposte)
        self.question_count = 0
        
    def answer_question(self, question: str, context: str = "") -> str:
        """
        Genera risposta simulata alla domanda basandosi sul profilo.
        
        Args:
            question: Domanda da rispondere
            context: Contesto conversazionale (domande/risposte precedenti)
        
        Returns:
            Risposta simulata del paziente
        """
        self.question_count += 1
        
        # System prompt per user simulator
        system_prompt = (
            f"Sei {self.name}, un paziente anziano di {self.age} anni.\n"
            f"Genere: {self.gender}\n"
        )
        
        if self.main_condition:
            system_prompt += f"Condizione principale: {self.main_condition}\n"
        
        if self.communication_style:
            system_prompt += f"Stile comunicativo: {self.communication_style}\n"
        
        system_prompt += (
            "\nRispondi alla domanda come farebbe un anziano vero:\n"
            "- Usa un linguaggio naturale e informale\n"
            "- Varia tra risposte dettagliate e brevi\n"
            "- Occasionalmente sii evasivo ('non so', 'non ricordo')\n"
            "- Esprimi emozioni quando appropriate (tristezza, ansia, gioia)\n"
            "- Sii coerente con il tuo profilo\n"
            "- Mantieni un tono anziano e autentico\n"
            "\nRispondi SOLO con la risposta del paziente, niente altro."
        )
        
        # Costruisci messaggio human con domanda e contesto
        human_content = f"Domanda: {question}"
        if context:
            human_content = f"Contesto conversazione:\n{context}\n\n{human_content}"
        
        system = SystemMessage(content=system_prompt)
        human = HumanMessage(content=human_content)
        
        try:
            result = self.llm.invoke([system, human])
            answer = (result.content or "").strip()
            
            # Fallback se risposta vuota
            if not answer:
                fallbacks = [
                    "Non saprei...",
                    "Boh, non mi ricordo bene.",
                    "Abbastanza bene direi.",
                    "Niente di particolare."
                ]
                answer = random.choice(fallbacks)
            
            return answer
            
        except Exception as e:
            print(f"[ERRORE] User simulator: {e}")
            return "Non so, non mi ricordo."


# ===== GENERATORE SESSIONI =====
class SessionGenerator:
    """Genera sessioni di evaluation alternate tra FULL e NO-ROUTING"""
    
    def __init__(self, output_dir: Path):
        self.ROOT = Path(__file__).resolve().parent
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup paths
        self.data_dir = self.ROOT / "data"
        self.profiles_dir = self.data_dir / "profiles"
        self.questions_path = self.data_dir / "diary_questions.json"
        self.schema_path = self.data_dir / "profile_schema.json"
        self.runtime_dir = self.ROOT / "runtime"
        
        # Carica domande
        self.diary_questions = json.loads(
            self.questions_path.read_text(encoding="utf-8")
        )
        
        # Tracking
        self.manifest = []
        
    def generate_all_sessions(self):
        """Genera tutte le 20 sessioni alternate"""
        print("=" * 70)
        print("  GENERAZIONE SESSIONI DI EVALUATION")
        print("=" * 70)
        print(f"Configurazioni: FULL (routing ON) e NO-ROUTING (routing OFF)")
        print(f"Sessioni per config: {NUM_SESSIONS_PER_CONFIG}")
        print(f"Totale sessioni: {NUM_SESSIONS_PER_CONFIG * 2}")
        print(f"Seed: {SEED}")
        print(f"User simulator: {SIMULATOR_MODEL} (temp={SIMULATOR_TEMPERATURE})")
        print("=" * 70 + "\n")
        
        # Set seed globale
        random.seed(SEED)
        
        # Genera sessioni alternate
        for i in range(NUM_SESSIONS_PER_CONFIG):
            # Scegli profilo (cicla tra i profili disponibili)
            profile_name = PROFILES_TO_USE[i % len(PROFILES_TO_USE)]
            
            # Alterna: FULL → NO-ROUTING → FULL → ...
            print(f"\n{'='*70}")
            print(f"  SESSIONE {2*i + 1}/{NUM_SESSIONS_PER_CONFIG * 2}: FULL (routing ON)")
            print(f"{'='*70}")
            self.generate_session(
                config_name="FULL",
                profile_name=profile_name,
                routing_enabled=True,
                session_index=2*i + 1
            )
            
            print(f"\n{'='*70}")
            print(f"  SESSIONE {2*i + 2}/{NUM_SESSIONS_PER_CONFIG * 2}: NO-ROUTING (routing OFF)")
            print(f"{'='*70}")
            self.generate_session(
                config_name="NO_ROUTING",
                profile_name=profile_name,
                routing_enabled=False,
                session_index=2*i + 2
            )
        
        # Salva manifest
        self.save_manifest()
        
        print("\n" + "=" * 70)
        print("  GENERAZIONE COMPLETATA")
        print("=" * 70)
        print(f"Sessioni generate: {len(self.manifest)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Manifest salvato: {self.output_dir / 'manifest.json'}")
        print("=" * 70)
    
    def generate_session(
        self,
        config_name: str,
        profile_name: str,
        routing_enabled: bool,
        session_index: int
    ):
        """Genera una singola sessione"""
        print(f"Profilo: {profile_name}")
        print(f"Config: {config_name}")
        print(f"Routing: {'ON' if routing_enabled else 'OFF'}\n")
        
        # Carica profilo
        profile_path = self.profiles_dir / f"{profile_name}.json"
        if not profile_path.exists():
            print(f"[ERRORE] Profilo non trovato: {profile_path}")
            return
        
        profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
        
        # ID sessione univoco
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patient_id = f"EVAL_{config_name}_{profile_name}_{timestamp}"
        
        # Crea user simulator per questo profilo
        user_sim = UserSimulator(profile_data, seed=SEED + session_index)
        
        # Setup retriever
        db_dir = self.runtime_dir / "chroma_profile_db" / profile_path.stem
        db_dir.parent.mkdir(parents=True, exist_ok=True)
        
        cfg = ProfileStoreConfig(
            profile_path=profile_path,
            schema_path=self.schema_path,
            db_dir=db_dir,
            k=3,
        )
        retriever = build_profile_retriever(cfg)
        
        # Inizializza LLM
        llm = ChatOllama(model="qwen2.5:7b", temperature=0.75)
        
        # Session logger
        sessions_dir = self.output_dir / "sessions"
        session_logger = SessionLogger(patient_id, sessions_dir)
        
        # Costruisci grafo
        graph = build_graph()
        
        # Stato iniziale con flag routing_enabled
        initial_state: DialogueState = {
            "profile_path": str(profile_path),
            "profile_context": "",
            "diary_questions": self.diary_questions,
            "current_index": 0,
            "current_question": None,
            "last_user_answer": None,
            "qa_history": [],
            "done": False,
            "retriever": retriever,
            "llm": llm,
            "assistant_reply": None,
            "skip_question_print": False,
            "question_mode": "MAIN",
            "pending_question": None,
            "branch_count_for_current": 0,
            "current_branch_type": None,
            "original_question_index": 0,
            "session_logger": session_logger,
            "signals": [],
            "routing_enabled": routing_enabled,  # FLAG CONFIGURAZIONE
        }
        
        # Wrapper per user simulator
        def answer_callback():
            # Ottieni contesto (ultime 2 Q/A)
            context = ""
            if initial_state.get("qa_history"):
                recent = initial_state["qa_history"][-2:]
                context = "\n".join([
                    f"Q: {qa['question']}\nA: {qa['answer']}"
                    for qa in recent
                ])
            
            # Ottieni domanda corrente
            q = initial_state.get("pending_question") or initial_state.get("current_question", "")
            
            # Genera risposta
            answer = user_sim.answer_question(q, context)
            return answer
        
        # Inietta callback nello state (per node_ask_and_read)
        initial_state["auto_answer_callback"] = answer_callback
        
        # Esegui grafo
        try:
            final_state = graph.invoke(initial_state, config={"recursion_limit": 400})
            
            # Popola session_logger
            for i, qa in enumerate(final_state["qa_history"], start=1):
                q_text = qa["question"]
                a_text = qa["answer"]
                assistant_reply = qa.get("assistant_reply", "")
                session_logger.log_qa(i, q_text, a_text, assistant_reply)
            
            # Salva sessione
            session_logger.save(profile_data, final_state.get("signals", []))
            
            # Aggiungi a manifest
            self.manifest.append({
                "session_index": session_index,
                "config": config_name,
                "profile_name": profile_name,
                "patient_id": patient_id,
                "routing_enabled": routing_enabled,
                "timestamp": timestamp,
                "total_questions": len(final_state["qa_history"]),
                "json_path": str(session_logger.json_path),
                "csv_path": str(session_logger.csv_path),
            })
            
            print(f"[OK] Sessione salvata: {session_logger.json_path.name}")
            print(f"     Domande totali: {len(final_state['qa_history'])}")
            
        except Exception as e:
            print(f"[ERRORE] Generazione sessione fallita: {e}")
            import traceback
            traceback.print_exc()
    
    def save_manifest(self):
        """Salva manifest delle sessioni generate"""
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps({
                "metadata": {
                    "seed": SEED,
                    "num_sessions_per_config": NUM_SESSIONS_PER_CONFIG,
                    "total_sessions": len(self.manifest),
                    "simulator_model": SIMULATOR_MODEL,
                    "simulator_temperature": SIMULATOR_TEMPERATURE,
                    "profiles_used": PROFILES_TO_USE,
                },
                "sessions": self.manifest
            }, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )


def main():
    ROOT = Path(__file__).resolve().parent
    output_dir = ROOT / "evaluation_results"
    
    generator = SessionGenerator(output_dir)
    generator.generate_all_sessions()


if __name__ == "__main__":
    main()
