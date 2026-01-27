# compute_metrics.py
"""
Calcola metriche di evaluation dai JSON di sessione generati.

Metriche:
M1) Completion Rate: % sessioni che completano tutte le domande
M2) Average Answer Length: lunghezza media risposte (in parole)
M3) Evasiveness Resolution Rate: % evasività risolta dopo follow-up (solo FULL)
M4) Branch Rate: branch_questions / total_questions (solo FULL)
"""
from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import statistics


# ===== KEYWORD EVASIVE (stesso set di response_classifier.py) =====
EVASIVE_KEYWORDS = [
    "no", "niente", "non ricordo", "non so", "nulla", 
    "non mi viene in mente", "boh", "mah"
]


def is_evasive(answer: str) -> bool:
    """Check se risposta è evasiva (keyword matching)"""
    normalized = answer.lower().strip()
    if not normalized or len(normalized) <= 15:
        return any(kw in normalized for kw in EVASIVE_KEYWORDS)
    return False


@dataclass
class SessionMetrics:
    """Metriche per una singola sessione"""
    session_index: int
    config: str  # "FULL" o "NO_ROUTING"
    profile_name: str
    patient_id: str
    routing_enabled: bool
    
    # M1: Completion
    total_questions: int
    expected_questions: int
    completed: bool  # True se total_questions == expected_questions
    
    # M2: Answer Length
    avg_answer_length_words: float
    
    # M3: Evasiveness Resolution (solo FULL)
    total_evasive_entries: int
    evasive_resolved: int
    evasiveness_resolution_rate: float  # resolved / total (o None se total=0)
    
    # M4: Branch Rate (solo FULL)
    branch_questions: int
    branch_rate: float  # branch / total (o 0 se routing disabled)
    
    # Extra info
    duration_seconds: int


@dataclass
class AggregatedMetrics:
    """Metriche aggregate per configurazione"""
    config: str
    num_sessions: int
    
    # M1
    completion_rate: float  # % completed
    
    # M2
    avg_answer_length_mean: float
    avg_answer_length_std: float
    
    # M3 (solo FULL)
    evasiveness_resolution_rate_mean: float
    evasiveness_resolution_rate_std: float
    
    # M4 (solo FULL)
    branch_rate_mean: float
    branch_rate_std: float


class MetricsComputer:
    """Computa metriche da JSON di sessione"""
    
    def __init__(self, manifest_path: Path, expected_questions: int = 8):
        """
        Args:
            manifest_path: Path al manifest.json generato
            expected_questions: Numero atteso di domande principali
        """
        self.manifest_path = manifest_path
        self.expected_questions = expected_questions
        
        # Carica manifest
        with open(manifest_path, encoding="utf-8") as f:
            self.manifest_data = json.load(f)
        
        self.sessions = self.manifest_data["sessions"]
        self.session_metrics: List[SessionMetrics] = []
    
    def compute_all_metrics(self):
        """Computa metriche per tutte le sessioni"""
        print("=" * 70)
        print("  CALCOLO METRICHE DI EVALUATION")
        print("=" * 70)
        print(f"Sessioni da analizzare: {len(self.sessions)}\n")
        
        for session_info in self.sessions:
            json_path = Path(session_info["json_path"])
            
            if not json_path.exists():
                print(f"[WARN] JSON non trovato: {json_path}")
                continue
            
            # Carica JSON sessione
            with open(json_path, encoding="utf-8") as f:
                session_data = json.load(f)
            
            # Computa metriche per questa sessione
            metrics = self.compute_session_metrics(session_info, session_data)
            self.session_metrics.append(metrics)
            
            print(f"[{metrics.config}] {metrics.profile_name}: "
                  f"Q={metrics.total_questions}, "
                  f"AvgLen={metrics.avg_answer_length_words:.1f}, "
                  f"Branches={metrics.branch_questions}")
        
        print(f"\n[OK] Metriche calcolate per {len(self.session_metrics)} sessioni")
    
    def compute_session_metrics(
        self,
        session_info: Dict,
        session_data: Dict
    ) -> SessionMetrics:
        """Computa metriche per una sessione singola"""
        
        config = session_info["config"]
        routing_enabled = session_info["routing_enabled"]
        qa_history = session_data.get("qa_history", [])
        branches = session_data.get("branches", {}).get("details", [])
        stats = session_data.get("statistics", {})
        metadata = session_data.get("session_metadata", {})
        
        # M1: Completion
        total_questions = len(qa_history)
        completed = (total_questions >= self.expected_questions)
        
        # M2: Average Answer Length (parole)
        answer_lengths = []
        for qa in qa_history:
            answer = qa.get("user_answer", "")
            word_count = len(answer.split())
            answer_lengths.append(word_count)
        
        avg_answer_length = (
            statistics.mean(answer_lengths) if answer_lengths else 0.0
        )
        
        # M3: Evasiveness Resolution Rate (solo se routing enabled)
        total_evasive_entries = 0
        evasive_resolved = 0
        evasiveness_resolution_rate = None
        
        if routing_enabled:
            # Identifica entry evasive in qa_history
            for i, qa in enumerate(qa_history):
                answer = qa.get("user_answer", "")
                
                if is_evasive(answer):
                    total_evasive_entries += 1
                    
                    # Check se risolto nelle successive risposte (max 2 follow-up)
                    resolved = False
                    for j in range(i + 1, min(i + 3, len(qa_history))):
                        next_answer = qa_history[j].get("user_answer", "")
                        if not is_evasive(next_answer) and len(next_answer.split()) > 5:
                            resolved = True
                            break
                    
                    if resolved:
                        evasive_resolved += 1
            
            # Calcola rate
            if total_evasive_entries > 0:
                evasiveness_resolution_rate = evasive_resolved / total_evasive_entries
            else:
                evasiveness_resolution_rate = None  # Non applicabile
        
        # M4: Branch Rate
        branch_questions = stats.get("branch_questions", len(branches))
        branch_rate = branch_questions / total_questions if total_questions > 0 else 0.0
        
        # Duration
        duration_seconds = metadata.get("duration_seconds", 0)
        
        return SessionMetrics(
            session_index=session_info["session_index"],
            config=config,
            profile_name=session_info["profile_name"],
            patient_id=session_info["patient_id"],
            routing_enabled=routing_enabled,
            total_questions=total_questions,
            expected_questions=self.expected_questions,
            completed=completed,
            avg_answer_length_words=avg_answer_length,
            total_evasive_entries=total_evasive_entries,
            evasive_resolved=evasive_resolved,
            evasiveness_resolution_rate=evasiveness_resolution_rate,
            branch_questions=branch_questions,
            branch_rate=branch_rate,
            duration_seconds=duration_seconds,
        )
    
    def aggregate_by_config(self) -> Dict[str, AggregatedMetrics]:
        """Aggrega metriche per configurazione"""
        configs = {}
        
        for config_name in ["FULL", "NO_ROUTING"]:
            config_sessions = [
                m for m in self.session_metrics if m.config == config_name
            ]
            
            if not config_sessions:
                continue
            
            # M1: Completion Rate
            completed_count = sum(1 for m in config_sessions if m.completed)
            completion_rate = completed_count / len(config_sessions)
            
            # M2: Average Answer Length
            avg_lengths = [m.avg_answer_length_words for m in config_sessions]
            avg_length_mean = statistics.mean(avg_lengths)
            avg_length_std = statistics.stdev(avg_lengths) if len(avg_lengths) > 1 else 0.0
            
            # M3: Evasiveness Resolution Rate (solo FULL)
            evasive_rates = [
                m.evasiveness_resolution_rate 
                for m in config_sessions 
                if m.evasiveness_resolution_rate is not None
            ]
            if evasive_rates:
                evasive_rate_mean = statistics.mean(evasive_rates)
                evasive_rate_std = statistics.stdev(evasive_rates) if len(evasive_rates) > 1 else 0.0
            else:
                evasive_rate_mean = 0.0
                evasive_rate_std = 0.0
            
            # M4: Branch Rate
            branch_rates = [m.branch_rate for m in config_sessions]
            branch_rate_mean = statistics.mean(branch_rates)
            branch_rate_std = statistics.stdev(branch_rates) if len(branch_rates) > 1 else 0.0
            
            configs[config_name] = AggregatedMetrics(
                config=config_name,
                num_sessions=len(config_sessions),
                completion_rate=completion_rate,
                avg_answer_length_mean=avg_length_mean,
                avg_answer_length_std=avg_length_std,
                evasiveness_resolution_rate_mean=evasive_rate_mean,
                evasiveness_resolution_rate_std=evasive_rate_std,
                branch_rate_mean=branch_rate_mean,
                branch_rate_std=branch_rate_std,
            )
        
        return configs
    
    def save_results(self, output_dir: Path):
        """Salva risultati in CSV e JSON"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. CSV con metriche per sessione
        csv_path = output_dir / "metrics_per_session.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if self.session_metrics:
                fieldnames = list(asdict(self.session_metrics[0]).keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for metrics in self.session_metrics:
                    writer.writerow(asdict(metrics))
        
        print(f"\n[OK] Salvato: {csv_path}")
        
        # 2. JSON con metriche aggregate
        aggregated = self.aggregate_by_config()
        json_path = output_dir / "metrics_aggregated.json"
        
        json_data = {
            "configurations": {
                config_name: asdict(metrics)
                for config_name, metrics in aggregated.items()
            },
            "sessions": [asdict(m) for m in self.session_metrics]
        }
        
        json_path.write_text(
            json.dumps(json_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        
        print(f"[OK] Salvato: {json_path}")
        
        # 3. CSV comparison table (per paper)
        comparison_csv = output_dir / "metrics_comparison.csv"
        with open(comparison_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Metric", "FULL (mean)", "FULL (std)", 
                "NO_ROUTING (mean)", "NO_ROUTING (std)"
            ])
            
            full = aggregated.get("FULL")
            no_routing = aggregated.get("NO_ROUTING")
            
            if full and no_routing:
                writer.writerow([
                    "M1: Completion Rate",
                    f"{full.completion_rate:.2%}",
                    "-",
                    f"{no_routing.completion_rate:.2%}",
                    "-"
                ])
                
                writer.writerow([
                    "M2: Avg Answer Length (words)",
                    f"{full.avg_answer_length_mean:.2f}",
                    f"{full.avg_answer_length_std:.2f}",
                    f"{no_routing.avg_answer_length_mean:.2f}",
                    f"{no_routing.avg_answer_length_std:.2f}"
                ])
                
                writer.writerow([
                    "M3: Evasiveness Resolution Rate",
                    f"{full.evasiveness_resolution_rate_mean:.2%}",
                    f"{full.evasiveness_resolution_rate_std:.2%}",
                    "N/A (routing disabled)",
                    "-"
                ])
                
                writer.writerow([
                    "M4: Branch Rate",
                    f"{full.branch_rate_mean:.2%}",
                    f"{full.branch_rate_std:.2%}",
                    f"{no_routing.branch_rate_mean:.2%}",
                    f"{no_routing.branch_rate_std:.2%}"
                ])
        
        print(f"[OK] Salvato: {comparison_csv}")
        
        # 4. Print summary to console
        print("\n" + "=" * 70)
        print("  SUMMARY - METRICHE AGGREGATE")
        print("=" * 70)
        
        for config_name, metrics in aggregated.items():
            print(f"\n{config_name}:")
            print(f"  Sessioni: {metrics.num_sessions}")
            print(f"  M1 - Completion Rate: {metrics.completion_rate:.2%}")
            print(f"  M2 - Avg Answer Length: {metrics.avg_answer_length_mean:.2f} ± {metrics.avg_answer_length_std:.2f} words")
            
            if config_name == "FULL":
                print(f"  M3 - Evasiveness Resolution: {metrics.evasiveness_resolution_rate_mean:.2%} ± {metrics.evasiveness_resolution_rate_std:.2%}")
            
            print(f"  M4 - Branch Rate: {metrics.branch_rate_mean:.2%} ± {metrics.branch_rate_std:.2%}")
        
        print("=" * 70)


def main():
    ROOT = Path(__file__).resolve().parent
    manifest_path = ROOT / "evaluation_results" / "manifest.json"
    
    if not manifest_path.exists():
        print(f"[ERRORE] Manifest non trovato: {manifest_path}")
        print("Esegui prima: python generate_evaluation_sessions.py")
        return
    
    # Computa metriche
    computer = MetricsComputer(manifest_path, expected_questions=8)
    computer.compute_all_metrics()
    
    # Salva risultati
    output_dir = ROOT / "evaluation_results"
    computer.save_results(output_dir)


if __name__ == "__main__":
    main()
