# src/session_logger.py
"""
Logging sessioni conversazione in CSV con privacy.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


class SessionLogger:
    """
    Logger per sessioni conversazione con privacy.
    Salva in CSV + metadata JSON.
    """
    
    def __init__(self, patient_id: str, sessions_dir: Path):
        self.patient_id = patient_id
        self.sessions_dir = sessions_dir
        self.start_time = datetime.now()
        self.conversation_log: List[Dict[str, Any]] = []
        self.branches_triggered: List[Dict[str, Any]] = []  # Track follow-up branches
        
        # Crea directory sessioni
        self.patient_dir = sessions_dir / patient_id
        self.patient_dir.mkdir(parents=True, exist_ok=True)
        
        # Nome file con timestamp
        timestamp_str = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.session_name = timestamp_str
        self.csv_path = self.patient_dir / f"{self.session_name}.csv"
        self.meta_path = self.patient_dir / f"{self.session_name}_meta.json"
        self.json_path = self.patient_dir / f"{self.session_name}.json"
    
    def log_qa(self, question_id: int, question: str, user_answer: str, assistant_reply: str):
        """
        Registra una coppia domanda-risposta.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "question_id": question_id,
            "question": question,
            "user_answer": user_answer,
            "assistant_reply": assistant_reply
        }
        self.conversation_log.append(entry)
    
    def log_branch(self, branch_type: str, theme_display: str, followup_question: str):
        """
        Traccia una diramazione (follow-up o deepening).
        
        Args:
            branch_type: Tipo di branch ("evasive", "ansia_panico", "dolore_fisico", etc.)
            theme_display: Nome human-readable ("Risposta evasiva", "Ansia/Panico", etc.)
            followup_question: Testo della domanda di follow-up
        """
        self.branches_triggered.append({
            "timestamp": datetime.now().isoformat(),
            "type": branch_type,
            "theme_display": theme_display,
            "followup_question": followup_question
        })
    
    def save(self, profile_data: Dict[str, Any]):
        """
        Salva la sessione in CSV e metadata in JSON.
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Salva CSV
        self._save_csv()
        
        # Salva metadata
        metadata = {
            "patient_id": self.patient_id,
            "session_start": self.start_time.isoformat(),
            "session_end": end_time.isoformat(),
            "duration_seconds": int(duration),
            "total_questions": len(self.conversation_log),
            "profile_summary": {
                "age": profile_data.get("age", "N/A"),
                "gender": profile_data.get("gender", "N/A"),
                "main_condition": profile_data.get("main_condition", "N/A")
            }
        }
        
        self.meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        
        # Salva JSON strutturato completo
        self._save_json()
        
        print(f"\n✅ Sessione salvata:")
        print(f"   CSV: {self.csv_path}")
        print(f"   Metadata: {self.meta_path}")
        print(f"   JSON: {self.json_path}")
    
    def _save_json(self):
        """
        Salva sessione completa in JSON strutturato.
        Include qa_history, timestamps, profile_id, branch metadata.
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        session_data = {
            "profile_id": self.patient_id,
            "session_metadata": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": int(duration)
            },
            "qa_history": self.conversation_log,
            "branches": {
                "total_count": len(self.branches_triggered),
                "details": self.branches_triggered
            },
            "statistics": {
                "total_questions": len(self.conversation_log),
                "branch_questions": len(self.branches_triggered)
            }
        }
        
        self.json_path.write_text(
            json.dumps(session_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    
    def _save_csv(self):
        """
        Salva conversation log in CSV.
        """
        if not self.conversation_log:
            return
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['timestamp', 'question_id', 'question', 'user_answer', 'assistant_reply']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for entry in self.conversation_log:
                writer.writerow(entry)
