"""
Script di test unificato per validare tutti i profili utente.
Uso: 
  python test_profiles.py          # Test veloci (senza ChromaDB)
  python test_profiles.py --full   # Test completi (con ChromaDB, richiede Ollama)
"""
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.profile_utils import load_profile_safe, get_safe_field


@dataclass
class TestResult:
    profile_name: str
    test_name: str
    passed: bool
    message: str
    details: str = ""


class ProfileTester:
    def __init__(self, full_mode: bool = False):
        self.full_mode = full_mode
        self.results: List[TestResult] = []
        self.data_dir = ROOT / "data"
        self.profiles_dir = self.data_dir / "profiles"
        self.schema_path = self.data_dir / "profile_schema.json"
        self.runtime_dir = ROOT / "runtime"
        
        self.schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        self.all_fields = set(self.schema.keys())
        
    def add_result(self, profile_name: str, test_name: str, passed: bool, 
                   message: str, details: str = ""):
        self.results.append(TestResult(
            profile_name=profile_name,
            test_name=test_name,
            passed=passed,
            message=message,
            details=details
        ))
    
    def test_json_validity(self, profile_path: Path) -> bool:
        try:
            profile = load_profile_safe(profile_path)
            if not profile:
                self.add_result(
                    profile_path.stem,
                    "JSON Validity",
                    False,
                    "File JSON vuoto o non caricabile"
                )
                return False
            
            self.add_result(
                profile_path.stem,
                "JSON Validity",
                True,
                "✓ JSON ben formato"
            )
            return True
        except Exception as e:
            self.add_result(
                profile_path.stem,
                "JSON Validity",
                False,
                f"Errore parsing JSON: {str(e)}"
            )
            return False
    
    def test_required_fields(self, profile_path: Path, profile: Dict[str, Any]) -> bool:
        required = ["name", "age"]
        missing = []
        
        for field in required:
            value = profile.get(field)
            if value is None or str(value).strip() == "":
                missing.append(field)
        
        if missing:
            self.add_result(
                profile_path.stem,
                "Required Fields",
                False,
                f"Campi obbligatori mancanti: {', '.join(missing)}"
            )
            return False
        
        self.add_result(
            profile_path.stem,
            "Required Fields",
            True,
            "✓ Campi obbligatori presenti"
        )
        return True
    
    def test_field_coverage(self, profile_path: Path, profile: Dict[str, Any]) -> Dict[str, Any]:
        present_fields = {k for k, v in profile.items() 
                         if v is not None and str(v).strip() != ""}
        missing_fields = self.all_fields - present_fields
        coverage = len(present_fields) / len(self.all_fields) * 100
        
        if coverage < 20:
            icon = "📉"
            status = "Molto scarsa"
        elif coverage < 50:
            icon = "📊"
            status = "Bassa"
        elif coverage < 80:
            icon = "📈"
            status = "Media"
        else:
            icon = "✨"
            status = "Alta"
        
        self.add_result(
            profile_path.stem,
            "Field Coverage",
            True,
            f"{icon} Copertura: {coverage:.1f}% ({status})",
            details=f"Presenti: {len(present_fields)}/{len(self.all_fields)}"
        )
        
        return {
            "coverage": coverage,
            "present": len(present_fields),
            "missing": len(missing_fields),
            "missing_list": sorted(missing_fields)
        }
    
    def test_unknown_fields(self, profile_path: Path, profile: Dict[str, Any]) -> bool:
        profile_fields = set(profile.keys())
        unknown = profile_fields - self.all_fields
        
        if unknown:
            self.add_result(
                profile_path.stem,
                "Unknown Fields",
                False,
                f"⚠️ Campi sconosciuti: {', '.join(unknown)}"
            )
            return False
        
        self.add_result(
            profile_path.stem,
            "Unknown Fields",
            True,
            "✓ Tutti i campi validi"
        )
        return True
    
    def test_data_types(self, profile_path: Path, profile: Dict[str, Any]) -> bool:
        issues = []
        
        age = profile.get("age")
        if age is not None:
            try:
                age_int = int(age)
                if age_int < 0 or age_int > 120:
                    issues.append(f"age fuori range: {age_int}")
            except (ValueError, TypeError):
                issues.append(f"age non numerico: {age}")
        
        if issues:
            self.add_result(
                profile_path.stem,
                "Data Types",
                False,
                f"⚠️ {'; '.join(issues)}"
            )
            return False
        
        self.add_result(
            profile_path.stem,
            "Data Types",
            True,
            "✓ Tipi di dato corretti"
        )
        return True
    
    def test_geriatric_context(self, profile_path: Path, profile: Dict[str, Any]) -> bool:
        warnings = []
        
        age = profile.get("age")
        if age:
            try:
                age_int = int(age)
                if age_int < 60:
                    warnings.append(f"Età {age_int} < 60 (fuori target geriatrico)")
            except:
                pass
        
        job = profile.get("job")
        if job and isinstance(job, str):
            active_keywords = ["attuale", "lavoro", "dipendente"]
            if any(kw in job.lower() for kw in active_keywords):
                warnings.append("Job suggerisce lavoro attivo (target=anziani)")
        
        if warnings:
            self.add_result(
                profile_path.stem,
                "Geriatric Context",
                False,
                f"⚠️ {'; '.join(warnings)}"
            )
            return False
        
        self.add_result(
            profile_path.stem,
            "Geriatric Context",
            True,
            "✓ Coerente con target geriatrico"
        )
        return True
    
    def test_chromadb_indexing(self, profile_path: Path) -> bool:
        if not self.full_mode:
            return None
        
        try:
            from src.profile_store import ProfileStoreConfig, build_profile_retriever
            
            db_dir = self.runtime_dir / "chroma_profile_db" / f"test_{profile_path.stem}"
            db_dir.parent.mkdir(parents=True, exist_ok=True)
            
            cfg = ProfileStoreConfig(
                profile_path=profile_path,
                schema_path=self.schema_path,
                db_dir=db_dir,
                collection_name=f"test_{profile_path.stem}"
            )
            
            retriever = build_profile_retriever(cfg)
            
            if retriever is None:
                raise Exception("Retriever None")
            
            self.add_result(
                profile_path.stem,
                "ChromaDB Indexing",
                True,
                "✓ Indicizzazione completata"
            )
            return True
            
        except Exception as e:
            self.add_result(
                profile_path.stem,
                "ChromaDB Indexing",
                False,
                f"❌ Errore: {str(e)[:50]}"
            )
            return False
    
    def test_retrieval_basic(self, profile_path: Path) -> bool:
        if not self.full_mode:
            return None
        
        try:
            from src.profile_store import ProfileStoreConfig, build_profile_retriever
            
            db_dir = self.runtime_dir / "chroma_profile_db" / f"test_{profile_path.stem}"
            
            cfg = ProfileStoreConfig(
                profile_path=profile_path,
                schema_path=self.schema_path,
                db_dir=db_dir,
                collection_name=f"test_{profile_path.stem}"
            )
            
            retriever = build_profile_retriever(cfg)
            
            test_queries = [
                "Qual è il nome del paziente?",
                "Quali sono i problemi di salute?",
                "Routine quotidiana"
            ]
            
            results_found = []
            for query in test_queries:
                try:
                    docs = retriever.invoke(query)
                    if docs:
                        results_found.append(query)
                except Exception:
                    pass
            
            if results_found:
                self.add_result(
                    profile_path.stem,
                    "Retrieval Basic",
                    True,
                    f"✓ Funzionante: {len(results_found)}/{len(test_queries)} query"
                )
                return True
            else:
                self.add_result(
                    profile_path.stem,
                    "Retrieval Basic",
                    False,
                    "❌ Nessuna query ha prodotto risultati"
                )
                return False
                
        except Exception as e:
            self.add_result(
                profile_path.stem,
                "Retrieval Basic",
                False,
                f"❌ Errore: {str(e)[:50]}"
            )
            return False
    
    def test_profile(self, profile_path: Path):
        print(f"\n{'─'*60}")
        print(f"📋 {profile_path.name}")
        print(f"{'─'*60}")
        
        passed = 0
        total = 0
        skipped = 0
        
        # Test 1: JSON Validity
        if not self.test_json_validity(profile_path):
            return (passed, 1, skipped)
        passed += 1
        total += 1
        
        profile = load_profile_safe(profile_path)
        
        # Test 2-6: Test base
        tests = [
            (self.test_required_fields, [profile_path, profile]),
            (self.test_field_coverage, [profile_path, profile]),
            (self.test_unknown_fields, [profile_path, profile]),
            (self.test_data_types, [profile_path, profile]),
            (self.test_geriatric_context, [profile_path, profile]),
        ]
        
        coverage_data = None
        for test_func, args in tests:
            result = test_func(*args)
            if test_func == self.test_field_coverage:
                coverage_data = result
                passed += 1
                total += 1
            elif result:
                passed += 1
                total += 1
            else:
                total += 1
        
        # Test 7-8: ChromaDB (solo in full mode)
        if self.full_mode:
            for test_func in [self.test_chromadb_indexing, self.test_retrieval_basic]:
                result = test_func(profile_path)
                if result is True:
                    passed += 1
                    total += 1
                elif result is False:
                    total += 1
        else:
            skipped = 2
        
        # Stampa risultati
        recent_results = [r for r in self.results if r.profile_name == profile_path.stem]
        display_count = 8 if self.full_mode else 6
        for result in recent_results[-display_count:]:
            status = "✓" if result.passed else "✗"
            print(f"  {status} {result.test_name}: {result.message}")
        
        if skipped > 0:
            print(f"  ⏭️  {skipped} test ChromaDB saltati (usa --full)")
        
        if coverage_data and coverage_data["coverage"] < 50:
            print(f"    └─ Campi mancanti: {', '.join(coverage_data['missing_list'][:5])}")
        
        return (passed, total, skipped)
    
    def run_all_tests(self):
        profiles = sorted(self.profiles_dir.glob("*.json"))
        
        if not profiles:
            print("❌ Nessun profilo trovato in data/profiles/")
            return
        
        mode_desc = "COMPLETI (con ChromaDB)" if self.full_mode else "VELOCI"
        print(f"\n{'═'*60}")
        print(f"🧪 TEST {mode_desc} - PROFILI GERIATRICI")
        print(f"{'═'*60}")
        print(f"Profili trovati: {len(profiles)}\n")
        
        if self.full_mode:
            print("⚠️  Modo completo: richiede Ollama attivo (può essere lento)\n")
        
        total_passed = 0
        total_tests = 0
        total_skipped = 0
        
        for profile_path in profiles:
            passed, total, skipped = self.test_profile(profile_path)
            total_passed += passed
            total_tests += total
            total_skipped += skipped
        
        self.print_summary(total_passed, total_tests, total_skipped)
    
    def print_summary(self, total_passed: int, total_tests: int, total_skipped: int):
        print(f"\n{'═'*60}")
        print("📊 RIEPILOGO FINALE")
        print(f"{'═'*60}\n")
        
        profiles = {}
        for result in self.results:
            if result.profile_name not in profiles:
                profiles[result.profile_name] = []
            profiles[result.profile_name].append(result)
        
        perfect = []
        good = []
        issues = []
        
        for profile_name in sorted(profiles.keys()):
            results = profiles[profile_name]
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            
            if passed == total:
                perfect.append(profile_name)
            elif passed >= total * 0.8:
                good.append(profile_name)
            else:
                issues.append(profile_name)
        
        if perfect:
            print(f"✅ Perfetti ({len(perfect)}):")
            for name in perfect:
                print(f"   • {name}")
        
        if good:
            print(f"\n✓ Buoni ({len(good)}):")
            for name in good:
                results = profiles[name]
                passed = sum(1 for r in results if r.passed)
                total = len(results)
                failures = [r for r in results if not r.passed]
                print(f"   • {name} ({passed}/{total})")
                for failure in failures:
                    print(f"     └─ {failure.message}")
        
        if issues:
            print(f"\n⚠️ Da rivedere ({len(issues)}):")
            for name in issues:
                results = profiles[name]
                passed = sum(1 for r in results if r.passed)
                total = len(results)
                failures = [r for r in results if not r.passed]
                print(f"   • {name} ({passed}/{total})")
                for failure in failures:
                    print(f"     └─ {failure.message}")
        
        print(f"\n{'─'*60}")
        percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        if percentage == 100:
            emoji = "🎉"
            msg = "Perfetto! Tutti i profili pronti."
        elif percentage >= 80:
            emoji = "👍"
            msg = "Ottimo! Solo piccoli aggiustamenti."
        elif percentage >= 60:
            emoji = "👌"
            msg = "Bene, ma serve qualche correzione."
        else:
            emoji = "⚠️"
            msg = "Attenzione: diversi problemi da risolvere."
        
        print(f"{emoji} TOTALE: {total_passed}/{total_tests} ({percentage:.1f}%)")
        if total_skipped > 0:
            print(f"⏭️  Test saltati: {total_skipped}")
        print(f"{msg}")
        print(f"{'─'*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test automatico profili geriatrici",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python test_profiles.py           # Test veloci (6 test per profilo)
  python test_profiles.py --full    # Test completi con ChromaDB (8 test per profilo)
        """
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Esegue anche i test ChromaDB/Ollama (lenti ma completi)'
    )
    
    args = parser.parse_args()
    
    tester = ProfileTester(full_mode=args.full)
    tester.run_all_tests()


if __name__ == "__main__":
    main()
