# src/profile_store.py
from __future__ import annotations

import json
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


@dataclass(frozen=True)
class ProfileStoreConfig:
    # Paths
    profile_path: Path
    schema_path: Path
    db_dir: Path

    # Chroma
    collection_name: str = "user_profile"

    # Ollama embeddings
    embed_model: str = "mxbai-embed-large"

    # Retrieval
    k: int = 5


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint_profile(profile: Dict[str, Any]) -> str:
    """
    Crea un fingerprint stabile del profilo (per capire se è cambiato).
    """
    dumped = json.dumps(profile, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def _ensure_rebuilt_if_profile_changed(cfg: ProfileStoreConfig, fp: str) -> None:
    """
    Se il profilo cambia, ricostruisco la DB (semplice e robusto per prototipo).
    """
    fp_file = cfg.db_dir / ".profile_fingerprint"
    if cfg.db_dir.exists() and fp_file.exists():
        old_fp = fp_file.read_text(encoding="utf-8").strip()
        if old_fp != fp:
            shutil.rmtree(cfg.db_dir, ignore_errors=True)

    cfg.db_dir.mkdir(parents=True, exist_ok=True)
    fp_file.write_text(fp, encoding="utf-8")


def _profile_to_documents(
    profile: Dict[str, Any],
    schema: Dict[str, Any],
    *,
    skip_empty_values: bool = True,
) -> List[Document]:
    """
    Converte il profilo in una lista di Document (1 campo = 1 documento).
    """
    docs: List[Document] = []

    for key, value in profile.items():
        # Normalizza value
        if value is None:
            value_str = ""
        else:
            value_str = str(value).strip()

        if skip_empty_values and value_str == "":
            continue

        desc = ""
        section = ""
        if isinstance(schema, dict) and key in schema and isinstance(schema[key], dict):
            desc = str(schema[key].get("description", "")).strip()
            section = str(schema[key].get("section", "")).strip()

        page_content_parts = [
            f"Profile field: {key}",
        ]
        if desc:
            page_content_parts.append(f"Description: {desc}")
        if section:
            page_content_parts.append(f"Section: {section}")
        page_content_parts.append(f"Value: {value_str}")

        page_content = "\n".join(page_content_parts)

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "key": key,
                    "section": section,
                },
            )
        )

    return docs


def build_profile_retriever(cfg: ProfileStoreConfig):
    """
    Crea (o carica) una Chroma persistente per il profilo e ritorna un retriever.
    """
    profile = _load_json(cfg.profile_path)
    schema = _load_json(cfg.schema_path)

    if not isinstance(profile, dict):
        raise ValueError("profile.json deve essere un oggetto JSON (dict).")

    fp = _fingerprint_profile(profile)
    _ensure_rebuilt_if_profile_changed(cfg, fp)

    embeddings = OllamaEmbeddings(model=cfg.embed_model)

    vector_store = Chroma(
        collection_name=cfg.collection_name,
        persist_directory=str(cfg.db_dir),
        embedding_function=embeddings,
    )

    # Se è una DB nuova (o appena ricostruita), inserisco i documenti
    existing = vector_store.get()
    if not existing.get("ids"):
        docs = _profile_to_documents(profile, schema, skip_empty_values=True)
        ids = [d.metadata["key"] for d in docs]
        if docs:
            vector_store.add_documents(documents=docs, ids=ids)

    return vector_store.as_retriever(search_kwargs={"k": cfg.k})


def retrieve_profile_context(
    retriever,
    query: str,
) -> str:
    """
    Utility: ritorna un testo pronto da mettere nel prompt (per i nodi LangGraph).
    """
    docs = retriever.invoke(query)
    # Format compatto, leggibile
    lines = []
    for d in docs:
        key = d.metadata.get("key", "unknown")
        lines.append(f"- {key}:\n{d.page_content}")
    return "\n\n".join(lines)
