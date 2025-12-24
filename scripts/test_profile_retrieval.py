# src/test_profile_retrieval.py
from pathlib import Path
from profile_store import ProfileStoreConfig, build_profile_retriever, retrieve_profile_context

def main():
    cfg = ProfileStoreConfig(
        profile_path=Path("data/profiles/demo_profile_01.json"),
        schema_path=Path("data/profile_schema.json"),
        db_dir=Path("runtime/chroma_profile_db"),
        embed_model="mxbai-embed-large",
        k=5,
    )

    retriever = build_profile_retriever(cfg)

    while True:
        q = input("\nQuery (Q per uscire): ").strip()
        if q == "":
            print("Inserisci una query (es. 'routine', 'comunicazione', 'obiettivo').")
            continue
        if q.lower() == "q":
            break
        ctx = retrieve_profile_context(retriever, q)
        print("\n--- PROFILE CONTEXT ---")
        print(ctx)

if __name__ == "__main__":
    main()
