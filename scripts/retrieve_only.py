from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ---------------- #

CHROMA_DIR = Path("data/vector_db")
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5

# ---------------------------------------- #


def main():
    # ---- Init Chroma ----
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    # ---- Load embedder ----
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("\nDocument Retrieval Test")
    print("-" * 40)

    while True:
        query = input("\nEnter a query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        query_embedding = embedder.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K
        )

        print("\nTop Results:\n")

        for i in range(len(results["documents"][0])):
            text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            pages = metadata.get("pages", "unknown")

            print(f"--- Result {i+1} | Pages: {pages} ---")
            print(text[:800])  # avoid terminal spam
            print()

    print("Exiting retrieval test.")


if __name__ == "__main__":
    main()
