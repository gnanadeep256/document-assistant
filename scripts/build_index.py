import json
from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore


def build_index():
    """Build the vector index from precomputed chunks.

    Expects `data/processed/chunks.json` to contain items of the form::

        {
            "id": "chunk_00000",
            "pages": [1, 2],
            "text": "..."
        }

    We preserve the chunk `id` and `pages` so downstream retrieval and the UI
    can provide page-level traceability back into the PDF.
    """
    with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Use existing chunk IDs when present; fall back to a generated one for
    # backwards compatibility with older chunk formats.
    ids = [c.get("id") or f"chunk_{i}" for i, c in enumerate(chunks)]
    texts = [c["text"] for c in chunks]

    # Store page span for each chunk in a Chroma-compatible way (no list values).
    # Chroma metadata must be scalar: str, int, float, bool, or None.
    metadatas = []
    for c in chunks:
        # Prefer "pages" (list of ints); fall back to legacy single "page".
        pages = c.get("pages")
        if not pages:
            legacy_page = c.get("page")
            if legacy_page is not None:
                pages = [legacy_page]

        # Normalize to a comma-separated string (e.g. "1,2") or None.
        if isinstance(pages, (list, tuple, set)):
            pages_value = ",".join(str(p) for p in pages)
        elif pages is None:
            pages_value = None
        else:
            pages_value = str(pages)

        metadatas.append({
            "pages": pages_value,
            "source": "pdf",
        })

    embedder = Embedder()
    embeddings = embedder.embed_texts(texts)

    store = VectorStore(
        collection_name="documents",
        persist_dir="data/vector_db",
    )

    store.add_documents(ids, texts, embeddings, metadatas)

    print("Vector index built successfully.")


if __name__ == "__main__":
    build_index()
