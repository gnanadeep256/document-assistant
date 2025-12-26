# import json
# from pathlib import Path

# import chromadb
# from sentence_transformers import SentenceTransformer
# from tqdm import tqdm

# # ---------------- CONFIG ---------------- #

# CHROMA_DIR = Path("data/vector_db")
# CHUNKS_FILE = Path("data/processed/chunks.json")

# COLLECTION_NAME = "documents"
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# BATCH_SIZE = 32
# RESET_COLLECTION = True  # set False after dev

# MIN_CHUNK_LENGTH = 100  # characters

# # ---------------------------------------- #


# def load_chunks():
#     if not CHUNKS_FILE.exists():
#         raise FileNotFoundError("chunks.json not found. Run chunker first.")

#     with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
#         chunks = json.load(f)

#     print(f"Loaded {len(chunks)} chunks")
#     return chunks


# def normalize_metadata(chunk):
#     """
#     Canonical metadata schema.
#     Every key MUST exist and MUST be Chroma-safe.
#     """

#     section_level = chunk.get("section_level")
#     if section_level is None:
#         section_level = -1

#     return {
#         "pages": ",".join(map(str, chunk.get("pages", []))),
#         "section_id": chunk.get("section_id") or "",
#         "section_parents": "|".join(chunk.get("section_parents", [])),
#         "section_level": int(section_level),
#         "structure_confidence": float(chunk.get("structure_confidence", 0.0)),
#         "source": chunk.get("source", "document"),
#     }


# def main():
#     chunks = load_chunks()

#     # ---- Init Chroma ----
#     client = chromadb.PersistentClient(path=str(CHROMA_DIR))

#     existing = [c.name for c in client.list_collections()]
#     if RESET_COLLECTION and COLLECTION_NAME in existing:
#         client.delete_collection(COLLECTION_NAME)
#         print("Existing collection deleted")

#     collection = client.get_or_create_collection(
#         name=COLLECTION_NAME,
#         metadata={"hnsw:space": "cosine"}
#     )

#     # ---- Load embedding model ----
#     print("Loading embedding model...")
#     embedder = SentenceTransformer(EMBEDDING_MODEL)

#     texts = []
#     metadatas = []
#     ids = []

#     skipped = 0

#     for chunk in chunks:
#         text = chunk.get("text", "").strip()

#         if len(text) < MIN_CHUNK_LENGTH:
#             skipped += 1
#             continue

#         texts.append(text)
#         metadatas.append(normalize_metadata(chunk))
#         ids.append(chunk["id"])

#     print(f"Prepared {len(texts)} chunks for embedding")
#     print(f"Skipped {skipped} short chunks")

#     # ---- Embed & store ----
#     print("Embedding chunks...")

#     for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Batches"):
#         batch_texts = texts[i:i + BATCH_SIZE]
#         batch_ids = ids[i:i + BATCH_SIZE]
#         batch_meta = metadatas[i:i + BATCH_SIZE]

#         embeddings = embedder.encode(
#             batch_texts,
#             show_progress_bar=False,
#             normalize_embeddings=True
#         ).tolist()

#         collection.add(
#             documents=batch_texts,
#             embeddings=embeddings,
#             metadatas=batch_meta,
#             ids=batch_ids
#         )

#     print("Indexing complete")
#     print(f"Stored {collection.count()} vectors")

#     # ---- Diagnostics (STRUCTURE-AWARE) ----
#     structured = sum(
#         1 for m in metadatas if m["structure_confidence"] >= 0.5
#     )

#     print(f"Structured chunks      : {structured}")
#     print(f"Unstructured chunks    : {len(metadatas) - structured}")


# if __name__ == "__main__":
#     main()

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------- CONFIG ---------------- #

CHROMA_DIR = Path("data/vector_db")
CHUNKS_FILE = Path("data/processed/chunks.json")

COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32
MIN_CHUNK_LENGTH = 100

# ---------------------------------------- #


def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError("chunks.json not found. Run chunker first.")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")
    return chunks


def normalize_metadata(chunk):
    section_level = chunk.get("section_level")
    if section_level is None:
        section_level = -1

    return {
        "pages": ",".join(map(str, chunk.get("pages", []))),
        "section_id": chunk.get("section_id") or "",
        "section_parents": "|".join(chunk.get("section_parents", [])),
        "section_level": int(section_level),
        "structure_confidence": float(chunk.get("structure_confidence", 0.0)),
        "source": chunk.get("source", "document"),
    }


def main():
    chunks = load_chunks()

    # ---- Init Chroma SAFELY ----
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print("Existing collection deleted safely")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # ---- Load embedding model ----
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    texts = []
    metadatas = []
    ids = []

    skipped = 0

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if len(text) < MIN_CHUNK_LENGTH:
            skipped += 1
            continue

        texts.append(text)
        metadatas.append(normalize_metadata(chunk))
        ids.append(chunk["id"])

    print(f"Prepared {len(texts)} chunks for embedding")
    print(f"Skipped {skipped} short chunks")

    print("Embedding chunks...")

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Batches"):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]
        batch_meta = metadatas[i:i + BATCH_SIZE]

        embeddings = embedder.encode(
            batch_texts,
            show_progress_bar=False,
            normalize_embeddings=True
        ).tolist()

        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_meta,
            ids=batch_ids
        )

    print("Indexing complete")
    print(f"Stored {collection.count()} vectors")

    structured = sum(
        1 for m in metadatas if m["structure_confidence"] >= 0.5
    )

    print(f"Structured chunks      : {structured}")
    print(f"Unstructured chunks    : {len(metadatas) - structured}")


if __name__ == "__main__":
    main()
