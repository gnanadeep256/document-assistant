import chromadb
from sentence_transformers import SentenceTransformer
from retrieval.query_intent import detect_intent, QueryIntent

CHROMA_DIR = "data/vector_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Retriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def search(self, query: str, k: int = 20):
        intent = detect_intent(query)

        query_embedding = self.embedder.encode(
            query, normalize_embeddings=True
        ).tolist()

        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        results = []
        for text, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0]
        ):
            score = self._score(query, intent, text, meta, dist)
            results.append({
                "text": text,
                "pages": meta.get("pages", "").split(","),
                "section_id": meta.get("section_id"),
                "section_title": meta.get("section_id"),
                "confidence": round(score, 3)
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:5]

    def _score(self, query, intent, text, meta, distance):
        # Base semantic score
        semantic = 1.0 - distance

        score = 0.65 * semantic

        # Intent bonus
        if intent == QueryIntent.DEFINITION and "is" in text.lower():
            score += 0.15
        elif intent == QueryIntent.WHY and "because" in text.lower():
            score += 0.15
        elif intent == QueryIntent.SECTION:
            score += 0.10

        # Structure bias (soft, never dominant)
        section_id = meta.get("section_id", "")
        if intent == QueryIntent.SECTION and section_id and section_id in query:
            score += 0.15

        # Penalize references unless explicitly asked
        if section_id and section_id.lower().startswith("9"):
            score -= 0.25

        return max(score, 0.0)
