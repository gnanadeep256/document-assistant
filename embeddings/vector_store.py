import chromadb

class VectorStore:
    def __init__(self, collection_name="documents", persist_dir="data/vector_db"):
        self.client = chromadb.PersistentClient(
            path=persist_dir
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(self, ids, texts, embeddings, metadatas):
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query(self, query_embedding, n_results=5):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
