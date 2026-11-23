from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain.schema import Document

class MiniLMEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_documents(self, texts):
        # Accepts List[str] or List[Document]
        if not texts:
            return []
        if isinstance(texts[0], Document):
            texts = [doc.page_content for doc in texts]
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, query):
        # Accepts str or Document
        if isinstance(query, Document):
            query = query.page_content
        return self.model.encode([query], normalize_embeddings=True)[0].tolist()
