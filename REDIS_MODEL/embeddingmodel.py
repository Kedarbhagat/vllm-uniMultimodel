from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

class MiniLMEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_documents(self, texts):
        # texts is List[str]
        return self.model.encode(texts, normalize_embeddings=True)

    def embed_query(self, query):
        # query is str
        return self.model.encode([query], normalize_embeddings=True)[0]
