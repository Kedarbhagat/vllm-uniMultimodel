from langchain.embeddings.base import Embeddings
from embedding_client import RemoteEmbeddingClient
from langchain.schema import Document

class RemoteEmbeddings(Embeddings):
    def __init__(self, base_url="http://localhost:9096"):
        self.client = RemoteEmbeddingClient(base_url=base_url)

    def embed_documents(self, texts):
        if not texts:
            return []
        if isinstance(texts[0], Document):
            texts = [doc.page_content for doc in texts]
        return self.client.embed_texts(texts)

    def embed_query(self, query):
        if isinstance(query, Document):
            query = query.page_content
        return self.client.embed_query(query)
