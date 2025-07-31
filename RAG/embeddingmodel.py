from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class MiniLMEmbeddings(Embeddings):
    # Class-level model instance (shared across all instances)
    _model = None
    _model_loaded = False
    
    def __init__(self):
        # Only load the model once for all instances
        if not MiniLMEmbeddings._model_loaded:
            logger.info("Loading SentenceTransformer model (one-time initialization)...")
            MiniLMEmbeddings._model = SentenceTransformer('all-MiniLM-L6-v2')
            MiniLMEmbeddings._model_loaded = True
            logger.info("SentenceTransformer model loaded successfully!")
        else:
            logger.info("Using cached SentenceTransformer model")
    
    @property
    def model(self):
        """Get the cached model instance"""
        return MiniLMEmbeddings._model
    
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
'''
# Alternative: Singleton pattern for even more control
class MiniLMEmbeddingsSingleton(Embeddings):
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info("Creating new MiniLMEmbeddings singleton instance")
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if MiniLMEmbeddingsSingleton._model is None:
            logger.info("Loading SentenceTransformer model (singleton initialization)...")
            MiniLMEmbeddingsSingleton._model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer model loaded successfully!")
    
    @property
    def model(self):
        return MiniLMEmbeddingsSingleton._model
    
    def embed_documents(self, texts):
        if not texts:
            return []
        if isinstance(texts[0], Document):
            texts = [doc.page_content for doc in texts]
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    
    def embed_query(self, query):
        if isinstance(query, Document):
            query = query.page_content
        return self.model.encode([query], normalize_embeddings=True)[0].tolist()
'''
# Factory function for easy usage
def get_minilm_embeddings():
    """Factory function to get cached embeddings instance"""
    return MiniLMEmbeddings()

# Global instance (if you prefer this approach)
_global_embeddings = None

def get_global_embeddings():
    """Get global cached embeddings instance"""
    global _global_embeddings
    if _global_embeddings is None:
        _global_embeddings = MiniLMEmbeddings()
    return _global_embeddings