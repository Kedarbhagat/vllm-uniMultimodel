import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging

logger = logging.getLogger(__name__)

class RemoteEmbeddingClient:
    def __init__(self, base_url="http://localhost:9096"):
        self.base_url = base_url.rstrip("/")
        
        # Create session with connection pooling
        self.session = requests.Session()
        
        # Configure retries and connection pooling
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy, 
            pool_connections=5, 
            pool_maxsize=10
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default timeout
        self.timeout = 10

    def embed_texts(self, texts, normalize=True):
        if not texts:
            return []
        
        payload = {"texts": texts, "normalize": normalize}
        try:
            response = self.session.post(
                f"{self.base_url}/embed", 
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            logger.error(f"Failed to embed texts via remote service: {e}", exc_info=True)
            raise

    def embed_query(self, query: str, normalize=True):
        return self.embed_texts([query], normalize=normalize)[0]