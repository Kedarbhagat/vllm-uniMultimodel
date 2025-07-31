import os
import logging
from pathlib import Path
from typing import List, Optional, Dict
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from remote_embeddings import RemoteEmbeddings
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)

# Directory structure
BASE_DATA_DIR = "llm_net_data"
VECTORSTORE_DIR = os.path.join(BASE_DATA_DIR, "vectorstores")

Path(VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)

class VectorStoreManager:
    """Singleton manager for vectorstores with caching"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self._embedder = None
            self._vectorstore_cache: Dict[str, Chroma] = {}
            self._cache_lock = threading.Lock()
            self.initialized = True
    
    def get_embedder(self) -> RemoteEmbeddings:
        """Get singleton embedding instance"""
        if self._embedder is None:
            self._embedder = RemoteEmbeddings(base_url="http://localhost:9096")
        return self._embedder
    
    def _get_cache_key(self, thread_id: str, document_id: str) -> str:
        return f"{thread_id}_{document_id}"
    
    def get_vectorstore(self, thread_id: str, document_id: str) -> Optional[Chroma]:
        """Get vectorstore from cache or load it"""
        cache_key = self._get_cache_key(thread_id, document_id)
        
        with self._cache_lock:
            if cache_key in self._vectorstore_cache:
                logger.info(f"Retrieved vectorstore from cache: {cache_key}")
                return self._vectorstore_cache[cache_key]
        
        # Load vectorstore if not in cache
        vectorstore = self._load_vectorstore_internal(thread_id, document_id)
        if vectorstore:
            with self._cache_lock:
                # Implement LRU-like behavior - keep only last 10 vectorstores
                if len(self._vectorstore_cache) >= 10:
                    # Remove oldest entry
                    oldest_key = next(iter(self._vectorstore_cache))
                    del self._vectorstore_cache[oldest_key]
                    logger.info(f"Evicted vectorstore from cache: {oldest_key}")
                
                self._vectorstore_cache[cache_key] = vectorstore
                logger.info(f"Cached vectorstore: {cache_key}")
        
        return vectorstore
    
    def _load_vectorstore_internal(self, thread_id: str, document_id: str) -> Optional[Chroma]:
        """Internal method to load vectorstore from disk"""
        try:
            if not thread_id or not document_id:
                raise ValueError("Both thread_id and document_id are required")

            collection_name = f"{thread_id}_{document_id}"
            persist_dir = os.path.join(VECTORSTORE_DIR, thread_id, document_id)

            if not os.path.exists(persist_dir):
                logger.warning(f"Vectorstore path does not exist: {persist_dir}")
                return None

            logger.info(f"Loading vectorstore from disk: {collection_name}")
            embedder = self.get_embedder()
            
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embedder,
                persist_directory=persist_dir
            )

            if not hasattr(vectorstore, '_collection') or not vectorstore._collection:
                logger.error(f"Failed to load collection {collection_name}")
                return None

            logger.info(f"Loaded vectorstore with {vectorstore._collection.count()} chunks")
            return vectorstore

        except Exception as e:
            logger.error(f"Error loading vectorstore: {e}", exc_info=True)
            return None
    
    def invalidate_cache(self, thread_id: str, document_id: str):
        """Remove vectorstore from cache"""
        cache_key = self._get_cache_key(thread_id, document_id)
        with self._cache_lock:
            if cache_key in self._vectorstore_cache:
                del self._vectorstore_cache[cache_key]
                logger.info(f"Invalidated cache for: {cache_key}")
    
    def clear_cache(self):
        """Clear all cached vectorstores"""
        with self._cache_lock:
            self._vectorstore_cache.clear()
            logger.info("Cleared all vectorstore cache")

# Global manager instance
vectorstore_manager = VectorStoreManager()

def build_vectorstore(
    docs: List[Document],
    thread_id: str,
    document_id: str,
) -> Optional[Chroma]:
    """
    Build a Chroma vectorstore for a single document under a thread.
    """
    try:
        if not thread_id or not document_id:
            raise ValueError("Both thread_id and document_id are required")

        non_empty_docs = [doc for doc in docs if doc.page_content.strip()]
        if not non_empty_docs:
            logger.warning("No valid content in documents")
            return None

        collection_name = f"{thread_id}_{document_id}"
        persist_dir = os.path.join(VECTORSTORE_DIR, thread_id, document_id)
        os.makedirs(persist_dir, exist_ok=True)

        logger.info(f"Creating vectorstore: {collection_name} at {persist_dir}")
        embedder = vectorstore_manager.get_embedder()

        vectorstore = Chroma.from_documents(
            documents=non_empty_docs,
            embedding=embedder,
            collection_name=collection_name,
            persist_directory=persist_dir
        )

        vectorstore.persist()
        logger.info(f"Stored {len(non_empty_docs)} chunks in {collection_name}")
        
        # Cache the newly created vectorstore
        cache_key = vectorstore_manager._get_cache_key(thread_id, document_id)
        with vectorstore_manager._cache_lock:
            vectorstore_manager._vectorstore_cache[cache_key] = vectorstore
        
        return vectorstore

    except Exception as e:
        logger.error(f"Failed to build vectorstore: {e}", exc_info=True)
        return None

def load_vectorstore(thread_id: str, document_id: str) -> Optional[Chroma]:
    """
    Load a Chroma vectorstore for a specific document (with caching).
    """
    return vectorstore_manager.get_vectorstore(thread_id, document_id)

def delete_vectorstore(thread_id: str, document_id: str) -> bool:
    """
    Delete a Chroma vectorstore for a document.
    """
    try:
        persist_dir = os.path.join(VECTORSTORE_DIR, thread_id, document_id)
        if not os.path.exists(persist_dir):
            logger.warning(f"Directory not found: {persist_dir}")
            return False

        import shutil
        shutil.rmtree(persist_dir)
        logger.info(f"Deleted vectorstore: {persist_dir}")
        
        # Invalidate cache
        vectorstore_manager.invalidate_cache(thread_id, document_id)
        return True

    except Exception as e:
        logger.error(f"Failed to delete vectorstore: {e}", exc_info=True)
        return False

# Utility functions for cache management
def clear_vectorstore_cache():
    """Clear all cached vectorstores"""
    vectorstore_manager.clear_cache()

def get_cache_stats():
    """Get cache statistics"""
    with vectorstore_manager._cache_lock:
        return {
            "cached_vectorstores": len(vectorstore_manager._vectorstore_cache),
            "cache_keys": list(vectorstore_manager._vectorstore_cache.keys())
        }