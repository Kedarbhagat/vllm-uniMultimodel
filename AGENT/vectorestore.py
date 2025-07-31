# vectorestore.py (FIXED Version)
import os
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Optional
import logging

from embeddingmodel import MiniLMEmbeddings

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = "chroma_data"

def build_vectorstore(
    docs: List[Document],
    collection_name: str = "default_collection",
    persist_directory: str = CHROMA_PERSIST_DIR
) -> Optional[Chroma]:
    """
    Builds or loads a ChromaDB vector store from a list of Langchain Document objects.
    Ensures only valid, non-empty documents are used for embedding.
    """
    try:
        non_empty_docs = [
            doc for doc in docs if doc.page_content and doc.page_content.strip()
        ]

        os.makedirs(persist_directory, exist_ok=True)
        embedder = MiniLMEmbeddings()

        if not non_empty_docs:
            logger.warning("No valid (non-empty) documents provided to build vector store. Initializing an empty ChromaDB client.")
            return Chroma(
                embedding_function=embedder,
                collection_name=collection_name,
                persist_directory=persist_directory
            )

        logger.info(f"Building ChromaDB vector store for collection '{collection_name}' in '{persist_directory}' with {len(non_empty_docs)} documents...")

        vectorstore = Chroma.from_documents(
            documents=non_empty_docs,
            embedding=embedder,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        vectorstore.persist()
        logger.info("ChromaDB vector store built and persisted successfully.")
        return vectorstore

    except Exception as e:
        logger.error(f"Error building vector store: {e}", exc_info=True)
        return None

# --- Example usage for local testing ---
if __name__ == "__main__":
    # Example usage for local testing:
    sample_docs = [
        Document(page_content="This is a test document about Python programming."),
        Document(page_content="Another document discussing machine learning."),
        Document(page_content=""),  # Empty
        Document(page_content="   "),  # Whitespace only
    ]
    test_collection = "my_test_collection"
    test_vs = build_vectorstore(sample_docs, collection_name=test_collection)

    if test_vs:
        print(f"Test vector store created for collection: {test_collection}")
        query = "What is python?"
        results = test_vs.similarity_search(query, k=1)
        print(f"\nResults for '{query}':")
        for doc in results:
            print(f"- {doc.page_content}")
    else:
        print("Failed to build test vector store.")