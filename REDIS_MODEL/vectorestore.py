# vectorestore.py
from langchain_community.vectorstores import FAISS # You're using FAISS now, which is great!
from langchain_core.documents import Document # Use langchain_core.documents.Document
from typing import List # Import List for type hinting

from embeddingmodel import MiniLMEmbeddings # Assuming this is your custom embedding model

def build_vectorstore(docs: List[Document]) -> FAISS: # Type hint that 'docs' is a list of Document objects
    """
    Builds and returns a FAISS vector store from a list of LangChain Document objects.
    """
    # The 'docs' parameter already contains LangChain Document objects from document_processor.py.
    # So, you don't need to re-wrap them. Just use the 'docs' list directly.
    
    embedder = MiniLMEmbeddings() # Instantiate your embedding model
    
    print("Building FAISS vector store...")
    # Pass the list of Document objects directly to FAISS.from_documents
    vectorstore = FAISS.from_documents(docs, embedder)
    print("FAISS vector store built successfully.")
    
    return vectorstore