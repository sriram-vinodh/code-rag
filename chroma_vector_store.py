from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from storage_interface import VectorStoreInterface

class ChromaVectorStore(VectorStoreInterface):
    """
    A ChromaDB implementation of the VectorStoreInterface for in-memory storage.
    """
    def __init__(self, embedding_function: Embeddings):
        self._vector_store: Chroma | None = None
        self._embedding_function = embedding_function

    def add_documents(self, documents: List[Document]):
        """
        Creates an in-memory Chroma vector store from documents.
        """
        print("Creating in-memory vector store...")
        self._vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self._embedding_function
        )
        print("Vector store created.")

    def as_retriever(self) -> BaseRetriever:
        """Returns a retriever for the Chroma vector store."""
        if self._vector_store is None:
            raise ValueError("Vector store not initialized. Call add_documents first.")
        return self._vector_store.as_retriever()