# Implementation migrated from storage_interface.py
from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

class VectorStoreInterface(ABC):
    """
    An interface for a vector store that can store and retrieve document embeddings.
    """
    @abstractmethod
    def add_documents(self, documents: List[Document]):
        """Adds documents to the vector store."""
        pass
    @abstractmethod
    def as_retriever(self) -> BaseRetriever:
        """Returns a retriever for the vector store."""
        pass
    @abstractmethod
    def clear(self):
        """Clears the vector store."""
        pass
    @abstractmethod
    def exists(self) -> bool:
        """Checks if the vector store exists and is not empty."""
        pass
# Moved from storage_interface.py
# VectorStoreInterface definition
