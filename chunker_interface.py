from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class ChunkerInterface(ABC):
    """
    An interface for chunking a list of documents into smaller pieces.
    """

    @abstractmethod
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Splits the documents into smaller chunks."""
        pass