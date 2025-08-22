from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
class ChunkerInterface(ABC):
    """
    An interface for chunking a list of documents into smaller pieces.
    """