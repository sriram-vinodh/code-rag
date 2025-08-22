from abc import ABC, abstractmethod
from typing import List, Dict, Iterator
from langchain_core.documents import Document
class ParserInterface(ABC):
    """
    An interface for parsing raw file data into structured Document objects.
    """
    @abstractmethod
    def parse_documents(self, file_data_iterator: Iterator[Dict[str, str]]) -> List[Document]:
        """
        Parses an iterator of file data into a list of Document objects.
        """
        pass