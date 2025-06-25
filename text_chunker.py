from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chunker_interface import ChunkerInterface

class RecursiveTextChunker(ChunkerInterface):
    """
    A class to chunk a list of documents using a recursive character text splitter.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        Initializes the TextChunker.

        Args:
            chunk_size (int): The maximum size of each chunk (in characters).
            chunk_overlap (int): The number of characters to overlap between chunks.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Splits the documents into smaller chunks."""
        return self.text_splitter.split_documents(documents)