
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .chunker_interface import ChunkerInterface

class RecursiveTextChunker(ChunkerInterface):
    """
    General-purpose chunker for text-based documents using LangChain's RecursiveCharacterTextSplitter.
    Splits text into semantically meaningful chunks, preserving paragraphs and sentences where possible.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits each document into smaller chunks using RecursiveCharacterTextSplitter.
        Returns a flat list of chunked Documents, preserving metadata.
        """
        all_chunks = []
        for doc in documents:
            splits = self.splitter.split_text(doc.page_content)
            for i, chunk in enumerate(splits):
                metadata = doc.metadata.copy()
                metadata["chunk_index"] = i
                all_chunks.append(Document(page_content=chunk, metadata=metadata))
        return all_chunks
