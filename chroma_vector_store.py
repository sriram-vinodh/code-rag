from typing import List
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import logging
import os
from storage_interface import VectorStoreInterface

class ChromaVectorStore(VectorStoreInterface):
    """
    A ChromaDB implementation of the VectorStoreInterface for persistent storage.
    """
    def __init__(self, embedding_function: Embeddings, persist_directory: str = "./chroma_db"):
        self._vector_store: Chroma | None = None
        self._embedding_function = embedding_function
        self._persist_directory = persist_directory
        self.logger = logging.getLogger(__name__)

    def add_documents(self, documents: List[Document]):
        """
        Adds documents to the Chroma vector store, either creating a new one
        or adding to an existing persistent one.
        """
        self._load_or_create_vector_store(documents)
        self._add_new_documents_to_store(documents)
        # The Chroma client with a persist_directory handles persistence automatically on writes.
        self.logger.info("Vector store updated and persisted.")

    def _load_or_create_vector_store(self, documents: List[Document]):
        """Loads an existing vector store or creates a new one."""
        if os.path.exists(self._persist_directory) and os.listdir(self._persist_directory):
            self.logger.info(f"Loading existing vector store from {self._persist_directory}...")
            self._vector_store = Chroma(
                persist_directory=self._persist_directory,
                embedding_function=self._embedding_function
            )
        else:
            self.logger.info(f"Creating new persistent vector store at {self._persist_directory}...")
            self._vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self._embedding_function,
                persist_directory=self._persist_directory
            )

    def _add_new_documents_to_store(self, documents: List[Document]):
        """Adds new documents to the initialized vector store."""
        if self._vector_store and documents:
            self.logger.info(f"Adding {len(documents)} new documents to existing vector store.")
            self._vector_store.add_documents(documents)

    def as_retriever(self) -> BaseRetriever:
        """Returns a retriever for the Chroma vector store."""
        if self._vector_store is None:
            # If vector store was not created via add_documents, try to load it # noqa
            if os.path.exists(self._persist_directory) and os.listdir(self._persist_directory): # noqa
                self.logger.info(f"Loading vector store for retrieval from {self._persist_directory}...")
                self._vector_store = Chroma(
                    persist_directory=self._persist_directory,
                    embedding_function=self._embedding_function
                )
            else:
                raise ValueError("Vector store not initialized and no persistent data found. Call add_documents first.")
        return self._vector_store.as_retriever()