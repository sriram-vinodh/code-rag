# Implementation migrated from chroma_vector_store.py
from typing import List
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import logging
import os
import shutil
from tqdm import tqdm
from .storage_interface import VectorStoreInterface

class ChromaVectorStore(VectorStoreInterface):
    """
    A ChromaDB implementation of the VectorStoreInterface for persistent storage.
    """
    def __init__(self, embedding_function: Embeddings, persist_directory: str = "./chroma_db", batch_size: int = 32):
        self._vector_store: Chroma | None = None
        self._embedding_function = embedding_function
        self._persist_directory = persist_directory
        self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size

    def add_documents(self, documents: List[Document]):
        if not documents:
            self.logger.info("No documents to add.")
            return
        if self._vector_store is None:
            self.logger.info(f"Initializing vector store at {self._persist_directory}...")
            self._vector_store = Chroma(
                persist_directory=self._persist_directory,
                embedding_function=self._embedding_function
            )
        self.logger.info(f"Adding {len(documents)} new documents to vector store.")
        for i in tqdm(range(0, len(documents), self.batch_size), desc="Embedding and adding documents"):
            batch = documents[i:i + self.batch_size]
            try:
                self._vector_store.add_documents(batch)
            except ValueError as e:
                if "dimension" in str(e):
                    error_message = (
                        "Embedding dimension mismatch. The model you are currently using produces embeddings "
                        "of a different size than the ones stored in the existing database. "
                        f"Please delete the directory '{self._persist_directory}' and restart the application."
                    )
                    self.logger.error(error_message, exc_info=True)
                    raise ValueError(error_message) from e
                else:
                    raise e
        self.logger.info("Vector store updated and persisted.")

    def as_retriever(self) -> BaseRetriever:
        if self._vector_store is None:
            if os.path.exists(self._persist_directory) and os.listdir(self._persist_directory):
                self.logger.info(f"Loading vector store for retrieval from {self._persist_directory}...")
                self._vector_store = Chroma(
                    persist_directory=self._persist_directory,
                    embedding_function=self._embedding_function
                )
            else:
                raise ValueError("Vector store not initialized and no persistent data found. Call add_documents first.")
        return self._vector_store.as_retriever()

    def clear(self):
        if os.path.exists(self._persist_directory):
            self.logger.info(f"Clearing vector store by deleting directory: {self._persist_directory}")
            shutil.rmtree(self._persist_directory)
            self._vector_store = None

    def exists(self) -> bool:
        return os.path.exists(self._persist_directory) and bool(os.listdir(self._persist_directory))
# Moved from chroma_vector_store.py
# ChromaVectorStore implementation
