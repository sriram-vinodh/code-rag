import argparse
import sys
import json

from loader_interface import LoaderInterface
from filesystem_loader import FileSystemLoader
from parser_interface import ParserInterface
from document_parser import DefaultDocumentParser
from chunker_interface import ChunkerInterface
from text_chunker import RecursiveTextChunker
from java_code_chunker import JavaCodeChunker
from rag_pipeline import RAGPipeline
from storage_interface import VectorStoreInterface
from chroma_vector_store import ChromaVectorStore
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate # Added for summarization
from langchain_core.output_parsers import StrOutputParser # Added for summarization
from langchain_community.embeddings import OllamaEmbeddings

class Application:
    """
    The main application class that orchestrates the RAG pipeline.
    """
    def __init__(self):
        self.config = self._load_config()
        self.args = self._parse_args()
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.doc_parser = None
        self.chunker = None
        self.java_chunker = None
        self.loader = None
        self.pipeline = None
        self.is_setup = False

    def _load_config(self, config_path="config.json"):
        """Loads the configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Configuration file not found at {config_path}")
        except json.JSONDecodeError:
            raise RuntimeError(f"Could not decode JSON from {config_path}")

    def _parse_args(self):
        """Parses command-line arguments."""
        data_config = self.config.get('data_processing', {})
        pipeline_config = self.config.get('pipeline_settings', {})

        parser = argparse.ArgumentParser(description="A modular RAG application with Ollama. Command-line arguments override config.json settings.")
        parser.add_argument("--docs-path", type=str, default=data_config.get('docs_path', './sample_data'), help="Path to the directory containing documents to process.")
        parser.add_argument("--model", type=str, default=pipeline_config.get('model', 'llama3'), help="The name of the Ollama model to use.")
        parser.add_argument("--extensions", nargs='+', default=data_config.get('allowed_extensions', ['.txt', '.md', '.py', '.json', '.html', '.htm', '.java']), help="List of file extensions to process (e.g., .txt .py .md).")
        parser.add_argument("--chunk-size", type=int, default=data_config.get('chunk_size', 1000), help="The maximum size of each chunk (in characters).")
        parser.add_argument("--chunk-overlap", type=int, default=data_config.get('chunk_overlap', 200), help="The number of characters to overlap between chunks.")
        return parser.parse_args()

    def _setup_components(self):
        """Initializes and wires up all the components of the application."""
        print("Initializing core components...")
        self.llm = Ollama(model=self.args.model)
        self.embeddings = OllamaEmbeddings(model=self.args.model)
        self.vector_store: VectorStoreInterface = ChromaVectorStore(embedding_function=self.embeddings)
        self.doc_parser: ParserInterface = DefaultDocumentParser()
        self.chunker: ChunkerInterface = RecursiveTextChunker(chunk_size=self.args.chunk_size, chunk_overlap=self.args.chunk_overlap)
        try:
            self.java_chunker: ChunkerInterface = JavaCodeChunker()
        except ImportError:
            print("\nWarning: `tree_sitter_languages` is not installed or failed to initialize.")
            print("Java files will be chunked using the default text-based chunker.\n")
            self.java_chunker = None
        self.loader: LoaderInterface = FileSystemLoader(directory_path=self.args.docs_path, allowed_extensions=self.args.extensions)
        print("Core components initialized.")

    def setup(self):
        """Performs the one-time setup of the RAG pipeline by processing documents."""
        if self.is_setup:
            return

        print("Performing one-time application setup...")
        self._setup_components()

        try:
            raw_files = self.loader.load()
            documents = self.doc_parser.parse_documents(raw_files)
            if not documents:
                print(f"Warning: No documents found in '{self.args.docs_path}'. The RAG pipeline will operate without custom context.")
                self.is_setup = True
                return

            print(f"Loaded and parsed {len(documents)} documents.")
            
            # --- Intelligent Chunking Strategy ---
            chunked_documents = []
            java_docs = [doc for doc in documents if doc.metadata.get("language") == "java"]
            other_docs = [doc for doc in documents if doc.metadata.get("language") != "java"]

            if other_docs:
                other_chunks = self.chunker.chunk_documents(other_docs)
                chunked_documents.extend(other_chunks)
                print(f"Split {len(other_docs)} non-Java documents into {len(other_chunks)} chunks using the default chunker.")

            if java_docs and self.java_chunker:
                java_chunks = self.java_chunker.chunk_documents(java_docs)
                chunked_documents.extend(java_chunks)
                print(f"Split {len(java_docs)} Java documents into {len(java_chunks)} semantic chunks.")
            elif java_docs: # Fallback if java_chunker is not available
                java_chunks = self.chunker.chunk_documents(java_docs)
                chunked_documents.extend(java_chunks)
                print(f"Split {len(java_docs)} Java documents into {len(java_chunks)} chunks using the default chunker (semantic chunker not available).")

            self.vector_store.add_documents(chunked_documents)

            retriever = self.vector_store.as_retriever()
            pipeline_config = self.config.get('pipeline_settings', {})
            prompt_template = pipeline_config.get('prompt_template', "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}")
            self.pipeline = RAGPipeline(llm=self.llm, retriever=retriever, prompt_template=prompt_template)

        except Exception as e:
            # Check for common Ollama connection errors
            if "503" in str(e) or "Connection refused" in str(e):
                print("\n---")
                print("FATAL: Could not connect to Ollama.")
                print("Please make sure the Ollama application is running and accessible before starting the server.")
                print(f"Original error: {e}")
                print("---\n")
                raise RuntimeError("Failed to connect to Ollama service.") from e
            
            raise RuntimeError(f"Failed to set up the RAG pipeline due to a data processing error: {e}") from e

        self.is_setup = True
        print("Application setup complete.")

    def ask(self, question: str) -> str:
        """Asks a question to the RAG pipeline. Will trigger setup on first run."""
        if not self.is_setup:
            self.setup()
        if not self.pipeline:
            return "The RAG pipeline is not available, likely because no documents were found during setup."
        return self.pipeline.ask(question)

    def run_cli(self):
        """Runs the application with an interactive command-line interface."""
        self.setup()
        self._start_qa_loop()

    def _start_qa_loop(self):
        """Starts the interactive question and answer session in the terminal."""
        print("\n--- RAG Q&A Terminal ---")
        print("Ask a question about your documents. Type 'exit' to quit.")
        while True:
            question = input("\nQuestion: ")
            if question.lower() == 'exit':
                break
            answer = self.pipeline.ask(question)
            print(f"\nAnswer: {answer}")

    def summarize_conversation(self, conversation_text: str) -> str:
        """Summarizes a given conversation text using the LLM."""
        if not self.llm:
            # This case should ideally not happen if setup() was successful
            return "LLM not initialized for summarization."
        
        summarize_prompt = ChatPromptTemplate.from_template(
            "Please summarize the following conversation:\n\n{conversation}\n\nSummary:"
        )
        summarize_chain = summarize_prompt | self.llm | StrOutputParser()
        try:
            return summarize_chain.invoke({"conversation": conversation_text})
        except Exception as e:
            print(f"Error during summarization: {e}")
            return "Failed to generate summary."