import argparse
import sys
import json
import logging
import multiprocessing

from loader_interface import LoaderInterface
from filesystem_loader import FileSystemLoader
from parser_interface import ParserInterface
from document_parser import DefaultDocumentParser
from chunker_interface import ChunkerInterface
from text_chunker import RecursiveTextChunker
from cpg_chunker import CpgCodeChunker
from java_code_chunker import JavaCodeChunker
from rag_pipeline import RAGPipeline
from storage_interface import VectorStoreInterface
from chroma_vector_store import ChromaVectorStore
from langchain_ollama import OllamaLLM # Updated import
from langchain_core.prompts import ChatPromptTemplate # Added for summarization
from langchain_core.output_parsers import StrOutputParser # Added for summarization
from langchain_ollama import OllamaEmbeddings # Updated import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Background CPG Generation Function ---
# This function runs in a separate process and must be defined at the top level
# so that it can be pickled by multiprocessing.
def _generate_cpgs_in_background(code_path, extensions, cpg_vector_store_path, model_name):
    """
    Generates CPGs for Java files in the background and stores them in a separate ChromaDB.
    """
    logger.info(f"Background CPG Generation Started for {code_path}")
    try:
        # Initialize components within the new process
        embeddings = OllamaEmbeddings(model=model_name) # Use updated OllamaEmbeddings
        cpg_vector_store = ChromaVectorStore(
            embedding_function=embeddings,
            persist_directory=cpg_vector_store_path
        )
        doc_parser = DefaultDocumentParser()
        cpg_chunker = CpgCodeChunker()
        code_loader = FileSystemLoader(directory_path=code_path, allowed_extensions=extensions)

        # Load and parse Java files
        raw_code_files = code_loader.load()
        java_docs = [doc for doc in doc_parser.parse_documents(raw_code_files) if doc.metadata.get("language") == "java"]

        if not java_docs:
            logger.info("No Java documents found for background CPG generation.")
            return

        logger.info(f"Found {len(java_docs)} Java documents for background CPG generation.")
        cpg_chunks = cpg_chunker.chunk_documents(java_docs)
        logger.info(f"Generated {len(cpg_chunks)} CPG chunks in background.")
        cpg_vector_store.add_documents(cpg_chunks)
        logger.info(f"CPG chunks stored persistently at {cpg_vector_store_path}")

    except Exception as e:
        logger.error(f"Error during background CPG generation: {e}", exc_info=True)
    finally:
        logger.info("--- Background CPG Generation Finished ---")

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
        self.code_loader = None
        self.doc_loader = None
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
        parser.add_argument("--code-path", type=str, default=data_config.get('code_path', './code'), help="Path to the directory containing source code.")
        parser.add_argument("--docs-path", type=str, default=data_config.get('docs_path', './docs'), help="Path to the directory containing documentation.")
        parser.add_argument("--model", type=str, default=pipeline_config.get('model', 'llama3'), help="The name of the Ollama model to use.")
        parser.add_argument("--extensions", nargs='+', default=data_config.get('allowed_extensions', ['.txt', '.md', '.py', '.json', '.html', '.htm', '.java']), help="List of file extensions to process (e.g., .txt .py .md).")
        parser.add_argument("--chunk-size", type=int, default=data_config.get('chunk_size', 1000), help="The maximum size of each chunk (in characters).")
        parser.add_argument("--chunk-overlap", type=int, default=data_config.get('chunk_overlap', 200), help="The number of characters to overlap between chunks.")
        parser.add_argument("--java-chunker-strategy", type=str, default=data_config.get('java_chunker_strategy', 'cpg'), choices=['ast', 'cpg'], help="The chunking strategy for Java files: 'ast' or 'cpg'.")
        parser.add_argument("--background-cpg-generation-enabled", type=bool, default=data_config.get('background_cpg_generation_enabled', False), help="Enable background CPG generation.")
        parser.add_argument("--cpg-vector-store-path", type=str, default=data_config.get('cpg_vector_store_path', './chroma_cpg_db'), help="Path to store the persistent CPG vector database.")
        parser.add_argument("--vector-store-path", type=str, default=data_config.get('vector_store_path', './chroma_db'), help="Path to store the persistent vector database.")
        # Use parse_known_args() to ignore unrecognized arguments, making it
        # compatible with environments like the uvicorn reloader.
        args, _ = parser.parse_known_args()
        return args

    def _initialize_llm_and_embeddings(self):
        """Initializes the Language Model and Embedding functions."""
        logger.info("Initializing LLM and Embeddings...")
        self.llm = OllamaLLM(model=self.args.model)
        self.embeddings = OllamaEmbeddings(model=self.args.model)
        logger.info("LLM and Embeddings initialized.")

    def _initialize_vector_store(self):
        """Initializes the persistent vector store."""
        logger.info("Initializing vector store...")
        self.vector_store: VectorStoreInterface = ChromaVectorStore(
            embedding_function=self.embeddings,
            persist_directory=self.args.vector_store_path
        )
        logger.info("Vector store initialized.")

    def _initialize_parsers_and_chunkers(self):
        """Initializes document parser and generic text chunker."""
        logger.info("Initializing document parser and generic chunker...")
        self.doc_parser: ParserInterface = DefaultDocumentParser()
        self.chunker: ChunkerInterface = RecursiveTextChunker(
            chunk_size=self.args.chunk_size,
            chunk_overlap=self.args.chunk_overlap
        )
        logger.info("Document parser and generic chunker initialized.")

    def _select_and_initialize_java_chunker(self):
        """Selects and initializes the appropriate Java chunker based on configuration."""
        self.java_chunker = None # Default to no specialized Java chunker
        java_chunker_classes = {
            'cpg': CpgCodeChunker,
            'ast': JavaCodeChunker
        }
        
        strategies_to_try = []
        if self.args.java_chunker_strategy:
            strategies_to_try.append(self.args.java_chunker_strategy)
        
        configured_fallbacks = self.config.get('data_processing', {}).get('java_chunker_fallback_order', ['cpg', 'ast'])
        for strategy in configured_fallbacks:
            if strategy not in strategies_to_try:
                strategies_to_try.append(strategy)

        for strategy_name in strategies_to_try:
            if strategy_name in java_chunker_classes:
                chunker_class = java_chunker_classes[strategy_name]
                try:
                    self.java_chunker = chunker_class()
                    logger.info(f"Initialized {strategy_name.upper()} chunker for Java files.")
                    break
                except (ImportError, FileNotFoundError, RuntimeError) as e:
                    logger.warning(f"Could not initialize {strategy_name.upper()} chunker ({e}).")
                    if strategy_name == self.args.java_chunker_strategy and len(strategies_to_try) > strategies_to_try.index(strategy_name) + 1:
                        logger.info(f"Attempting to use next available chunker from fallback order: {strategies_to_try[strategies_to_try.index(strategy_name)+1:]}")
            else:
                logger.warning(f"Unknown Java chunker strategy '{strategy_name}' specified in config. Skipping.")

        if not self.java_chunker:
            logger.warning("No specialized Java chunker could be initialized. Java files will be chunked using the default text-based chunker.")

    def _start_background_cpg_generation(self):
        """Initiates background CPG generation if enabled in configuration."""
        if self.args.background_cpg_generation_enabled:
            logger.info("Starting background CPG generation process...")
            cpg_process = multiprocessing.Process(target=_generate_cpgs_in_background, args=(self.args.code_path, self.args.extensions, self.args.cpg_vector_store_path, self.args.model))
            cpg_process.start()
            logger.info("Background CPG generation process initiated. Main application will continue startup.")

    def _setup_components(self):
        """Initializes and wires up all the components of the application."""
        logger.info("Initializing core components...")
        self._initialize_llm_and_embeddings()
        self._initialize_vector_store()
        self._initialize_parsers_and_chunkers()
        self._select_and_initialize_java_chunker()
        self._start_background_cpg_generation()
        self.code_loader: LoaderInterface = FileSystemLoader(directory_path=self.args.code_path, allowed_extensions=self.args.extensions)
        self.doc_loader: LoaderInterface = FileSystemLoader(directory_path=self.args.docs_path, allowed_extensions=self.args.extensions)
        logger.info("Core components initialized.")

    def setup(self):
        """Performs the one-time setup of the RAG pipeline by processing documents."""
        if self.is_setup:
            return

        logger.info("Performing one-time application setup...")
        self._setup_components()

        try:
            parsed_code_docs, parsed_docs = self._load_and_parse_all_documents()
            documents = parsed_code_docs + parsed_docs

            if not documents:
                self._handle_no_documents_found()
                return

            logger.info(f"Loaded and parsed {len(documents)} documents.")
            
            chunked_documents = self._chunk_documents_by_type(documents)
            logger.info(f"Total documents chunked: {len(chunked_documents)}.")

            self._add_chunks_to_vector_store_and_build_pipeline(chunked_documents)

        except Exception as e:
            self._handle_setup_exception(e)

        self.is_setup = True
        logger.info("Application setup complete.")

    def _load_and_parse_all_documents(self):
        """Loads and parses documents from code and docs directories."""
        logger.info("Loading and parsing documents from specified paths...")
        raw_code_files = self.code_loader.load()
        raw_doc_files = self.doc_loader.load()
        
        parsed_code_docs = self.doc_parser.parse_documents(raw_code_files)
        parsed_docs = self.doc_parser.parse_documents(raw_doc_files)
        return parsed_code_docs, parsed_docs

    def _handle_no_documents_found(self):
        """Handles the scenario where no documents are found during setup."""
        logger.warning(f"No documents found in '{self.args.code_path}' or '{self.args.docs_path}'. The RAG pipeline will operate without custom context.")
        try:
            self.vector_store.as_retriever()
            logger.info("Loaded existing vector store despite no new documents.")
        except ValueError:
            logger.warning("No documents to process and no existing vector store found. RAG pipeline will be limited.")

    def _chunk_documents_by_type(self, documents: list) -> list:
        """Chunks documents based on their type (Java vs. others)."""
        chunked_documents = []
        java_docs = [doc for doc in documents if doc.metadata.get("language") == "java"]
        other_docs = [doc for doc in documents if doc.metadata.get("language") != "java"]

        if other_docs:
            other_chunks = self.chunker.chunk_documents(other_docs)
            chunked_documents.extend(other_chunks)
            logger.info(f"Split {len(other_docs)} non-Java documents into {len(other_chunks)} chunks using the default chunker.")

        if java_docs:
            if self.java_chunker:
                try:
                    java_chunks = self.java_chunker.chunk_documents(java_docs)
                    chunked_documents.extend(java_chunks)
                    logger.info(f"Split {len(java_docs)} Java documents into {len(java_chunks)} semantic chunks using the '{self.java_chunker.__class__.__name__}' strategy.")
                except Exception as e:
                    logger.warning(f"The '{self.java_chunker.__class__.__name__}' failed during chunking: {e}. Falling back to default text chunker.")
                    java_chunks = self.chunker.chunk_documents(java_docs)
                    chunked_documents.extend(java_chunks)
                    logger.info(f"Split {len(java_docs)} Java documents into {len(java_chunks)} chunks using the default chunker as a fallback.")
            else:
                java_chunks = self.chunker.chunk_documents(java_docs)
                chunked_documents.extend(java_chunks)
                logger.info(f"Split {len(java_docs)} Java documents into {len(java_chunks)} chunks using the default chunker (specialized Java chunker not available or failed).")
        return chunked_documents

    def _add_chunks_to_vector_store_and_build_pipeline(self, chunked_documents: list):
        """Adds chunks to the vector store and builds the RAG pipeline."""
        self.vector_store.add_documents(chunked_documents)
        retriever = self.vector_store.as_retriever()
        pipeline_config = self.config.get('pipeline_settings', {})
        prompt_template = pipeline_config.get('prompt_template', "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}")
        self.pipeline = RAGPipeline(llm=self.llm, retriever=retriever, prompt_template=prompt_template)

    def _handle_setup_exception(self, e: Exception):
        """Handles exceptions during the setup process."""
        if "503" in str(e) or "Connection refused" in str(e):
            logger.critical("Could not connect to Ollama. Please make sure the Ollama application is running and accessible.")
            logger.error(f"Original error: {e}", exc_info=True)
            raise RuntimeError("Failed to connect to Ollama service.") from e
        
        raise RuntimeError(f"Failed to set up the RAG pipeline due to a data processing error: {e}") from e

    def setup(self):
        """Performs the one-time setup of the RAG pipeline by processing documents."""
        if self.is_setup:
            return

        logger.info("Performing one-time application setup...")
        self._setup_components()

        try:
            parsed_code_docs, parsed_docs = self._load_and_parse_all_documents()
            documents = parsed_code_docs + parsed_docs

            if not documents:
                self._handle_no_documents_found()
                return

            logger.info(f"Loaded and parsed {len(documents)} documents.")
            
            chunked_documents = self._chunk_documents_by_type(documents)
            logger.info(f"Total documents chunked: {len(chunked_documents)}.")

            self._add_chunks_to_vector_store_and_build_pipeline(chunked_documents)

        except Exception as e:
            self._handle_setup_exception(e)

        self.is_setup = True
        logger.info("Application setup complete.")

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
        logger.info("\n--- RAG Q&A Terminal ---")
        logger.info("Ask a question about your documents. Type 'exit' to quit.")
        while True:
            question = input("\nQuestion: ")
            if question.lower() == 'exit':
                break
            answer = self.pipeline.ask(question)
            logger.info(f"Answer: {answer}")

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
            logger.error(f"Error during summarization: {e}", exc_info=True)
            return "Failed to generate summary."