import logging
import sys
import json
import os
import argparse
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from rag.flow.rag_pipeline import RAGPipeline
from rag.retriever.neo4j_graph_retriever import Neo4jGraphRetriever
from rag.retriever.cypher_query_helper import CypherQueryHelper
# Configure logging
def setup_logging():
    # Create formatters
    console_formatter = logging.Formatter('%(message)s')  # Simple format for readability
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
    
    # Console handler for execution flow (INFO and below)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(lambda record: record.levelno <= logging.INFO)
    
    # File handler for issues (WARNING and above)
    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler('logs/issues.log')
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(file_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add new handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

setup_logging()
logger = logging.getLogger(__name__)

class Application:
    """
    The main application class that orchestrates the RAG pipeline.
    This implementation focuses on a step-based execution flow with graph-based context retrieval.
    """
    def __init__(self, model_name: str = "llama3"):
        """
        Initialize the application with the specified model.
        
        Args:
            model_name: The name of the Ollama model to use
        """
        # Load configuration first
        self.config = self._load_config()
        
        # Parse arguments (which can override config)
        self.args = self._parse_args()
        
        # Configure logging with parsed arguments
        self._configure_logging()
        
        # Initialize components
        self.llm = None
        self.embeddings = None
        self.pipeline = None
        self.is_setup = False
        
        # Initialize graph retriever
        self.graph_retriever = Neo4jGraphRetriever()

    def _configure_logging(self):
        """Sets up application-wide logging."""
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
        for handler in logging.root.handlers:
            handler.setFormatter(formatter)
        logger.info("Logging configured")

    def _initialize_llm_and_embeddings(self):
        """Initializes the Language Model and Embedding functions."""
        logger.info("Initializing LLM and Embeddings...")
        self.llm = OllamaLLM(model=self.args.model)
        self.embeddings = OllamaEmbeddings(model=self.args.model)
        logger.info("LLM and Embeddings initialized.")

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
        logging_config = self.config.get('logging', {})

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
        parser.add_argument("--log-level", type=str.upper, default=logging_config.get('level', 'INFO'), choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level.")
        parser.add_argument("--force-reload", action="store_true", default=False, help="Force re-indexing of all documents, ignoring existing vector store data.")
        # Use parse_known_args() to ignore unrecognized arguments, making it
        # compatible with environments like the uvicorn reloader.
        args, _ = parser.parse_known_args()
        return args

    def _configure_logging(self):
        """Sets the application-wide logging level based on arguments."""
        log_level = self.args.log_level.upper()
        # Get the root logger and set its level. This will affect all loggers.
        logging.getLogger().setLevel(log_level)

        # Create a more detailed formatter for DEBUG mode
        if log_level == 'DEBUG':
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
        else:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

        # Update all existing handlers with the new formatter
        for handler in logging.root.handlers:
            handler.setFormatter(formatter)

        logger.info(f"Logging level set to {log_level}")

    def _initialize_llm_and_embeddings(self):
        """Initializes the Language Model and Embedding functions."""
        logger.info("Initializing LLM and Embeddings...")
        self.llm = OllamaLLM(model=self.args.model)
        self.embeddings = OllamaEmbeddings(model=self.args.model)
        logger.info("LLM and Embeddings initialized.")

    def _setup_components(self):
        """Initializes core components of the application."""
        logger.info("Initializing core components...")
        self._initialize_llm_and_embeddings()
        logger.info("Core components initialized.")

    def setup(self):
        """Performs the one-time setup of the RAG pipeline."""
        if self.is_setup:
            return

        logger.info("Performing application setup...")
        try:
            self._setup_components()
            
            # Initialize graph retriever
            if self.graph_retriever:
                logger.info("Initializing graph retriever...")
                self.graph_retriever.connect()
                logger.info("Graph retriever initialized.")
            
            # Build the pipeline
            self._build_pipeline_from_vector_store()
            
        except Exception as e:
            self._handle_setup_exception(e)

        self.is_setup = True
        logger.info("Application setup complete.")

    def _process_and_embed_documents(self):
        """Loads, parses, chunks, and embeds documents into the vector store."""
        logger.info("Vector store not found or is empty. Processing and embedding documents...")
        parsed_code_docs, parsed_docs = self._load_and_parse_all_documents()
        documents = parsed_code_docs + parsed_docs

        if not documents:
            logger.warning("No documents found to process. The vector store will be empty.")
            return

        logger.info(f"Loaded and parsed {len(documents)} documents.")
        chunked_documents = self._chunk_documents_by_type(documents)
        logger.info(f"Total documents chunked: {len(chunked_documents)}.")
        self.vector_store.add_documents(chunked_documents)

    def _load_and_parse_all_documents(self):
        """Loads and parses documents from code and docs directories."""
        raw_code_files = self.code_loader.load()
        raw_doc_files = self.doc_loader.load()
        
        parsed_code_docs = self.doc_parser.parse_documents(raw_code_files)
        parsed_docs = self.doc_parser.parse_documents(raw_doc_files)
        return parsed_code_docs, parsed_docs

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

    def _build_pipeline_from_vector_store(self):
        """Builds the RAG pipeline using the graph retriever."""
        logger.info("Building RAG pipeline...")
        self.pipeline = RAGPipeline(
            llm=self.llm,
            embeddings=self.embeddings,
            neo4j_retriever=self.graph_retriever
        )

    def _handle_setup_exception(self, e: Exception):
        """Handles exceptions during the setup process."""
        if "503" in str(e) or "Connection refused" in str(e):
            logger.critical("Could not connect to Ollama. Please make sure the Ollama application is running and accessible.")
            logger.error(f"Original error: {e}", exc_info=True)
            raise RuntimeError("Failed to connect to Ollama service.") from e
        
        raise RuntimeError(f"Failed to set up the RAG pipeline due to a data processing error: {e}") from e

    def _get_graph_context(self, question: str) -> str:
        """Attempts to retrieve relevant graph context from Neo4j for the question."""
        if not self.graph_retriever:
            logger.warning("Graph retriever is not initialized. Skipping graph context retrieval.")
            return ""

        cypher = CypherQueryHelper.build_flexible_query(question)
        if cypher:
            logger.info(f"Constructed Cypher query: {cypher}")
            try:
                results = self.graph_retriever.query(cypher)
                logger.info(f"Neo4j query result: {json.dumps(results, indent=2)}")
                # Extract method/class names from Neo4j result
                search_terms = set()
                for record in results:
                    for key, value in record.items():
                        if isinstance(value, dict) and 'name' in value:
                            search_terms.add(value['name'])
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict) and 'name' in item:
                                    search_terms.add(item['name'])
                logger.info(f"Search terms extracted from graphDB: {search_terms}")
                return '\n'.join(search_terms)
            except Exception as e:
                logger.warning(f"Neo4j graph retrieval failed: {e}")
        return ""

    def ask(self, question: str) -> str:
        """
        Asks a question to the RAG pipeline.
        This method processes the query through the step-based execution flow.
        """
        if not self.is_setup:
            self.setup()
            
        if not self.pipeline:
            return "The RAG pipeline is not available, likely because the setup failed."
            
        if not self.graph_retriever:
            logger.warning("Graph retriever is not available. Some functionality may be limited.")
            
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
            answer = self.ask(question)
            logger.info(f"Answer: {answer}")