from typing import Optional, Dict
import logging
import json
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from .types import ExecutionResult
from .prompt_processor import PromptProcessor
from .step_executor import StepExecutor
from .task_execution_plan import TaskExecutionPlan

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    The main RAG pipeline that orchestrates the execution of user queries.
    """
    def __init__(
        self,
        llm: BaseLanguageModel,
        embeddings: Optional[Embeddings] = None,
        neo4j_retriever: Optional[object] = None  # Neo4jGraphRetriever type
    ):
        """Initialize the RAG pipeline with required components."""
        self.llm = llm
        self.embeddings = embeddings
        self.neo4j_retriever = neo4j_retriever
        self.prompt_processor = PromptProcessor(llm)
        self.step_executor = StepExecutor(llm, neo4j_retriever)
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates from the configured file."""
        try:
            with open("config.json") as f:
                config = json.load(f)
            templates_file = config.get("pipeline_settings", {}).get("templates_file", "prompt_templates.json")
            
            with open(templates_file) as f:
                templates = json.load(f)
            logger.info(f"Loaded prompt templates from {templates_file}")
            return templates.get("rag", {})
        except Exception as e:
            logger.error(f"Failed to load prompt templates: {e}", exc_info=True)
            # Return a minimal default template
            return {
                "answer": "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}"
            }

    def process_query(self, query: str) -> str:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            query: The user's question or request
            
        Returns:
            str: The final response after executing all steps
        """
        try:
            # Generate execution plan
            plan = self.prompt_processor.get_execution_plan(query)
            logger.info(f"Generated execution plan with {len(plan)} steps")

            # Execute the plan
            result = self.step_executor.execute_plan(plan)
            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return f"An error occurred while processing your query: {str(e)}"

    def ask(self, question: str, only_graph: bool = False) -> str:
        """
        Process a user query through the RAG pipeline.
        This is the main entry point for question answering.
        
        Args:
            question: The user's question or request
            only_graph: If True, only use graph-based retrieval
            
        Returns:
            str: The final response after executing all steps
        """
        return self.process_query(question)
