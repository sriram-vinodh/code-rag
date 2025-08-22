import logging
import json
from typing import Dict, Optional, List
from helper_models.nl_to_cypher import generate_cypher
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from .task_execution_plan import TaskExecutionPlan
from .types import ExecutionResult, PlanStep

logger = logging.getLogger(__name__)

class StepExecutor:
    def __init__(self, llm: BaseLanguageModel, neo4j_retriever):
        """Initialize the step executor with LLM, retriever and load templates."""
        self.llm = llm
        self.neo4j_retriever = neo4j_retriever
        self.knowledge_sources = {
            "graph_database": self._fetch_graph_context
        }
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
            logger.error(f"Failed to load prompt templates: {e}")
            return {}

    def _get_cypher_query(self, question: str) -> Optional[str]:
        try:
            logger.info("\n=== 🤔 Generating Cypher Query ===\nInput: %s", question)
            cypher_query = generate_cypher(question)
            logger.info("\n=== 🔍 Generated Cypher Query ===\n%s\n==============================", cypher_query)
            return cypher_query
        except Exception as e:
            logger.error(f"❌ Failed to generate Cypher query: {e}")
            return None

    def _fetch_or_reconstruct_context(self, step: PlanStep) -> Optional[str]:
        """Fetch context from the knowledge source or reconstruct it if fetching fails."""
        # First try the specified knowledge source
        if step.knowledge_source:
            fetcher = self.knowledge_sources.get(step.knowledge_source)
            if fetcher:
                try:
                    context = fetcher(step.required_context_description)
                    if context:
                        return context
                except Exception as e:
                    logger.warning(f"Failed to fetch context from primary source: {e}")
        
        # If primary fetch fails, try to reconstruct from accumulated context
        try:
            # Create a targeted prompt to reconstruct the required context
            reconstruction_prompt = ChatPromptTemplate.from_template(
                self.templates.get("context_reconstruction", 
                "Given the context needed: {context_desc}\n"
                "And the information we have:\n{available_context}\n"
                "Please reconstruct or derive the relevant information to answer the context requirement.")
            )
            
            # Use the LLM to reconstruct context from available information
            result = self.llm.predict(reconstruction_prompt.format(
                context_desc=step.required_context_description,
                available_context=step.previous_step_result or "No previous context available"
            ))
            
            logger.info("🔄 Reconstructed context using available information")
            return result
            
        except Exception as e:
            logger.error(f"Failed to reconstruct context: {e}")
            return None

    def _fetch_graph_context(self, context_description: str) -> str:
        """Fetch context from the Neo4j graph database."""
        if not self.neo4j_retriever:
            logger.error("❌ Neo4j retriever not initialized")
            return ""
            
        logger.info("\n=== 🔍 Graph Context Request ===\nQuery: %s", context_description)
        
        cypher_query = self._get_cypher_query(context_description)
        if not cypher_query:
            logger.warning("⚠️ No Cypher query generated, skipping graph context fetch")
            return ""
        
        try:
            logger.info("\n=== 🔄 Executing Neo4j Query ===\n%s", cypher_query)
            results = self.neo4j_retriever.query(cypher_query)
            
            if not results:
                logger.warning("⚠️ No results returned from Neo4j query")
                return ""
            
            # Format results for better readability
            context = json.dumps(results, indent=2)
            logger.info("\n=== 📊 Retrieved Graph Context ===\nResults count: %d\nResults:\n%s\n==============================", 
                      len(results), context)
            
            # Create a more readable summary of the results
            summary = []
            for result in results:
                if isinstance(result, dict):
                    # Extract relevant information based on result structure
                    if "name" in result:
                        summary.append(f"Name: {result['name']}")
                    if "type" in result:
                        summary.append(f"Type: {result['type']}")
                    if "properties" in result:
                        props = result["properties"]
                        summary.append("Properties:")
                        for k, v in props.items():
                            summary.append(f"  - {k}: {v}")
                
                summary.append("---")
            
            formatted_context = "\n".join(summary)
            logger.info("\n=== 📝 Formatted Context ===\n%s\n==============================", formatted_context)
            
            return formatted_context
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch graph context: {e}", exc_info=True)
            return ""

    def _execute_step(self, step: PlanStep, accumulated_context: List[str]) -> ExecutionResult:
        try:
            logger.info("\n=== 🔄 Executing Step ===\nDescription: %s\nComplexity: %s", 
                       step.step, step.complexity)

            # Build context section
            context_parts = []
            
            # Try to fetch or reconstruct context
            if step.needs_context and step.required_context_description:
                context = self._fetch_or_reconstruct_context(step)
                if context:
                    logger.info("\n--- 📊 Retrieved/Reconstructed Context ---\n%s\n------------------------", context)
                    context_parts.append(f"Retrieved context:\n{context}")
                else:
                    logger.warning("⚠️ Failed to fetch or reconstruct context")
            
            # Add accumulated context
            if accumulated_context:
                context_parts.append("Previous context:\n" + "\n".join(accumulated_context))

            context_section = "\n".join(context_parts) if context_parts else "No additional context available."
            previous_result_section = f"Previous step result: {step.previous_step_result}" if step.previous_step_result else "This is the first step."

            # Execute step with LLM
            step_template = self.templates.get("step_execution", "")
            if not step_template:
                logger.error("Step execution template not found")
                return ExecutionResult(
                    success=False,
                    message="Failed to load step execution template",
                    context=None
                )

            prompt = ChatPromptTemplate.from_template(step_template).format(
                step_description=step.step,
                context_section=context_section,
                previous_result_section=previous_result_section
            )
            
            logger.info("\n=== 📝 Step Execution Prompt ===\n%s\n=========================", prompt)

            result = self.llm.predict(prompt)
            logger.info("\n=== ✅ Step Result ===\n%s\n===================", result)
            
            return ExecutionResult(
                success=True,
                message=result,
                context=context_section
            )

        except Exception as e:
            logger.error(f"❌ Step execution failed: {e}")
            return ExecutionResult(
                success=False,
                message=f"Failed to execute step: {str(e)}",
                context=None
            )

    def execute_plan(self, plan: TaskExecutionPlan) -> str:
        """Execute each step in the plan, accumulating context and results."""
        accumulated_context = []
        final_results = []

        logger.info("\n=== 🚀 Starting Plan Execution ===\nTotal steps: %d", plan.total_steps)
        
        while (step := plan.get_next_step()) is not None:
            current_step = plan.current_step + 1
            logger.info("\n=== 🔄 Step %d/%d ===\nDescription: %s\nNeeds Context: %s\nComplexity: %s", 
                       current_step, plan.total_steps, step.step, 
                       step.needs_context, step.complexity)
            
            if step.needs_context:
                if not step.knowledge_source or not step.required_context_description:
                    logger.warning("⚠️ Step needs context but missing source or description")
                    step.needs_context = False
                else:
                    logger.info("🔍 Will fetch context from: %s", step.knowledge_source)
            
            result = self._execute_step(step, accumulated_context)
            plan.update_step_result(result)
            
            if result.success:
                if result.context:
                    logger.info("📦 Adding context to accumulation")
                    accumulated_context.append(f"Context from step {current_step}:\n{result.context}")
                final_results.append(result.message)
                logger.info("✅ Step completed successfully")
            else:
                logger.warning("⚠️ Step %d failed, attempting to continue", current_step)
            
            plan.advance()

        logger.info("\n=== 🏁 Plan Execution Complete ===\nSuccessfully executed %d steps", len(final_results))
        return "\n\n".join(final_results)
