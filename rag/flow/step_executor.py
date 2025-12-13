import logging
import json
from typing import Dict, Optional, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from .task_execution_plan import TaskExecutionPlan
from .types import ExecutionResult, PlanStep

logger = logging.getLogger(__name__)

class StepExecutor:
    def __init__(self, llm: BaseLanguageModel, neo4j_retriever, mcp_client=None):
        """Initialize the step executor with LLM, retriever, MCP client and load templates."""
        self.llm = llm
        self.neo4j_retriever = neo4j_retriever
        self.mcp_client = mcp_client
        self.knowledge_sources = {
            "graph_database": self._fetch_graph_context
        }
        self.templates = self._load_templates()
        self.cached_schema = None  # Cache the schema to avoid repeated fetches

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

    def _get_schema(self) -> str:
        """Get and cache the Neo4j schema."""
        if self.cached_schema:
            return self.cached_schema
            
        try:
            if self.mcp_client:
                print("Fetching schema from Neo4j via MCP...")
                schema_result = self.mcp_client.get_schema(sample_size=1000)
                
                if schema_result and isinstance(schema_result, dict) and "content" in schema_result:
                    content = schema_result["content"]
                    if isinstance(content, list) and len(content) > 0:
                        schema_text = content[0].get("text", "")
                        self.cached_schema = schema_text
                        print(f"Schema retrieved successfully")
                        return schema_text
            
            # Fallback to default schema description
            default_schema = """
Node Types: Class, Method, Field, Interface, Package
Relationships: HAS_METHOD, HAS_FIELD, CALLS, IMPLEMENTS_INTERFACE, CONTAINS
Properties: name, signature, type, file_path, code
"""
            self.cached_schema = default_schema
            return default_schema
            
        except Exception as e:
            logger.error(f"Failed to get schema: {e}")
            return "Schema unavailable"
    
    def _generate_cypher_from_nl(self, question: str) -> Optional[str]:
        """Convert natural language question to Cypher query using LLM with schema context."""
        try:
            print("\n" + "="*50)
            print("Translating Natural Language to Cypher")
            print(f"Question: {question}")
            print("="*50 + "\n")
            
            # Get the cypher generation template (includes schema)
            cypher_template = self.templates.get("cypher_generation", "")
            if not cypher_template:
                logger.error("Cypher generation template not found")
                return None
            
            # Create enhanced question with schema
            enhanced_prompt = cypher_template.format(question=question)
            
            # Use LLM to generate Cypher
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(enhanced_prompt)
                cypher_query = response.content if hasattr(response, 'content') else str(response)
            else:
                cypher_query = self.llm.predict(enhanced_prompt)
            
            # Clean up the query
            cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
            
            if not cypher_query:
                logger.warning("Generated empty Cypher query")
                return None
            
            print(f"\nGenerated Cypher Query:\n{cypher_query}\n")
            
            # Validate the query before returning
            if self._validate_cypher_query(cypher_query):
                return cypher_query
            else:
                logger.error("Generated Cypher query failed validation")
                return None
            
        except Exception as e:
            logger.error(f"Failed to generate Cypher query: {e}")
            return None
    
    def _validate_cypher_query(self, cypher_query: str) -> bool:
        """Validate Cypher query syntax using Neo4j EXPLAIN."""
        try:
            print("\n" + "="*50)
            print("Validating Cypher Query")
            print("="*50 + "\n")
            
            # Use EXPLAIN to validate syntax without executing
            explain_query = f"EXPLAIN {cypher_query}"
            
            if self.mcp_client:
                logger.info("Validating query via MCP")
                result = self.mcp_client.execute_read_query(explain_query)
                if result is not None:
                    print("✓ Query validation passed\n")
                    return True
                else:
                    print("✗ Query validation failed via MCP\n")
                    return False
            
            elif self.neo4j_retriever:
                logger.info("Validating query via direct Neo4j connection")
                result = self.neo4j_retriever.query(explain_query)
                if result is not None:
                    print("✓ Query validation passed\n")
                    return True
                else:
                    print("✗ Query validation failed\n")
                    return False
            
            logger.warning("No connection available for validation, skipping")
            return True  # Allow query if no validation possible
            
        except Exception as e:
            logger.error(f"Query validation error: {e}")
            print(f"✗ Validation error: {e}\n")
            return False
    
    def _validate_cypher_via_fallback(self, cypher_query: str) -> bool:
        """Validate Cypher query syntax using direct Neo4j connection."""
        try:
            explain_query = f"EXPLAIN {cypher_query}"
            
            if self.neo4j_retriever:
                result = self.neo4j_retriever.query(explain_query)
                if result is not None:
                    logger.info("Query validation passed via direct Neo4j")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Query validation error: {e}")
            return False
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
            
            logger.info("Reconstructed context using available information")
            return result
            
        except Exception as e:
            logger.error(f"Failed to reconstruct context: {e}")
            return None

    def _fetch_graph_context(self, context_description: str) -> str:
        """
        Fetch context from Neo4j graph database using natural language question.
        
        Process:
        MCP handles the full pipeline: NL -> Cypher translation -> validation -> execution
        
        Note: Only structural queries are supported (not behavioral analysis).
        """
        print("\n" + "="*50)
        print("Fetching Graph Context via MCP")
        print(f"Question: {context_description}")
        print("="*50 + "\n")
        
        try:
            results = None
            
            # Use MCP for full pipeline if available
            if self.mcp_client:
                logger.info("Processing query via MCP (NL -> Cypher -> Execute)")
                results = self.mcp_client.query_with_natural_language(context_description, self.llm)
            
            # Fallback: manual translation + direct Neo4j execution
            elif self.neo4j_retriever:
                logger.info("MCP not available, using fallback: manual translation + direct Neo4j")
                cypher_query = self._generate_cypher_from_nl(context_description)
                if cypher_query and self._validate_cypher_via_fallback(cypher_query):
                    results = self.neo4j_retriever.query(cypher_query)
            
            if not results:
                print("\nNo results returned from query")
                return "No matching data found in the graph database."
            
            # Format results
            print("\n" + "="*50)
            print(f"Retrieved {len(results)} result(s)")
            print("="*50 + "\n")
            
            formatted_context = self._format_graph_results(results)
            print(formatted_context)
            print("\n" + "="*50 + "\n")
            
            return formatted_context
            
        except Exception as e:
            logger.error(f"Error fetching graph context: {e}", exc_info=True)
            return f"Error querying graph database: {str(e)}"
    
    def _format_graph_results(self, results: List[Dict]) -> str:
        """Format graph query results for readability."""
        formatted_lines = []
        
        for i, result in enumerate(results, 1):
            formatted_lines.append(f"\n--- Result {i} ---")
            
            if not isinstance(result, dict):
                formatted_lines.append(str(result))
                continue
            
            for key, value in result.items():
                # Handle node objects
                if isinstance(value, dict):
                    formatted_lines.append(f"\n{key}:")
                    for prop_key, prop_value in value.items():
                        # Truncate code snippets for readability
                        if prop_key == "code" and isinstance(prop_value, str) and len(prop_value) > 500:
                            formatted_lines.append(f"  {prop_key}: {prop_value[:500]}... (truncated)")
                        else:
                            formatted_lines.append(f"  {prop_key}: {prop_value}")
                
                # Handle lists (relationships, collections)
                elif isinstance(value, list):
                    formatted_lines.append(f"\n{key}: ({len(value)} items)")
                    for idx, item in enumerate(value[:10], 1):  # Limit to 10 items
                        if isinstance(item, dict):
                            name = item.get('name', item.get('signature', f'item_{idx}'))
                            formatted_lines.append(f"  {idx}. {name}")
                            # Include code if available but truncated
                            if 'code' in item and item['code']:
                                code_preview = item['code'][:200] if len(item['code']) > 200 else item['code']
                                formatted_lines.append(f"     Code: {code_preview}{'...' if len(item['code']) > 200 else ''}")
                        else:
                            formatted_lines.append(f"  {idx}. {item}")
                    
                    if len(value) > 10:
                        formatted_lines.append(f"  ... and {len(value) - 10} more")
                
                # Handle simple values
                else:
                    formatted_lines.append(f"{key}: {value}")
        
        return "\n".join(formatted_lines)

    def _execute_step(self, step: PlanStep, accumulated_context: List[str]) -> ExecutionResult:
        try:
            logger.info("\n=== Executing Step ===\nDescription: %s\nComplexity: %s", 
                       step.step, step.complexity)

            # Build context section
            context_parts = []
            
            # Try to fetch or reconstruct context
            if step.needs_context and step.required_context_description:
                context = self._fetch_or_reconstruct_context(step)
                if context:
                    logger.info("\n--- Retrieved/Reconstructed Context ---\n%s\n------------------------", context)
                    context_parts.append(f"Retrieved context:\n{context}")
                else:
                    logger.warning("Failed to fetch or reconstruct context")
            
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
            
            logger.info("\n=== Step Execution Prompt ===\n%s\n=========================", prompt)

            result = self.llm.predict(prompt)
            logger.info("\n=== Step Result ===\n%s\n===================", result)
            
            return ExecutionResult(
                success=True,
                message=result,
                context=context_section
            )

        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return ExecutionResult(
                success=False,
                message=f"Failed to execute step: {str(e)}",
                context=None
            )

    def _format_plan(self, plan: TaskExecutionPlan) -> str:
        """Format the execution plan for display"""
        lines = [
            "="*50,
            "Execution Plan",
            "="*50,
            f"Total Steps: {plan.total_steps}\n"
        ]
        
        for i, step in enumerate(plan.steps, 1):
            lines.extend([
                f"Step {i}/{plan.total_steps}:",
                f"  Description: {step.step}",
                f"  Complexity: {step.complexity}",
                f"  Needs Context: {step.needs_context}"
            ])
            if step.needs_context:
                lines.extend([
                    f"  Context Source: {step.knowledge_source}",
                    f"  Context Description: {step.required_context_description}"
                ])
            lines.append("-"*40)
        
        return "\n".join(lines)

    def _format_step_status(self, step: PlanStep, current: int, total: int) -> str:
        """Format the current step status for display"""
        lines = [
            "="*50,
            f"Executing Step {current}/{total}",
            "="*50,
            f"Description: {step.step}",
            f"Complexity: {step.complexity}",
            f"Needs Context: {step.needs_context}"
        ]
        
        if step.needs_context:
            lines.extend([
                f"Context Source: {step.knowledge_source}",
                f"Context Description: {step.required_context_description}"
            ])
            
        if step.previous_step_result:
            lines.extend([
                "\nPrevious Step Result:",
                "-"*20,
                step.previous_step_result
            ])
            
        return "\n".join(lines)

    def execute_plan(self, plan: TaskExecutionPlan) -> str:
        """Execute each step in the plan, accumulating context and results."""
        accumulated_context = []
        final_results = []

        # Display the full plan at start
        print("\n" + self._format_plan(plan) + "\n")
        
        while (step := plan.get_next_step()) is not None:
            current_step = plan.current_step + 1
            
            # Display current step status
            print("\n" + self._format_step_status(step, current_step, plan.total_steps) + "\n")
            
            if step.needs_context:
                if not step.knowledge_source or not step.required_context_description:
                    print("Step needs context but missing source or description")
                    step.needs_context = False
                else:
                    print(f"Fetching context from: {step.knowledge_source}")
            
            result = self._execute_step(step, accumulated_context)
            plan.update_step_result(result)
            
            # Display step result
            print("\n" + "="*50)
            if result.success:
                print("Step Completed Successfully")
                if result.context:
                    print("\nNew Context Added:")
                    print("-"*40)
                    print(result.context)
                    accumulated_context.append(f"Context from step {current_step}:\n{result.context}")
                print("\nStep Result:")
                print("-"*40)
                print(result.message)
                final_results.append(result.message)
            else:
                print(f"Step {current_step} failed: {result.message}")
            print("="*50 + "\n")
            
            plan.advance()

        # Display execution summary
        print("\n" + "="*50)
        print(f"Plan Execution Complete")
        print(f"Successfully executed {len(final_results)}/{plan.total_steps} steps")
        print("="*50 + "\n")
        
        return "\n\n".join(final_results)
