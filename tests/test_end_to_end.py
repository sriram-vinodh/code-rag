import unittest
from unittest.mock import MagicMock, patch
import json
from rag.flow.rag_pipeline import RAGPipeline
from rag.flow.types import PlanStep

class TestEndToEnd(unittest.TestCase):

    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_retriever = MagicMock()
        self.mock_mcp = MagicMock()
        
        # Initialize pipeline with mocks
        self.pipeline = RAGPipeline(
            llm=self.mock_llm,
            neo4j_retriever=self.mock_retriever,
            mcp_client=self.mock_mcp
        )

    @patch("rag.flow.rag_pipeline.PromptProcessor")
    def test_enhance_cachekey_flow(self, MockPromptProcessor):
        """
        End-to-end test for the user request: 'Enhance class CacheKey to autogenerate Keys'.
        Mocks LLM and Neo4j to simulate a successful execution flow.
        """
        user_query = "Enhance class CacheKey to autogenerate Keys"

        # --- Mock 1: Plan Generation ---
        # Patch PromptProcessor to return our desired plan directly
        from rag.flow.task_execution_plan import TaskExecutionPlan
        
        plan_data = {
            "total_steps": 2,
            "steps": [
                {
                    "step": "Retrieve the CacheKey class definition and its methods",
                    "complexity": "low",
                    "needs_context": True,
                    "knowledge_source": "graph_database",
                    "required_context_description": "Get the CacheKey class and its methods"
                },
                {
                    "step": "Modify CacheKey to add key autogeneration logic",
                    "complexity": "medium",
                    "needs_context": True,
                    "knowledge_source": "hybrid",
                    "required_context_description": "Current implementation of CacheKey",
                    "target_symbols": ["CacheKey"],
                    "edit_strategy": "symbolic"
                }
            ]
        }
        
        # Configure the mock instance returned by the class mock
        mock_processor_instance = MockPromptProcessor.return_value
        mock_processor_instance.get_execution_plan.return_value = TaskExecutionPlan.from_dict(plan_data)
        
        # Re-initialize pipeline to use the mocked processor
        # (RAGPipeline creates PromptProcessor in __init__)
        self.pipeline = RAGPipeline(
            llm=self.mock_llm,
            neo4j_retriever=self.mock_retriever,
            mcp_client=self.mock_mcp
        )
        # Ensure the pipeline uses our mock instance (it should if patch works correctly on class init)
        # But RAGPipeline instantiates PromptProcessor(llm). 
        # The patch replaces the class, so instantiation returns the mock instance.
        
        # --- Mock 2: Step Execution & Context Retrieval ---
        # The StepExecutor will call the LLM for:
        # 1. Cypher generation (if using fallback) or MCP query
        # 2. Step execution (interpreting context and generating result)
        # 3. Meta-validation (checking if step is complete)
        
        # We'll use side_effect to return different responses based on the input prompt
        def llm_side_effect(prompt):
            prompt_str = str(prompt)
            
            # 2. Cypher Generation (for context retrieval)
            if "convert it to a cypher query" in prompt_str.lower():
                return "MATCH (c:Class {name: 'CacheKey'}) RETURN c"
            
            # 3. Step 1 Execution (Retrieve context)
            if "Retrieve the CacheKey class" in prompt_str:
                return "I have retrieved the CacheKey class. It has methods for key management."
            
            # 4. Step 2 Execution (Modify code)
            if "Modify CacheKey" in prompt_str:
                return """I will add a generateKey method.
                <code_edit>
                  <type>symbolic</type>
                  <target>CacheKey/generateKey</target>
                  <file>code/source/agent/com/zoho/zic/cache/CacheKey.java</file>
                  <body>
                  public String generateKey() {
                      return UUID.randomUUID().toString();
                  }
                  </body>
                </code_edit>
                """
            
            # 5. Meta-validation
            if "validating if a task step" in prompt_str.lower():
                return json.dumps({
                    "is_complete": True,
                    "confidence": 0.9,
                    "reason": "Step completed successfully"
                })
                
            return "Generic LLM response"

        self.mock_llm.predict.side_effect = llm_side_effect
        # Handle invoke() as well if used
        self.mock_llm.invoke.side_effect = lambda p: MagicMock(content=llm_side_effect(p))

        # --- Mock 3: Neo4j/MCP Context Retrieval ---
        # Mock MCP client to return graph data
        self.mock_mcp.query_with_natural_language.return_value = [
            {
                "name": "CacheKey",
                "type": "Class",
                "file_path": "code/source/agent/com/zoho/zic/cache/CacheKey.java",
                "code": "public class CacheKey { ... }"
            }
        ]
        
        # --- Execute Pipeline ---
        result = self.pipeline.process_query(user_query)

        # --- Assertions ---
        self.assertIn("I have retrieved the CacheKey class", result)
        self.assertIn("generateKey", result)
        
        # Verify MCP was queried
        self.mock_mcp.query_with_natural_language.assert_called()


if __name__ == "__main__":
    unittest.main()
