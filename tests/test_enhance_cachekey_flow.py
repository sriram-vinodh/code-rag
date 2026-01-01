"""
Comprehensive Integration Test for the "Enhance class CacheKey to auto-set TTL" flow.

This test validates the complete RAG pipeline execution flow including:
1. Prompt Processing - Plan generation from user query
2. Step Execution - Context fetching and LLM-based step execution
3. Tool Usage - Serena tool request parsing and execution
4. Code Modification - Code edit block parsing
5. Completion Validation - Step completion checks

Usage:
    python -m pytest tests/test_enhance_cachekey_flow.py -v
    python tests/test_enhance_cachekey_flow.py  # Direct execution
"""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.flow.rag_pipeline import RAGPipeline
from rag.flow.prompt_processor import PromptProcessor
from rag.flow.step_executor import StepExecutor
from rag.flow.task_execution_plan import TaskExecutionPlan
from rag.flow.types import PlanStep, StepComplexity, ExecutionResult


class TestEnhanceCacheKeyFlow(unittest.TestCase):
    """Test suite for 'Enhance class CacheKey to auto-set TTL' prompt flow."""

    # =========================================================================
    # Test Fixtures
    # =========================================================================
    
    USER_QUERY = "Enhance class CacheKey to auto-set TTL"
    
    EXPECTED_PLAN = {
        "total_steps": 3,
        "steps": [
            {
                "step": "Analyze current implementation of class CacheKey",
                "needs_context": True,
                "complexity": "low",
                "required_context_description": "What are the fields and methods in class CacheKey?",
                "knowledge_source": "hybrid"
            },
            {
                "step": "Identify usage patterns",
                "needs_context": True,
                "complexity": "medium",
                "required_context_description": "Show me where class CacheKey is used",
                "knowledge_source": "graph_database"
            },
            {
                "step": "Suggest implementation changes for auto-set TTL",
                "needs_context": False,
                "complexity": "high"
            }
        ]
    }
    
    MOCK_CACHEKEY_CONTEXT = """
--- Result 1 ---
className: CacheKey
filePath: src/main/java/com/example/cache/CacheKey.java
package: com.example.cache

methods:
  1. getKey
     signature: public String getKey()
     code:
     public String getKey() {
         return this.prefix + ":" + this.name;
     }
     
  2. setExpiry
     signature: public void setExpiry(long seconds)
     code:
     public void setExpiry(long seconds) {
         this.expirySeconds = seconds;
     }
     
  3. hasExpiry
     signature: public boolean hasExpiry()
     code:
     public boolean hasExpiry() {
         return this.expirySeconds > 0;
     }

fields:
  - prefix (String)
  - name (String)
  - expirySeconds (long)
"""

    MOCK_USAGE_CONTEXT = """
--- Result 1 ---
calledBy: (3 items)
  1. CacheService.get - uses CacheKey to fetch from cache
  2. CacheService.put - uses CacheKey to store in cache
  3. SessionManager.createSession - creates CacheKey for session
"""

    # =========================================================================
    # Phase 1: Plan Generation Tests
    # =========================================================================

    @patch("rag.flow.prompt_processor.ChatPromptTemplate")
    @patch("rag.flow.prompt_processor.JsonOutputParser")
    def test_plan_generation_structure(self, MockParser, MockPrompt):
        """Test that the planner generates a valid execution plan structure."""
        mock_llm = MagicMock()
        
        # Mock the chain to return our expected plan
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = self.EXPECTED_PLAN
        
        # Setup the chain construction: prompt | llm | parser
        mock_prompt_instance = MagicMock()
        MockPrompt.from_template.return_value = mock_prompt_instance
        mock_prompt_instance.__or__ = MagicMock(return_value=MagicMock())
        mock_prompt_instance.__or__.return_value.__or__ = MagicMock(return_value=mock_chain)
        
        processor = PromptProcessor(mock_llm)
        plan = processor.get_execution_plan(self.USER_QUERY)
        
        # Validate plan structure
        self.assertIsInstance(plan, TaskExecutionPlan)
        self.assertEqual(plan.total_steps, 3)
        
        # Validate first step has context requirements
        first_step = plan.steps[0]
        self.assertTrue(first_step.needs_context)
        self.assertEqual(first_step.knowledge_source, "hybrid")
        self.assertIn("CacheKey", first_step.required_context_description)

    @patch("rag.flow.prompt_processor.ChatPromptTemplate")
    @patch("rag.flow.prompt_processor.JsonOutputParser")
    def test_plan_generation_uses_correct_knowledge_sources(self, MockParser, MockPrompt):
        """Test that the planner selects appropriate knowledge sources."""
        mock_llm = MagicMock()
        
        # Mock the chain
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = self.EXPECTED_PLAN
        
        mock_prompt_instance = MagicMock()
        MockPrompt.from_template.return_value = mock_prompt_instance
        mock_prompt_instance.__or__ = MagicMock(return_value=MagicMock())
        mock_prompt_instance.__or__.return_value.__or__ = MagicMock(return_value=mock_chain)
        
        processor = PromptProcessor(mock_llm)
        plan = processor.get_execution_plan(self.USER_QUERY)
        
        # Step 1 should use hybrid for comprehensive context
        self.assertEqual(plan.steps[0].knowledge_source, "hybrid")
        
        # Step 2 should use graph_database for usage patterns
        self.assertEqual(plan.steps[1].knowledge_source, "graph_database")
        
        # Step 3 should not need context (synthesis step)
        self.assertFalse(plan.steps[2].needs_context)

    @patch("rag.flow.prompt_processor.ChatPromptTemplate")
    @patch("rag.flow.prompt_processor.JsonOutputParser")
    def test_plan_generation_handles_malformed_json(self, MockParser, MockPrompt):
        """Test graceful handling when LLM returns malformed JSON."""
        mock_llm = MagicMock()
        
        # Mock the chain to raise an exception
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("Invalid JSON")
        
        mock_prompt_instance = MagicMock()
        MockPrompt.from_template.return_value = mock_prompt_instance
        mock_prompt_instance.__or__ = MagicMock(return_value=MagicMock())
        mock_prompt_instance.__or__.return_value.__or__ = MagicMock(return_value=mock_chain)
        
        processor = PromptProcessor(mock_llm)
        
        # Should return fallback plan, not raise
        plan = processor.get_execution_plan(self.USER_QUERY)
        
        # Fallback plan has 1 step
        self.assertEqual(plan.total_steps, 1)
        self.assertFalse(plan.steps[0].needs_context)

    # =========================================================================
    # Phase 2: Context Fetching Tests
    # =========================================================================

    def test_hybrid_context_fetching(self):
        """Test that hybrid knowledge source combines graph and Serena context."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_mcp = MagicMock()
        
        # Configure MCP to return graph context
        mock_mcp.query_with_natural_language.return_value = [
            {
                "className": "CacheKey",
                "file_path": "src/main/java/com/example/cache/CacheKey.java",
                "methods": [{"name": "getKey", "signature": "public String getKey()"}]
            }
        ]
        
        executor = StepExecutor(mock_llm, mock_retriever, mock_mcp)
        
        # Create a step requiring hybrid context
        step = PlanStep(
            step="Analyze current implementation of class CacheKey",
            needs_context=True,
            complexity=StepComplexity.LOW,
            required_context_description="What are the fields and methods in class CacheKey?",
            knowledge_source="hybrid"
        )
        
        # Fetch context
        context = executor._fetch_hybrid_context(step.required_context_description)
        
        # Should contain graph context
        self.assertIn("CacheKey", context)
        # MCP should have been called
        mock_mcp.query_with_natural_language.assert_called()

    def test_graph_context_fetching_no_results(self):
        """Test handling when graph database returns no results."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_mcp = MagicMock()
        
        # Configure MCP to return empty results
        mock_mcp.query_with_natural_language.return_value = []
        
        executor = StepExecutor(mock_llm, mock_retriever, mock_mcp)
        
        context = executor._fetch_graph_context("Find class NonExistent")
        
        # Should return informative message about no results
        self.assertIn("No matching data found", context)
        self.assertIn("CONTEXT_INSUFFICIENT_REFRESH", context)

    # =========================================================================
    # Phase 3: Step Execution Tests
    # =========================================================================

    def test_step_execution_with_context(self):
        """Test step execution with provided context."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_mcp = MagicMock()
        
        # Configure LLM responses
        def llm_side_effect(prompt):
            prompt_str = str(prompt)
            
            # Step execution response
            if "Current step:" in prompt_str and "CacheKey" in prompt_str:
                return """Analysis of CacheKey class:

The CacheKey class manages cache key generation with the following structure:
- Fields: prefix, name, expirySeconds
- Methods: getKey(), setExpiry(), hasExpiry()

Key findings:
1. The class lacks automatic TTL configuration
2. TTL must be manually set via setExpiry()

Recommendation: Add a default TTL in the constructor."""
            
            # Meta-validation response
            if "validating if a task step" in prompt_str.lower():
                return json.dumps({
                    "is_complete": True,
                    "confidence": 0.9,
                    "reason": "Analysis is comprehensive"
                })
            
            return "Generic response"
        
        mock_llm.predict.side_effect = llm_side_effect
        
        # Configure MCP to return context
        mock_mcp.query_with_natural_language.return_value = [
            {"className": "CacheKey", "methods": [{"name": "getKey"}]}
        ]
        
        executor = StepExecutor(mock_llm, mock_retriever, mock_mcp)
        
        step = PlanStep(
            step="Analyze current implementation of class CacheKey",
            needs_context=True,
            complexity=StepComplexity.LOW,
            required_context_description="What are the fields and methods in class CacheKey?",
            knowledge_source="graph_database"
        )
        
        result = executor._execute_single_step(step, [])
        
        self.assertTrue(result.success)
        self.assertIn("CacheKey", result.message)

    def test_step_execution_context_retry(self):
        """Test that context retry mechanism triggers on insufficient context."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_mcp = MagicMock()
        
        call_count = [0]
        
        def llm_side_effect(prompt):
            prompt_str = str(prompt)
            call_count[0] += 1
            
            # First call: Return insufficient context flag
            if call_count[0] == 1 and "Current step:" in prompt_str:
                return """[CONTEXT_INSUFFICIENT_REFRESH]
The provided context only shows the class structure but I need the actual method implementations.
Please retrieve the code for each method in CacheKey."""
            
            # Retry call: Return complete analysis
            if call_count[0] > 1 and "Current step:" in prompt_str:
                return """Analysis complete. CacheKey has getKey(), setExpiry(), hasExpiry() methods."""
            
            # Meta-validation
            if "validating if a task step" in prompt_str.lower():
                if call_count[0] <= 2:
                    return json.dumps({"is_complete": False, "confidence": 0.3, "reason": "Incomplete"})
                return json.dumps({"is_complete": True, "confidence": 0.9, "reason": "Complete"})
            
            return "Generic"
        
        mock_llm.predict.side_effect = llm_side_effect
        mock_mcp.query_with_natural_language.return_value = [{"name": "CacheKey"}]
        
        executor = StepExecutor(mock_llm, mock_retriever, mock_mcp)
        
        step = PlanStep(
            step="Analyze CacheKey",
            needs_context=True,
            complexity=StepComplexity.LOW,
            required_context_description="CacheKey class",
            knowledge_source="graph_database"
        )
        
        result = executor._execute_single_step(step, [], max_context_retries=2)
        
        # Should have retried at least once
        self.assertGreater(call_count[0], 1)
        self.assertTrue(result.success)

    # =========================================================================
    # Phase 4: Tool Request Parsing Tests
    # =========================================================================

    def test_tool_request_parsing(self):
        """Test parsing of XML-style tool requests from LLM output."""
        executor = StepExecutor(MagicMock(), MagicMock(), MagicMock())
        
        llm_output = """I need more information about the CacheKey class.

<tool_request><name>find_symbol</name><args>{"name_path": "CacheKey", "include_body": true}</args></tool_request>

Let me also check the directory structure:

<tool_request><name>list_dir</name><args>{"relative_path": "src/cache", "recursive": false}</args></tool_request>
"""
        
        requests = executor._parse_tool_requests(llm_output)
        
        self.assertEqual(len(requests), 2)
        
        # Check first request
        self.assertEqual(requests[0]["name"], "find_symbol")
        self.assertEqual(requests[0]["args"]["name_path"], "CacheKey")
        self.assertTrue(requests[0]["args"]["include_body"])
        
        # Check second request
        self.assertEqual(requests[1]["name"], "list_dir")
        self.assertEqual(requests[1]["args"]["relative_path"], "src/cache")

    def test_tool_request_parsing_no_requests(self):
        """Test parsing when no tool requests are present."""
        executor = StepExecutor(MagicMock(), MagicMock(), MagicMock())
        
        llm_output = """Based on the context, CacheKey needs a TTL field.
I recommend adding a defaultTtl field to the constructor."""
        
        requests = executor._parse_tool_requests(llm_output)
        
        self.assertEqual(len(requests), 0)

    def test_tool_request_parsing_malformed(self):
        """Test graceful handling of malformed tool requests."""
        executor = StepExecutor(MagicMock(), MagicMock(), MagicMock())
        
        llm_output = """
<tool_request><name>find_symbol</name><args>not valid json</args></tool_request>
<tool_request><name>list_dir</name><args>{"relative_path": "src"}</args></tool_request>
"""
        
        requests = executor._parse_tool_requests(llm_output)
        
        # Should skip malformed and return valid one
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["name"], "list_dir")

    # =========================================================================
    # Phase 5: Code Edit Parsing Tests
    # =========================================================================

    def test_code_edit_parsing(self):
        """Test parsing of code edit blocks from LLM output."""
        executor = StepExecutor(MagicMock(), MagicMock(), MagicMock())
        
        llm_output = """To add auto-TTL, I'll modify the CacheKey class:

<code_edit>
  <type>symbolic</type>
  <target>CacheKey/__init__</target>
  <file>src/cache/CacheKey.java</file>
  <body>
    public CacheKey(String prefix, String name) {
        this.prefix = prefix;
        this.name = name;
        this.expirySeconds = 3600; // Default 1 hour TTL
    }
  </body>
  <explanation>Add default TTL to constructor</explanation>
</code_edit>
"""
        
        edits = executor._parse_code_edits(llm_output)
        
        self.assertEqual(len(edits), 1)
        edit = edits[0]
        
        from rag.flow.types import EditStrategy
        self.assertEqual(edit.edit_type, EditStrategy.SYMBOLIC)
        self.assertEqual(edit.target_symbol, "CacheKey/__init__")
        self.assertEqual(edit.file_path, "src/cache/CacheKey.java")
        self.assertIn("expirySeconds = 3600", edit.new_body)
        self.assertEqual(edit.explanation, "Add default TTL to constructor")

    def test_code_edit_parsing_insertion(self):
        """Test parsing of insertion-type code edits."""
        executor = StepExecutor(MagicMock(), MagicMock(), MagicMock())
        
        llm_output = """
<code_edit>
  <type>insertion</type>
  <target>CacheKey/getKey</target>
  <file>src/cache/CacheKey.java</file>
  <body>
    public long getDefaultTtl() {
        return this.expirySeconds;
    }
  </body>
  <explanation>Insert getter for default TTL after getKey method</explanation>
</code_edit>
"""
        
        edits = executor._parse_code_edits(llm_output)
        
        self.assertEqual(len(edits), 1)
        from rag.flow.types import EditStrategy
        self.assertEqual(edits[0].edit_type, EditStrategy.INSERTION)

    # =========================================================================
    # Phase 6: Completion Validation Tests
    # =========================================================================

    def test_completion_validation_success(self):
        """Test step completion validation for successful execution."""
        mock_llm = MagicMock()
        mock_llm.predict.return_value = json.dumps({
            "is_complete": True,
            "confidence": 0.95,
            "reason": "Analysis is thorough and actionable"
        })
        
        executor = StepExecutor(mock_llm, MagicMock(), MagicMock())
        
        step = PlanStep(
            step="Analyze CacheKey",
            needs_context=True,
            complexity=StepComplexity.LOW
        )
        
        result = """The CacheKey class has three main methods:
1. getKey() - Returns the computed cache key
2. setExpiry() - Sets TTL manually
3. hasExpiry() - Checks if TTL is set

To add auto-TTL, modify the constructor to set a default expirySeconds value."""
        
        is_complete, confidence, reason = executor.isStepComplete(step, result, self.MOCK_CACHEKEY_CONTEXT)
        
        self.assertTrue(is_complete)
        self.assertGreater(confidence, 0.8)

    def test_completion_validation_hedging_detection(self):
        """Test that hedging language triggers incomplete status."""
        mock_llm = MagicMock()
        # Meta-validation won't be called if hedging is detected first
        
        executor = StepExecutor(mock_llm, MagicMock(), MagicMock())
        
        step = PlanStep(
            step="Analyze CacheKey",
            needs_context=True,
            complexity=StepComplexity.LOW
        )
        
        result = """I don't have enough information about the CacheKey class.
I cannot determine the current implementation without seeing the actual code."""
        
        is_complete, confidence, reason = executor.isStepComplete(step, result, None)
        
        self.assertFalse(is_complete)
        self.assertLess(confidence, 0.5)
        self.assertIn("hedging", reason.lower())

    def test_completion_validation_context_not_used(self):
        """Test detection when result doesn't reference provided context."""
        mock_llm = MagicMock()
        mock_llm.predict.return_value = json.dumps({
            "is_complete": True,
            "confidence": 0.9,
            "reason": "Complete"
        })
        
        executor = StepExecutor(mock_llm, MagicMock(), MagicMock())
        
        step = PlanStep(
            step="Analyze CacheKey",
            needs_context=True,
            complexity=StepComplexity.LOW
        )
        
        # Result that doesn't reference anything from context
        result = """A cache class typically stores key-value pairs.
You should add a timeout parameter to manage expiration."""
        
        # Context with specific identifiers
        context = """CacheKey class:
- getComputedKey() method
- shardPrefix field
- RedisConnectionPool dependency"""
        
        is_complete, confidence, reason = executor.isStepComplete(step, result, context)
        
        # Should detect that context wasn't used
        self.assertFalse(is_complete)
        self.assertIn("context", reason.lower())

    # =========================================================================
    # Phase 7: Full Pipeline Integration Test
    # =========================================================================

    @patch("rag.flow.rag_pipeline.PromptProcessor")
    def test_full_pipeline_execution(self, MockPromptProcessor):
        """Test complete pipeline execution from query to result."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_mcp = MagicMock()
        
        # Configure PromptProcessor mock
        mock_processor_instance = MockPromptProcessor.return_value
        mock_processor_instance.get_execution_plan.return_value = TaskExecutionPlan.from_dict(self.EXPECTED_PLAN)
        
        # Configure LLM responses for each phase
        execution_call_count = [0]
        
        def llm_side_effect(prompt):
            prompt_str = str(prompt)
            execution_call_count[0] += 1
            
            # Step 1: Analysis
            if "Analyze current implementation" in prompt_str:
                return f"""Analysis of CacheKey:
                
Based on the context, CacheKey has:
- getKey() method
- setExpiry() method for manual TTL
- expirySeconds field

The class lacks automatic TTL configuration."""
            
            # Step 2: Usage patterns
            if "Identify usage patterns" in prompt_str:
                return """Usage analysis:

CacheKey is used in:
1. CacheService.get() - for retrieving cached items
2. CacheService.put() - for storing items
3. SessionManager - for session caching

All callers assume manual TTL setting."""
            
            # Step 3: Implementation suggestions
            if "Suggest implementation" in prompt_str:
                return """Implementation recommendation:

Add auto-TTL by modifying CacheKey constructor:

<code_edit>
  <type>symbolic</type>
  <target>CacheKey/CacheKey</target>
  <file>src/cache/CacheKey.java</file>
  <body>
    public CacheKey(String prefix, String name) {
        this(prefix, name, 3600); // Default 1 hour
    }
    
    public CacheKey(String prefix, String name, long ttlSeconds) {
        this.prefix = prefix;
        this.name = name;
        this.expirySeconds = ttlSeconds;
    }
  </body>
  <explanation>Add default TTL with optional override</explanation>
</code_edit>"""
            
            # Meta-validation
            if "validating if a task step" in prompt_str.lower():
                return json.dumps({
                    "is_complete": True,
                    "confidence": 0.9,
                    "reason": "Step completed successfully"
                })
            
            return "Generic response"
        
        mock_llm.predict.side_effect = llm_side_effect
        mock_llm.invoke.side_effect = lambda p: MagicMock(content=llm_side_effect(str(p)))
        
        # Configure MCP responses
        mock_mcp.query_with_natural_language.return_value = [
            {"className": "CacheKey", "methods": [{"name": "getKey"}]}
        ]
        
        # Create pipeline
        pipeline = RAGPipeline(
            llm=mock_llm,
            neo4j_retriever=mock_retriever,
            mcp_client=mock_mcp
        )
        
        # Execute
        result = pipeline.process_query(self.USER_QUERY)
        
        # Verify pipeline completed
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        
        # Verify all steps were executed
        self.assertGreater(execution_call_count[0], 0)
        
        # Verify code edit was generated
        self.assertIn("code_edit", result.lower())

    # =========================================================================
    # Utility Tests
    # =========================================================================

    def test_hedging_language_detection(self):
        """Test the hedging language detection utility."""
        executor = StepExecutor(MagicMock(), MagicMock(), MagicMock())
        
        hedging_texts = [
            "I don't have enough information to complete this",
            "I cannot determine the implementation without more context",
            "It's unclear from the provided context",
            "More information is needed about the class",
            "I would need to see the actual code",
            "unable to proceed without more details"
        ]
        
        for text in hedging_texts:
            detected, reason = executor._detect_hedging_language(text)
            self.assertTrue(detected, f"Should detect hedging in: {text}")
        
        non_hedging_texts = [
            "The CacheKey class has three methods for key management",
            "Based on the code, TTL should be set in the constructor",
            "I recommend adding a default timeout parameter"
        ]
        
        for text in non_hedging_texts:
            detected, reason = executor._detect_hedging_language(text)
            self.assertFalse(detected, f"Should not detect hedging in: {text}")

    def test_context_usage_detection(self):
        """Test detection of context usage in results."""
        executor = StepExecutor(MagicMock(), MagicMock(), MagicMock())
        
        context = """CacheKey class:
Methods: getComputedKey, setShardPrefix, hasRedisConnection
Fields: shardId, connectionPool, maxRetries"""
        
        # Result that uses context
        good_result = """The CacheKey class uses getComputedKey() for key generation
and shardId for partitioning. The connectionPool manages Redis connections."""
        
        used, score = executor._check_context_usage(good_result, context)
        self.assertTrue(used)
        self.assertGreater(score, 0.2)
        
        # Result that ignores context
        bad_result = """A typical cache implementation uses string keys.
You should add timeout handling for better performance."""
        
        used, score = executor._check_context_usage(bad_result, context)
        self.assertFalse(used)
        self.assertLess(score, 0.2)


class TestEnhanceCacheKeyIntegration(unittest.TestCase):
    """Integration tests that test larger flows."""

    @patch("rag.flow.prompt_processor.ChatPromptTemplate")
    @patch("rag.flow.prompt_processor.JsonOutputParser")
    def test_plan_to_execution_flow(self, MockParser, MockPrompt):
        """Test the complete flow from plan generation to step execution."""
        mock_llm = MagicMock()
        
        # Plan generation response
        plan_data = {
            "total_steps": 2,
            "steps": [
                {
                    "step": "Analyze CacheKey",
                    "needs_context": True,
                    "complexity": "low",
                    "required_context_description": "CacheKey fields and methods",
                    "knowledge_source": "graph_database"
                },
                {
                    "step": "Suggest TTL implementation",
                    "needs_context": False,
                    "complexity": "medium"
                }
            ]
        }
        
        # Mock the chain for plan generation
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = plan_data
        
        mock_prompt_instance = MagicMock()
        MockPrompt.from_template.return_value = mock_prompt_instance
        mock_prompt_instance.__or__ = MagicMock(return_value=MagicMock())
        mock_prompt_instance.__or__.return_value.__or__ = MagicMock(return_value=mock_chain)
        
        call_count = [0]
        
        def llm_side_effect(prompt):
            call_count[0] += 1
            prompt_str = str(prompt)
            
            # Step execution
            if "Current step:" in prompt_str:
                if "Analyze" in prompt_str:
                    return "CacheKey has getKey(), setExpiry() methods. Needs default TTL."
                if "Suggest" in prompt_str:
                    return "Add defaultTtlSeconds = 3600 in constructor."
            
            # Validation
            if "validating if a task step" in prompt_str.lower():
                return json.dumps({"is_complete": True, "confidence": 0.9, "reason": "OK"})
            
            return "Response"
        
        mock_llm.predict.side_effect = llm_side_effect
        mock_llm.invoke.side_effect = lambda p: MagicMock(content=llm_side_effect(str(p)))
        
        # Mock MCP
        mock_mcp = MagicMock()
        mock_mcp.query_with_natural_language.return_value = [{"name": "CacheKey"}]
        
        # Create and run pipeline
        pipeline = RAGPipeline(
            llm=mock_llm,
            neo4j_retriever=MagicMock(),
            mcp_client=mock_mcp
        )
        
        result = pipeline.process_query("Enhance CacheKey for auto TTL")
        
        # Verify multiple LLM calls were made
        self.assertGreater(call_count[0], 2)
        
        # Result should contain suggestions
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
