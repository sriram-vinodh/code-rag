"""Unit tests for RAGPipeline."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from rag.flow.rag_pipeline import RAGPipeline
from rag.flow.types import PlanStep, StepComplexity


@pytest.mark.unit
class TestRAGPipeline:
    """Test suite for RAGPipeline class."""
    
    def test_initialization(self, mock_llm, mock_neo4j_retriever, mock_mcp_client):
        """Test RAGPipeline initialization."""
        pipeline = RAGPipeline(
            llm=mock_llm,
            embeddings=None,
            neo4j_retriever=mock_neo4j_retriever,
            mcp_client=mock_mcp_client
        )
        
        assert pipeline.llm == mock_llm
        assert pipeline.neo4j_retriever == mock_neo4j_retriever
        assert pipeline.mcp_client == mock_mcp_client
        assert pipeline.prompt_processor is not None
        assert pipeline.step_executor is not None
    
    def test_initialization_without_optional_components(self, mock_llm):
        """Test RAGPipeline can initialize without optional components."""
        pipeline = RAGPipeline(llm=mock_llm)
        
        assert pipeline.llm == mock_llm
        assert pipeline.neo4j_retriever is None
        assert pipeline.mcp_client is None
    
    def test_load_templates_success(self, mock_llm, tmp_path):
        """Test template loading from file."""
        # Create test config and templates
        config = {"pipeline_settings": {"templates_file": "test_templates.json"}}
        templates = {
            "rag": {
                "answer": "Test template: {context} {question}",
                "cypher_generation": "Generate: {question}"
            }
        }
        
        config_path = tmp_path / "config.json"
        templates_path = tmp_path / "test_templates.json"
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        with open(templates_path, 'w') as f:
            json.dump(templates, f)
        
        # Change to tmp directory for the test
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            pipeline = RAGPipeline(llm=mock_llm)
            assert "answer" in pipeline.templates
            assert pipeline.templates["answer"] == "Test template: {context} {question}"
        finally:
            os.chdir(original_dir)
    
    def test_load_templates_fallback_on_error(self, mock_llm, tmp_path):
        """Test template loading falls back to defaults on error."""
        # Create config pointing to non-existent template file
        config = {"pipeline_settings": {"templates_file": "nonexistent.json"}}
        config_path = tmp_path / "config.json"
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            pipeline = RAGPipeline(llm=mock_llm)
            # Should have fallback template
            assert "answer" in pipeline.templates
            assert "{context}" in pipeline.templates["answer"]
            assert "{question}" in pipeline.templates["answer"]
        finally:
            os.chdir(original_dir)
    
    def test_process_query_success(self, mock_llm, mock_neo4j_retriever):
        """Test successful query processing."""
        from rag.flow.types import PlanStep, StepComplexity
        from rag.flow.task_execution_plan import TaskExecutionPlan
        
        # Create proper plan instead of Mock
        plan_steps = [
            PlanStep(
                step="Test step",
                needs_context=False,
                complexity=StepComplexity.LOW
            )
        ]
        plan = TaskExecutionPlan(total_steps=1, steps=plan_steps)
        
        mock_llm.responses = ["Step executed", "Final answer"]
        
        pipeline = RAGPipeline(
            llm=mock_llm,
            neo4j_retriever=mock_neo4j_retriever
        )
        
        with patch.object(pipeline.prompt_processor, 'get_execution_plan') as mock_plan, \
             patch.object(pipeline.step_executor, 'execute_plan') as mock_execute:
            
            mock_plan.return_value = plan  # ✅ Returns TaskExecutionPlan
            mock_execute.return_value = "Final answer"
            
            result = pipeline.process_query("What is the meaning of life?")
            
            assert result == "Final answer"
            mock_plan.assert_called_once_with("What is the meaning of life?")
            mock_execute.assert_called_once()
    
    def test_process_query_handles_exception(self, mock_llm):
        """Test query processing handles exceptions gracefully."""
        pipeline = RAGPipeline(llm=mock_llm)
        
        with patch.object(pipeline.prompt_processor, 'get_execution_plan') as mock_plan:
            mock_plan.side_effect = Exception("Test error")
            
            result = pipeline.process_query("What causes this error?")
            
            assert "error occurred" in result.lower()
            assert "Test error" in result
    
    def test_ask_delegates_to_process_query(self, mock_llm):
        """Test ask() method delegates to process_query()."""
        pipeline = RAGPipeline(llm=mock_llm)
        
        with patch.object(pipeline, 'process_query') as mock_process:
            mock_process.return_value = "Expected result"
            
            result = pipeline.ask("Test question")
            
            assert result == "Expected result"
            mock_process.assert_called_once_with("Test question")
    
    def test_ask_with_only_graph_parameter(self, mock_llm):
        """Test ask() accepts only_graph parameter."""
        pipeline = RAGPipeline(llm=mock_llm)
        
        with patch.object(pipeline, 'process_query') as mock_process:
            mock_process.return_value = "Graph result"
            
            # Parameter is accepted but delegated to process_query
            result = pipeline.ask("Graph question", only_graph=True)
            
            assert result == "Graph result"
            mock_process.assert_called_once_with("Graph question")


@pytest.mark.unit
class TestRAGPipelineIntegration:
    """Integration tests for RAGPipeline with its components."""
    
    def test_full_query_flow_with_mocks(self, mock_llm, mock_neo4j_retriever, test_templates):
        """Test full query processing flow with properly mocked components."""
        from rag.flow.types import PlanStep, StepComplexity
        from rag.flow.task_execution_plan import TaskExecutionPlan
        
        # Create a proper plan to return
        plan_steps = [
            PlanStep(
                step="Find class TestClass",
                needs_context=True,
                complexity=StepComplexity.MEDIUM,
                knowledge_source="graph_database"
            )
        ]
        plan = TaskExecutionPlan(total_steps=1, steps=plan_steps)
        
        # Setup LLM responses properly
        mock_llm.responses = [
            "Final answer: TestClass is a sample class used for testing."
        ]
        
        # Create pipeline with mocks
        pipeline = RAGPipeline(
            llm=mock_llm,
            neo4j_retriever=mock_neo4j_retriever
        )
        
        # Mock the plan generation to return our proper plan
        with patch.object(pipeline.prompt_processor, 'get_execution_plan', return_value=plan):
            result = pipeline.ask("What is TestClass?")
        
        # Should have called LLM for step execution
        assert mock_llm.call_count >= 1
        assert isinstance(result, str)
        assert len(result) > 0
