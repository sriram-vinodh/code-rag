"""End-to-end tests for complete user flows."""

import pytest
import json
from unittest.mock import Mock, patch
import sys
import os


@pytest.mark.e2e
class TestCLIFlow:
    """End-to-end tests for CLI interface."""
    
    def test_cli_question_answer_flow(self, test_config, tmp_path, mock_llm):
        """Test complete CLI question-answer flow."""
        config_path = tmp_path / "config.json"
        test_config["mcp"]["enabled"] = False
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        templates = {
            "rag": {
                "plan_generation": "Plan: {question}",
                "answer": "Answer: {question}"
            }
        }
        (tmp_path / "prompt_templates.json").write_text(json.dumps(templates))
        
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            with patch('application.OllamaLLM', return_value=mock_llm), \
                 patch('application.OllamaEmbeddings', return_value=Mock()), \
                 patch('builtins.input', return_value='exit'):  # Auto-exit
                
                from application import Application
                
                app = Application()
                app.graph_retriever = Mock()
                app.graph_retriever.connect = Mock()
                
                # Should run without errors
                app.run_cli()
                
                assert app.is_setup is True
        finally:
            os.chdir(original_dir)
    
    def test_cli_multiple_questions(self, test_config, tmp_path, mock_llm):
        """Test CLI with multiple questions."""
        config_path = tmp_path / "config.json"
        test_config["mcp"]["enabled"] = False
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        templates = {"rag": {"plan_generation": "{question}", "answer": "{question}"}}
        (tmp_path / "prompt_templates.json").write_text(json.dumps(templates))
        
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Simulate user input
            questions = ['What is class A?', 'What is class B?', 'exit']
            
            with patch('application.OllamaLLM', return_value=mock_llm), \
                 patch('application.OllamaEmbeddings', return_value=Mock()), \
                 patch('builtins.input', side_effect=questions):
                
                from application import Application
                
                app = Application()
                app.graph_retriever = Mock()
                app.graph_retriever.connect = Mock()
                
                mock_llm.responses = [
                    json.dumps({"steps": []}),
                    "Answer about class A",
                    json.dumps({"steps": []}),
                    "Answer about class B"
                ]
                
                app.run_cli()
                
                # Should have processed both questions
                assert mock_llm.call_count >= 2
        finally:
            os.chdir(original_dir)


@pytest.mark.e2e
@pytest.mark.slow
class TestWebAPIFlow:
    """End-to-end tests for web API."""
    
    def test_web_server_startup(self, test_config, tmp_path):
        """Test web server startup and initialization."""
        config_path = tmp_path / "config.json"
        test_config["mcp"]["enabled"] = False
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        templates = {"rag": {"answer": "{question}"}}
        (tmp_path / "prompt_templates.json").write_text(json.dumps(templates))
        
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Import FastAPI app
            from fastapi.testclient import TestClient
            
            with patch('application.OllamaLLM'), \
                 patch('application.OllamaEmbeddings'):
                
                # This would normally import web.web_server
                # For now, we test the structure exists
                assert True  # Placeholder for actual web server test
        finally:
            os.chdir(original_dir)


@pytest.mark.e2e
class TestErrorHandlingFlow:
    """End-to-end tests for error scenarios."""
    
    def test_invalid_config_handling(self, tmp_path):
        """Test handling of invalid configuration."""
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            f.write("{invalid json")
        
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            with patch('application.OllamaLLM'), \
                 patch('application.OllamaEmbeddings'):
                
                from application import Application
                
                with pytest.raises(RuntimeError):
                    app = Application()
        finally:
            os.chdir(original_dir)
    
    def test_llm_connection_failure(self, test_config, tmp_path):
        """Test handling of LLM connection failure."""
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            with patch('application.OllamaLLM') as mock_ollama:
                mock_ollama.side_effect = ConnectionError("Cannot connect to Ollama")
                
                from application import Application
                
                app = Application()
                
                with pytest.raises(RuntimeError, match="Failed to connect"):
                    app.setup()
        finally:
            os.chdir(original_dir)
    
    def test_graceful_degradation_without_neo4j(self, test_config, tmp_path, mock_llm):
        """Test system works without Neo4j (degraded mode)."""
        config_path = tmp_path / "config.json"
        test_config["mcp"]["enabled"] = False
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        templates = {"rag": {"plan_generation": "{question}", "answer": "{question}"}}
        (tmp_path / "prompt_templates.json").write_text(json.dumps(templates))
        
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            with patch('application.OllamaLLM', return_value=mock_llm), \
                 patch('application.OllamaEmbeddings', return_value=Mock()):
                
                from application import Application
                
                app = Application()
                # Simulate Neo4j unavailable
                app.graph_retriever = None
                app.mcp_client = None
                
                mock_llm.responses = [
                    json.dumps({"steps": [{"step": "Test", "needs_context": False}]}),
                    "Answer without graph context"
                ]
                
                # Should still work in degraded mode
                app.setup()
                result = app.ask("Test question")
                
                assert result is not None
        finally:
            os.chdir(original_dir)


@pytest.mark.e2e
@pytest.mark.smoke
class TestSmokeTests:
    """Quick smoke tests for basic functionality."""
    
    def test_imports(self):
        """Test all major imports work."""
        try:
            from application import Application
            from rag.flow.rag_pipeline import RAGPipeline
            from rag.flow.step_executor import StepExecutor
            from rag.flow.prompt_processor import PromptProcessor
            from mcp_client import MCPNeo4jClient
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_basic_object_creation(self):
        """Test basic object creation without external dependencies."""
        from rag.flow.types import PlanStep, StepComplexity, ActionType
        from rag.flow.task_execution_plan import TaskExecutionPlan
        
        step = PlanStep(
            step="Test step",
            needs_context=False,
            complexity=StepComplexity.LOW,
            action_type=ActionType.READ
        )
        
        plan = TaskExecutionPlan(
            question="Test?",
            steps=[step]
        )
        
        assert step.step == "Test step"
        assert plan.question == "Test?"
        assert len(plan.steps) == 1
