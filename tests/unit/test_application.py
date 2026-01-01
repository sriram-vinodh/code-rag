"""Unit tests for Application class."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from application import Application


@pytest.mark.unit
class TestApplication:
    """Test suite for Application class."""
    
    def test_initialization(self, test_config, tmp_path):
        """Test Application initialization."""
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            with patch('application.OllamaLLM'), \
                 patch('application.OllamaEmbeddings'), \
                 patch('application.Neo4jGraphRetriever'), \
                 patch('application.MCPNeo4jClient'):
                
                app = Application()
                
                assert app.config is not None
                assert app.llm is None  # Not initialized until setup
                assert app.pipeline is None
                assert app.is_setup is False
        finally:
            os.chdir(original_dir)
    
    def test_load_config_success(self, test_config, tmp_path):
        """Test configuration loading."""
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            with patch('application.OllamaLLM'), \
                 patch('application.OllamaEmbeddings'), \
                 patch('application.Neo4jGraphRetriever'), \
                 patch('application.MCPNeo4jClient'):
                
                app = Application()
                config = app._load_config()
                
                assert config == test_config
                assert "data_processing" in config
                assert "pipeline_settings" in config
        finally:
            os.chdir(original_dir)
    
    def test_load_config_missing_file(self, tmp_path):
        """Test configuration loading with missing file."""
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            with patch('application.OllamaLLM'), \
                 patch('application.OllamaEmbeddings'), \
                 patch('application.Neo4jGraphRetriever'), \
                 patch('application.MCPNeo4jClient'):
                
                with pytest.raises(RuntimeError, match="Configuration file not found"):
                    app = Application()
        finally:
            os.chdir(original_dir)
    
    def test_parse_args(self, test_config, tmp_path):
        """Test argument parsing."""
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        import os
        import sys
        original_dir = os.getcwd()
        original_argv = sys.argv
        try:
            os.chdir(tmp_path)
            sys.argv = ['test', '--model', 'test-model', '--log-level', 'DEBUG']
            
            with patch('application.OllamaLLM'), \
                 patch('application.OllamaEmbeddings'), \
                 patch('application.Neo4jGraphRetriever'), \
                 patch('application.MCPNeo4jClient'):
                
                app = Application()
                args = app._parse_args()
                
                assert args.model == 'test-model'
                assert args.log_level == 'DEBUG'
        finally:
            os.chdir(original_dir)
            sys.argv = original_argv
    
    def test_setup_initializes_components(self, test_config, tmp_path, mock_llm):
        """Test setup initializes all components."""
        config_path = tmp_path / "config.json"
        test_config["mcp"]["enabled"] = False  # Disable MCP for simpler test
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            with patch('application.OllamaLLM', return_value=mock_llm), \
                 patch('application.OllamaEmbeddings', return_value=Mock()), \
                 patch('application.RAGPipeline') as mock_pipeline_class:
                
                app = Application()
                app.graph_retriever = Mock()
                app.graph_retriever.connect = Mock()
                
                app.setup()
                
                assert app.is_setup is True
                assert app.llm is not None
                assert app.embeddings is not None
                mock_pipeline_class.assert_called_once()
        finally:
            os.chdir(original_dir)
    
    def test_ask_calls_pipeline(self, test_config, tmp_path, mock_llm):
        """Test ask method calls pipeline."""
        config_path = tmp_path / "config.json"
        test_config["mcp"]["enabled"] = False
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            with patch('application.OllamaLLM', return_value=mock_llm), \
                 patch('application.OllamaEmbeddings', return_value=Mock()):
                
                app = Application()
                app.is_setup = True
                app.pipeline = Mock()
                app.pipeline.ask = Mock(return_value="Mock answer")
                
                result = app.ask("Test question")
                
                assert result == "Mock answer"
                app.pipeline.ask.assert_called_once_with("Test question")
        finally:
            os.chdir(original_dir)
    
    def test_ask_without_setup(self, test_config, tmp_path):
        """Test ask triggers setup if not done."""
        config_path = tmp_path / "config.json"
        test_config["mcp"]["enabled"] = False
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            with patch('application.OllamaLLM'), \
                 patch('application.OllamaEmbeddings'), \
                 patch('application.RAGPipeline'):
                
                app = Application()
                app.graph_retriever = Mock()
                app.graph_retriever.connect = Mock()
                
                with patch.object(app, 'setup') as mock_setup:
                    app.pipeline = Mock()
                    app.pipeline.ask = Mock(return_value="Answer")
                    
                    app.ask("Question")
                    
                    mock_setup.assert_called_once()
        finally:
            os.chdir(original_dir)
    
    def test_run_cli_starts_qa_loop(self, test_config, tmp_path):
        """Test run_cli starts the QA loop."""
        config_path = tmp_path / "config.json"
        test_config["mcp"]["enabled"] = False
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            with patch('application.OllamaLLM'), \
                 patch('application.OllamaEmbeddings'):
                
                app = Application()
                
                with patch.object(app, 'setup') as mock_setup, \
                     patch.object(app, '_start_qa_loop') as mock_qa:
                    
                    app.run_cli()
                    
                    mock_setup.assert_called_once()
                    mock_qa.assert_called_once()
        finally:
            os.chdir(original_dir)


@pytest.mark.unit
class TestApplicationErrorHandling:
    """Test error handling in Application."""
    
    def test_setup_handles_ollama_connection_error(self, test_config, tmp_path):
        """Test setup handles Ollama connection errors."""
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            with patch('application.OllamaLLM') as mock_ollama:
                mock_ollama.side_effect = Exception("503 Service Unavailable")
                
                app = Application()
                
                with pytest.raises(RuntimeError, match="Failed to connect to Ollama"):
                    app.setup()
        finally:
            os.chdir(original_dir)
