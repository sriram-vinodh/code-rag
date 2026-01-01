"""Integration tests for RAG Pipeline with real components."""

import pytest
import json
from unittest.mock import Mock, patch
from rag.flow.rag_pipeline import RAGPipeline
from rag.flow.types import PlanStep


@pytest.mark.integration
class TestRAGPipelineIntegration:
    """Integration tests for RAG pipeline with mocked LLM."""
    
    def test_full_query_flow(self, mock_llm, mock_neo4j_retriever, test_templates, tmp_path):
        """Test complete query flow from question to answer."""
        # Setup test environment
        config = {
            "pipeline_settings": {"templates_file": "prompt_templates.json"}
        }
        templates = {
            "rag": {
                "plan_generation": "Create plan for: {question}",
                "cypher_generation": "Generate Cypher: {question}",
                "step_execution": "Execute: {step}",
                "answer": "Answer: {context}\nQuestion: {question}"
            }
        }
        
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            (tmp_path / "config.json").write_text(json.dumps(config))
            (tmp_path / "prompt_templates.json").write_text(json.dumps(templates))
            
            # Setup LLM responses for the flow
            plan_response = json.dumps({
                "steps": [
                    {
                        "step": "Find TestClass definition",
                        "needs_context": True,
                        "complexity": "medium",
                        "knowledge_source": "graph_database"
                    }
                ]
            })
            
            cypher_response = "MATCH (c:Class {name: 'TestClass'}) RETURN c"
            execution_response = "Found TestClass in module test.py"
            final_answer = "TestClass is a test class used for unit testing."
            
            mock_llm.responses = [
                plan_response,
                cypher_response,
                execution_response,
                final_answer
            ]
            
            # Create pipeline and execute
            pipeline = RAGPipeline(
                llm=mock_llm,
                neo4j_retriever=mock_neo4j_retriever
            )
            
            result = pipeline.ask("What is TestClass?")
            
            # Verify result
            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0
            
            # Verify LLM was called
            assert mock_llm.call_count > 0
        finally:
            os.chdir(original_dir)
    
    def test_multi_step_execution(self, mock_llm, mock_neo4j_retriever, test_templates, tmp_path):
        """Test execution of multi-step plan."""
        config = {
            "pipeline_settings": {"templates_file": "prompt_templates.json"}
        }
        templates = {
            "rag": {
                "plan_generation": "Plan: {question}",
                "cypher_generation": "Cypher: {question}",
                "step_execution": "Execute: {step}",
                "answer": "{context}\n{question}"
            }
        }
        
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            (tmp_path / "config.json").write_text(json.dumps(config))
            (tmp_path / "prompt_templates.json").write_text(json.dumps(templates))
            
            # Multi-step plan
            plan_response = json.dumps({
                "steps": [
                    {
                        "step": "Find class",
                        "needs_context": True,
                        "complexity": "low",
                        "knowledge_source": "graph_database"
                    },
                    {
                        "step": "Find methods",
                        "needs_context": True,
                        "complexity": "medium",
                        "knowledge_source": "graph_database"
                    },
                    {
                        "step": "Analyze relationships",
                        "needs_context": False,
                        "complexity": "high"
                    }
                ]
            })
            
            mock_llm.responses = [
                plan_response,
                "MATCH (c:Class) RETURN c",  # Step 1 cypher
                "Step 1 result",
                "MATCH (m:Method) RETURN m",  # Step 2 cypher
                "Step 2 result",
                "Step 3 result",
                "Final comprehensive answer"
            ]
            
            pipeline = RAGPipeline(
                llm=mock_llm,
                neo4j_retriever=mock_neo4j_retriever
            )
            
            result = pipeline.ask("Analyze TestClass")
            
            assert result is not None
            # Should have processed multiple steps
            assert mock_llm.call_count >= 3
        finally:
            os.chdir(original_dir)
    
    def test_error_recovery(self, mock_llm, mock_neo4j_retriever, test_templates, tmp_path):
        """Test error handling and recovery during execution."""
        config = {
            "pipeline_settings": {"templates_file": "prompt_templates.json"}
        }
        templates = {"rag": {"plan_generation": "{question}"}}
        
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            (tmp_path / "config.json").write_text(json.dumps(config))
            (tmp_path / "prompt_templates.json").write_text(json.dumps(templates))
            
            # Invalid JSON plan response (should handle gracefully)
            mock_llm.responses = ["Invalid JSON {not valid}"]
            
            pipeline = RAGPipeline(
                llm=mock_llm,
                neo4j_retriever=mock_neo4j_retriever
            )
            
            result = pipeline.ask("Test question")
            
            # Should return error message, not crash
            assert "error" in result.lower() or len(result) > 0
        finally:
            os.chdir(original_dir)


@pytest.mark.integration
class TestApplicationIntegration:
    """Integration tests for Application class."""
    
    def test_application_setup_and_query(self, test_config, tmp_path, mock_llm):
        """Test Application setup and query execution."""
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
        
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            with patch('application.OllamaLLM', return_value=mock_llm), \
                 patch('application.OllamaEmbeddings', return_value=Mock()):
                
                from application import Application
                
                app = Application()
                app.graph_retriever = Mock()
                app.graph_retriever.connect = Mock()
                
                # Setup
                app.setup()
                assert app.is_setup is True
                
                # Execute query
                mock_llm.responses = [
                    json.dumps({"steps": []}),
                    "The answer is 42"
                ]
                
                result = app.ask("What is the answer?")
                assert result is not None
        finally:
            os.chdir(original_dir)


@pytest.mark.integration
@pytest.mark.slow
class TestDocumentProcessingPipeline:
    """Integration tests for document processing pipeline."""
    
    def test_file_loading_and_parsing(self, temp_workspace, test_templates):
        """Test file loading and parsing integration."""
        from rag.loader.filesystem_loader import FileSystemLoader
        from rag.parser.document_parser import DefaultDocumentParser
        
        loader = FileSystemLoader(
            directory_path=str(temp_workspace / "code"),
            allowed_extensions=[".py", ".java"]
        )
        
        documents = loader.load()
        
        assert len(documents) > 0
        
        parser = DefaultDocumentParser()
        parsed_docs = parser.parse_documents(documents)
        
        assert len(parsed_docs) == len(documents)
        for doc in parsed_docs:
            assert hasattr(doc, 'page_content')
            assert hasattr(doc, 'metadata')
    
    def test_document_chunking(self, sample_documents, test_templates):
        """Test document chunking integration."""
        from rag.chunking.text_chunker import RecursiveTextChunker
        
        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)
        
        chunks = chunker.chunk_documents(sample_documents)
        
        assert len(chunks) >= len(sample_documents)
        for chunk in chunks:
            assert hasattr(chunk, 'page_content')
            assert len(chunk.page_content) <= 100 + 20  # Account for overlap
