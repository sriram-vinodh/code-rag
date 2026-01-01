"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides:
- Shared fixtures available to all tests
- Test configuration
- Mock objects and utilities
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Provide test configuration."""
    return {
        "data_processing": {
            "code_path": "./test_code",
            "docs_path": "./test_docs",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "allowed_extensions": [".py", ".java", ".md"]
        },
        "pipeline_settings": {
            "model": "llama3",
            "templates_file": "prompt_templates.json",
            "max_retries": 3
        },
        "neo4j": {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "test_password",
            "database": "test_db"
        },
        "mcp": {
            "enabled": True,
            "neo4j_server": {
                "command": "uvx",
                "args": ["mcp-neo4j-cypher@0.5.1", "--transport", "stdio"],
                "env": {
                    "NEO4J_URI": "bolt://localhost:7687",
                    "NEO4J_USERNAME": "neo4j",
                    "NEO4J_PASSWORD": "test_password"
                }
            }
        },
        "logging": {
            "level": "WARNING"  # Quiet during tests
        }
    }


@pytest.fixture
def test_config_file(test_config, tmp_path):
    """Create a temporary config.json file."""
    config_path = tmp_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(test_config, f)
    return str(config_path)


@pytest.fixture
def test_templates() -> Dict[str, Any]:
    """Provide test prompt templates."""
    return {
        "rag": {
            "cypher_generation": "Generate Cypher for: {question}",
            "step_execution": "Execute: {step}",
            "answer": "Answer based on context: {context}\n\nQuestion: {question}",
            "plan_generation": "Create plan for: {question}"
        }
    }


@pytest.fixture
def test_templates_file(test_templates, tmp_path):
    """Create a temporary prompt_templates.json file."""
    templates_path = tmp_path / "prompt_templates.json"
    with open(templates_path, 'w') as f:
        json.dump(test_templates, f)
    return str(templates_path)


# ============================================================================
# Mock LLM Fixtures
# ============================================================================

class MockLLMResponse:
    """Mock LLM response object."""
    def __init__(self, content: str):
        self.content = content
    
    def __str__(self):
        return self.content


class MockLLM:
    """Mock Language Model for testing."""
    
    def __init__(self, responses: List[str] = None):
        """Initialize with predetermined responses."""
        self.responses = responses or ["Mock response"]
        self.call_count = 0
        self.invocations = []
    
    def invoke(self, prompt: str) -> MockLLMResponse:
        """Mock invoke method."""
        self.invocations.append(prompt)
        response_idx = min(self.call_count, len(self.responses) - 1)
        response = self.responses[response_idx]
        self.call_count += 1
        return MockLLMResponse(response)
    
    def predict(self, prompt: str) -> str:
        """Mock predict method (legacy API)."""
        result = self.invoke(prompt)
        return result.content


@pytest.fixture
def mock_llm():
    """Provide a mock LLM with default responses."""
    return MockLLM([
        "This is a test response",
        "Here is another response",
        "Final mock response"
    ])


@pytest.fixture
def mock_llm_with_plan():
    """Provide a mock LLM that returns execution plans."""
    plan_json = json.dumps({
        "steps": [
            {
                "step": "Find class definition",
                "needs_context": True,
                "complexity": "medium",
                "knowledge_source": "graph_database"
            },
            {
                "step": "Analyze methods",
                "needs_context": True,
                "complexity": "low",
                "knowledge_source": "serena_ide"
            }
        ]
    })
    return MockLLM([plan_json])


# ============================================================================
# Mock Database Fixtures
# ============================================================================

@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver."""
    driver = MagicMock()
    
    # Mock session
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    
    # Mock query results
    mock_result = MagicMock()
    mock_result.data.return_value = [
        {"class": {"name": "TestClass"}, "method": {"name": "testMethod"}}
    ]
    session.run.return_value = mock_result
    
    return driver


@pytest.fixture
def mock_neo4j_retriever(mock_neo4j_driver):
    """Mock Neo4jGraphRetriever."""
    from unittest.mock import Mock
    
    retriever = Mock()
    retriever.driver = mock_neo4j_driver
    retriever.connected = True
    
    def mock_query(cypher, params=None):
        return [
            {"class": {"name": "TestClass"}, "method": {"name": "testMethod"}}
        ]
    
    retriever.query = Mock(side_effect=mock_query)
    retriever.connect = Mock()
    retriever.close = Mock()
    
    return retriever


# ============================================================================
# Mock MCP Client Fixtures
# ============================================================================

@pytest.fixture
def mock_mcp_client():
    """Mock MCPNeo4jClient."""
    client = MagicMock()
    client.connected = True
    
    # Mock schema response
    client.get_schema.return_value = {
        "content": [
            {
                "text": "Node Types: Class, Method\nRelationships: HAS_METHOD"
            }
        ]
    }
    
    # Mock query response
    client.execute_cypher.return_value = {
        "content": [
            {
                "text": json.dumps([
                    {"class": {"name": "TestClass"}}
                ])
            }
        ]
    }
    
    return client


# ============================================================================
# Mock Serena Client Fixtures
# ============================================================================

@pytest.fixture
def mock_serena_client():
    """Mock SerenaClient."""
    client = MagicMock()
    
    # Mock tool responses
    client.find_symbol.return_value = {
        "symbols": [
            {
                "name_path": "TestClass/testMethod",
                "file_path": "test.py",
                "body": "def testMethod(): pass"
            }
        ]
    }
    
    client.get_symbols_overview.return_value = {
        "symbols": [
            {"name": "TestClass", "kind": 5}
        ]
    }
    
    return client


# ============================================================================
# File System Fixtures
# ============================================================================

@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    
    # Create sample structure
    code_dir = workspace / "code"
    code_dir.mkdir()
    
    docs_dir = workspace / "docs"
    docs_dir.mkdir()
    
    # Sample Python file
    (code_dir / "sample.py").write_text("""
class SampleClass:
    def sample_method(self):
        return "Hello, World!"
""")
    
    # Sample Java file
    (code_dir / "Sample.java").write_text("""
public class Sample {
    public String sampleMethod() {
        return "Hello, World!";
    }
}
""")
    
    # Sample doc
    (docs_dir / "README.md").write_text("""
# Sample Documentation

This is a test document.
""")
    
    yield workspace
    
    # Cleanup is automatic with tmp_path


@pytest.fixture
def sample_documents():
    """Provide sample document objects for testing."""
    from langchain_core.documents import Document
    
    return [
        Document(
            page_content="Sample Python code",
            metadata={"source": "test.py", "language": "python"}
        ),
        Document(
            page_content="Sample Java code",
            metadata={"source": "Test.java", "language": "java"}
        ),
        Document(
            page_content="Sample documentation",
            metadata={"source": "README.md", "language": "markdown"}
        )
    ]


# ============================================================================
# Execution Plan Fixtures
# ============================================================================

@pytest.fixture
def sample_plan_steps():
    """Provide sample PlanStep objects."""
    from rag.flow.types import PlanStep, StepComplexity, ActionType
    
    return [
        PlanStep(
            step="Find class definition",
            needs_context=True,
            complexity=StepComplexity.MEDIUM,
            knowledge_source="graph_database",
            action_type=ActionType.READ
        ),
        PlanStep(
            step="Analyze method implementation",
            needs_context=True,
            complexity=StepComplexity.HIGH,
            knowledge_source="serena_ide",
            action_type=ActionType.ANALYZE
        ),
        PlanStep(
            step="Generate summary",
            needs_context=False,
            complexity=StepComplexity.LOW,
            action_type=ActionType.READ
        )
    ]


@pytest.fixture
def sample_execution_plan(sample_plan_steps):
    """Provide sample TaskExecutionPlan."""
    from rag.flow.task_execution_plan import TaskExecutionPlan
    
    return TaskExecutionPlan(
        question="What does TestClass do?",
        steps=sample_plan_steps
    )


# ============================================================================
# Application Fixtures
# ============================================================================

@pytest.fixture
def mock_application(mock_llm, mock_neo4j_retriever, mock_mcp_client, test_config):
    """Provide a mocked Application instance."""
    with patch('application.OllamaLLM', return_value=mock_llm), \
         patch('application.OllamaEmbeddings', return_value=Mock()), \
         patch('application.Neo4jGraphRetriever', return_value=mock_neo4j_retriever), \
         patch('application.MCPNeo4jClient', return_value=mock_mcp_client):
        
        from application import Application
        app = Application()
        app.config = test_config
        app.llm = mock_llm
        app.graph_retriever = mock_neo4j_retriever
        app.mcp_client = mock_mcp_client
        
        return app


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def capture_logs(caplog):
    """Fixture to easily capture and assert on logs."""
    import logging
    caplog.set_level(logging.DEBUG)
    return caplog


@pytest.fixture
def mock_time():
    """Mock time.time() for deterministic timing tests."""
    with patch('time.time') as mock:
        mock.return_value = 1000.0
        yield mock


# ============================================================================
# Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    # Reset query monitor if it exists
    try:
        from rag.pipeline.query_monitor import QueryMonitor
        if hasattr(QueryMonitor, '_instance'):
            QueryMonitor._instance = None
    except ImportError:
        pass
    
    yield
    
    # Cleanup after test
