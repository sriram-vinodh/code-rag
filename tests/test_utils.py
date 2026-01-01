"""Test utilities and helpers."""

from typing import List, Dict, Any
import json
from unittest.mock import Mock


class ResponseSequence:
    """Helper to create sequences of LLM responses."""
    
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.index = 0
    
    def __call__(self, *args, **kwargs):
        """Return next response in sequence."""
        if self.index >= len(self.responses):
            raise ValueError("No more responses in sequence")
        response = self.responses[self.index]
        self.index += 1
        return Mock(content=response)


class GraphQueryBuilder:
    """Helper to build test Cypher queries."""
    
    @staticmethod
    def match_class(class_name: str) -> str:
        """Build MATCH query for class."""
        return f"MATCH (c:Class {{name: '{class_name}'}}) RETURN c"
    
    @staticmethod
    def match_method(class_name: str, method_name: str) -> str:
        """Build MATCH query for method."""
        return f"""
        MATCH (c:Class {{name: '{class_name}'}})-[:HAS_METHOD]->(m:Method {{name: '{method_name}'}})
        RETURN c, m
        """
    
    @staticmethod
    def match_calls(method_name: str) -> str:
        """Build query for method calls."""
        return f"""
        MATCH (m:Method {{name: '{method_name}'}})-[:CALLS]->(called:Method)
        RETURN m, called
        """


class PlanBuilder:
    """Helper to build execution plans."""
    
    @staticmethod
    def simple_plan(question: str, num_steps: int = 1) -> Dict[str, Any]:
        """Build a simple execution plan."""
        steps = []
        for i in range(num_steps):
            steps.append({
                "step": f"Step {i+1}",
                "needs_context": i % 2 == 0,
                "complexity": "low" if i < 2 else "medium",
                "knowledge_source": "graph_database" if i % 2 == 0 else None
            })
        
        return {
            "steps": steps
        }
    
    @staticmethod
    def complex_plan(question: str) -> Dict[str, Any]:
        """Build a complex multi-source plan."""
        return {
            "steps": [
                {
                    "step": "Search graph for class definition",
                    "needs_context": True,
                    "complexity": "medium",
                    "knowledge_source": "graph_database",
                    "action_type": "read"
                },
                {
                    "step": "Retrieve source code via Serena",
                    "needs_context": True,
                    "complexity": "high",
                    "knowledge_source": "serena_ide",
                    "action_type": "read",
                    "target_symbols": ["TestClass"]
                },
                {
                    "step": "Analyze dependencies",
                    "needs_context": True,
                    "complexity": "high",
                    "knowledge_source": "hybrid"
                },
                {
                    "step": "Synthesize answer",
                    "needs_context": False,
                    "complexity": "medium"
                }
            ]
        }


class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def sample_code_snippet(language: str = "python") -> str:
        """Generate sample code snippet."""
        snippets = {
            "python": """
class SampleClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
""",
            "java": """
public class SampleClass {
    private int value;
    
    public SampleClass() {
        this.value = 42;
    }
    
    public int getValue() {
        return value;
    }
}
"""
        }
        return snippets.get(language, snippets["python"])
    
    @staticmethod
    def sample_graph_result() -> List[Dict[str, Any]]:
        """Generate sample Neo4j query result."""
        return [
            {
                "class": {
                    "name": "SampleClass",
                    "file_path": "sample.py",
                    "type": "Class"
                },
                "methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self)",
                        "type": "Method"
                    },
                    {
                        "name": "get_value",
                        "signature": "get_value(self)",
                        "type": "Method"
                    }
                ]
            }
        ]
    
    @staticmethod
    def sample_serena_symbol() -> Dict[str, Any]:
        """Generate sample Serena symbol result."""
        return {
            "symbols": [
                {
                    "name_path": "SampleClass/get_value",
                    "relative_path": "sample.py",
                    "file_path": "/path/to/sample.py",
                    "kind": 6,  # Method
                    "body": "def get_value(self):\n    return self.value",
                    "start_line": 5,
                    "end_line": 6
                }
            ]
        }


def assert_valid_cypher(query: str):
    """Assert a string is a valid-looking Cypher query."""
    assert isinstance(query, str)
    assert len(query) > 0
    
    # Basic Cypher validation
    query_upper = query.upper()
    assert any(keyword in query_upper for keyword in ['MATCH', 'CREATE', 'MERGE', 'RETURN'])


def assert_valid_plan(plan: Dict[str, Any]):
    """Assert a plan has valid structure."""
    assert "steps" in plan
    assert isinstance(plan["steps"], list)
    assert len(plan["steps"]) > 0
    
    for step in plan["steps"]:
        assert "step" in step
        assert isinstance(step["step"], str)


def create_mock_llm_with_responses(*responses: str):
    """Create a mock LLM that returns predetermined responses."""
    from tests.conftest import MockLLM
    return MockLLM(list(responses))
