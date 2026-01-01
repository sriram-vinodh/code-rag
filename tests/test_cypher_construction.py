import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_client import MCPNeo4jClient


class DummyLLM:
    """Simple stub capturing prompt content and returning a fixed response."""

    def __init__(self, response: str):
        self.response = response
        self.last_prompt = None

    def predict(self, prompt: str):
        self.last_prompt = prompt
        return self.response


IMPLEMENTATION_Q = "How does the authenticate method work?"
STRUCTURAL_Q = "What methods are in class UserService?"
SEARCH_Q = "Find all methods with more than 5 parameters"


def _make_client():
    return MCPNeo4jClient(config_path="config.json")


def test_implementation_template_injects_code_requirements():
    llm = DummyLLM("MATCH (m) RETURN m.code")
    client = _make_client()

    cypher = client._generate_cypher_from_nl(IMPLEMENTATION_Q, llm)

    assert cypher.startswith("MATCH")
    assert llm.last_prompt is not None
    assert "WHERE m.code IS NOT NULL" in llm.last_prompt
    assert "ALWAYS include m.code" in llm.last_prompt or "m.code" in llm.last_prompt


def test_structural_template_omits_code_requirement():
    llm = DummyLLM("MATCH (n) RETURN n")
    client = _make_client()

    cypher = client._generate_cypher_from_nl(STRUCTURAL_Q, llm)

    assert cypher.startswith("MATCH")
    assert llm.last_prompt is not None
    assert "WITHOUT code" in llm.last_prompt
    assert "WHERE m.code IS NOT NULL" not in llm.last_prompt


def test_search_template_highlights_filtering():
    llm = DummyLLM("MATCH (m) RETURN m")
    client = _make_client()

    cypher = client._generate_cypher_from_nl(SEARCH_Q, llm)

    assert cypher.startswith("MATCH")
    assert llm.last_prompt is not None
    assert "FILTER/SEARCH" in llm.last_prompt


def test_invalid_llm_output_is_rejected():
    llm = DummyLLM("hi")  # Too short / missing Cypher keyword
    client = _make_client()

    cypher = client._generate_cypher_from_nl(STRUCTURAL_Q, llm)

    assert cypher is None
