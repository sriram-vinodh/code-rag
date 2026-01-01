import os
import sys

# Ensure project root on path for imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rag.flow.step_executor import StepExecutor


def test_extract_symbols_from_metadata_with_class_and_file():
    metadata = {
        "methods": [
            {
                "name": "login",
                "class_name": "UserService",
                "file_path": "src/UserService.java",
                "signature": "login(String, String)",
            }
        ]
    }

    executor = StepExecutor.__new__(StepExecutor)
    symbols = executor._extract_symbols_from_metadata(metadata)

    assert len(symbols) == 1
    sym = symbols[0]
    assert sym["name_path"] == "UserService/login"
    assert sym["relative_path"] == "src/UserService.java"
    assert sym["signature"] == "login(String, String)"


def test_extract_symbols_from_metadata_without_class():
    metadata = {
        "methods": [
            {
                "name": "orphanMethod",
                "file_path": "src/Utility.java",
            }
        ]
    }

    executor = StepExecutor.__new__(StepExecutor)
    symbols = executor._extract_symbols_from_metadata(metadata)

    assert len(symbols) == 1
    sym = symbols[0]
    assert sym["name_path"] == "orphanMethod"
    assert sym["relative_path"] == "src/Utility.java"


def test_extract_symbols_from_metadata_empty():
    executor = StepExecutor.__new__(StepExecutor)
    symbols = executor._extract_symbols_from_metadata({})
    assert symbols == []


class DummyLLM:
    pass


class DummyMCP:
    def query_with_natural_language(self, description, llm, return_metadata=False):
        return {
            "raw_results": [
                {"name": "login"}
            ],
            "metadata": {
                "methods": [
                    {
                        "name": "login",
                        "class_name": "UserService",
                        "file_path": "src/UserService.java",
                        "signature": "login(String, String)",
                    }
                ]
            },
        }


class DummySerena:
    def __init__(self):
        self.calls = []

    def find_symbol(self, name_path: str, relative_path: str = "", include_body: bool = False, depth: int = 0):
        self.calls.append((name_path, relative_path, include_body, depth))
        return {"body": f"def {name_path.replace('/', '_')}(self): pass"}


def test_hybrid_context_prefers_metadata_and_fetches_serena():
    executor = StepExecutor(DummyLLM(), neo4j_retriever=None, mcp_client=DummyMCP(), serena_client=DummySerena())

    context = executor._fetch_hybrid_context("describe login flow")

    # Should have pulled from metadata and fetched code via Serena
    assert "Serena Code" in context
    assert "UserService/login" in context
    assert "src/UserService.java" in context
    assert executor.serena_client.calls, "Serena client should be invoked with metadata-derived symbol"
