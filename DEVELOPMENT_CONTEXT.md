# Development Context - Code-RAG System

**Last Updated**: December 22, 2025  
**Purpose**: Comprehensive reference for developers working on code-RAG

---

## Quick Start for Developers

### Understanding the System (5 minutes)

1. **Read**: [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and components
2. **Review**: [TEST_FAILURE_ANALYSIS.md](TEST_FAILURE_ANALYSIS.md) - Current test status
3. **Reference**: [TESTING_FRAMEWORK.md](TESTING_FRAMEWORK.md) - How to write and run tests

### Setting Up Development Environment (15 minutes)

```bash
# 1. Clone and enter directory
cd /Users/sriram-14910/code/code-rag

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt

# 4. Start required services
ollama serve                    # Terminal 1
neo4j start                     # Terminal 2 or Docker

# 5. Run tests to verify setup
./run_tests.sh smoke
```

### Making Your First Change (30 minutes)

```bash
# 1. Create a feature branch
git checkout -b feature/my-feature

# 2. Make changes to code
# ... edit files ...

# 3. Write tests
# ... add tests in tests/unit/ ...

# 4. Run tests
./run_tests.sh unit

# 5. Check coverage
pytest --cov=rag --cov-report=html
open htmlcov/index.html

# 6. Commit and push
git add .
git commit -m "Add: my feature"
git push origin feature/my-feature
```

---

## System Overview

### What is Code-RAG?

Code-RAG is a **Retrieval-Augmented Generation** system that answers natural language questions about codebases by:

1. **Understanding** the question using an LLM
2. **Planning** how to answer it (multi-step decomposition)
3. **Retrieving** relevant code from Neo4j graph database
4. **Generating** answers using LLM with retrieved context

### Key Components

```
User Question
    ↓
Application (application.py)
    ↓
RAGPipeline (rag/flow/rag_pipeline.py)
    ↓
PromptProcessor → Generates execution plan
    ↓
StepExecutor → Executes each step
    ↓
Knowledge Sources (Neo4j, Serena, Vector Store)
    ↓
Final Answer
```

### Technology Stack

- **Language**: Python 3.10+
- **LLM**: Ollama (llama3.2:3b)
- **Graph DB**: Neo4j
- **Vector Store**: ChromaDB
- **Testing**: pytest
- **MCP**: Model Context Protocol for Neo4j communication

---

## Architecture Reference

For detailed architecture information, see [ARCHITECTURE.md](ARCHITECTURE.md).

### Core Modules

| Module | Responsibility | Key Files |
|--------|---------------|-----------|
| **Application** | Orchestration, config, CLI | `application.py`, `main.py` |
| **Pipeline** | Query processing flow | `rag/flow/rag_pipeline.py` |
| **Processor** | Plan generation | `rag/flow/prompt_processor.py` |
| **Executor** | Step execution | `rag/flow/step_executor.py` |
| **MCP Client** | Neo4j communication | `mcp_client.py` |
| **Types** | Data models | `rag/flow/types.py` |

### Data Flow Example

```
Question: "What classes implement Parser?"

1. PromptProcessor generates plan:
   [
     PlanStep("Find Parser interface"),
     PlanStep("Find implementing classes")
   ]

2. StepExecutor processes Step 1:
   - Fetch context from Neo4j graph
   - LLM generates: "Parser is an interface..."
   
3. StepExecutor processes Step 2:
   - Fetch context from Neo4j (with previous result)
   - LLM generates: "Classes: TextParser, JavaParser"

4. Return combined answer to user
```

---

## Current Status

### Test Results (Last Run: Dec 22, 2025)

- **Overall**: 73% passing (38/52 tests)
- **Unit Tests**: 79% passing (30/38)
- **Integration Tests**: 17% passing (1/6) ⚠️
- **E2E Tests**: 88% passing (7/8)

For detailed failure analysis, see [TEST_FAILURE_ANALYSIS.md](TEST_FAILURE_ANALYSIS.md).

### Known Issues

#### Priority 1 (Critical) - Blocks Integration Testing

**Template Loading Failures** (5 tests failing)
- **Cause**: Integration tests don't provide template files
- **Impact**: RAG pipeline cannot generate plans
- **Fix**: Use `test_templates` fixture in integration tests
- **Effort**: 2-4 hours
- **Status**: Not started

#### Priority 2 (High) - API Mismatches

**MCP Client API Mismatches** (4 tests failing)
- **Cause**: Tests use wrong method names
- **Fix**: Update test calls:
  - `disconnect()` → `close()`
  - `execute_cypher()` → `execute_read_query()`
- **Effort**: 1-2 hours
- **Status**: Not started

**Data Model Mismatches** (2 tests failing)
- **Cause**: Tests missing required constructor parameters
- **Fix**: Add `complexity` to `PlanStep`, `total_steps` to `TaskExecutionPlan`
- **Effort**: 30 minutes
- **Status**: Not started

**Mock Configuration Issues** (2 tests failing)
- **Cause**: Mocks return `Mock` objects instead of proper types
- **Fix**: Configure mocks with correct return types
- **Effort**: 1 hour
- **Status**: Not started

### Recommended Fix Order

1. ✅ **Quick Wins** (2-3 hours total):
   - Fix PlanStep/TaskExecutionPlan constructors
   - Fix MCP Client method names
   - Fix mock return values

2. ✅ **Template Infrastructure** (2-4 hours):
   - Create test_templates fixture
   - Update integration tests
   - Verify template loading

3. ⚠️ **Investigation** (2-3 hours):
   - Debug remaining E2E failures
   - Add regression tests

---

## Testing Guide

For complete testing documentation, see [TESTING_FRAMEWORK.md](TESTING_FRAMEWORK.md).

### Running Tests

```bash
# Quick smoke tests
./run_tests.sh smoke

# All unit tests
./run_tests.sh unit

# All tests
./run_tests.sh all

# With coverage
./run_tests.sh all coverage
```

### Writing Tests

**Unit Test Template**:
```python
# tests/unit/test_my_component.py
import pytest
from unittest import mock
from rag.module import MyComponent


class TestMyComponent:
    def test_initialization(self):
        """Test component initializes correctly."""
        component = MyComponent()
        assert component is not None
    
    def test_method_with_mock(self, mock_dependency):
        """Test method behavior with mocked dependency."""
        component = MyComponent(mock_dependency)
        result = component.do_something()
        
        assert result == "expected"
        mock_dependency.some_method.assert_called_once()
```

**Integration Test Template**:
```python
# tests/integration/test_integration.py
import pytest


class TestIntegration:
    def test_component_interaction(
        self, 
        mock_llm, 
        mock_neo4j, 
        test_templates
    ):
        """Test real components with mocked externals."""
        from rag.flow.rag_pipeline import RAGPipeline
        
        pipeline = RAGPipeline(
            llm=mock_llm,
            neo4j_retriever=mock_neo4j,
            templates=test_templates
        )
        
        result = pipeline.ask("test question")
        assert len(result) > 0
```

### Using Fixtures

Common fixtures (from `tests/conftest.py`):

- `test_config` - Test configuration dict
- `test_config_file` - Temporary config file path
- `test_templates` - Prompt template dict
- `mock_llm` - Mock language model
- `mock_neo4j_retriever` - Mock Neo4j retriever
- `mock_mcp_client` - Mock MCP client
- `sample_plan_steps` - Sample PlanStep list
- `sample_execution_plan` - Sample TaskExecutionPlan

---

## Common Development Tasks

### Task 1: Add a New Query Type

**Goal**: Support a new type of query (e.g., security analysis)

**Steps**:

1. **Add enum value** in `rag/pipeline/query_classifier.py`:
   ```python
   class QueryType(Enum):
       # ... existing types
       SECURITY = "security"
   ```

2. **Add patterns** in same file:
   ```python
   QUERY_PATTERNS = {
       QueryType.SECURITY: [
           r"security\s+(issue|vulnerability)",
           r"(vulnerable|insecure)\s+code"
       ]
   }
   ```

3. **Add template** in `prompt_templates.json`:
   ```json
   {
     "rag": {
       "cypher_generation_security": "Generate Cypher to find security issues..."
     }
   }
   ```

4. **Update template mapping** in `mcp_client.py`:
   ```python
   template_map = {
       QueryType.SECURITY: "cypher_generation_security",
       # ... existing mappings
   }
   ```

5. **Write tests**:
   ```python
   def test_security_query_classification():
       classification = classify_query("find security vulnerabilities")
       assert classification.query_type == QueryType.SECURITY
   ```

### Task 2: Add a New Knowledge Source

**Goal**: Integrate a new source of context (e.g., GitHub API)

**Steps**:

1. **Create client** in `rag/retrieval/github_client.py`:
   ```python
   class GitHubClient:
       def get_file_content(self, repo: str, path: str) -> str:
           # Implementation
   ```

2. **Add to StepExecutor** in `rag/flow/step_executor.py`:
   ```python
   def __init__(self, llm, neo4j_retriever, github_client=None):
       self.github_client = github_client
   
   def _fetch_context_from_source(self, step):
       if step.knowledge_source == "github":
           return self._fetch_from_github(step)
   
   def _fetch_from_github(self, step):
       # Use self.github_client
   ```

3. **Update PlanStep** to support "github" knowledge source

4. **Write tests**:
   ```python
   def test_fetch_from_github(mock_github_client):
       executor = StepExecutor(llm=mock_llm, github_client=mock_github_client)
       step = PlanStep(
           step="Get README",
           needs_context=True,
           knowledge_source="github"
       )
       context = executor._fetch_from_github(step)
       assert "README" in context
   ```

### Task 3: Fix a Failing Test

**Goal**: Fix `test_process_query_success` (mock configuration issue)

**Problem**: Mock returns `Mock` instead of list

**Solution**:

1. **Locate test** in `tests/unit/test_rag_pipeline.py`:
   ```python
   def test_process_query_success(mock_llm, test_templates):
       # Current code (broken):
       mock_processor.generate_plan.return_value = mock.Mock()
   ```

2. **Fix mock configuration**:
   ```python
   def test_process_query_success(mock_llm, test_templates):
       from rag.flow.types import PlanStep, StepComplexity
       
       # Create realistic plan
       plan = [
           PlanStep(
               step="Answer question",
               needs_context=False,
               complexity=StepComplexity.LOW
           )
       ]
       
       mock_processor = mock.Mock()
       mock_processor.generate_plan.return_value = plan  # ✅ Returns list
       
       # ... rest of test
   ```

3. **Run test**:
   ```bash
   pytest tests/unit/test_rag_pipeline.py::TestRAGPipeline::test_process_query_success -v
   ```

4. **Verify fixed**:
   ```bash
   ./run_tests.sh unit
   ```

### Task 4: Add Code Modification Support

**Goal**: Enable Serena to modify code based on plan steps

**Steps**:

1. **Ensure Serena client is available** in `StepExecutor`

2. **Check for modify action**:
   ```python
   def _execute_step(self, step, context):
       if step.action_type == ActionType.MODIFY:
           return self._execute_modification(step, context)
   ```

3. **Implement modification**:
   ```python
   def _execute_modification(self, step, context):
       # Use Serena to modify code
       if not self.serena_client:
           return ExecutionResult(
               success=False,
               message="Serena client not available"
           )
       
       # Parse modification instructions from LLM
       # Call Serena client
       # Return result
   ```

4. **Add governance tracking**:
   ```python
   step.governance_metadata = {
       "user_approved": False,
       "auto_applied": False,
       "review_required": True
   }
   ```

---

## Code Style Guidelines

### Python Style

- **PEP 8** compliance
- **Type hints** for function signatures
- **Docstrings** for all public methods (Google style)
- **Line length**: 100 characters max

**Example**:
```python
def process_query(
    self, 
    question: str, 
    use_graph: bool = True
) -> str:
    """Process a user question through the RAG pipeline.
    
    Args:
        question: The user's natural language question
        use_graph: Whether to use Neo4j graph database for context
    
    Returns:
        The generated answer as a string
    
    Raises:
        ValueError: If question is empty
    """
    if not question:
        raise ValueError("Question cannot be empty")
    
    # Implementation
```

### Testing Style

- **Test names**: Describe what they test
- **Arrange-Act-Assert**: Clear structure
- **One assertion** per test (guideline)
- **Fixtures** for common setup

### Git Commit Messages

```
Type: Brief description (50 chars)

Detailed explanation if needed (wrap at 72 chars).

- Bullet points for multiple changes
- Reference issues: Fixes #123

Types: Add, Fix, Update, Remove, Refactor, Test, Docs
```

**Examples**:
```
Add: New query classification for security queries

Fix: Template loading in integration tests

Update: MCP client to use correct method names

Test: Add coverage for edge cases in plan generation
```

---

## Debugging Tips

### Common Issues

#### 1. "Template not found" Error

```python
# Problem: Templates not loaded
ERROR - Planner template not found

# Solution: Provide templates explicitly
processor = PromptProcessor(llm=llm, templates=test_templates)
```

#### 2. Mock Returns Mock Instead of Value

```python
# Problem: Mock not configured
mock_obj.method.return_value = mock.Mock()  # ❌

# Solution: Return proper type
mock_obj.method.return_value = []  # ✅ List
mock_obj.method.return_value = {"key": "value"}  # ✅ Dict
```

#### 3. Test Depends on External Service

```python
# Problem: Test calls real Ollama
llm = ChatOllama(model="llama3.2")

# Solution: Use mock
def test_with_mock(mock_llm):
    # mock_llm is injected
```

### Debugging Tools

```bash
# Run with debugger
pytest --pdb

# Show local variables on failure
pytest -l

# Verbose output
pytest -vv

# Show print statements
pytest -s

# Run specific test
pytest tests/unit/test_file.py::TestClass::test_method
```

---

## Resources

### Documentation Files

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [TEST_FAILURE_ANALYSIS.md](TEST_FAILURE_ANALYSIS.md) - Test status and failures
- [TESTING_FRAMEWORK.md](TESTING_FRAMEWORK.md) - Testing guide
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Quick testing reference

### External Documentation

- [Pytest Documentation](https://docs.pytest.org/)
- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [Model Context Protocol](https://modelcontextprotocol.io/)

### Code Examples

- `tests/unit/` - Unit test examples
- `tests/integration/` - Integration test examples
- `tests/test_utils.py` - Test helper utilities

---

## Getting Help

### When Stuck

1. **Check documentation** in this file and linked docs
2. **Review test examples** in `tests/` directory
3. **Look at similar code** in the codebase
4. **Run tests** to understand expected behavior
5. **Use debugger** to step through code

### Common Questions

**Q: How do I run just one test?**
```bash
pytest tests/unit/test_file.py::test_name -v
```

**Q: How do I see what a fixture provides?**
```python
def test_inspect_fixture(some_fixture):
    import pprint
    pprint.pprint(some_fixture)
    assert False  # Force test to fail and show output
```

**Q: How do I mock a complex object?**
```python
# Use test_utils.py helpers or create custom mock class
from tests.test_utils import MockLLM

llm = MockLLM(["response1", "response2"])
```

**Q: How do I test async code?**
```python
@pytest.mark.asyncio
async def test_async():
    result = await async_function()
    assert result
```

---

## Contribution Checklist

Before submitting a PR:

- [ ] Code follows style guidelines
- [ ] All new code has tests
- [ ] All tests pass (`./run_tests.sh all`)
- [ ] Coverage is maintained or improved
- [ ] Documentation is updated
- [ ] Commit messages are descriptive
- [ ] No debugging code left in (print statements, etc.)
- [ ] Type hints added for new functions
- [ ] Docstrings added for public methods

---

## Project Structure Quick Reference

```
code-rag/
├── application.py              # Main app orchestrator
├── main.py                     # CLI entry point
├── mcp_client.py               # Neo4j MCP client
├── config.json                 # Configuration
├── prompt_templates.json       # LLM prompts
│
├── rag/                        # Core modules
│   ├── flow/                   # Pipeline
│   │   ├── rag_pipeline.py
│   │   ├── prompt_processor.py
│   │   ├── step_executor.py
│   │   ├── types.py
│   │   └── task_execution_plan.py
│   │
│   ├── retrieval/              # Context retrieval
│   ├── storage/                # Data storage
│   ├── loading/                # Document loading
│   ├── parsing/                # Code parsing
│   ├── chunking/               # Code chunking
│   └── pipeline/               # Enhancements
│       ├── query_monitor.py
│       ├── query_classifier.py
│       └── graph_metadata.py
│
├── tests/                      # Test suite
│   ├── conftest.py             # Fixtures
│   ├── test_utils.py           # Helpers
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── e2e/                    # E2E tests
│
└── docs/                       # Documentation
    ├── ARCHITECTURE.md
    ├── TEST_FAILURE_ANALYSIS.md
    ├── TESTING_FRAMEWORK.md
    └── DEVELOPMENT_CONTEXT.md  # This file
```

---

**Last Updated**: December 22, 2025  
**Maintained By**: Development Team  
**Questions**: See individual documentation files or code comments
