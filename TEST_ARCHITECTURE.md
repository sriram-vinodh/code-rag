# Test Architecture & Strategy

## Testing Framework
**pytest** - Chosen for its rich ecosystem, fixtures, parametrization, and plugin support.

## Test Categories

### 1. Unit Tests (`tests/unit/`)
Test individual components in isolation with mocked dependencies.

**Testable Components:**
- ✅ `RAGPipeline` - Query processing logic
- ✅ `StepExecutor` - Step execution logic
- ✅ `PromptProcessor` - Prompt generation
- ✅ `TaskExecutionPlan` - Plan creation and validation
- ✅ `Neo4jGraphRetriever` - Query construction (mocked DB)
- ✅ `MCPNeo4jClient` - MCP communication (mocked server)
- ✅ `QueryClassifier` - Query classification
- ✅ `QueryMonitor` - Metrics tracking
- ✅ `GraphMetadata` - Metadata extraction
- ✅ Parsers (DocumentParser) - Document parsing
- ✅ Loaders (FileSystemLoader) - File loading
- ✅ Chunkers (TextChunker, JavaCodeChunker) - Text chunking

**Non-Testable (Quantified):**
- ❌ LLM output quality - Non-deterministic
- ❌ Embeddings accuracy - Model-dependent
- ❌ Graph traversal completeness - Data-dependent

### 2. Integration Tests (`tests/integration/`)
Test component interactions with real or stubbed external services.

**Testable Flows:**
- ✅ RAGPipeline + StepExecutor + PromptProcessor integration
- ✅ Application + RAGPipeline full flow (mocked LLM)
- ✅ Neo4j retriever + MCP client (with test database)
- ✅ File loading → Parsing → Chunking pipeline
- ✅ Session management + Version control

**Non-Testable:**
- ❌ Real Ollama LLM responses - Requires running service
- ❌ Production Neo4j queries - Requires live database

### 3. End-to-End Tests (`tests/e2e/`)
Test complete user flows with minimal mocking.

**Testable Scenarios:**
- ✅ CLI question answering (mocked LLM)
- ✅ Web API request/response (mocked LLM)
- ✅ Document ingestion pipeline
- ✅ Error handling and recovery

**Non-Testable:**
- ❌ Real-time performance benchmarks - Environment-dependent
- ❌ Actual LLM reasoning quality - Non-deterministic

### 4. Fixture Tests (`tests/fixtures/`)
Reusable test data and mock objects.

## Testing Principles

1. **Isolation**: Each test is independent
2. **Speed**: Unit tests run in <5s, integration <30s
3. **Determinism**: No flaky tests due to LLM randomness
4. **Coverage**: Aim for 80%+ code coverage on business logic
5. **Maintainability**: Clear naming, documented fixtures
6. **No Pollution**: Tests don't modify application code

## Mock Strategy

### External Dependencies to Mock:
- **Ollama LLM**: Mock with deterministic responses
- **Neo4j Database**: Use in-memory or test database
- **MCP Server**: Mock subprocess communication
- **Serena Client**: Mock tool calls
- **File System**: Use temporary directories

### Not Mocked (Real):
- Data structures (PlanStep, ExecutionResult, etc.)
- Business logic algorithms
- Template loading (use test templates)
- JSON parsing

## Test Data Management

- **Sample documents**: Small, representative files
- **Mock LLM responses**: Realistic but controlled
- **Test schemas**: Simplified Neo4j graph
- **Fixture data**: JSON files in `tests/fixtures/data/`

## CI/CD Integration

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rag --cov=application --cov=mcp_client --cov-report=html

# Run specific category
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run fast tests only (exclude slow)
pytest -m "not slow"
```

## Coverage Goals

- **Core Logic**: 90%+ (RAG pipeline, executors, processors)
- **Integration Glue**: 70%+ (Application, clients)
- **Utilities**: 80%+ (helpers, parsers, loaders)
- **Overall Target**: 80%+
