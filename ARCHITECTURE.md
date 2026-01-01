# Code-RAG Architecture Documentation

**Version**: 2.0  
**Last Updated**: December 22, 2025  
**Purpose**: Retrieval-Augmented Generation system for code understanding using Neo4j graph database and LLM reasoning

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Layers](#architecture-layers)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Module Structure](#module-structure)
6. [External Dependencies](#external-dependencies)
7. [Configuration](#configuration)
8. [Extension Points](#extension-points)

---

## System Overview

### Purpose
Code-RAG is a Retrieval-Augmented Generation (RAG) system designed to answer natural language questions about codebases by combining:
1. **Graph Database** (Neo4j) - Stores code structure and relationships
2. **Vector Store** (Chroma) - Semantic search over code chunks
3. **LLM** (Ollama/Llama) - Natural language understanding and generation
4. **IDE Agent** (Serena) - Code navigation and modification

### Key Capabilities
- ✅ Natural language queries about code structure
- ✅ Multi-step query decomposition and execution
- ✅ Graph-based context retrieval (classes, methods, dependencies)
- ✅ Semantic vector search for implementation details
- ✅ Cypher query generation from natural language
- ✅ Code modification via IDE agent integration
- ✅ Query monitoring and instrumentation
- ✅ Context insufficiency detection and retry logic

### Architecture Style
- **Pattern**: Modular Pipeline Architecture
- **Paradigm**: Event-driven with step-based execution
- **Integration**: MCP (Model Context Protocol) for Neo4j communication
- **Testability**: Dependency injection with mock-friendly interfaces

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │     CLI      │  │  Web API     │  │   Jupyter    │         │
│  │  (main.py)   │  │ (Flask/REST) │  │   Notebook   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                            │
│  ┌──────────────────────────────────────────────────────┐      │
│  │          Application (application.py)                 │      │
│  │  - Orchestration   - Configuration   - Setup         │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PIPELINE LAYER                             │
│  ┌──────────────────────────────────────────────────────┐      │
│  │            RAGPipeline (rag_pipeline.py)              │      │
│  │  ┌────────────────┐  ┌──────────────────────────┐   │      │
│  │  │ Prompt         │  │     Step Executor         │   │      │
│  │  │ Processor      │──│  (step_executor.py)       │   │      │
│  │  └────────────────┘  └──────────────────────────┘   │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE LAYER                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Neo4j     │  │   Vector    │  │   Serena    │            │
│  │   Graph     │  │   Store     │  │  IDE Agent  │            │
│  │  Retriever  │  │  (Chroma)   │  │             │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INTEGRATION LAYER                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ MCP Client  │  │    Ollama   │  │   Serena    │            │
│  │  (Neo4j)    │  │     LLM     │  │     MCP     │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Application Layer

#### `application.py` - Main Application Orchestrator
**Responsibility**: Application lifecycle and configuration management

**Key Methods**:
```python
class Application:
    def __init__(self, config_path: str = "config.json")
    def load_config(self) -> Dict[str, Any]
    def setup(self) -> bool
    def ask(self, question: str) -> str
    def run(self)  # CLI entry point
```

**Key Responsibilities**:
- Load and validate configuration
- Initialize all components (LLM, Neo4j, Vector Store, Serena)
- Manage component lifecycle (setup, teardown)
- Handle CLI argument parsing
- Route queries to RAG pipeline
- Error handling and logging setup

**Dependencies**:
- `RAGPipeline` - Core query processing
- `Ollama` (langchain) - LLM provider
- `MCPNeo4jClient` - Neo4j MCP connection
- `SerenaClient` - IDE agent connection

**Configuration Keys**:
```json
{
  "llm": {
    "base_url": "http://localhost:11434",
    "model": "llama3.2:3b",
    "temperature": 0.1
  },
  "pipeline_settings": {
    "templates_file": "prompt_templates.json",
    "log_level": "INFO"
  }
}
```

---

### 2. Pipeline Layer

#### `rag/flow/rag_pipeline.py` - Query Processing Pipeline
**Responsibility**: High-level query orchestration

**Key Methods**:
```python
class RAGPipeline:
    def process_query(
        self, 
        question: str,
        use_graph: bool = True,
        use_vector: bool = False
    ) -> str
    
    def ask(self, question: str, only_graph: bool = False) -> str
```

**Processing Flow**:
```
1. question → PromptProcessor.generate_plan()
   ↓
2. plan (List[PlanStep]) → StepExecutor.execute_plan()
   ↓
3. accumulated results → final answer
```

**Configuration**:
- `templates_file`: Path to prompt templates JSON
- `llm`: Language model instance
- `neo4j_retriever`: Graph context retriever
- `serena_client`: Optional IDE agent

---

#### `rag/flow/prompt_processor.py` - Plan Generation
**Responsibility**: Convert questions into execution plans

**Key Methods**:
```python
class PromptProcessor:
    def generate_plan(self, question: str) -> List[PlanStep]
```

**Plan Generation Process**:
```
1. Load "planner" template from config
2. Inject question into template
3. LLM generates JSON plan
4. Validate with Pydantic (Generation model)
5. Parse into List[PlanStep]
```

**Output Example**:
```python
[
    PlanStep(
        step="Find all classes in the codebase",
        needs_context=True,
        complexity=StepComplexity.MEDIUM,
        knowledge_source="graph_database",
        action_type=ActionType.READ
    ),
    PlanStep(
        step="Analyze class relationships",
        needs_context=True,
        complexity=StepComplexity.HIGH,
        knowledge_source="graph_database",
        action_type=ActionType.ANALYZE
    )
]
```

**Template Structure**:
```json
{
  "rag": {
    "planner": "You are a query planner...\nQuestion: {question}\n..."
  }
}
```

---

#### `rag/flow/step_executor.py` - Step Execution Engine
**Responsibility**: Execute individual plan steps, fetch context, generate answers

**Key Methods**:
```python
class StepExecutor:
    def execute_plan(self, plan: TaskExecutionPlan) -> str
    def _execute_step(self, step: PlanStep, context: List[str]) -> ExecutionResult
    def _fetch_context_from_source(self, step: PlanStep) -> Optional[str]
```

**Execution Flow**:
```
For each step in plan:
  1. Check if context needed (step.needs_context)
  2. If yes → fetch from knowledge_source
     - "graph_database" → Neo4j Cypher query
     - "serena_ide" → Serena symbol search
     - "hybrid" → Both sources
  3. Build prompt with step + context + accumulated results
  4. LLM generates answer
  5. Accumulate context for next step
  6. Return final aggregated result
```

**Knowledge Source Routing**:
```python
def _fetch_context_from_source(self, step: PlanStep) -> Optional[str]:
    source = step.knowledge_source
    
    if source == "graph_database":
        return self._fetch_from_graph_database(step)
    elif source == "serena_ide":
        return self._fetch_from_serena(step)
    elif source == "hybrid":
        graph_ctx = self._fetch_from_graph_database(step)
        serena_ctx = self._fetch_from_serena(step)
        return self._merge_contexts(graph_ctx, serena_ctx)
```

**Context Fetching Process**:
```
graph_database:
  1. Generate Cypher from natural language (LLM)
  2. Execute via MCPNeo4jClient
  3. Format results as context

serena_ide:
  1. Use SerenaClient.find_symbol()
  2. Get symbol metadata + body
  3. Format as context

hybrid:
  1. Fetch both concurrently
  2. Merge with deduplication
  3. Return combined context
```

---

### 3. Knowledge Layer

#### `mcp_client.py` - Neo4j MCP Client
**Responsibility**: Communicate with Neo4j via Model Context Protocol

**Key Methods**:
```python
class MCPNeo4jClient:
    def connect(self)
    def get_schema(self, sample_size: int = 1000) -> Optional[Dict]
    def execute_read_query(self, query: str, params: Dict = None) -> List[Dict]
    def execute_write_query(self, query: str, params: Dict = None) -> List[Dict]
    def query_with_natural_language(
        self, 
        question: str, 
        llm, 
        return_metadata: bool = False
    ) -> List[Dict]
    def close(self)
```

**MCP Protocol**:
```
Client Process (Python)         MCP Server Process (Node.js/uvx)
     │                                    │
     ├─ connect() ───────────────────────▶ Start subprocess
     │                                    │ (uvx mcp-neo4j-cypher)
     │                                    │
     ├─ initialize ──────────────────────▶ Setup protocol
     │◀─ capabilities ────────────────────┤
     │                                    │
     ├─ tools/call: get_schema ──────────▶ Query Neo4j
     │◀─ JSON response ───────────────────┤
     │                                    │
     ├─ tools/call: read_cypher ─────────▶ Execute query
     │◀─ Results ─────────────────────────┤
     │                                    │
     ├─ close() ─────────────────────────▶ Terminate
```

**Natural Language Query Process**:
```python
def query_with_natural_language(question, llm):
    # 1. Classify query type
    classification = classify_query(question)
    
    # 2. Select template based on type
    template = select_template(classification.query_type)
    
    # 3. Generate Cypher (with retries)
    for attempt in range(max_retries):
        cypher = llm.invoke(template.format(question=question))
        
        # 4. Validate via EXPLAIN
        if validate_cypher(cypher):
            break
        else:
            # Include error feedback in next attempt
            question += f"\nError: {last_error}"
    
    # 5. Execute via MCP
    return execute_read_query(cypher)
```

**Query Classification** (Phase 2 Enhancement):
```python
class QueryType(Enum):
    STRUCTURAL = "structural"      # Classes, packages, dependencies
    IMPLEMENTATION = "implementation"  # Method bodies, logic
    SEARCH = "search"              # Find by name/pattern
    TRACE = "trace"                # Call chains, data flow
    MODIFICATION = "modification"  # Recent changes
    HYBRID = "hybrid"              # Multiple types
```

**Instrumentation**: Integrated with `QueryMonitor` for tracking:
- Query classification
- Template selection
- Cypher generation iterations
- Execution time
- Results metadata

---

#### `rag/retrieval/neo4j_graph_retriever.py` - Graph Context Retriever
**Responsibility**: High-level graph query interface (legacy, being replaced by MCP)

**Key Methods**:
```python
class Neo4jGraphRetriever:
    def get_relevant_context(
        self, 
        query: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]
```

**Status**: Legacy component, gradually being replaced by direct `MCPNeo4jClient` usage.

---

#### `rag/storage/chroma_vector_store.py` - Vector Storage
**Responsibility**: Semantic search over code chunks

**Key Methods**:
```python
class ChromaVectorStore:
    def add_documents(self, documents: List[Document])
    def search(self, query: str, k: int = 5) -> List[Document]
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5
    ) -> List[Tuple[Document, float]]
```

**Use Cases**:
- Finding implementation details when structure is known
- Semantic code search
- Similar code detection
- Documentation lookup

**Current Status**: Implemented but not actively used in main query flow (graph-first strategy).

---

### 4. Data Models

#### `rag/flow/types.py` - Core Data Types

**PlanStep** - Represents one step in execution plan
```python
@dataclass
class PlanStep:
    # Core fields
    step: str                              # Human-readable description
    needs_context: bool                    # Whether to fetch context
    complexity: StepComplexity             # LOW, MEDIUM, HIGH
    
    # Context management
    required_context_description: Optional[str]
    fetched_context: Optional[str]
    knowledge_source: Optional[str]        # "graph_database", "serena_ide", "hybrid"
    result: Optional[str]
    previous_step_result: Optional[str]
    
    # Code modification (Serena integration)
    action_type: Optional[ActionType]      # READ, ANALYZE, MODIFY, etc.
    requires_code_edit: bool = False
    target_symbols: Optional[List[str]]
    edit_strategy: Optional[EditStrategy]
    
    # Context insufficiency tracking (Phase 1)
    is_complete: bool = False
    completion_confidence: float = 0.0
    context_corpus: List[Dict[str, Any]]   # Accumulated context
    attempted_queries: List[str]
    retry_history: List[Dict[str, Any]]
```

**TaskExecutionPlan** - Container for plan steps
```python
@dataclass
class TaskExecutionPlan:
    total_steps: int                       # Total number of steps
    steps: List[PlanStep]                  # The actual steps
    current_step: int = 0                  # Current position
    
    def get_next_step(self) -> Optional[PlanStep]
    def advance(self) -> bool
    def update_step_result(self, result: ExecutionResult)
```

**ExecutionResult** - Result of step execution
```python
@dataclass
class ExecutionResult:
    success: bool
    message: str
    context: Optional[str]
    
    # Code modification tracking
    code_changes: Optional[List[str]]      # Modified file paths
    serena_used: bool = False
```

**Enums**:
```python
class StepComplexity(Enum):
    LOW = "low"        # Simple, single query
    MEDIUM = "medium"  # Multiple queries or analysis
    HIGH = "high"      # Complex reasoning or editing

class ActionType(Enum):
    READ = "read"            # Read-only
    ANALYZE = "analyze"      # Deep analysis
    MODIFY = "modify"        # Code changes
    REFACTOR = "refactor"    # Restructure
    CREATE = "create"        # New code
    DELETE = "delete"        # Remove code

class EditStrategy(Enum):
    SYMBOLIC = "symbolic"      # Replace symbol body
    FILE_BASED = "file_based"  # String replacement
    REGEX = "regex"            # Regex replacement
    INSERTION = "insertion"    # Insert before/after
    RENAME = "rename"          # Rename across codebase
```

---

### 5. Integration Layer

#### Serena MCP Client (Planned)
**Purpose**: IDE agent for code navigation and modification

**Capabilities**:
- Symbol search (`find_symbol`)
- Read symbol bodies
- Code modification (replace, insert, rename)
- File operations

**Integration Point**: `StepExecutor._fetch_from_serena()`

#### Ollama LLM
**Purpose**: Language model for understanding and generation

**Models Supported**:
- llama3.2:3b (default, fast)
- llama3.2:1b (lightweight)
- codellama (code-specialized)

**Usage**:
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2:3b",
    temperature=0.1,
    num_predict=2048
)

response = llm.invoke("prompt here")
answer = response.content
```

---

## Data Flow

### Query Processing Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. User Input                                               │
│    "What classes implement the Parser interface?"          │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Application.ask(question)                                │
│    - Log question                                           │
│    - Route to RAGPipeline                                   │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. RAGPipeline.process_query(question)                      │
│    - Call PromptProcessor.generate_plan()                   │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. PromptProcessor.generate_plan(question)                  │
│    - Load "planner" template                                │
│    - LLM generates JSON plan                                │
│    - Parse into List[PlanStep]                              │
│                                                             │
│    Example Output:                                          │
│    [                                                        │
│      PlanStep(                                              │
│        step="Find Parser interface definition",             │
│        needs_context=True,                                  │
│        complexity=MEDIUM,                                   │
│        knowledge_source="graph_database"                    │
│      ),                                                     │
│      PlanStep(                                              │
│        step="Find classes implementing Parser",             │
│        needs_context=True,                                  │
│        complexity=HIGH,                                     │
│        knowledge_source="graph_database"                    │
│      )                                                      │
│    ]                                                        │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Create TaskExecutionPlan                                 │
│    plan = TaskExecutionPlan(                                │
│        total_steps=2,                                       │
│        steps=[...]                                          │
│    )                                                        │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. StepExecutor.execute_plan(plan)                          │
│                                                             │
│    Loop: for each step in plan                              │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Execute Step 1: "Find Parser interface"                 │
│                                                             │
│    a) Check step.needs_context → True                       │
│    b) Check step.knowledge_source → "graph_database"        │
│    c) Call _fetch_from_graph_database(step)                 │
│       ├─ Generate Cypher from NL description               │
│       ├─ LLM: "Find Parser interface" →                    │
│       │        "MATCH (i:Interface {name:'Parser'})        │
│       │         RETURN i"                                  │
│       ├─ Execute via MCPNeo4jClient                        │
│       └─ Return: [{id: "Parser", methods: [...]}]          │
│                                                             │
│    d) fetched_context = format_graph_results(results)       │
│    e) Build step prompt:                                    │
│       "Step: Find Parser interface definition               │
│        Context: {fetched_context}                           │
│        Previous: N/A                                        │
│        Provide answer."                                     │
│                                                             │
│    f) LLM generates answer                                  │
│    g) Store result in step.result                           │
│    h) Add to accumulated_context list                       │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. Execute Step 2: "Find classes implementing Parser"      │
│                                                             │
│    a) Check step.needs_context → True                       │
│    b) Call _fetch_from_graph_database(step)                 │
│       ├─ LLM: "Find classes implementing Parser" →         │
│       │        "MATCH (c:Class)-[:IMPLEMENTS]->(i:Interface│
│       │         {name:'Parser'})                            │
│       │         RETURN c.name, c.file_path"                │
│       ├─ Execute Cypher                                    │
│       └─ Return: [                                         │
│              {name: "TextParser", path: "text_parser.py"}, │
│              {name: "JavaParser", path: "java_parser.py"}  │
│           ]                                                │
│                                                             │
│    c) Build prompt with:                                    │
│       - Current step description                            │
│       - Fetched graph context                               │
│       - Previous step result (from Step 1)                  │
│       - Accumulated context                                 │
│                                                             │
│    d) LLM generates final answer                            │
│    e) Return result                                         │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 9. Aggregate Results                                        │
│    final_answer = "\n\n".join([                             │
│        step1.result,                                        │
│        step2.result                                         │
│    ])                                                       │
│                                                             │
│    Example Output:                                          │
│    "The Parser interface is defined in parser.py with       │
│     methods parse() and validate(). Two classes implement   │
│     it: TextParser (text_parser.py) and JavaParser          │
│     (java_parser.py)."                                      │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 10. Return to User                                          │
│     - Display in CLI / Web UI / Notebook                    │
│     - Log to file if configured                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
code-rag/
├── application.py                 # Main application orchestrator
├── main.py                        # CLI entry point
├── mcp_client.py                  # Neo4j MCP client
├── config.json                    # Application configuration
├── prompt_templates.json          # LLM prompt templates
│
├── rag/                           # Core RAG modules
│   ├── flow/                      # Query processing pipeline
│   │   ├── rag_pipeline.py        # Main pipeline orchestrator
│   │   ├── prompt_processor.py    # Plan generation
│   │   ├── step_executor.py       # Step execution engine
│   │   ├── types.py               # Data models (PlanStep, etc.)
│   │   └── task_execution_plan.py # Plan container
│   │
│   ├── retrieval/                 # Context retrieval
│   │   └── neo4j_graph_retriever.py  # Graph context (legacy)
│   │
│   ├── storage/                   # Data storage
│   │   └── chroma_vector_store.py # Vector store
│   │
│   ├── loading/                   # Document loading
│   │   ├── file_loader.py
│   │   └── filesystem_loader.py
│   │
│   ├── parsing/                   # Code parsing
│   │   ├── document_parser.py
│   │   └── java_ast_parser.py
│   │
│   ├── chunking/                  # Code chunking
│   │   ├── text_chunker.py
│   │   ├── java_code_chunker.py
│   │   └── cpg_chunker.py
│   │
│   └── pipeline/                  # Query pipeline enhancements
│       ├── query_monitor.py       # Instrumentation
│       ├── query_classifier.py    # Query classification
│       └── graph_metadata.py      # Result metadata extraction
│
├── tests/                         # Test suite
│   ├── conftest.py                # Pytest fixtures
│   ├── test_utils.py              # Test helpers
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── e2e/                       # End-to-end tests
│
└── docs/                          # Documentation
    ├── ARCHITECTURE.md            # This file
    ├── TEST_FAILURE_ANALYSIS.md   # Test results
    └── TESTING_GUIDE.md           # Testing documentation
```

---

## External Dependencies

### Required Services

#### 1. Ollama (LLM Provider)
- **Purpose**: Language model inference
- **URL**: http://localhost:11434
- **Models**: llama3.2:3b, llama3.2:1b, codellama
- **Install**: `brew install ollama` (macOS)
- **Start**: `ollama serve`

#### 2. Neo4j (Graph Database)
- **Purpose**: Code structure storage
- **URL**: bolt://localhost:7687
- **Auth**: neo4j / password
- **Install**: `brew install neo4j` or Docker
- **Schema**: Nodes (Class, Method, Field, Interface, Package)
           Relationships (IMPLEMENTS, EXTENDS, CALLS, CONTAINS)

#### 3. MCP Server (Model Context Protocol)
- **Purpose**: Neo4j communication protocol
- **Command**: `uvx mcp-neo4j-cypher@0.5.1 --transport stdio`
- **Tools Provided**:
  - `get_neo4j_schema`
  - `read_neo4j_cypher`
  - `write_neo4j_cypher`

### Optional Services

#### 4. Serena MCP (IDE Agent)
- **Purpose**: Code navigation and modification
- **Status**: Planned integration
- **Protocol**: MCP over stdio

#### 5. ChromaDB (Vector Store)
- **Purpose**: Semantic code search
- **Status**: Implemented but not in main flow
- **Storage**: Local file system

---

## Configuration

### config.json Structure

```json
{
  "llm": {
    "base_url": "http://localhost:11434",
    "model": "llama3.2:3b",
    "temperature": 0.1,
    "num_predict": 2048,
    "timeout": 120
  },
  
  "neo4j": {
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "password"
  },
  
  "mcp": {
    "neo4j_server": {
      "command": "uvx",
      "args": [
        "mcp-neo4j-cypher@0.5.1",
        "--transport",
        "stdio"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "password"
      }
    }
  },
  
  "pipeline_settings": {
    "templates_file": "prompt_templates.json",
    "max_context_length": 8000,
    "log_level": "INFO",
    "enable_monitoring": true
  },
  
  "vector_store": {
    "type": "chroma",
    "persist_directory": "./chroma_db",
    "collection_name": "code_chunks"
  }
}
```

### prompt_templates.json Structure

```json
{
  "rag": {
    "planner": "You are a query planner for a code RAG system...",
    "step_executor": "Execute this step...",
    "cypher_generation": "Generate Cypher query for...",
    "cypher_generation_structural": "Generate Cypher for structural query...",
    "cypher_generation_implementation": "Generate Cypher for implementation query...",
    "cypher_generation_search": "Generate Cypher for search query...",
    "cypher_generation_trace": "Generate Cypher for trace query...",
    "cypher_generation_modification": "Generate Cypher for modification query..."
  }
}
```

---

## Extension Points

### 1. Adding New Knowledge Sources

To add a new knowledge source (e.g., GitHub API, Confluence):

```python
# In step_executor.py
def _fetch_context_from_source(self, step: PlanStep) -> Optional[str]:
    source = step.knowledge_source
    
    if source == "github_api":
        return self._fetch_from_github(step)
    # ... existing sources

def _fetch_from_github(self, step: PlanStep) -> Optional[str]:
    # Implement GitHub API integration
    pass
```

### 2. Adding New Query Types

To support new query classifications:

```python
# In rag/pipeline/query_classifier.py
class QueryType(Enum):
    # ... existing types
    SECURITY = "security"  # Security analysis queries

# Add pattern matching
QUERY_PATTERNS = {
    QueryType.SECURITY: [
        r"security\s+(issue|vulnerability|risk)",
        r"(vulnerable|insecure|unsafe)\s+code"
    ]
}

# Add template mapping in prompt_templates.json
{
  "rag": {
    "cypher_generation_security": "Generate Cypher to find security issues..."
  }
}
```

### 3. Adding New Action Types

To support new step actions:

```python
# In rag/flow/types.py
class ActionType(Enum):
    # ... existing types
    TEST = "test"      # Generate tests
    DOCUMENT = "document"  # Generate documentation

# In step_executor.py
def _execute_step(self, step: PlanStep, context: List[str]):
    if step.action_type == ActionType.TEST:
        return self._generate_tests(step, context)
    elif step.action_type == ActionType.DOCUMENT:
        return self._generate_documentation(step, context)
```

### 4. Custom Prompt Templates

Templates use string formatting with placeholders:

```python
# In prompt_templates.json
{
  "rag": {
    "custom_template": """
You are analyzing {language} code.

Task: {task}
Context: {context}
Code: {code}

Provide detailed analysis.
"""
  }
}

# In code:
template = templates["custom_template"]
prompt = template.format(
    language="Python",
    task="Find bugs",
    context=context_string,
    code=code_string
)
```

---

## Performance Considerations

### Query Optimization
- **Graph Queries**: Use indexed properties (name, type)
- **LLM Calls**: Cache frequent queries, batch when possible
- **Vector Search**: Limit results (k=5 default)

### Scalability
- **Neo4j**: Can handle millions of nodes
- **Chroma**: Scales to hundreds of thousands of chunks
- **Ollama**: Limited by GPU/CPU, consider batching

### Monitoring
- Query execution time tracked via `QueryMonitor`
- Template selection logged
- Cypher generation iterations recorded
- Results metadata extracted

---

## Security Considerations

### Credentials
- Store in environment variables or config files (not in code)
- Neo4j credentials in `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- Consider using secrets management (Vault, AWS Secrets Manager)

### Code Execution
- **Serena Integration**: Code modifications need governance
- **Action Type Tracking**: All modifications logged
- **User Approval**: Consider requiring approval for MODIFY/DELETE actions

### Input Validation
- Validate user queries before processing
- Sanitize inputs to prevent injection
- Rate limit queries if exposed via API

---

## Future Enhancements

### Planned Features
1. **Serena Integration**: Full IDE agent support for code modification
2. **Multi-language Support**: Expand beyond Java (Python, TypeScript, etc.)
3. **Diff Generation**: Generate code diffs for review
4. **Test Generation**: Automated test case creation
5. **Documentation Generation**: Auto-generate docstrings
6. **Code Review**: Automated review comments
7. **Refactoring Suggestions**: Intelligent refactoring recommendations

### Architecture Improvements
1. **Event Sourcing**: Track all queries and modifications
2. **Caching Layer**: Redis for frequent queries
3. **Async Execution**: Parallel step execution where possible
4. **Plugin System**: Extensible knowledge source plugins
5. **Web UI**: React frontend for query interface

---

## Troubleshooting

### Common Issues

#### 1. "Template not found" error
**Cause**: `prompt_templates.json` missing or incorrect path  
**Fix**: Verify file exists and `templates_file` in config is correct

#### 2. MCP connection fails
**Cause**: Neo4j not running or credentials wrong  
**Fix**: Check Neo4j service, verify credentials in config

#### 3. LLM timeout
**Cause**: Ollama not responding or model not loaded  
**Fix**: Run `ollama list` and `ollama pull llama3.2:3b`

#### 4. Empty results from graph
**Cause**: No data in Neo4j  
**Fix**: Ingest codebase first (separate process)

---

## Glossary

- **RAG**: Retrieval-Augmented Generation - combining retrieval and LLM generation
- **MCP**: Model Context Protocol - protocol for tool/data access
- **Cypher**: Neo4j's query language (like SQL for graphs)
- **Plan**: Sequence of steps to answer a query
- **Step**: Single unit of work in a plan
- **Context**: Retrieved information used to answer a step
- **Knowledge Source**: Where context comes from (graph, vector store, IDE)
- **Action Type**: Category of operation (READ, MODIFY, etc.)
- **Complexity**: Estimated difficulty of a step (LOW, MEDIUM, HIGH)

---

**End of Architecture Documentation**
