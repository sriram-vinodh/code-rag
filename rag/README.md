# Modular RAG Application (Refactored)

## Package Structure

- `rag/` - Main package root
  - `chunking/` - Chunkers for text/code (AST, CPG, etc.)
  - `parser/` - Document and code parsers
  - `loader/` - Data loaders (filesystem, etc.)
  - `retriever/` - Retrieval logic (Neo4j, Cypher helpers, etc.)
  - `storage/` - Vector store interfaces and implementations
  - `pipeline/` - RAG pipeline orchestration
  - `__main__.py` - CLI entry point

## How to Run

```sh
python -m rag
```

## Main Components

- `application.py` (root): Main orchestration, CLI, and setup logic.
- `rag/chunking/`: Chunkers for splitting documents/code.
- `rag/parser/`: Parsers for documents and code.
- `rag/loader/`: Loaders for files and data sources.
- `rag/retriever/`: Knowledge graph and retrieval helpers.
- `rag/storage/`: Vector store interfaces and ChromaDB integration.
- `rag/pipeline/`: RAG pipeline logic.

## Notes
- All imports in `application.py` and other modules now use the new package structure.
- See `config.json` for configuration options.
