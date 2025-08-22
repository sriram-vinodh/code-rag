# rag.chunking

This package contains chunkers for splitting documents and code into semantically meaningful pieces.

- `chunker_interface.py`: Abstract base class for chunkers.
- `text_chunker.py`: Recursive text chunker for generic documents.
- `cpg_chunker.py`: Chunker for code property graphs (CPG).
- `java_code_chunker.py`: Chunker for Java code using AST or CPG.
