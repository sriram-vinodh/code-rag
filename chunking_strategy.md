# Document Chunking Strategy

In a Retrieval-Augmented Generation (RAG) system, the way documents are split into smaller, manageable pieces (chunks) is crucial for the quality of the retrieved context and, consequently, the accuracy of the LLM's answers. This application employs an intelligent, multi-strategy chunking approach to optimize context retrieval for various document types.

## Why Intelligent Chunking?

Simple character-based splitting can break logical units, especially in structured data like code. For example, splitting a Java method in half would render both halves less useful. By applying different chunking strategies based on the document's content type, we ensure that each chunk represents a semantically meaningful unit, improving the relevance of retrieved information.

## Chunking Components

The application utilizes the following chunking components:

1.  **`RecursiveTextChunker` (Default)**
    *   **Purpose**: This is the general-purpose chunker used for most text-based documents.
    *   **Mechanism**: It employs LangChain's `RecursiveCharacterTextSplitter`. This splitter attempts to split text using a list of characters (`\n\n`, `\n`, ` `, etc.) in order until the chunks are small enough. This helps to keep paragraphs and sentences together.
    *   **Configuration**: The `chunk_size` and `chunk_overlap` parameters (defined in `config.json` or via command-line arguments) control the maximum size of each chunk and the amount of overlap between consecutive chunks, respectively.
    *   **Applicability**: Used for `.txt`, `.md`, `.html`, `.htm`, `.json`, and any other file types not explicitly handled by a specialized chunker.

2.  **`JavaCodeChunker` (Specialized for Java)**
    *   **Purpose**: This application provides two advanced, semantic chunking strategies for Java source code (`.java`), selectable via the `--java-chunker-strategy` argument.
    *   **Strategy 1: `ast`**
        *   **Mechanism**: This chunker uses `tree-sitter` to parse Java code into an Abstract Syntax Tree (AST). It traverses the AST to extract individual methods. Each chunk includes the method's code, its class signature, class-level fields, and Javadocs, providing rich intra-class context.
        *   **Mechanism (Updated)**: This chunker now uses `javalang` (a pure-Python Java parser) to parse code into an AST. It extracts methods, class signatures, fields, and Javadocs. This approach is more robust as it has no external build dependencies.
        *   **Mechanism**: This is a more advanced strategy that uses **Code Property Graphs (CPGs)**. It requires the external tool `joern` to be installed. A CPG combines the AST with control-flow and data-flow graphs.
        *   **Enriched Context**: The chunker runs `joern` to build a CPG of the source code. It then extracts each method and enriches it with inter-procedural context: the names of other methods that call it (callers) and methods that it calls (callees). This provides the LLM with a deeper understanding of how methods interact across the codebase.
    *   **Configurable Fallback Hierarchy**: The application attempts to initialize the Java chunker based on the `--java-chunker-strategy` argument. If this primary strategy's chunker fails to initialize (e.g., due to missing `joern` or a broken `tree-sitter` installation), the system will then attempt to use chunkers from a configurable fallback order. This order is defined by the `java_chunker_fallback_order` list in `config.json` (defaulting to `["cpg", "ast"]`). If all specialized Java chunkers fail, Java files will be processed by the `RecursiveTextChunker` as the ultimate fallback.

## Orchestration in `Application.setup()`

The `Application` class's `setup()` method orchestrates the data ingestion and chunking process:

1.  **Document Loading**: The application loads files from two distinct locations specified by `--code-path` (default: `./code`) and `--docs-path` (default: `./docs`).
2.  **Parsing**: All loaded files are parsed by `DefaultDocumentParser`. During this step, `.java` files are tagged with `metadata["language"] = "java"` to identify them for specialized processing.
3.  **Strategy Selection**: Based on the `--java-chunker-strategy` argument, the appropriate chunker (`JavaCodeChunker` for `ast` or `CpgCodeChunker` for `cpg`) is initialized.
3.  **Categorization**: Documents are then categorized into `java_docs` and `other_docs` based on this `language` metadata. The `java_chunker_strategy` argument determines the *preferred* chunker, but the system will attempt fallbacks if initialization fails.
3.  **Chunker Assignment**:
    *   `other_docs` are passed to the `RecursiveTextChunker`.
    *   `java_docs` are passed to the selected specialized Java chunker (CPG or AST).
    *   If the initially preferred specialized chunker is not available or fails to initialize, the system attempts to use the next available specialized chunker in the configurable fallback hierarchy. If no specialized Java chunker can be initialized, `java_docs` are processed by the `RecursiveTextChunker` as a final fallback.
4.  **Vector Store Addition**: All generated chunks (from both chunkers) are then added to the vector store for embedding and retrieval.

## Background CPG Generation

To provide the most contextual information without blocking the main application's startup, the application can optionally generate CPGs for Java code in the background.

*   **Enabling**: This feature is controlled by the `background_cpg_generation_enabled` setting in `config.json` (default: `false`).
*   **Mechanism**:
    *   When enabled, a separate `multiprocessing.Process` is spawned during the application's `setup` phase.
    *   This background process independently loads Java source files, generates CPGs using the `CpgCodeChunker`, and stores these CPG-based chunks in a *separate* persistent ChromaDB instance.
    *   The path for this separate CPG vector store is configured via `cpg_vector_store_path` in `config.json` (default: `./chroma_cpg_db`).
*   **Impact on Main Application**:
    *   The main application's startup is *not* blocked by this background process.
    *   The primary vector store (configured by `vector_store_path`) will be populated using the default or configured `java_chunker_strategy` (e.g., AST-based chunks).
    *   The CPGs generated in the background are currently stored in a separate database and are *not automatically integrated* into the main application's retriever for the current session. This feature is intended for pre-computation and persistence of the most detailed code representation, which can be leveraged in future enhancements (e.g., a combined retriever, or a mechanism to refresh the main store with CPGs).
*   **Error Handling**: Errors during background CPG generation are logged to `stderr` of the background process and do not crash the main application.

This allows the application to be responsive while still preparing the most advanced code representations for future use.

## RAG Query Pipeline

Once the vector store is populated, the application uses a sophisticated RAG pipeline to answer user questions. This pipeline includes a "relevance gate" to ensure that only questions pertaining to the indexed documents are processed.

1.  **Relevance Gate**:
    *   When a user asks a question, it is first sent to a relevance classifier.
    *   This classifier is a small LLM chain that determines if the question is 'RELEVANT' (related to the codebase/docs) or 'IRRELEVANT' (a general knowledge question, greeting, etc.).
    *   If the question is deemed 'IRRELEVANT', the pipeline stops, and a message is returned to the user indicating that the chatbot can only answer questions about the provided context.

2.  **Context Retrieval**:
    *   If the question is 'RELEVANT', it is passed to the vector store's retriever.
    *   The retriever performs a similarity search to find the most relevant document chunks from the vector store.

3.  **Prompt Augmentation and Generation**:
    *   The retrieved chunks (the "context") and the original question are inserted into a prompt template.
    *   This augmented prompt is then sent to the primary language model (e.g., `phi3:latest`).
    *   The LLM generates an answer based *only* on the provided context and the question.

This two-step process ensures that the RAG system stays focused on the provided data, prevents it from answering out-of-scope questions, and improves the overall quality and reliability of its responses.


## Persistent Vector Storage

To avoid re-processing and re-embedding documents on every application startup, the application now utilizes a persistent vector database.

*   **Database**: ChromaDB is used for vector storage.
*   **Persistence**: The vector store is configured to save its data to disk at a specified path (default: `./chroma_db`). This path can be configured via the `vector_store_path` setting in `config.json` or overridden with the `--vector-store-path` command-line argument.
*   **Behavior**:
    *   On startup, the `ChromaVectorStore` first checks if an existing vector store is present at the configured `vector_store_path`.
    *   If an existing store is found, new documents (from the `code_path` and `docs_path`) are added to it.
    *   If no existing store is found, a new one is created from the processed documents.
    *   After processing, the updated vector store is explicitly persisted to disk using `_vector_store.persist()`, ensuring that changes are saved for future sessions.
    *   If no new documents are found in the configured `code_path` or `docs_path`, the application will still attempt to load the existing vector store for retrieval, allowing the RAG pipeline to function with previously indexed data.

This significantly reduces startup time for subsequent runs, as document processing and embedding only need to occur for new or modified documents (though the current implementation re-adds all documents, Chroma handles deduplication internally to some extent).



This layered approach ensures that the RAG system can effectively understand and retrieve context from diverse document types, particularly benefiting from the semantic understanding of code.