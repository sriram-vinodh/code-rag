from typing import List
from langchain_core.documents import Document
from chunker_interface import ChunkerInterface
from java_ast_parser import JavaAstParser, TypeDeclaration, MethodDeclaration
import logging

logger = logging.getLogger(__name__)

class JavaCodeChunker(ChunkerInterface):
    def __init__(self):
        pass

    """A semantic chunker for Java code that uses an AST parser."""
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunks Java documents semantically by method."""
        all_chunks = []
        for doc in documents:
            if doc.metadata.get("language") == "java":
                try:
                    # Attempt to chunk the Java document semantically
                    all_chunks.extend(self._chunk_single_java_document(doc))
                except Exception as e:
                    # If any error occurs (e.g., parsing a malformed file), log it and fall back
                    logger.warning(f"Failed to process {doc.metadata.get('source')} with AST chunker due to: {e}. Returning original document as one chunk.")
                    all_chunks.append(doc)
        return all_chunks

    def _chunk_single_java_document(self, document: Document) -> List[Document]:
        """Parses and chunks a single Java document using the AST parser."""
        parser = self._initialize_parser(document.page_content)
        chunks = []
        imports_and_package = parser.get_imports_and_package()
        
        chunks.extend(self._extract_chunks_from_parser(parser, document, imports_and_package))

        if not chunks and document.page_content.strip():
            chunks.append(document) # Fallback for files with no recognized methods

        return chunks

    def _initialize_parser(self, code_content: str) -> JavaAstParser:
        """Initializes the Java AST parser."""
        return JavaAstParser(code_content)

    def _extract_chunks_from_parser(self, parser: JavaAstParser, original_document: Document, imports_and_package: str) -> List[Document]:
        """Extracts chunks from the parsed Java AST."""
        chunks = []
        for type_decl in parser.get_type_declarations():
            for method_decl in type_decl.methods:
                chunks.append(self._create_document_from_declarations(
                    original_document=original_document,
                    imports_and_package=imports_and_package,
                    type_decl=type_decl,
                    method_decl=method_decl
                ))
        return chunks

    def _create_document_from_declarations(
        self,
        original_document: Document,
        imports_and_package: str,
        type_decl: TypeDeclaration,
        method_decl: MethodDeclaration
    ) -> Document:
        """Builds a Document chunk from parsed AST declarations.""" # noqa
        chunk_parts = [imports_and_package, type_decl.javadoc]
        class_body = self._format_class_body(type_decl, method_decl)
        chunk_parts.append(f"{type_decl.signature}\n{class_body}\n}}")
        page_content = "\n\n".join(filter(None, chunk_parts))

        metadata = original_document.metadata.copy()
        metadata.update({
            "class_name": type_decl.name,
            "method_name": method_decl.name,
            # javalang positions are 1-based.
            "start_line": method_decl.node.position.line if method_decl.node.position else None,
            # Estimate end line by counting newlines in the method's content.
            "end_line": (method_decl.node.position.line + method_decl.content.count('\n')) if method_decl.node.position else None,
        })
        return Document(page_content=page_content, metadata=metadata)

    def _format_class_body(self, type_decl: TypeDeclaration, method_decl: MethodDeclaration) -> str:
        """Formats the content of the class body for the chunk."""
        indented_javadoc = ('    ' + method_decl.javadoc.replace('\n', '\n    ')) if method_decl.javadoc else ''
        indented_method_content = '    ' + method_decl.content.replace('\n', '\n    ')
        
        fields_str = "\n".join(type_decl.fields) if type_decl.fields else None
        method_str = f"{indented_javadoc}\n{indented_method_content}".strip()

        return "\n".join(filter(None, [fields_str, method_str]))
