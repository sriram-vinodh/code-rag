from typing import List
from langchain_core.documents import Document
from chunker_interface import ChunkerInterface
from tree_sitter import Node

try:
    from tree_sitter_languages import get_parser
except ImportError:
    # This is a fallback for the print statement in application.py to work correctly
    # The application will handle the case where the library is not installed.
    def get_parser(language):
        raise ImportError(f"Language parser for '{language}' not available. Please `pip install tree_sitter_languages`.")

class JavaCodeChunker(ChunkerInterface):
    """A semantic chunker for Java code using tree-sitter."""

    def __init__(self):
        """Initializes the JavaCodeChunker."""
        try:
            self.parser = get_parser("java")
        except (ImportError, OSError) as e:
            print(f"Error initializing Java parser: {e}")
            raise ImportError("JavaCodeChunker requires `tree_sitter_languages`. Please `pip install tree_sitter_languages` and ensure git is installed.")

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunks Java documents semantically by method."""
        all_chunks = []
        for doc in documents:
            if doc.metadata.get("language") == "java":
                all_chunks.extend(self._chunk_code(doc))
        return all_chunks

    def _chunk_code(self, document: Document) -> List[Document]:
        """Parses and chunks a single Java document."""
        content_bytes = bytes(document.page_content, "utf8")
        tree = self.parser.parse(content_bytes)
        root_node = tree.root_node

        chunks = []
        imports_and_package = self._get_imports_and_package(root_node)

        # Query to find all top-level and nested type declarations
        query = self.parser.language.query("""
        [
            (class_declaration) @type
            (interface_declaration) @type
            (enum_declaration) @type
        ]
        """)
        type_declaration_nodes = [match[0] for match in query.captures(root_node)]

        for type_node in type_declaration_nodes:
            type_name = self._get_node_name(type_node) or "UnnamedType"
            type_body = next((child for child in type_node.children if child.type.endswith('_body')), None)
            if not class_body:
                continue

            # --- Gather Class-Level Context ---
            class_javadoc = self._get_javadoc(type_node)
            field_nodes = [child for child in type_body.children if child.type == 'field_declaration']
            fields_text = "\n".join(f"    {node.text.decode('utf8')}" for node in field_nodes)
            type_signature = type_node.text.decode('utf8').split('{', 1)[0] + '{'

            # --- Create Chunks for each Method ---
            method_nodes = [child for child in type_body.children if child.type == 'method_declaration']

            for method_node in method_nodes:
                method_name = self._get_node_name(method_node) or "unnamedMethod"
                method_javadoc = self._get_javadoc(method_node)
                method_text = method_node.text.decode('utf8')

                # --- Assemble the Rich Chunk Content ---
                chunk_parts = [imports_and_package]
                if class_javadoc:
                    chunk_parts.append(class_javadoc)

                class_body_parts = []
                if fields_text:
                    class_body_parts.append(fields_text)

                method_and_javadoc = []
                if method_javadoc:
                    method_and_javadoc.append("    " + method_javadoc.replace('\n', '\n    '))
                method_and_javadoc.append("    " + method_text.replace('\n', '\n    '))
                class_body_parts.append("\n".join(method_and_javadoc))

                class_body_content = "\n\n".join(filter(None, class_body_parts))
                chunk_parts.append(f"{type_signature}\n{class_body_content}\n}}")

                chunk_content = "\n\n".join(filter(None, chunk_parts))

                metadata = document.metadata.copy()
                metadata.update({
                    "class_name": type_name,
                    "method_name": method_name,
                    "start_line": method_node.start_point[0] + 1,
                    "end_line": method_node.end_point[0] + 1,
                })

                chunks.append(Document(page_content=chunk_content, metadata=metadata))

        if not chunks and document.page_content.strip():
            # Fallback for files with no recognized methods (e.g., simple classes, scripts)
            chunks.append(document)

        return chunks

    def _get_javadoc(self, node: Node) -> str | None:
        """Extracts the Javadoc comment for a given node if it exists."""
        previous_sibling = node.prev_sibling
        if previous_sibling and previous_sibling.type == 'block_comment':
            comment_text = previous_sibling.text.decode('utf8')
            if comment_text.startswith('/**'):
                return comment_text
        return None

    def _get_imports_and_package(self, root_node: Node) -> str:
        """Extracts package and import statements from the root node."""
        header_nodes = []
        package_nodes = self._find_nodes_by_type(root_node, "package_declaration")
        if package_nodes:
            header_nodes.append(package_nodes[0].text.decode('utf8'))
        
        import_nodes = self._find_nodes_by_type(root_node, "import_declaration")
        header_nodes.extend(node.text.decode('utf8') for node in import_nodes)
            
        return "\n".join(header_nodes)

    def _find_nodes_by_type(self, node: Node, node_type: str) -> List[Node]:
        """Recursively finds all descendant nodes of a given type."""
        query = self.parser.language.query(f"({node_type}) @node")
        return [match[0] for match in query.captures(node)]

    def _get_node_name(self, node: Node) -> str | None:
        """Extracts the identifier/name from a declaration node."""
        name_node = next((child for child in node.children if child.type == 'identifier'), None)
        return name_node.text.decode('utf8') if name_node else None