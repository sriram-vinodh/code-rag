from dataclasses import dataclass, field
from typing import Iterator, List, Optional
import logging
import javalang.tree
import javalang.tokenizer
import javalang.parse
logger = logging.getLogger(__name__)
@dataclass
class MethodDeclaration:
    """Represents a parsed Java method."""
    name: str
    node: javalang.tree.Node
    javadoc: Optional[str] = None
    content: str = ""

@dataclass
class TypeDeclaration:
    """Represents a parsed Java type (class, interface, enum)."""
    name: str
    node: javalang.tree.Node
    signature: str
    javadoc: Optional[str] = None
    fields: List[str] = field(default_factory=list)
    methods: List[MethodDeclaration] = field(default_factory=list)

class JavaAstParser:
    """A parser for Java code that uses tree-sitter to extract semantic structures."""

    def __init__(self, code_content: str):
        """
        Initializes the parser and parses the given Java code content using javalang.
        Args:
            code_content (str): The Java source code to parse.
        """
        self.code_content = code_content
        try:
            # javalang.parse.parse expects a string of code
            self.tree: javalang.tree.CompilationUnit = javalang.parse.parse(code_content)
        except javalang.tokenizer.LexerError as e:
            logger.error(f"Lexer error parsing Java code: {e}", exc_info=True)
            raise ValueError(f"Syntax error in Java code: {e}") from e
        except javalang.parser.JavaSyntaxError as e:
            logger.error(f"Syntax error parsing Java code: {e}", exc_info=True)
            raise ValueError(f"Syntax error in Java code: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error parsing Java code with javalang: {e}", exc_info=True)
            raise RuntimeError(f"Failed to parse Java code with javalang: {e}") from e
        
        # For compatibility with previous structure, we can simulate root_node if needed,
        # but javalang's CompilationUnit is the direct AST root.
        self.root_node = self.tree # Use the CompilationUnit as the root for traversal

    def get_imports_and_package(self) -> str:
        """Extracts package and import statements from the root node."""
        header_nodes_text = []
        if self.root_node.package:
            header_nodes_text.append(f"package {self.root_node.package.name};")
        for imp in self.root_node.imports:
            header_nodes_text.append(f"import {imp.path}{'.*' if imp.wildcard else ''};")
        return "\n".join(header_nodes_text)

    def get_type_declarations(self) -> Iterator[TypeDeclaration]:
        """
        Finds all type declarations (class, interface, enum) in the code
        and yields them as structured TypeDeclaration objects.
        """
        for type_decl in self.root_node.types:
            yield self._process_single_type_node(type_decl)

    def _process_single_type_node(self, type_node: javalang.tree.Node) -> TypeDeclaration:
        """
        Processes a single javalang type declaration node to extract its details.
        Note: javalang nodes don't directly store the full text of a method/field.
        We reconstruct it using start/end positions from the original code.
        """
        type_name = type_node.name
        type_javadoc = type_node.documentation
        
        # Reconstruct signature (basic attempt, might need refinement for complex cases)
        # javalang nodes have .position (line, col) but not .end_position directly for the whole node.
        # We'll use the start of the node up to the first '{' for a basic signature.
        signature_start_line = type_node.position[0] - 1
        signature_start_col = type_node.position[1]
        
        # Find the start of the body to get the signature
        type_content_lines = self.code_content.splitlines()
        signature_line = type_content_lines[signature_start_line]
        signature_text = signature_line[signature_start_col:]
        
        # Find the first '{' to delimit the signature
        brace_idx = signature_text.find('{')
        if brace_idx != -1:
            signature = signature_text[:brace_idx+1] # Include the brace
        else:
            signature = signature_text.strip() + " {" # Fallback if no brace found on same line

        methods = []
        fields = []

        for member in type_node.body:
            if isinstance(member, javalang.tree.MethodDeclaration):
                method_name = member.name
                method_javadoc = member.documentation
                method_content = self._get_node_content(member)
                methods.append(MethodDeclaration(name=method_name, node=member, javadoc=method_javadoc, content=method_content))
            elif isinstance(member, javalang.tree.FieldDeclaration):
                # For fields, we just get the full declaration text
                fields.append(self._get_node_content(member))

        return TypeDeclaration(name=type_name, node=type_node, signature=signature, javadoc=type_javadoc, fields=fields, methods=methods)

    def _get_node_content(self, node: javalang.tree.Node) -> str:
        """
        Extracts the raw code content for a given javalang AST node.
        This is a common workaround as javalang nodes don't store raw text.
        """
        # javalang nodes have a .position attribute (line, column)
        # We need to find the end position by looking at the next sibling or parent's end
        # This is a simplified approach; a more robust one would involve tokenizing
        # and finding the exact start/end tokens.
        
        start_line, start_col = node.position
        
        # Find the end of the node's content by iterating through tokens
        # This is a heuristic and might not be perfect for all cases.
        tokens = list(javalang.tokenizer.tokenize(self.code_content))
        
        node_tokens = []
        found_start = False
        
        for token in tokens:
            if token.position == (start_line, start_col) and not found_start:
                found_start = True
            
            if found_start:
                node_tokens.append(token.value)
                # Heuristic to find the end of a method/field:
                # Look for the closing brace '}' for methods, or semicolon ';' for fields.
                if (isinstance(node, javalang.tree.MethodDeclaration) and token.value == '}') or \
                   (isinstance(node, javalang.tree.FieldDeclaration) and token.value == ';'):
                    break
        
        return "".join(node_tokens)