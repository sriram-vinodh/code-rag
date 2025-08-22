
from typing import List
from langchain_core.documents import Document
from .chunker_interface import ChunkerInterface

try:
    import javalang
except ImportError:
    javalang = None

class JavaCodeChunker(ChunkerInterface):
    """
    AST-based chunker for Java code using javalang. Extracts methods with class context.
    """
    def __init__(self):
        if javalang is None:
            raise ImportError("The 'javalang' package is required for JavaCodeChunker. Install with 'pip install javalang'.")

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits Java source code into chunks at the method level, including class and field context, using javalang's AST.
        """
        all_chunks = []
        for doc in documents:
            code = doc.page_content
            try:
                tree = javalang.parse.parse(code)
            except Exception:
                all_chunks.append(doc)
                continue

            # Map line numbers to code for fast lookup
            lines = code.splitlines()

            for _, class_node in filter(lambda n: isinstance(n[1], javalang.tree.ClassDeclaration), tree.filter(javalang.tree.ClassDeclaration)):
                class_name = class_node.name
                class_fields = [decl.declarators[0].name for decl in class_node.fields] if hasattr(class_node, 'fields') else []
                class_javadoc = getattr(class_node, 'documentation', None)

                for method in class_node.methods:
                    # Use AST to get method signature and body
                    method_lines = self._extract_method_lines(method, lines)
                    chunk_code = '\n'.join(method_lines) if method_lines else str(method)
                    metadata = doc.metadata.copy()
                    metadata.update({
                        "java_class": class_name,
                        "java_method": method.name,
                        "class_fields": class_fields,
                    })
                    if class_javadoc:
                        metadata["class_javadoc"] = class_javadoc
                    all_chunks.append(Document(page_content=chunk_code, metadata=metadata))
        return all_chunks

    def _extract_method_lines(self, method_node, lines):
        """
        Extracts the lines of code for a method using its position and braces, using the AST for structure.
        """
        if hasattr(method_node, 'position') and method_node.position:
            start = method_node.position.line - 1
            brace_count = 0
            in_method = False
            method_lines = []
            for i in range(start, len(lines)):
                line = lines[i]
                if '{' in line:
                    in_method = True
                if in_method:
                    method_lines.append(line)
                    brace_count += line.count('{')
                    brace_count -= line.count('}')
                    if brace_count == 0:
                        break
            return method_lines
        return None

    def _extract_method_code(self, code: str, method_node) -> str:
        # javalang does not provide source ranges, so fallback to string extraction
        # This is a best-effort approach
        if hasattr(method_node, 'position') and method_node.position:
            lines = code.splitlines()
            start = method_node.position.line - 1
            # Try to find the end of the method (naive)
            end = start + 1
            brace_count = 0
            in_method = False
            for i in range(start, len(lines)):
                line = lines[i]
                brace_count += line.count('{')
                brace_count -= line.count('}')
                if '{' in line:
                    in_method = True
                if in_method and brace_count == 0:
                    end = i + 1
                    break
            return '\n'.join(lines[start:end])
        return str(method_node)
