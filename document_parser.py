from typing import List, Dict, Iterator
from langchain_core.documents import Document
from parser_interface import ParserInterface

try:
    from bs4 import BeautifulSoup
    BS4_IS_AVAILABLE = True
except ImportError:
    BS4_IS_AVAILABLE = False

class DefaultDocumentParser(ParserInterface):
    """
    A default implementation to parse raw file content into structured Document objects.
    Handles HTML files by extracting text content if beautifulsoup4 is installed.
    """

    def _parse_html(self, content: str, file_path: str) -> str:
        """Parses HTML content to extract text."""
        if not BS4_IS_AVAILABLE:
            print(f"Warning: HTML file '{file_path}' detected, but `beautifulsoup4` is not installed. "
                  "Content will be treated as plain text. For better results, please `pip install beautifulsoup4`.")
            return content
        
        # Use BeautifulSoup to parse the HTML and extract text
        soup = BeautifulSoup(content, 'html.parser')
        return soup.get_text(separator=" ", strip=True)

    def parse_documents(self, file_data_iterator: Iterator[Dict[str, str]]) -> List[Document]:
        """
        Parses an iterator of file data into a list of Document objects.

        Args:
            file_data_iterator (Iterator[Dict[str, str]]): An iterator yielding dictionaries
                                                            with 'file_path' and 'content'.

        Returns:
            List[Document]: A list of LangChain Document objects.
        """
        documents = []
        for file_data in file_data_iterator:
            file_path = file_data['file_path']
            content = file_data['content']
            metadata = {"source": file_path}

            if file_path.lower().endswith(('.html', '.htm')):
                page_content = self._parse_html(content, file_path)
            elif file_path.lower().endswith('.java'):
                page_content = content
                metadata["language"] = "java"
            else:
                page_content = content
            
            documents.append(Document(page_content=page_content, metadata=metadata))
        return documents