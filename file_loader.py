import os
from typing import List, Dict, Iterator
class FileLoader:
    """
    A class to load files from a specified directory with given extensions.
    """
    def __init__(self, directory_path: str, allowed_extensions: List[str]):
        """
        Initializes the FileLoader.

        Args:
            directory_path (str): The path to the directory to scan for files.
            allowed_extensions (List[str]): A list of file extensions to load (e.g., ['.txt', '.py']).
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")
        self.directory_path = directory_path
        self.allowed_extensions = allowed_extensions

    def load_files(self) -> Iterator[Dict[str, str]]:
        """
        Walks through the directory and yields the content of files with allowed extensions.

        Yields:
            Iterator[Dict[str, str]]: An iterator of dictionaries, where each dictionary
                                      contains the 'file_path' and 'content' of a file.
        """
        for root, _, files in os.walk(self.directory_path):
            for file in files:
                if any(file.endswith(ext) for ext in self.allowed_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            yield {"file_path": file_path, "content": f.read()}
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")