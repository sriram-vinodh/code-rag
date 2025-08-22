# Implementation migrated from filesystem_loader.py
import os
from typing import List, Dict, Iterator
from .loader_interface import LoaderInterface

class FileSystemLoader(LoaderInterface):
    """
    A class to load files from a specified directory on the local filesystem.
    """
    def __init__(self, directory_path: str, allowed_extensions: List[str]):
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")
        self.directory_path = directory_path
        self.allowed_extensions = allowed_extensions

    def load(self) -> Iterator[Dict[str, str]]:
        for root, _, files in os.walk(self.directory_path):
            for file in files:
                if any(file.endswith(ext) for ext in self.allowed_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            yield {"file_path": file_path, "content": f.read()}
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
# Moved from filesystem_loader.py
# FileSystemLoader implementation
