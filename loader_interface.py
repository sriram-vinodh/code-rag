from abc import ABC, abstractmethod
from typing import Dict, Iterator
class LoaderInterface(ABC):
    """
    An interface for loading data from a source.
    """
    @abstractmethod
    def load(self) -> Iterator[Dict[str, str]]:
        """
        Loads data and yields it as an iterator of dictionaries.
        Each dictionary should contain 'file_path' (or a unique identifier) and 'content'.
        """