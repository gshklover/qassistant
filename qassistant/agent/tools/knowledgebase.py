"""
Simple knowledge base tool implementation with embedding-based retrieval.
"""
import dataclasses
from typing import Callable


@dataclasses.dataclass(slots=True)
class Content:
    """
    Standardized content format for knowledge base entries.
    """
    content: str
    
    # common metadata fields:
    document_url: str | None = None  # original source URL for the content
    chunk_index: int | None = None   # index of this chunk within the original document
    total_chunks: int | None = None  # total number of chunks in the original document
    
    # custom metadata fields:
    metadata: dict = None


ROOT = '/'


class KnowledgeBase:
    """
    A simple tool that allows the agent to store and retrieve information in a key-value store.
    Uses specified async embedding function to embed available text.
    """
    def __init__(self, embedding: Callable[[str], list[float]]):
        self._store = {}
        self._embedder = embedding

    def search(self, text: str, top_k: int = 10) -> list[Content]:
        """
        Retrieve a value from the knowledge base by specified query text.
        """
        pass

    def add_file(self, path: str, destination: str = '/') -> None:
        """
        Add a new entry to the knowledge base with the specified key and value.
        """
        pass

    def add_directory(self, path: str, destination: str = '/') -> None:
        """
        Add multiple entries to the knowledge base from a directory.
        """
        pass

    def list_documents(path: str = '/') -> list[dict]:
        """
        List documents / folders under specified path.
        """
        pass

    def read(self, path: str, start: int = 0, end: int = None) -> str:
        """
        Read content of the specified document path. Supports optional chunking via start/end parameters.
        """
        pass
