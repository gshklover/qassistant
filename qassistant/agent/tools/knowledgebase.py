"""
Simple knowledge base tool implementation with embedding-based retrieval.
"""
import dataclasses


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


class KnowledgeBase:
    """
    A simple tool that allows the agent to store and retrieve information in a key-value store.
    """
    def __init__(self):
        self._store = {}

    def query(self, text: str, top_k: int = 1) -> list[Content]:
        """
        Retrieve a value from the knowledge base by key.
        """
        pass
