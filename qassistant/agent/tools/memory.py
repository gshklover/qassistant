"""
Implements long-term memory tool for the agent.
"""
import os

from .knowledgebase import Content, KnowledgeBase


_DEFAULT_LOCATION = os.path.expanduser("~/.qassistant/memory.json")


class Memory(KnowledgeBase):
    """
    Memory tool that allows storing and retrieving persistent information across agent sessions.
    """
    def __init__(self, location: str = _DEFAULT_LOCATION):
        self._location = location

    async def add_content(self, content: str, category: str | None = None) -> None:
        """
        Add a piece of content to memory with optional category.
        """
        document_url = f"memory://{category or 'uncategorized'}"
        await self._add_content_to_document(document_url, content)
