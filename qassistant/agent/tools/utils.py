"""
Common utilities for tools
"""
import importlib.metadata


DEFAULT_LENGTH = 1024


def truncate(text: str, max_length: int = DEFAULT_LENGTH) -> str:
    """
    Truncate the input text to the specified maximum length, adding an ellipsis if truncation occurs.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + '...'


def get_mcp_servers():
    """
    Get MCP servers registered using mcp-servers entry points
    """
    entries = importlib.metadata.entry_points(group='mcp-servers')
    print(entries)
