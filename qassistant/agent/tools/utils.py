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
