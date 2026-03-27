"""
Common definitions for the agent package.

Defines the BaseAgent interface used by concrete Agent implementations.
"""
from abc import ABC, abstractmethod
import dataclasses
import pandas
from typing import Any


class BaseAgent(ABC):
    """
    Abstract base class describing the public agent interface.
    Implementations must provide async lifecycle methods and a message API.
    """

    @abstractmethod
    async def start(self):
        """
        Start the agent and allocate any resources required.
        """

    @abstractmethod
    async def stop(self):
        """
        Stop the agent and release resources.
        """

    @abstractmethod
    async def reset(self):
        """
        Reset the agent session to a clean state.
        """

    @abstractmethod
    async def send(self, message: str) -> Any:
        """
        Send a message to the agent and return a response object.
        """


@dataclasses.dataclass(slots=True)
class Content:
    """
    Base content unit for a rich message.
    """
    pass


@dataclasses.dataclass(slots=True)
class TextContent(Content):
    """
    Simple text content unit.
    """
    text: str
    format: str = "plain"  # e.g. "plain", "markdown", "html"


@dataclasses.dataclass(slots=True)
class CodeContent(Content):
    """
    Code snippet content unit.
    """
    code: str
    language: str | None = None  # optional language hint for syntax highlighting


@dataclasses.dataclass(slots=True)
class ImageContent(Content):
    """
    Image content unit.
    """
    image_data: bytes  # raw image data
    format: str = "png"  # e.g. "png", "jpeg"
    alt_text: str | None = None  # optional alt text for accessibility


@dataclasses.dataclass(slots=True)
class TableContent(Content):
    """
    Table content unit for structured data.
    """
    table: pandas.DataFrame


@dataclasses.dataclass(slots=True)
class SectionContent(Content):
    """
    Section content unit for grouping related content together.
    """
    title: str | None = None
    contents: list[Content] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class ProgressContent(Content):
    """
    Progress update content unit for streaming responses.
    """
    progress: float  # value between 0.0 and 1.0 indicating completion percentage
    message: str | None = None  # optional message describing the current progress


class Role(str):
    """
    Enum-like class for message roles.
    """
    USER = 'user'
    ASSISTANT = 'assistant'
    SYSTEM = 'system'
    TOOL = 'tool'


@dataclasses.dataclass(slots=True)
class Message:
    """
    Standardized message format for agent communication.
    """
    role: str = Role.USER  # e.g. 'user', 'assistant', 'system'
    content: list[Content] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    complete: bool = True # indicates if this message is a complete response or part of a stream
