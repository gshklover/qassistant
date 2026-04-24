"""
Common definitions for the agent package.
"""
from abc import ABC, abstractmethod
import dataclasses
import enum
import pandas
from typing import Any


@dataclasses.dataclass(slots=True)
class Content:
    """
    Base content unit for a rich message.
    """
    metadata: dict[str, Any] = None


@dataclasses.dataclass(slots=True)
class TextContent(Content):
    """
    Simple text content unit.
    """
    text: str = ''
    format: str = "markdown"  # e.g. "plain", "markdown", "html"


@dataclasses.dataclass(slots=True)
class CodeContent(Content):
    """
    Code snippet content unit.
    """
    code: str = ''
    language: str | None = None  # optional language hint for syntax highlighting


@dataclasses.dataclass(slots=True)
class ImageContent(Content):
    """
    Image content unit.
    """
    image_data: bytes = dataclasses.field(default_factory=lambda: b'')  # raw image data
    format: str = "png"  # e.g. "png", "jpeg"
    alt_text: str | None = None  # optional alt text for accessibility


@dataclasses.dataclass(slots=True)
class TableContent(Content):
    """
    Table content unit for structured data.
    """
    table: pandas.DataFrame = None


@dataclasses.dataclass(slots=True)
class SectionContent(Content):
    """
    Section content unit for grouping related content together.
    """
    title: str | None = None
    contents: list[Content] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class ToolCallContent(Content):
    """
    Tool call content unit for rendering tool execution details.
    """
    tool_name: str = ""
    arguments: str = ""
    result: str = ""


class Role(str):
    """
    Enum-like class for message roles.
    """
    USER = 'user'
    ASSISTANT = 'assistant'
    SYSTEM = 'system'
    TOOL = 'tool'


class MessageState(enum.StrEnum):
    """
    Enum-line class for message states
    """
    PROCESSING = 'processing'  # local processing
    THINKING = 'thinking'      # set while calling LLM model
    EXECUTING ='executing'     # set while executing a tool
    COMPLETE = 'complete'      # final state after processing is done
    FAILED = 'failed'          # set when an error occurred while handling a message


@dataclasses.dataclass(slots=True)
class Message:
    """
    Standardized message format for agent communication.
    """
    role: str = Role.USER  # e.g. 'user', 'assistant', 'system'
    content: list[Content] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    state: MessageState = MessageState.COMPLETE  # only makes sense for assistant messages

    def append(self, content: Content):
        """
        Append new content to the message.
        """
        self.content.append(content)
