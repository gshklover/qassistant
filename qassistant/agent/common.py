"""
Common definitions for the agent package.

Defines the BaseAgent interface used by concrete Agent implementations.
"""
from abc import ABC, abstractmethod
import dataclasses
import pandas
from typing import Any, Callable


class AyncSignal:
    """
    Basic async signal implementation for event handling and change notification.
    """
    __slots__ = ['_subscribers']

    def __init__(self):
        self._subscribers = []

    def connect(self, callback: Callable):
        """
        Connect a new subscriber callback to the signal.
        """
        self._subscribers.append(callback)

    def disconnect(self, callback: Callable):
        """
        Disconnect a subscriber callback from the signal.
        """
        self._subscribers.remove(callback)

    async def emit(self, *args, **kwargs):
        """
        Emit the signal and call all subscriber callbacks with the provided arguments.
        """
        for callback in self._subscribers:
            await callback(*args, **kwargs)


class AgentEventHandler(ABC):
    """
    Abstract base class for handling agent events such as tool calls, permission requests, and errors.
    """

    async def on_tool_execution_start(self, tool_name: str, arguments: dict, tool_call_id: str):
        """
        Emitted when the agent starts executing a tool.
        """
        pass

    async def on_tool_execution_complete(self, tool_name: str, tool_call_id: str, success: bool, result: Any, interaction_id: str):
        """
        Emitted when the agent completes executing a tool.
        """
        pass

    async def on_assistant_message_delta(self):
        """
        Streaming delta for assistant message.
        """
        pass

    async def on_assistant_message(self, content: str, interaction_id: str, reasoning_text: str, tool_requests: list):
        """
        Handles response from the assistant emitted at the end of the assistant turn.
        """
        pass

    async def on_assistant_reasoning_delta(self):
        """
        Handle `assistant.reasoning_delta`.
        """
        pass

    async def on_assistant_streaming_delta(self):
        """
        Handle `assistant.streaming_delta`.
        """
        pass

    async def on_assistant_turn_end(self, turn_id: str, interaction_id: str):
        """
        Handle end of assistant turn.
        """
        pass

    async def on_assistant_turn_start(self, turn_id: str, interaction_id: str):
        """
        Single interaction may contain multiple assistant reasoning & tool execution turns.
        """
        pass

    async def on_session_idle(self):
        """
        Emitted when the session agent becomes idle after handling user message.
        """
        pass

    async def on_session_task_complete(self):
        """
        Handle `session.task_complete`.
        """
        pass

    async def on_session_error(self):
        """
        Handle `session.error`.
        """
        pass

    async def on_unknown_event(self):
        """
        Handle any event type that has no dedicated hook.
        """
        pass

    async def on_user_message(self, content: str, interaction_id: str):
        """
        Emitted when user sends a new message to the agent.
        """
        pass


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
