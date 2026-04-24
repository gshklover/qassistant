"""
Common definitions for the agent package.

Defines the BaseSession interface used by concrete session implementations.
"""
from abc import ABC, abstractmethod
import dataclasses
import enum
import pandas
from typing import Any


class AgentEventHandler:
    """
    Lifecycle handler interface for AgentAPI session list events.
    """

    async def on_session_created(self, session_id: str):
        """
        Called when a session is created.
        """
        return

    async def on_session_deleted(self, session_id: str):
        """
        Called when a session is deleted.
        """
        return

    async def on_session_updated(self, session_id: str):
        """
        Called when a session is updated.
        """
        return


class SessionEventHandler(ABC):
    """
    Abstract base class for handling agent events such as tool calls, permission requests, and errors.
    """

    async def on_tool_execution_start(
        self,
        tool_name: str | None,
        arguments: Any,
        tool_call_id: str | None,
        interaction_id: str | None,
    ):
        """
        Emitted when the agent starts executing a tool.
        """
        return

    async def on_tool_execution_partial_result(
        self,
        tool_call_id: str | None,
        partial_output: str | None,
    ):
        """
        Emitted when a running tool produces a streaming partial result.
        """
        return

    async def on_tool_execution_progress(
        self,
        tool_call_id: str | None,
        progress_message: str | None,
    ):
        """
        Emitted when a running tool reports progress.
        """
        return

    async def on_tool_execution_complete(
        self,
        tool_call_id: str | None,
        success: bool | None,
        result: Any,
        error: Any,
        interaction_id: str | None,
    ):
        """
        Emitted when the agent completes executing a tool.
        """
        return

    async def on_assistant_message_delta(
        self,
        delta_content: str | None,
        message_id: str | None,
        interaction_id: str | None,
    ):
        """
        Streaming delta for assistant message.
        """
        return

    async def on_assistant_message(
        self,
        content: str | None,
        message_id: str | None,
        interaction_id: str | None,
        reasoning_text: str | None,
        tool_requests: list[Any] | None,
    ):
        """
        Handles response from the assistant emitted at the end of the assistant turn.
        """
        return

    async def on_assistant_reasoning(
        self,
        content: str | None,
        reasoning_id: str | None,
        interaction_id: str | None,
        reasoning_text: str | None,
    ):
        """
        Emitted for a completed assistant reasoning block.
        """
        return

    async def on_assistant_reasoning_delta(
        self,
        delta_content: str | None,
        reasoning_id: str | None,
        interaction_id: str | None,
    ):
        """
        Handle `assistant.reasoning_delta`.
        """
        return

    async def on_assistant_streaming_delta(
        self,
        total_response_size_bytes: float | None,
        interaction_id: str | None,
    ):
        """
        Handle `assistant.streaming_delta`.
        """
        return

    async def on_assistant_turn_end(self, turn_id: str | None):
        """
        Handle end of assistant turn.
        """
        return

    async def on_assistant_turn_start(self, turn_id: str | None, interaction_id: str | None):
        """
        Single interaction may contain multiple assistant reasoning & tool execution turns.
        """
        return

    async def on_session_idle(self, background_tasks: Any):
        """
        Emitted when the session agent becomes idle after handling user message.
        """
        return

    async def on_session_task_complete(self, summary: str | None):
        """
        Handle `session.task_complete`.
        """
        return

    async def on_session_error(
        self,
        error_type: str | None,
        message: str | None,
        error: Any,
        status_code: int | None,
        url: str | None,
    ):
        """
        Handle `session.error`.
        """
        return

    async def on_session_usage(self, usage_percentage: float):
        """
        Emitted when session usage information is received.
        The usage_percentage is a float between 0.0 and 100.0 indicating
        the context window utilization (current_tokens / token_limit).
        """
        return

    async def on_unknown_event(self, event_type: str, event: Any):
        """
        Handle any event type that has no dedicated hook.
        """
        return

    async def on_user_message(
        self,
        content: str | None,
        interaction_id: str | None,
        attachments: list[Any] | None,
    ):
        """
        Emitted when user sends a new message to the agent.
        """
        return


class BaseSession(ABC):
    """
    Abstract base class describing the public agent interface.
    Implementations must provide async lifecycle methods and a message API.
    """
    @property
    def model(self) -> str:
        """
        Return the name of the underlying model used by the agent.
        """
        return ''

    @property
    def workspace_path(self) -> str:
        """
        Return agent workspace path
        """
        return ''

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

    @abstractmethod
    async def submit(self, message: str) -> Any:
        """
        Submit a message to the agent and return immediately while streaming events continue asynchronously.
        """


# Backward-compatible alias.
BaseAgent = BaseSession


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
