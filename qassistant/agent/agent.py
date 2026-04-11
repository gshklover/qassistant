"""
Agent implementation. Provides a small adapter around GenAI SDK and tool execution.
"""
import asyncio
import copilot
from copilot.generated.session_events import SessionEventType
import dataclasses
import functools
import inspect
import json
import keyring
from openai import AsyncOpenAI
import os
import traceback
from typing import Callable

from .common import AgentEventHandler, BaseAgent
from .tools.pythonshell import PythonShell


# agent defaults:
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TIMEOUT = 300.0  # per-turn timeout in seconds

_MODEL_LIST_CACHE: list[str] | None = None
_MODEL_LIST_LOCK = asyncio.Lock()
_SHARED_CLIENT: copilot.CopilotClient | None = None
_SHARED_CLIENT_LOCK = asyncio.Lock()


async def _get_client() -> copilot.CopilotClient:
    """
    Return a shared Copilot client that is started and ready.
    """
    global _SHARED_CLIENT

    if _SHARED_CLIENT is not None and _SHARED_CLIENT.get_state() in ("connected", "connecting"):
        return _SHARED_CLIENT

    async with _SHARED_CLIENT_LOCK:
        if _SHARED_CLIENT is None:
            _SHARED_CLIENT = copilot.CopilotClient()

        if _SHARED_CLIENT.get_state() not in ("connected", "connecting"):
            await _SHARED_CLIENT.start()

        return _SHARED_CLIENT


async def list_models() -> list[str]:
    """
    List available model ids from the Copilot API.
    """
    global _MODEL_LIST_CACHE

    if _MODEL_LIST_CACHE is not None:
        return list(_MODEL_LIST_CACHE)

    async with _MODEL_LIST_LOCK:
        if _MODEL_LIST_CACHE is not None:
            return list(_MODEL_LIST_CACHE)

        client = await _get_client()
        models = await client.list_models()

        _MODEL_LIST_CACHE = [model.id for model in models]
        return list(_MODEL_LIST_CACHE)


# map event type to handler name & argument mapping
_EVENT_METHOD_BY_TYPE = {
    SessionEventType.TOOL_EXECUTION_START: (
        "on_tool_execution_start", {
            "tool_name": "tool_name",
            "arguments": "arguments",
            "tool_call_id": "tool_call_id",
            "interaction_id": "interaction_id",
        },
    ),
    SessionEventType.TOOL_EXECUTION_PARTIAL_RESULT: (
        "on_tool_execution_partial_result", {
            "tool_call_id": "tool_call_id",
            "partial_output": "partial_output",
        },
    ),
    SessionEventType.TOOL_EXECUTION_PROGRESS: (
        "on_tool_execution_progress", {
            "tool_call_id": "tool_call_id",
            "progress_message": "progress_message",
        },
    ),
    SessionEventType.TOOL_EXECUTION_COMPLETE: (
        "on_tool_execution_complete", {
            "tool_call_id": "tool_call_id",
            "success": "success",
            "result": "result",
            "error": "error",
            "interaction_id": "interaction_id",
        },
    ),
    SessionEventType.ASSISTANT_MESSAGE: (
        "on_assistant_message", {
            "content": "content",
            "message_id": "message_id",
            "interaction_id": "interaction_id",
            "reasoning_text": "reasoning_text",
            "tool_requests": "tool_requests",
        },
    ),
    SessionEventType.ASSISTANT_MESSAGE_DELTA: (
        "on_assistant_message_delta", {
            "delta_content": "delta_content",
            "message_id": "message_id",
            "interaction_id": "interaction_id",
        },
    ),
    SessionEventType.ASSISTANT_REASONING: (
        "on_assistant_reasoning", {
            "content": "content",
            "reasoning_id": "reasoning_id",
            "interaction_id": "interaction_id",
            "reasoning_text": "reasoning_text",
        },
    ),
    SessionEventType.ASSISTANT_REASONING_DELTA: (
        "on_assistant_reasoning_delta", {
            "delta_content": "delta_content",
            "reasoning_id": "reasoning_id",
            "interaction_id": "interaction_id",
        },
    ),
    SessionEventType.ASSISTANT_STREAMING_DELTA: (
        "on_assistant_streaming_delta", {
            "total_response_size_bytes": "total_response_size_bytes",
            "interaction_id": "interaction_id",
        },
    ),
    SessionEventType.ASSISTANT_TURN_END: (
        "on_assistant_turn_end", {
            "turn_id": "turn_id",
        },
    ),
    SessionEventType.ASSISTANT_TURN_START: (
        "on_assistant_turn_start", {
            "turn_id": "turn_id",
            "interaction_id": "interaction_id",
        },
    ),
    SessionEventType.SESSION_IDLE: (
        "on_session_idle", {
            "background_tasks": "background_tasks",
        },
    ),
    SessionEventType.SESSION_TASK_COMPLETE: (
        "on_session_task_complete", {
            "summary": "summary",
        },
    ),
    SessionEventType.SESSION_ERROR: (
        "on_session_error", {
            "error_type": "error_type",
            "message": "message",
            "error": "error",
            "status_code": "status_code",
            "url": "url",
        },
    ),
    SessionEventType.USER_MESSAGE: (
        "on_user_message", {
            "content": "content",
            "interaction_id": "interaction_id",
            "attachments": "attachments",
        },
    ),
}


def as_tool(func: Callable, **kwargs) -> Callable:
    """
    Decorator to mark a function as a tool for the agent.
    """
    sig = inspect.signature(func)

    ArgType = dataclasses.make_dataclass(
        cls_name="Args",
        fields=[(param.name, param.annotation) for param in sig.parameters.values()],
    )

    @functools.wraps(func)
    def wrapper(arg):
        return func(**arg.to_dict())

    wrapper.__name__ = kwargs.get("name", func.__name__)
    wrapper.__annotations__.setdefault("return", func.__annotations__.get("return", None))
    wrapper.__annotations__["arg"] = ArgType

    return copilot.define_tool(**kwargs)(wrapper)


class Agent(BaseAgent):
    """
    Base agent class implementation using copilot SDK.
    Wraps around a single session with start/stop/reset and tool integration.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        tools: list[str | Callable] | None = None,
        event_handlers: list[AgentEventHandler] | None = None,
    ):
        """
        Initialize the agent with the specified model, tools, and event handlers.
        """
        self._model = model
        self._shell = PythonShell()  # shared python shell instance for tool execution
        self._session = None
        self._workspace_path = os.getcwd()
        self._tools = [
            # execution shell:
            as_tool(self._shell.execute, name='python_shell_execute'),
            as_tool(self._shell.get_variables, name='python_shell_get_variables'),
            *[as_tool(tool) for tool in (tools or ())],
        ]
        self._event_handlers = list(event_handlers or ())
        self._config = dict(
            on_permission_request=copilot.session.PermissionHandler.approve_all,
            model=model,
            tools=self._tools,
            streaming=True,
        )

    @property
    def model(self) -> str:
        """
        Get the current model name.
        """
        return self._model

    @model.setter
    def model(self, new_model: str):
        """
        Update the model name in the agent configuration. Note that this will not affect an already running session.
        """
        self._model = new_model
        self._config['model'] = new_model

    @property
    def running(self) -> bool:
        """
        Check if the agent session is currently active.
        """
        return self._session is not None

    @property
    def workspace_path(self) -> str:
        """
        Return current work area used by the session shell.
        """
        return self._workspace_path

    async def start(self, session_id: str = None):
        """
        Start the agent by initializing the Copilot client.

        :param session_id: The session ID to resume. If not specified, a new session is created.
        """
        if self._session is not None:
            raise RuntimeError("Agent is already running")

        client = await _get_client()
        if session_id is not None:
            self._session = await client.resume_session(working_directory=self._workspace_path, session_id=session_id, **self._config)
        else:
            self._session = await client.create_session(working_directory=self._workspace_path, **self._config)
        self._session.on(self._on_event)

    async def stop(self):
        """
        Stop the agent and clean up resources.
        """
        if self._session:
            await self._session.disconnect()
            self._session = None

    async def reset(self):
        """
        Reset the agent session.
        """
        if self._session:
            await self._session.disconnect()

        client = await _get_client()
        self._session = await client.create_session(**self._config)
        self._session.on(self._on_event)

    async def send(self, message: str) -> copilot.session.SessionEvent:
        """
        Send a message to the agent and return the response.
        """
        if not self._session:
            raise RuntimeError("Agent is not running. Call start() first.")

        res = await self._session.send_and_wait(message, timeout=DEFAULT_TIMEOUT)
        return res

    async def submit(self, message: str) -> str:
        """
        Submit a message to the agent and return immediately with the message ID.
        Response chunks and completion are delivered through registered event handlers.
        """
        if not self._session:
            raise RuntimeError("Agent is not running. Call start() first.")

        return await self._session.send(message)

    async def abort(self):
        """
        Abort the current agent turn.
        """
        if self._session:
            await self._session.abort()

    async def set_workspace(self, path: str):
        """
        Set the workspace to the specified path.
        """
        self._workspace_path = path
        if self._session is not None:
            session_id = self._session.session_id
            await self.stop()
            await self.start(session_id=session_id)

    async def __aenter__(self):
        """
        Start running the agent.
        Example:
            ```
            async with Agent() as agent:
                response = await agent.send("Hello")
            ```
        """
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Stop running the agent.
        """
        await self.stop()

    async def _handle_event(self, event: copilot.session.SessionEvent):
        """
        Route an event to all configured handlers using event-type-specific hooks.
        """
        event_info = _EVENT_METHOD_BY_TYPE.get(event.type, None)
        if event_info is None:
            for event_handler in self._event_handlers:
                await event_handler.on_unknown_event(event.type.value, event)
            return

        method_name, attributes = event_info
        args = {
            arg_name: getattr(event.data, attr_name, None)
            for arg_name, attr_name in attributes.items()
        }

        for event_handler in self._event_handlers:
            handler_method = getattr(event_handler, method_name, None)
            if handler_method is not None:
                try:
                    await handler_method(**args)
                except Exception:
                    traceback.print_exc()

    def _on_event(self, event: copilot.session.SessionEvent):
        """
        Session callback that logs non-streaming events and dispatches to handlers.
        """
        # if event.type.value not in (
        #     "assistant.streaming_delta",
        #     # "assistant.message_delta",
        #     # "assistant.reasoning_delta",
        # ):
        #     data = {k: v for k, v in event.data.to_dict().items() if v is not None}
        #     print(f"\n{event.type.value} ({event.id}, parent={event.parent_id}): {data}")

        if not self._event_handlers:
            return

        asyncio.create_task(self._handle_event(event))


class Model:
    """
    Wraps around github models API to provide chat completions and text embedding.
    """
    def __init__(self, chat_model: str = DEFAULT_MODEL, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        """
        Initialize the client with authentication and model configuration.
        """
        config_path = os.path.expanduser('~/.copilot/config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Failed to find ~/.copilot/config.json")

        with open(config_path) as stream:
            config = json.load(stream)
            host = config['logged_in_users'][0]['host']
            user_name = config['logged_in_users'][0]['login']

        # access Windows Credential Manager to get copilot-cli token:
        # "keyring" attempts to decode it using utf-16 by default, we reverse the encoding
        token_utf16 = keyring.get_password(f'copilot-cli/{host}:{user_name}', f'{host}:{user_name}')
        if not token_utf16:
            raise ValueError(f'Failed to get token for {host}:{user_name}')
        token = token_utf16.encode('utf-16')[2:].decode('ascii')

        self._client = AsyncOpenAI(
            base_url='https://models.github.ai/inference',
            api_key=token,
        )
        self._chat_model = f'openai/{chat_model}'
        self._embedding_model = f'openai/{embedding_model}'

    # NOT SUPPORTED BY GITHUB MODELS
    # async def complete(self, prompt: str) -> str:
    #     """
    #     Get a completion for the input prompt.
    #     Not supported by newer models.
    #     """
    #     response = await self._client.completions.create(prompt=prompt, model=self._chat_model, echo=False)
    #     return response.choices[0].text

    async def chat(self, messages: list[dict]) -> str:
        """
        Send a chat message to the model and return the response.
        """
        response = await self._client.chat.completions.create(messages=messages, model=self._chat_model)
        return response.choices[0].message.content

    async def embed(self, text: str) -> list[float]:
        """
        Get embedding for the input text.
        """
        response = await self._client.embeddings.create(input=text, model=self._embedding_model)
        return response.data[0].embedding
