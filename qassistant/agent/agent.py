"""
Agent implementation. Provides a small adapter around GenAI SDK and tool execution.
"""
import asyncio
import copilot
from copilot.generated.session_events import SessionEventType
from copilot.generated.rpc import SessionAgentSelectParams
import dataclasses
import functools
import httpx
import inspect
import json
import keyring
import yaml
from openai import AsyncOpenAI
import os
import pathlib
import traceback
from typing import Any, Callable, Sequence

from .common import AgentEventHandler, SessionEventHandler, BaseSession
from .tools.pythonshell import PythonShell


# agent defaults:
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
DEFAULT_TIMEOUT = 300.0  # per-turn timeout in seconds


class AgentAPI:
    """
    Thin wrapper around a shared CopilotClient instance.
    """

    def __init__(self, event_handlers: list[AgentEventHandler] | None = None):
        """
        Initialize the client
        """
        self._models = None
        self._event_handlers = list(event_handlers or ())
        self._client = copilot.CopilotClient(auto_start=True)
        self._client.on("session.created", lambda event: self._dispatch_event("on_session_created", event))
        self._client.on("session.deleted", lambda event: self._dispatch_event("on_session_deleted", event))
        self._client.on("session.updated", lambda event: self._dispatch_event("on_session_updated", event))

    async def list_models(self) -> list[copilot.client.ModelInfo]:
        """
        List available models from the Copilot API.
        Returns list of copilot.client.ModelInfo objects with 'id', 'name', 'summary', and 'capabilities' fields.
        """
        if self._models is None:
            self._models = await self._client.list_models()
        return self._models

    async def list_sessions(self) -> list[copilot.client.SessionMetadata]:
        """
        List existing sessions from the Copilot API.
        """
        return await self._client.list_sessions()

    async def create_session(
        self,
        *,
        working_directory: str = "",
        on_permission_request: Any = None,
        model: str = DEFAULT_MODEL,
        tools: list[Any] | None = None,
        streaming: bool = True,
        custom_agents: list[Any] | None = None,
        agent: str | None = None,
    ):
        """
        Create a Copilot session.
        """
        return await self._client.create_session(
            working_directory=working_directory,
            on_permission_request=on_permission_request,
            model=model,
            tools=tools,
            streaming=streaming,
            custom_agents=custom_agents,
            agent=agent,
        )

    async def resume_session(
        self,
        *,
        session_id: str,
        working_directory: str = "",
        on_permission_request: Any = None,
        model: str = DEFAULT_MODEL,
        tools: list[Any] | None = None,
        streaming: bool = True,
        custom_agents: list[Any] | None = None,
        agent: str | None = None,
    ):
        """
        Resume a Copilot session.
        """
        return await self._client.resume_session(
            session_id=session_id,
            working_directory=working_directory,
            on_permission_request=on_permission_request,
            model=model,
            tools=tools,
            streaming=streaming,
            custom_agents=custom_agents,
            agent=agent,
        )

    def _dispatch_event(self, method_name: str, event):
        """
        Dispatch lifecycle notifications to registered AgentAPI event handlers.
        """
        session_id = getattr(event, "sessionId", "")
        for event_handler in self._event_handlers:
            handler_method = getattr(event_handler, method_name, None)
            if handler_method is None:
                continue

            async def call_handler(method=handler_method, sid=session_id):
                try:
                    await method(sid)
                except Exception:
                    traceback.print_exc()

            asyncio.create_task(call_handler())


class CustomAgentConfig(copilot.session.CustomAgentConfig):
    """
    Extends custom agent definition with icon settings
    """
    icon: str


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
    SessionEventType.SESSION_USAGE_INFO: (
        "on_session_usage", {
            "token_limit": "token_limit",
            "current_tokens": "current_tokens",
            "messages_length": "messages_length",
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


class Session(BaseSession):
    """
    Base agent class implementation using copilot SDK.
    Wraps around a single session with start/stop/reset and tool integration.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        tools: list[str | Callable] | None = None,
        event_handlers: list[SessionEventHandler] | None = None,
        workspace_path: str = "",
    ):
        """
        Initialize the agent with the specified model, tools, and event handlers.
        """
        self._model = model
        self._api = AgentAPI()
        self._shell = PythonShell()  # shared python shell instance for tool execution
        self._session = None
        self._workspace_path = workspace_path or os.getcwd()
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
        self._usage = 0

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

    @property
    def usage(self) -> float:
        """
        Returns the current session usage percentage.
        """
        return self._usage

    async def start(self, session_id: str = None):
        """
        Start the agent by initializing the Copilot client.

        :param session_id: The session ID to resume. If not specified, a new session is created.
        """
        if self._session is not None:
            raise RuntimeError("Agent is already running")

        on_permission_request = self._config.get("on_permission_request")
        model = self._config.get("model", DEFAULT_MODEL)
        tools = self._config.get("tools")
        streaming = self._config.get("streaming", True)
        custom_agents = self._config.get("custom_agents")
        agent = self._config.get("agent")

        if session_id is not None:
            self._session = await self._api.resume_session(
                session_id=session_id,
                working_directory=self._workspace_path,
                on_permission_request=on_permission_request,
                model=model,
                tools=tools,
                streaming=streaming,
                custom_agents=custom_agents,
                agent=agent,
            )
        else:
            self._session = await self._api.create_session(
                working_directory=self._workspace_path,
                on_permission_request=on_permission_request,
                model=model,
                tools=tools,
                streaming=streaming,
                custom_agents=custom_agents,
                agent=agent,
            )

        if self._config.get('agent', None):
            await self._session.rpc.agent.select(SessionAgentSelectParams(name=self._config.get('agent')))

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

        self._session = await self._api.create_session(
            on_permission_request=self._config.get("on_permission_request"),
            model=self._config.get("model", DEFAULT_MODEL),
            tools=self._config.get("tools"),
            streaming=self._config.get("streaming", True),
            custom_agents=self._config.get("custom_agents"),
            agent=self._config.get("agent"),
        )
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
        await self._restart_session()

    async def set_agents(self, agents: list[CustomAgentConfig], agent: str = ""):
        """
        Set the available custom agents and optionally select one by name.
        Updates the session configuration and restarts the session.
        """
        self._config['custom_agents'] = agents or None
        self._config['agent'] = agent or None
        await self._restart_session()

    async def _restart_session(self):
        """
        Restart the current session, preserving the session id.
        """
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

        # Transform SESSION_USAGE_INFO into a single usage_percentage
        if event.type == SessionEventType.SESSION_USAGE_INFO:
            token_limit = args.pop("token_limit", None) or 0.0
            current_tokens = args.pop("current_tokens", None) or 0.0
            # args.pop("messages_length", None)
            usage_percentage = (current_tokens / token_limit * 100.0) if token_limit > 0 else 0.0
            args = {"usage_percentage": usage_percentage}
            self._usage = usage_percentage

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


def get_token() -> str:
    """
    Read the GitHub Copilot CLI token from the local config and keyring.
    """
    # Linux:
    config_path = os.path.expanduser('~/.copilot.cfg')
    if os.path.exists(config_path):
        with open(config_path) as stream:
            return stream.read().strip()

    config_path = os.path.expanduser('~/.copilot/config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Failed to find ~/.copilot/config.json")

    with open(config_path) as stream:
        config = json.load(stream)
        host = config['loggedInUsers'][0]['host']
        user_name = config['loggedInUsers'][0]['login']

    # access Windows Credential Manager to get copilot-cli token:
    # "keyring" attempts to decode it using utf-16 by default, we reverse the encoding
    token_utf16 = keyring.get_password(f'copilot-cli/{host}:{user_name}', f'{host}:{user_name}')
    if not token_utf16:
        raise ValueError(f'Failed to get token for {host}:{user_name}')
    return token_utf16.encode('utf-16')[2:].decode('ascii')


class Model:
    """
    Wraps around github models API to provide chat completions and text embedding.
    """
    def __init__(self, chat_model: str = f'openai/{DEFAULT_MODEL}', embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        """
        Initialize the client with authentication and model configuration.
        """
        token = get_token()

        self._client = AsyncOpenAI(
            base_url='https://models.github.ai/inference',
            api_key=token,
        )
        self._chat_model = chat_model
        self._embedding_model = embedding_model

    @property
    def chat_model(self) -> str:
        """
        Get the current chat model name.
        """
        return self._chat_model

    @chat_model.setter
    def chat_model(self, new_model: str):
        """
        Update the chat model name.
        """
        self._chat_model = new_model

    @property
    def embedding_model(self) -> str:
        """
        Get the current embedding model name.
        """
        return self._embedding_model

    @embedding_model.setter
    def embedding_model(self, new_model: str):
        """
        Update the embedding model name.
        """
        self._embedding_model = new_model

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

    @staticmethod
    async def list_models() -> list[dict]:
        """
        Fetch the list of supported models from the GitHub Models catalog.
        Returns a list of dicts with 'id', 'name', 'summary', and 'capabilities' fields.
        """
        token = get_token()

        async with httpx.AsyncClient() as client:
            response = await client.get(
                'https://models.github.ai/catalog/models',
                headers={'Authorization': f'Bearer {token}'},
            )
            response.raise_for_status()
            data = response.json()

        return [
            {
                'id': item.get('id', ''),
                'name': item.get('name', ''),
                'summary': item.get('summary', ''),
                'capabilities': item.get('capabilities', []),
            }
            for item in data
        ]


def read_agent_md(file: pathlib.Path) -> CustomAgentConfig:
    """
    Read a single *.agent.md file and return a CustomAgentConfig.
    Parses YAML frontmatter for name and description, uses the body as prompt.
    """
    text = file.read_text(encoding="utf-8").strip()

    name = ""
    description = ""
    tools = []
    icon = ""
    prompt = text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            frontmatter = yaml.safe_load(parts[1]) or {}
            prompt = parts[2].strip()
            name = frontmatter.get("name", "")
            description = frontmatter.get("description", "")
            icon = frontmatter.get("icon", "")
            tools = frontmatter.get("tools", [])

    if not name:
        name = file.stem.removesuffix(".agent")

    return CustomAgentConfig(
        name=name,
        description=description,
        prompt=prompt,
        icon=icon,
        tools=tools
    )


def load_agents(directory: str = None) -> Sequence[CustomAgentConfig]:
    """
    Read agents configuration from specified directory.
    If not specified, reads ~/.copilot/agents directory.
    """
    agents_dir = pathlib.Path(directory) if directory else pathlib.Path.home() / ".copilot" / "agents"
    if not agents_dir.is_dir():
        return []

    result = [read_agent_md(f) for f in sorted(agents_dir.glob("*.agent.md"))]

    # check all sub-directories with agent.md files:
    for subdir in agents_dir.iterdir():
        if subdir.is_dir() and (subdir / "agent.md").exists() and (subdir / "agent.md").is_file():
            result.append(read_agent_md(subdir / "agent.md"))

    return result
