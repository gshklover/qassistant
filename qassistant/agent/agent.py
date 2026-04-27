"""
Agent implementation. Provides a small adapter around GenAI SDK and tool execution.

NOTES:
    * copilot events description: https://docs.github.com/en/copilot/how-tos/copilot-sdk/use-copilot-sdk/streaming-events
"""
import asyncio
import traceback

import copilot
from copilot.generated.session_events import SessionEventType
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
from PySide6.QtCore import QObject, Signal
from typing import Any, Callable, Sequence

from .tools.pythonshell import PythonShell
from .common import Message, Role, TextContent

# agent defaults:
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
DEFAULT_TIMEOUT = 300.0  # per-turn timeout in seconds
CLIENT_CONNECTING_TIMEOUT = 5.0  # seconds to wait while client is already connecting
CLIENT_CONNECTING_POLL_INTERVAL = 0.05  # seconds between state checks


class AgentAPI(QObject):
    """
    Thin wrapper around a shared CopilotClient instance.
    Inherits QObject to expose session lifecycle events as Qt signals.
    """

    sessionCreated = Signal(str)
    sessionDeleted = Signal(str)
    sessionUpdated = Signal(str)

    def __init__(self, parent: QObject = None):
        """
        Initialize the client
        """
        super().__init__(parent)
        self._models = None
        self._client_start_lock = asyncio.Lock()
        self._client = copilot.CopilotClient(auto_start=True)
        self._client.on("session.created", lambda event: self.sessionCreated.emit(getattr(event, "sessionId", "")))
        self._client.on("session.deleted", lambda event: self.sessionDeleted.emit(getattr(event, "sessionId", "")))
        self._client.on("session.updated", lambda event: self.sessionUpdated.emit(getattr(event, "sessionId", "")))

    @property
    def client(self) -> copilot.CopilotClient:
        """
        Get the underlying CopilotClient instance.
        """
        return self._client

    async def _ensure_client_started(self):
        """
        Ensure the shared Copilot client is started before using read-only API calls.
        """
        if self._client.get_state() == "connected":
            return

        async with self._client_start_lock:
            state = self._client.get_state()

            if state == "connecting":
                for _ in range(int(CLIENT_CONNECTING_TIMEOUT / CLIENT_CONNECTING_POLL_INTERVAL)):
                    await asyncio.sleep(CLIENT_CONNECTING_POLL_INTERVAL)
                    state = self._client.get_state()
                    if state != "connecting":
                        break

            if state == "connecting":
                raise TimeoutError(f"Copilot client remained in 'connecting' state for {CLIENT_CONNECTING_TIMEOUT}s")

            if state == "connected":
                return

            await self._client.start()

    async def list_models(self) -> list[copilot.client.ModelInfo]:
        """
        List available models from the Copilot API.
        Returns list of copilot.client.ModelInfo objects with 'id', 'name', 'summary', and 'capabilities' fields.
        """
        await self._ensure_client_started()
        if self._models is None:
            self._models = await self._client.list_models()
        return self._models

    async def list_sessions(self) -> list[copilot.client.SessionMetadata]:
        """
        List existing sessions from the Copilot API.
        """
        await self._ensure_client_started()
        return await self._client.list_sessions()

    async def delete_session(self, session_id: str):
        """
        Delete an existing session by id.
        """
        await self._ensure_client_started()
        return await self._client.delete_session(session_id=session_id)

    async def create_session(
        self,
        *,
        workspace_directory: str = "",
        model: str = DEFAULT_MODEL,
        custom_agents: list[Any] | None = None,
        agent: str | None = None,
    ) -> Session:
        shell = PythonShell()  # shared python shell instance for tool execution
        tools = [
            # execution shell:
            as_tool(shell.execute, name='python_shell_execute'),
            as_tool(shell.get_variables, name='python_shell_get_variables'),
            # *[as_tool(tool) for tool in (tools or ())],
        ]
        config = {
            'tools': tools,
            'streaming': True,
            'on_permission_request': copilot.session.PermissionHandler.approve_all,
        }
        return Session(
            session=await self._client.create_session(
                working_directory=workspace_directory,
                model=model,
                custom_agents=custom_agents,
                agent=agent,
                **config
            ),
            shell=shell,
            session_config=config,
            resume_callback=self._client.resume_session,
            custom_agents=custom_agents,
            agent=agent
        )

    async def resume_session(
        self,
        session_id: str,
        workspace_directory: str = "",
        model: str = DEFAULT_MODEL,
        custom_agents: list[Any] | None = None,
        agent: str | None = None
    ) -> Session:
        """
        Resume an existing session by id.
        """
        shell = PythonShell()  # shared python shell instance for tool execution
        tools = [
            # execution shell:
            as_tool(shell.execute, name='python_shell_execute'),
            as_tool(shell.get_variables, name='python_shell_get_variables'),
            # *[as_tool(tool) for tool in (tools or ())],
        ]
        config = {
            'tools': tools,
            'streaming': True,
            'on_permission_request': copilot.session.PermissionHandler.approve_all,
        }
        return Session(
            session=await self._client.resume_session(
                session_id,
                working_directory=workspace_directory,
                model=model,
                custom_agents=custom_agents,
                agent=agent,
                **config
            ),
            shell=shell,
            session_config=config
        )


class CustomAgentConfig(copilot.session.CustomAgentConfig):
    """
    Extends custom agent definition with icon settings
    """
    icon: str


# map event type to handler name & argument mapping
_EVENT_ARGS = {
    SessionEventType.TOOL_EXECUTION_START: {
        "tool_name": "tool_name",
        "arguments": "arguments",
        "tool_call_id": "tool_call_id",
        "interaction_id": "interaction_id",
    },
    SessionEventType.TOOL_EXECUTION_PARTIAL_RESULT: {
        "tool_call_id": "tool_call_id",
        "partial_output": "partial_output",
    },
    SessionEventType.TOOL_EXECUTION_PROGRESS: {
        "tool_call_id": "tool_call_id",
        "progress_message": "progress_message",
    },
    SessionEventType.TOOL_EXECUTION_COMPLETE: {
        "tool_call_id": "tool_call_id",
        "success": "success",
        "result": "result",
        "error": "error",
        "interaction_id": "interaction_id",
    },
    SessionEventType.ASSISTANT_MESSAGE: {
        "content": "content",
        "message_id": "message_id",
        "interaction_id": "interaction_id",
        "reasoning_text": "reasoning_text",
        "tool_requests": "tool_requests",
    },
    SessionEventType.ASSISTANT_MESSAGE_DELTA: {
        "delta_content": "delta_content",
        "message_id": "message_id",
        "interaction_id": "interaction_id",
    },
    SessionEventType.ASSISTANT_REASONING: {
        "content": "content",
        "reasoning_id": "reasoning_id",
        "interaction_id": "interaction_id",
        "reasoning_text": "reasoning_text",
    },
    SessionEventType.ASSISTANT_REASONING_DELTA: {
        "delta_content": "delta_content",
        "reasoning_id": "reasoning_id",
        "interaction_id": "interaction_id",
    },
    SessionEventType.ASSISTANT_STREAMING_DELTA: {
        "total_response_size_bytes": "total_response_size_bytes",
        "interaction_id": "interaction_id",
    },
    SessionEventType.ASSISTANT_TURN_END: {
        "turn_id": "turn_id",
    },
    SessionEventType.ASSISTANT_TURN_START: {
        "turn_id": "turn_id",
        "interaction_id": "interaction_id",
    },
    SessionEventType.SESSION_IDLE: {
        "background_tasks": "background_tasks",
    },
    SessionEventType.SESSION_TASK_COMPLETE: {
        "summary": "summary",
    },
    SessionEventType.SESSION_ERROR: {
        "error_type": "error_type",
        "message": "message",
        "error": "error",
        "status_code": "status_code",
        "url": "url",
    },
    SessionEventType.SESSION_USAGE_INFO: {
        "token_limit": "token_limit",
        "current_tokens": "current_tokens",
        "messages_length": "messages_length",
    },
    SessionEventType.USER_MESSAGE: {
        "content": "content",
        "interaction_id": "interaction_id",
        "attachments": "attachments",
    }
}

_EVENT_SIGNAL_BY_METHOD = {
    SessionEventType.TOOL_EXECUTION_START: "toolExecutionStart",
    SessionEventType.TOOL_EXECUTION_PARTIAL_RESULT: "toolExecutionPartialResult",
    SessionEventType.TOOL_EXECUTION_PROGRESS: "toolExecutionProgress",
    SessionEventType.TOOL_EXECUTION_COMPLETE: "toolExecutionComplete",
    SessionEventType.ASSISTANT_MESSAGE: "assistantMessage",
    SessionEventType.ASSISTANT_MESSAGE_DELTA: "assistantMessageDelta",
    SessionEventType.ASSISTANT_REASONING: "assistantReasoning",
    SessionEventType.ASSISTANT_REASONING_DELTA: "assistantReasoningDelta",
    SessionEventType.ASSISTANT_STREAMING_DELTA: "assistantStreamingDelta",
    SessionEventType.ASSISTANT_TURN_END: "assistantTurnEnd",
    SessionEventType.ASSISTANT_TURN_START: "assistantTurnStart",
    SessionEventType.SESSION_IDLE: "sessionIdle",
    SessionEventType.SESSION_TASK_COMPLETE: "sessionTaskComplete",
    SessionEventType.SESSION_ERROR: "sessionError",
    SessionEventType.SESSION_USAGE_INFO: "sessionUsage",
    SessionEventType.USER_MESSAGE: "userMessage",
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


class Session(QObject):
    """
    Wraps around a single session.
    Exposes session events using Qt signals.
    """

    # Signature: (tool_name: str | None, arguments: Any, tool_call_id: str | None, interaction_id: str | None)
    toolExecutionStart = Signal(object, object, object, object)
    # Signature: (tool_call_id: str | None, partial_output: str | None)
    toolExecutionPartialResult = Signal(object, object)
    # Signature: (tool_call_id: str | None, progress_message: str | None)
    toolExecutionProgress = Signal(object, object)
    # Signature: (tool_call_id: str | None, success: bool | None, result: Any, error: Any, interaction_id: str | None)
    toolExecutionComplete = Signal(object, object, object, object, object)
    # Signature: (delta_content: str | None, message_id: str | None, interaction_id: str | None)
    assistantMessageDelta = Signal(object, object, object)
    # Signature: (content: str | None, message_id: str | None, interaction_id: str | None, reasoning_text: str | None, tool_requests: list[Any] | None)
    assistantMessage = Signal(object, object, object, object, object)
    # Signature: (content: str | None, reasoning_id: str | None, interaction_id: str | None, reasoning_text: str | None)
    assistantReasoning = Signal(object, object, object, object)
    # Signature: (delta_content: str | None, reasoning_id: str | None, interaction_id: str | None)
    assistantReasoningDelta = Signal(object, object, object)
    # Signature: (total_response_size_bytes: float | None, interaction_id: str | None)
    assistantStreamingDelta = Signal(object, object)
    # Signature: (turn_id: str | None)
    assistantTurnEnd = Signal(object)
    # Signature: (turn_id: str | None, interaction_id: str | None)
    assistantTurnStart = Signal(object, object)
    # Signature: (background_tasks: Any)
    sessionIdle = Signal(object)
    # Signature: (summary: str | None)
    sessionTaskComplete = Signal(object)
    # Signature: (error_type: str | None, message: str | None, error: Any, status_code: int | None, url: str | None)
    sessionError = Signal(object, object, object, object, object)
    # Signature: (usage_percentage: float)
    sessionUsage = Signal(float)
    # Signature: (event_type: str, event: Any)
    unknownEvent = Signal(str, object)
    # Signature: (content: str | None, interaction_id: str | None, attachments: list[Any] | None)
    userMessage = Signal(object, object, object)

    def __init__(
        self,
        session: copilot.CopilotSession,
        shell: PythonShell = None,
        model: str = DEFAULT_MODEL,
        workspace_path: str = "",
        session_config: dict[str, Any] = None,
        resume_callback = None,  # pointer to client.resume_session to allow session restart on workspace change
        custom_agents: list[Any] | None = None,
        agent: str | None = None,
        parent: QObject = None
    ):
        """
        Initialize the agent with the specified model, tools, and event handlers.
        """
        super().__init__(parent)
        self._session = session
        self._shell = shell
        self._session_config = session_config
        self._model = model
        self._workspace_path = workspace_path or os.getcwd()
        self._usage = 0
        self._resume_callback = resume_callback
        self._session.on(self._on_event)
        self._custom_agents = custom_agents
        self._agent = agent

    @property
    def session(self) -> copilot.CopilotSession:
        """
        Returns copilot session object associated with the agent.
        """
        return self._session

    @property
    def session_id(self) -> str:
        """
        Get the current session ID, or empty string if no active session.
        """
        return self._session.session_id if self._session else ""

    @property
    def model(self) -> str:
        """
        Get the current model name.
        """
        return self._model

    @property
    def custom_agents(self) -> list[Any] | None:
        """
        Get the list of available custom agents.
        """
        return self._custom_agents

    @property
    def agent(self) -> str | None:
        """
        Get the currently selected custom agent name, or None if no agent selected.
        """
        return self._agent

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

    async def start(self):
        """
        Start the session
        """
        pass

    async def stop(self):
        """
        Stop the agent and clean up resources.
        """
        await self._session.disconnect()

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

    async def reset(self):
        """
        Reset session contents. Not implemented yet - no clear API to implement this using copilot client.
        """
        pass

    async def set_model(self, model: str):
        """
        Update the model used by the session. This will restart the session with the new model.
        """
        self._model = model
        await self._session.set_model(model)

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
        self._session_config['custom_agents'] = agents or None
        self._session_config['agent'] = agent or None
        await self._restart_session()

    async def get_messages(self) -> list[Message]:
        """
        Get all events from the session events history.
        """
        events = await self._session.get_messages()

        # convert events to list of user & assistant messages
        result = []
        for event in events:
            if event.type == SessionEventType.USER_MESSAGE:
                result.append(Message(
                    role=Role.USER,
                    content=[TextContent(text=event.data.content)]
                ))
            elif event.type == SessionEventType.ASSISTANT_MESSAGE:
                result.append(Message(
                    role=Role.ASSISTANT,
                    content=[TextContent(text=event.data.content)]
                ))

        return result

    async def _restart_session(self):
        """
        Restart the current session, preserving the session id.
        Called when changing workspace directory.
        """
        if self._resume_callback is None:
            return

        session_id = self._session.session_id

        try:
            await self._session.disconnect()
            new_session = await self._resume_callback(
                session_id=session_id,
                working_directory=self._workspace_path,
                model=self._model,
                **self._session_config
            )
            new_session.on(self._on_event)
            self._session = new_session
        except:  # noqa
            traceback.print_exc()
            return

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
        event_info = _EVENT_ARGS.get(event.type, None)
        if event_info is None:
            self._emitEventSignal('unknownEvent', {
                "event_type": event.type.value,
                "event": event,
            })
            return

        args = {
            arg_name: getattr(event.data, attr_name, None)
            for arg_name, attr_name in event_info.items()
        }

        # Transform SESSION_USAGE_INFO into a single usage_percentage
        if event.type == SessionEventType.SESSION_USAGE_INFO:
            token_limit = args.pop("token_limit", None) or 0.0
            current_tokens = args.pop("current_tokens", None) or 0.0
            # args.pop("messages_length", None)
            usage_percentage = (current_tokens / token_limit * 100.0) if token_limit > 0 else 0.0
            args = {"usage_percentage": usage_percentage}
            self._usage = usage_percentage

        self._emitEventSignal(event.type, args)

    def _emitEventSignal(self, method_name: SessionEventType | str, args: dict[str, Any]):
        """
        Emit the Qt signal corresponding to the SessionEventHandler callback name.
        """
        signal_name = _EVENT_SIGNAL_BY_METHOD.get(method_name)
        if not signal_name:
            return

        signal = getattr(self, signal_name, None)
        if signal is None:
            return

        signal.emit(*args.values())

    def _on_event(self, event: copilot.session.SessionEvent):
        """
        Session callback that logs non-streaming events and dispatches to handlers.
        """
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
