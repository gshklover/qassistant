"""
Agent implementation. Provides a small adapter around GenAI SDK and tool execution.
"""
import copilot
from typing import Callable

from .common import BaseAgent


DEFAULT_MODEL = "gpt-5-mini"


class Agent(BaseAgent):
    """
    Base agent class implementation using copilot SDK.
    Wraps around a single session with start/stop/reset and tool integratio
    """
    
    def __init__(self, model: str = DEFAULT_MODEL, tools: list[str | Callable] = None):
        """
        Initialize the agent with the specified model and tools.
        """
        self._client = copilot.CopilotClient()
        self._model = model
        self._session = None
        self._tools = [copilot.define_tool()(tool) for tool in (tools or ())]
        self._config = dict(
            on_permission_request=copilot.PermissionHandler.approve_all,
            model=model,
            tools=self._tools,
            streaming=True
        )

    @property
    def running(self) -> bool:
        """
        Check if the agent session is currently active.
        """
        return self._session is not None

    async def start(self):
        """
        Start the agent by initializing the Copilot client.
        """
        if self._session is not None:
            raise RuntimeError("Agent is already running")
        
        await self._client.start()
        self._session = await self._client.create_session(**self._config)
        self._session.on(self._on_event)

    async def stop(self):
        """
        Stop the agent and clean up resources.
        """
        if self._session:
            await self._session.disconnect()
            self._session = None
        await self._client.stop()

    async def reset(self):
        """
        Reset the agent session.
        """
        if self._session:
            await self._session.disconnect()
        
        self._session = await self._client.create_session(**self._config)

    async def send(self, message: str):
        """
        Send a message to the agent and return the response.
        """
        if not self._session:
            raise RuntimeError("Agent is not running. Call start() first.")
        
        response = await self._session.send_and_wait(message)
        return response

    async def __aenter__(self):
        """
        Start running the agent
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
        Stop running the agent
        """
        await self.stop()

    def _on_event(self, event: copilot.SessionEvent):
        """
        Handle session events such as tool invocations or permission requests.
        Majority of the event are streaming events with very little data.
        """
        if event.type.value in ('assistant.streaming_delta', 'assistant.message_delta', 'assistant.reasoning_delta'):
            return
        
        data = {k: v for k, v in event.data.to_dict().items() if v is not None}
        print(f'\n{event.type.value} ({event.id}, parent={event.parent_id}): {data}')
