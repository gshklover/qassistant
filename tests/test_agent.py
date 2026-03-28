"""
Unit tests for the Agent event handler integration.
"""
import asyncio
from types import SimpleNamespace
import unittest
from uuid import uuid4

from copilot.generated.session_events import SessionEventType

from qassistant.agent import Agent, AgentEventHandler


class RecordingEventHandler(AgentEventHandler):
    """
    Collects event payloads for validation.
    """

    def __init__(self):
        self.tool_execution_start_events = []
        self.tool_execution_partial_result_events = []
        self.tool_execution_progress_events = []
        self.tool_execution_complete_events = []
        self.assistant_message_delta_events = []
        self.assistant_message_events = []
        self.assistant_reasoning_events = []
        self.assistant_reasoning_delta_events = []
        self.assistant_streaming_delta_events = []
        self.assistant_turn_start_events = []
        self.assistant_turn_end_events = []
        self.session_idle_events = []
        self.session_task_complete_events = []
        self.session_error_events = []
        self.unknown_events = []
        self.user_message_events = []
        self.idle_event = asyncio.Event()

    async def on_tool_execution_start(self, tool_name, arguments, tool_call_id, interaction_id):
        self.tool_execution_start_events.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "tool_call_id": tool_call_id,
            "interaction_id": interaction_id,
        })

    async def on_tool_execution_partial_result(self, tool_call_id, partial_output):
        self.tool_execution_partial_result_events.append({
            "tool_call_id": tool_call_id,
            "partial_output": partial_output,
        })

    async def on_tool_execution_progress(self, tool_call_id, progress_message):
        self.tool_execution_progress_events.append({
            "tool_call_id": tool_call_id,
            "progress_message": progress_message,
        })

    async def on_tool_execution_complete(self, tool_call_id, success, result, error, interaction_id):
        self.tool_execution_complete_events.append({
            "tool_call_id": tool_call_id,
            "success": success,
            "result": result,
            "error": error,
            "interaction_id": interaction_id,
        })

    async def on_assistant_message_delta(self, delta_content, message_id, interaction_id):
        self.assistant_message_delta_events.append({
            "delta_content": delta_content,
            "message_id": message_id,
            "interaction_id": interaction_id,
        })

    async def on_assistant_message(self, content, message_id, interaction_id, reasoning_text, tool_requests):
        self.assistant_message_events.append({
            "content": content,
            "message_id": message_id,
            "interaction_id": interaction_id,
            "reasoning_text": reasoning_text,
            "tool_requests": tool_requests,
        })

    async def on_assistant_reasoning(self, content, reasoning_id, interaction_id, reasoning_text):
        self.assistant_reasoning_events.append({
            "content": content,
            "reasoning_id": reasoning_id,
            "interaction_id": interaction_id,
            "reasoning_text": reasoning_text,
        })

    async def on_assistant_reasoning_delta(self, delta_content, reasoning_id, interaction_id):
        self.assistant_reasoning_delta_events.append({
            "delta_content": delta_content,
            "reasoning_id": reasoning_id,
            "interaction_id": interaction_id,
        })

    async def on_assistant_streaming_delta(self, total_response_size_bytes, interaction_id):
        self.assistant_streaming_delta_events.append({
            "total_response_size_bytes": total_response_size_bytes,
            "interaction_id": interaction_id,
        })

    async def on_assistant_turn_start(self, turn_id, interaction_id):
        self.assistant_turn_start_events.append({
            "turn_id": turn_id,
            "interaction_id": interaction_id,
        })

    async def on_assistant_turn_end(self, turn_id):
        self.assistant_turn_end_events.append({
            "turn_id": turn_id,
        })

    async def on_session_idle(self, background_tasks):
        self.session_idle_events.append({
            "background_tasks": background_tasks,
        })
        self.idle_event.set()

    async def on_session_task_complete(self, summary):
        self.session_task_complete_events.append({
            "summary": summary,
        })

    async def on_session_error(self, error_type, message, error, status_code, url):
        self.session_error_events.append({
            "error_type": error_type,
            "message": message,
            "error": error,
            "status_code": status_code,
            "url": url,
        })

    async def on_unknown_event(self, event_type, event):
        self.unknown_events.append({
            "event_type": event_type,
            "event": event,
        })

    async def on_user_message(self, content, interaction_id, attachments):
        self.user_message_events.append({
            "content": content,
            "interaction_id": interaction_id,
            "attachments": attachments,
        })


class TestAgentEventDispatch(unittest.IsolatedAsyncioTestCase):
    """
    Validate event dispatch logic without requiring a live Copilot session.
    """

    async def test_dispatch_maps_sdk_payload_fields(self):
        handler = RecordingEventHandler()
        agent = Agent(event_handlers=[handler])

        tool_call_id = str(uuid4())
        interaction_id = str(uuid4())
        message_id = str(uuid4())
        reasoning_id = str(uuid4())

        await agent._handle_event(SimpleNamespace(
            type=SessionEventType.TOOL_EXECUTION_START,
            data=SimpleNamespace(
                tool_name="view",
                arguments={"path": "/tmp"},
                tool_call_id=tool_call_id,
                interaction_id=interaction_id,
            ),
        ))
        await agent._handle_event(SimpleNamespace(
            type=SessionEventType.ASSISTANT_MESSAGE_DELTA,
            data=SimpleNamespace(
                delta_content="chunk",
                message_id=message_id,
                interaction_id=interaction_id,
            ),
        ))
        await agent._handle_event(SimpleNamespace(
            type=SessionEventType.ASSISTANT_REASONING_DELTA,
            data=SimpleNamespace(
                delta_content="thought",
                reasoning_id=reasoning_id,
                interaction_id=interaction_id,
            ),
        ))
        await agent._handle_event(SimpleNamespace(
            type=SessionEventType.TOOL_EXECUTION_COMPLETE,
            data=SimpleNamespace(
                tool_call_id=tool_call_id,
                success=True,
                result={"content": "ok"},
                error=None,
                interaction_id=interaction_id,
            ),
        ))

        self.assertEqual(handler.tool_execution_start_events[0]["tool_name"], "view")
        self.assertEqual(handler.tool_execution_start_events[0]["arguments"], {"path": "/tmp"})
        self.assertEqual(handler.assistant_message_delta_events[0]["delta_content"], "chunk")
        self.assertEqual(handler.assistant_reasoning_delta_events[0]["delta_content"], "thought")
        self.assertTrue(handler.tool_execution_complete_events[0]["success"])


class TestAgentEventHandlerIntegration(unittest.IsolatedAsyncioTestCase):
    """
    Validate live event handler integration with the Copilot session.
    """

    async def test_list_tmp_emits_tool_and_streaming_events(self):
        handler = RecordingEventHandler()

        async with Agent(event_handlers=[handler]) as agent:
            response = await agent.send(
                "List the contents of /tmp directory, then summarize what you found in one short paragraph."
            )
            await asyncio.wait_for(handler.idle_event.wait(), timeout=15)
            await asyncio.sleep(0.25)

        self.assertIsNotNone(response)
        self.assertTrue(handler.user_message_events)
        self.assertTrue(handler.assistant_turn_start_events)
        self.assertTrue(handler.assistant_turn_end_events)
        self.assertTrue(handler.session_idle_events)
        self.assertTrue(handler.assistant_message_events)
        self.assertTrue(handler.assistant_message_delta_events)
        self.assertTrue(handler.assistant_streaming_delta_events)

        view_start_event = next(
            (
                event
                for event in handler.tool_execution_start_events
                if event["tool_name"] == "view"
                and isinstance(event["arguments"], dict)
                and event["arguments"].get("path") == "/tmp"
            ),
            None,
        )
        self.assertIsNotNone(view_start_event)

        complete_event = next(
            (
                event
                for event in handler.tool_execution_complete_events
                if event["tool_call_id"] == view_start_event["tool_call_id"]
            ),
            None,
        )
        self.assertIsNotNone(complete_event)
        self.assertTrue(complete_event["success"])

        assistant_message = handler.assistant_message_events[-1]["content"] or ""
        self.assertTrue(assistant_message)
        self.assertIn("/tmp", assistant_message)

        # reasoning_event_count = (
        #     len(handler.assistant_reasoning_events) + len(handler.assistant_reasoning_delta_events)
        # )
        # self.assertGreater(
        #     reasoning_event_count,
        #     0,
        #     "Expected at least one reasoning event while streaming the response.",
        # )


if __name__ == "__main__":
    unittest.main()
