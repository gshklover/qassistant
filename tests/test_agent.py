"""
Unit tests for the Agent event handler integration.
"""
import asyncio
import datetime

from copilot.generated.session_events import SessionEventType
from types import SimpleNamespace
import unittest
from uuid import uuid4


from qassistant.agent import Agent, AgentEventHandler, list_models


class _RecordingEventsHandler(AgentEventHandler):
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
        self.session_usage_events = []
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

    async def on_session_usage(self, usage_percentage):
        self.session_usage_events.append({
            "usage_percentage": usage_percentage,
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


class TestAgent(unittest.IsolatedAsyncioTestCase):
    """
    Test Agent implementation
    """

    async def test_list_models(self):
        """
        Validate that the agent can retrieve a list of available models from the API.
        """
        models = await list_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)

    async def test_dispatch_maps_sdk_payload_fields(self):
        """
        Validate event dispatch logic without requiring a live Copilot session.
        """
        handler = _RecordingEventsHandler()
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

        await agent._handle_event(SimpleNamespace(
            type=SessionEventType.SESSION_USAGE_INFO,
            data=SimpleNamespace(
                token_limit=200000.0,
                current_tokens=50000.0,
                messages_length=10.0,
            ),
        ))

        self.assertEqual(handler.tool_execution_start_events[0]["tool_name"], "view")
        self.assertEqual(handler.tool_execution_start_events[0]["arguments"], {"path": "/tmp"})
        self.assertEqual(handler.assistant_message_delta_events[0]["delta_content"], "chunk")
        self.assertEqual(handler.assistant_reasoning_delta_events[0]["delta_content"], "thought")
        self.assertTrue(handler.tool_execution_complete_events[0]["success"])
        self.assertEqual(len(handler.session_usage_events), 1)
        self.assertAlmostEqual(handler.session_usage_events[0]["usage_percentage"], 25.0)

    async def test_dispatch_session_usage_high_utilization(self):
        """
        Validate that usage_percentage is correctly computed from current_tokens / token_limit.
        """
        handler = _RecordingEventsHandler()
        agent = Agent(event_handlers=[handler])

        await agent._handle_event(SimpleNamespace(
            type=SessionEventType.SESSION_USAGE_INFO,
            data=SimpleNamespace(
                token_limit=100000.0,
                current_tokens=80000.0,
                messages_length=25.0,
            ),
        ))

        self.assertEqual(len(handler.session_usage_events), 1)
        self.assertAlmostEqual(handler.session_usage_events[0]["usage_percentage"], 80.0)

    async def test_dispatch_session_usage_zero_token_limit(self):
        """
        Validate that usage_percentage is 0.0 when token_limit is zero.
        """
        handler = _RecordingEventsHandler()
        agent = Agent(event_handlers=[handler])

        await agent._handle_event(SimpleNamespace(
            type=SessionEventType.SESSION_USAGE_INFO,
            data=SimpleNamespace(
                token_limit=0.0,
                current_tokens=0.0,
                messages_length=0.0,
            ),
        ))

        self.assertEqual(len(handler.session_usage_events), 1)
        self.assertAlmostEqual(handler.session_usage_events[0]["usage_percentage"], 0.0)

    async def test_list_tmp_emits_tool_and_streaming_events(self):
        """
        Validate live event handler integration with the Copilot session.
        """

        handler = _RecordingEventsHandler()

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

    async def test_agent_shell(self):
        """
        Test using python shell with the agent
        """
        handler = _RecordingEventsHandler()

        async with Agent(event_handlers=[handler]) as agent:
            response = await agent.send(
                "Call the pyshell_execute tool and execute this exact Python code:\n"
                "def fib(n):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n"
                "print('110', fib(110))\n"
                "print('151', fib(151))\n"
                "print('173', fib(173))\n"
                "Do not use bash. Then briefly summarize the three computed values."
            )
            await asyncio.wait_for(handler.idle_event.wait(), timeout=15)
            await asyncio.sleep(0.25)

            self.assertIsNotNone(response)

            pyshell_start_event = next(
                (
                    event
                    for event in handler.tool_execution_start_events
                    if event["tool_name"] == "pyshell_execute"
                ),
                None,
            )
            self.assertIsNotNone(
                pyshell_start_event,
                f"Expected pyshell_execute start event. Saw: {[e['tool_name'] for e in handler.tool_execution_start_events]}",
            )

            pyshell_complete_event = next(
                (
                    event
                    for event in handler.tool_execution_complete_events
                    if event["tool_call_id"] == pyshell_start_event["tool_call_id"]
                ),
                None,
            )
            self.assertIsNotNone(pyshell_complete_event)

    async def test_submit(self):
        """
        Validate submit() dispatches message through session.send and updates workspace path.
        """
        done_event = asyncio.Event()

        class StreamingEventHandler(AgentEventHandler):
            def __init__(self):
                self.streaming_response = ""

            async def on_assistant_message_delta(self, delta_content, message_id, interaction_id):
                self.streaming_response += delta_content

            async def on_session_idle(self, background_tasks):
                done_event.set()

            async def on_session_error(self, error_type, message, error, status_code, url):
                done_event.set()

        handler = StreamingEventHandler()
        agent = Agent(event_handlers=[handler])
        async with agent:
            message_id = await agent.submit("compute 2 + 2")
            await asyncio.wait_for(done_event.wait(), timeout=15)

        self.assertTrue(handler.streaming_response)
        self.assertIn("4", handler.streaming_response)


class TestModel(unittest.IsolatedAsyncioTestCase):
    """
    Validate Model chat and embed methods against the live GitHub Models API.
    """

    async def test_list_models_returns_catalog(self):
        """
        Test that list_models() returns a non-empty list of model dicts with expected fields.
        """
        from qassistant.agent.agent import Model

        models = await Model.list_models()
        print(models)

        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        for model in models:
            self.assertIn('id', model)
            self.assertIn('name', model)
            self.assertIn('summary', model)
            self.assertIn('capabilities', model)
            self.assertIsInstance(model['id'], str)
            self.assertTrue(model['id'])

        # gpt-4.1-nano appears to be one of the fastest models
        # github_model = Model()
        # for model in models:
        #     github_model.chat_model = model['id']
        #     try:
        #         start = datetime.datetime.now()
        #         _ = await github_model.chat([{'role': 'user', 'content': 'compute 2 + 2'}])
        #         end = datetime.datetime.now()
        #     except:
        #         continue
        #     print(f'{model["id"]}: {(end - start).total_seconds():.2f} seconds')


    async def test_chat_returns_answer_to_arithmetic(self):
        """
        Test that chat() returns a response containing the correct answer to a simple arithmetic question.
        """
        from qassistant.agent.agent import Model
        client = Model()

        result = await client.chat([{'role': 'user', 'content': 'What is 2+2? Reply with just the number.'}])

        self.assertIn('4', result)

    async def test_embed_returns_vector(self):
        """
        Test that embed() returns a non-empty list of floats.
        """
        from qassistant.agent.agent import Model
        client = Model()

        result = await client.embed('hello world')

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIsInstance(result[0], float)

    # async def test_complete(self):
    #     """
    #     Test that complete() returns a response containing the correct answer to a simple arithmetic question.
    #     """
    #     from qassistant.agent.agent import Model
    #     client = Model(chat_model='o1-mini')
    #
    #     result = await client.complete('2 + 2 = ')
    #     self.assertIn('4', result)


if __name__ == "__main__":
    unittest.main()
