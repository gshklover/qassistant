"""
Unit tests for AgentAPI client startup behavior.
"""
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from qassistant.agent.agent import AgentAPI


class TestAgentAPI(unittest.IsolatedAsyncioTestCase):
    """
    Validate AgentAPI startup guards for read-only list APIs.
    """

    async def test_list_models_starts_client_once_and_caches_models(self):
        """
        Ensure list_models() starts the client before first use and reuses cached models.
        """
        with patch("qassistant.agent.agent.copilot.CopilotClient") as mock_client_cls:
            call_order: list[str] = []
            state = {"value": "disconnected"}

            mock_client = MagicMock()
            mock_client.on = MagicMock()
            mock_client.get_state = MagicMock(side_effect=lambda: state["value"])

            async def start_side_effect():
                call_order.append("start")
                state["value"] = "connected"

            async def list_models_side_effect():
                call_order.append("list_models")
                return [{"id": "m1"}]

            mock_client.start = AsyncMock(side_effect=start_side_effect)
            mock_client.list_models = AsyncMock(side_effect=list_models_side_effect)
            mock_client_cls.return_value = mock_client

            api = AgentAPI()
            first_result = await api.list_models()
            second_result = await api.list_models()

            self.assertEqual(first_result, [{"id": "m1"}])
            self.assertEqual(second_result, [{"id": "m1"}])
            self.assertEqual(call_order, ["start", "list_models"])
            mock_client.start.assert_awaited_once()
            mock_client.list_models.assert_awaited_once()

    async def test_list_sessions_starts_client_before_listing(self):
        """
        Ensure list_sessions() starts the client once before session listing calls.
        """
        with patch("qassistant.agent.agent.copilot.CopilotClient") as mock_client_cls:
            call_order: list[str] = []
            state = {"value": "disconnected"}

            mock_client = MagicMock()
            mock_client.on = MagicMock()
            mock_client.get_state = MagicMock(side_effect=lambda: state["value"])

            async def start_side_effect():
                call_order.append("start")
                state["value"] = "connected"

            async def list_sessions_side_effect():
                call_order.append("list_sessions")
                return [{"id": "s1"}]

            mock_client.start = AsyncMock(side_effect=start_side_effect)
            mock_client.list_sessions = AsyncMock(side_effect=list_sessions_side_effect)
            mock_client_cls.return_value = mock_client

            api = AgentAPI()
            first_result = await api.list_sessions()
            second_result = await api.list_sessions()

            self.assertEqual(first_result, [{"id": "s1"}])
            self.assertEqual(second_result, [{"id": "s1"}])
            self.assertEqual(call_order, ["start", "list_sessions", "list_sessions"])
            mock_client.start.assert_awaited_once()
            self.assertEqual(mock_client.list_sessions.await_count, 2)


if __name__ == "__main__":
    unittest.main()

