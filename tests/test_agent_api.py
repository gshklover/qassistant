"""
Unit tests for AgentAPI client startup behavior.
"""
import asyncio
import os
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from qassistant.agent.agent import AgentAPI


class TestAgentAPI(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
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

    async def test_list_models_waits_for_connecting_state(self):
        """
        Ensure list_models() waits for an in-flight connect and avoids a duplicate start.
        """
        with patch("qassistant.agent.agent.CLIENT_CONNECTING_TIMEOUT", 0.05), \
                patch("qassistant.agent.agent.CLIENT_CONNECTING_POLL_INTERVAL", 0.001), \
                patch("qassistant.agent.agent.copilot.CopilotClient") as mock_client_cls:
            state = {"value": "connecting"}

            mock_client = MagicMock()
            mock_client.on = MagicMock()
            mock_client.get_state = MagicMock(side_effect=lambda: state["value"])
            mock_client.start = AsyncMock()
            mock_client.list_models = AsyncMock(return_value=[{"id": "m1"}])
            mock_client_cls.return_value = mock_client

            async def transition_to_connected():
                await asyncio.sleep(0.01)
                state["value"] = "connected"

            transition_task = asyncio.create_task(transition_to_connected())
            try:
                api = AgentAPI()
                result = await api.list_models()
            finally:
                await transition_task

            self.assertEqual(result, [{"id": "m1"}])
            mock_client.start.assert_not_awaited()
            mock_client.list_models.assert_awaited_once()

    async def test_list_sessions_connecting_timeout_raises(self):
        """
        Ensure list_sessions() raises TimeoutError if client remains connecting beyond threshold.
        """
        with patch("qassistant.agent.agent.CLIENT_CONNECTING_TIMEOUT", 0.01), \
                patch("qassistant.agent.agent.CLIENT_CONNECTING_POLL_INTERVAL", 0.001), \
                patch("qassistant.agent.agent.copilot.CopilotClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.on = MagicMock()
            mock_client.get_state = MagicMock(return_value="connecting")
            mock_client.start = AsyncMock()
            mock_client.list_sessions = AsyncMock(return_value=[{"id": "s1"}])
            mock_client_cls.return_value = mock_client

            api = AgentAPI()
            with self.assertRaises(TimeoutError):
                await api.list_sessions()

            mock_client.start.assert_not_awaited()
            mock_client.list_sessions.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()

