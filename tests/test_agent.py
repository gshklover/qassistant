"""
Unit tests for the Agent core.
"""
import unittest

from qassistant.agent import Agent
import copilot


class TestCopilotSDK(unittest.IsolatedAsyncioTestCase):
    """
    Basic tests for Copilot SDK availability.
    """
    async def test_copilot_sdk(self):
        """
        Test that the Copilot SDK is available and can be used.
        """
        client = copilot.CopilotClient()
        await client.start()

        try:
            # create a new session:
            session = await client.create_session({
                "on_permission_request": copilot.PermissionHandler.approve_all,
                "model": "gpt-5-mini",
                "streaming": True
            })
            # send a message and wait for response:
            async with session:
                response = await session.send_and_wait({'prompt': "Compute 2 + 2"})
                print(response)
        finally:
            await client.stop()



class TestAgent(unittest.TestCase):
    """
    Basic tests for Agent functionality.
    """

    def test_start_and_send(self):
        """
        Test basic session start and message sending.
        """
        agent = Agent()
        session = agent.start_session("s1")
        self.assertEqual(session.session_id, "s1")
        res = agent.send_message("s1", "hello")
        self.assertIn("content", res)
        self.assertTrue(res["content"].startswith("Echo:"))


if __name__ == "__main__":
    unittest.main()
