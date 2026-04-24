"""
Unit tests for the Agent event handler integration.
"""
import unittest


class TestSession(unittest.IsolatedAsyncioTestCase):
    """
    Test Agent implementation
    """


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
