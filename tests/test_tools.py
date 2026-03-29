"""
Unit tests for agent tools.
"""
import sys
import unittest

from qassistant.agent.tools.pythonshell import PythonShell


class TestPythonShell(unittest.TestCase):
    """
    Validate PythonShell execution behavior.
    """

    def test_execute_captures_stdout_stderr_and_display_in_order(self):
        """
        Test that stdout, stderr, and display outputs are captured in the correct order.
        """
        shell = PythonShell()

        result = shell.execute("\n".join([
            "import sys",
            "from IPython.display import display",
            "",
            "print('stdout before')",
            "print('stderr line', file=sys.stderr)",
            "display(123)",
            "print('stdout after')",
        ]))

        self.assertTrue(result.success)
        self.assertEqual(
            result.output,
            ['stdout before', 'stderr line', '123', 'stdout after'],
        )

    def test_execute_fails_on_parse_error(self):
        """
        Test that execution fails gracefully when code has syntax errors.
        """
        shell = PythonShell()

        result = shell.execute("this is not valid python !!!")

        self.assertFalse(result.success)
        self.assertIn("syntax", result.error.lower())

    def test_execute_fails_on_runtime_error(self):
        """
        Test that execution fails gracefully when code raises a runtime error.
        """
        shell = PythonShell()

        result = shell.execute("\n".join([
            "x = 10",
            "y = 0",
            "z = x / y  # division by zero",
        ]))

        self.assertFalse(result.success)
        self.assertIn("division", result.error.lower())

    def test_execute_extracts_simple_assignment(self):
        """
        Test that simple variable assignments are correctly executed and stored.
        """
        shell = PythonShell()

        result = shell.execute("x = 42")

        self.assertTrue(result.success)
        self.assertEqual(result.result, 42)

    def test_execute_extracts_dict_indexed_assignment(self):
        """
        Test that dictionary creation and indexed assignments are correctly executed.
        """
        shell = PythonShell()

        result = shell.execute("\n".join([
            "data = {'key': 'value', 'number': 123}",
            "data['key2'] = 'another'",
        ]))

        self.assertTrue(result.success)
        self.assertDictEqual(
            result.result,
            {'key': 'value', 'number': 123, 'key2': 'another'},
        )

