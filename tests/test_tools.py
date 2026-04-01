"""  
Unit tests for agent tools.
"""
import sys
import tempfile
import unittest
from pathlib import Path

from qassistant.agent.tools.knowledgebase import KnowledgeBase
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


class TestKnowledgeBase(unittest.IsolatedAsyncioTestCase):
    """
    Validate KnowledgeBase chunking behavior.
    """

    @staticmethod
    async def _embed(text: str) -> list[float]:
        """
        Dummy async embedding function that returns the length of the text as a single float in a list.
        """
        return [float(len(text))]

    async def test_add_file(self):
        """
        Test that adjacent chunks include the configured overlap text.
        """
        knowledge_base = KnowledgeBase(self._embed, chunk_size=14, overlap=3)
        file_content = "alpha\n\nbeta\n\ngamma"

        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', suffix='.txt') as stream:
            stream.write(file_content)
            path = stream.name

        try:
            await knowledge_base.add_file(path)
            filename = Path(path).name
            document = knowledge_base.get_document(f'file:///{filename}')

            self.assertEqual(len(document.content), 2)
            self.assertEqual(document.content[0].content, 'alpha\n\nbeta')
            self.assertEqual(document.content[1].content, 'etagamma')
        finally:
            Path(path).unlink()

    async def test_add_directory(self):
        """
        Test that add_directory recursively stores files in subdirectories with proper URL hierarchy.
        """
        knowledge_base = KnowledgeBase(self._embed)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test directory structure:
            # tmpdir/
            #   file1.txt
            #   subdir1/
            #     file2.txt
            #   subdir1/subdir2/
            #     file3.txt

            tmpdir_path = Path(tmpdir)
            (tmpdir_path / 'subdir1' / 'subdir2').mkdir(parents=True)

            (tmpdir_path / 'file1.txt').write_text('content1')
            (tmpdir_path / 'subdir1' / 'file2.txt').write_text('content2')
            (tmpdir_path / 'subdir1' / 'subdir2' / 'file3.txt').write_text('content3')

            await knowledge_base.add_directory(tmpdir, destination='/docs')

            # Verify all files were added with correct URLs
            doc1 = knowledge_base.get_document('file:///docs/file1.txt')
            self.assertIsNotNone(doc1)
            self.assertEqual(doc1.content[0].content, 'content1')

            doc2 = knowledge_base.get_document('file:///docs/subdir1/file2.txt')
            self.assertIsNotNone(doc2)
            self.assertEqual(doc2.content[0].content, 'content2')

            doc3 = knowledge_base.get_document('file:///docs/subdir1/subdir2/file3.txt')
            self.assertIsNotNone(doc3)
            self.assertEqual(doc3.content[0].content, 'content3')

            # Verify list_documents works correctly
            root_list = knowledge_base.list_documents('/docs')
            self.assertEqual(len(root_list), 2)  # file1.txt and subdir1
            names = {item['name'] for item in root_list}
            self.assertIn('file1.txt', names)
            self.assertIn('subdir1', names)

