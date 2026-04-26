"""
Unit tests for GUI widget helpers.
"""
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from qassistant.agent.common import Message, TextContent
from qassistant.gui.widgets import SearchPopup, _completeWithAutoCompleters, PathAutoCompleter


class TestPathAutoCompleter(unittest.TestCase):
    """
    Validate path completion behavior.
    """

    def test_complete_with_path_prefix(self):
        """
        Completes a plain path prefix to the matching file path.
        """
        completer = PathAutoCompleter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "alpha.txt"
            target.write_text("x", encoding="utf-8")

            result = completer([], f"{root.as_posix()}/alp")

            self.assertEqual(result, [target.as_posix()])

    def test_complete_with_leading_text(self):
        """
        Completes the trailing path token while preserving leading text.
        """
        completer = PathAutoCompleter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "alpha.txt"
            target.write_text("x", encoding="utf-8")

            result = completer([], f"open {root.as_posix()}/alp")

            self.assertEqual(result, [f"open {target.as_posix()}"])

    def test_complete_directory_with_leading_text(self):
        """
        Completes directory paths and keeps trailing slash with leading text.
        """
        completer = PathAutoCompleter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_dir = root / "docs"
            target_dir.mkdir()

            result = completer([], f"read {root.as_posix()}/do")

            self.assertEqual(result, [f"read {target_dir.as_posix()}/"])

    def test_complete_multiple_matches_returns_only_common_prefix(self):
        """
        Returns only common-prefix completion when there are multiple path matches.
        """
        completer = PathAutoCompleter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "alpha.txt").write_text("x", encoding="utf-8")
            (root / "alpine.txt").write_text("x", encoding="utf-8")

            result = completer([], f"open {root.as_posix()}/al")

            self.assertEqual(result, [f"open {root.as_posix()}/alp"])

    def test_complete_multiple_matches_without_extension_returns_empty(self):
        """
        Returns no completion when multiple matches do not share a longer prefix.
        """
        completer = PathAutoCompleter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "alpha.txt").write_text("x", encoding="utf-8")
            (root / "atom.txt").write_text("x", encoding="utf-8")

            result = completer([], f"open {root.as_posix()}/a")

            self.assertEqual(result, [])


class TestAutoCompleterAggregation(unittest.TestCase):
    """
    Validate callable auto-completer aggregation behavior.
    """

    def test_multiple_auto_completers_aggregate_in_order(self):
        """
        Aggregate results from multiple completers while preserving order.
        """
        messages = [Message(role="user", content=[TextContent(text="hello")])]

        def first_completer(chat_messages: list[Message], text: str) -> list[str]:
            self.assertIs(chat_messages, messages)
            self.assertEqual(text, "op")
            return []

        def second_completer(chat_messages: list[Message], text: str) -> list[str]:
            return ["open", "operate"]

        def third_completer(chat_messages: list[Message], text: str) -> list[str]:
            return ["operate", "option"]

        result = _completeWithAutoCompleters(
            [first_completer, second_completer, third_completer],
            messages,
            "op",
        )

        self.assertEqual(result, ["open", "operate", "option"])

    def test_single_auto_completer_callable_is_supported(self):
        """
        Accept a single callable auto-completer without wrapping it in a list.
        """
        result = _completeWithAutoCompleters(
            lambda messages, text: [f"{text}!"],
            [],
            "hello",
        )

        self.assertEqual(result, ["hello!"])


class TestSearchPopup(unittest.TestCase):
    """
    Validate SearchPopup query and action behavior.
    """

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_text_roundtrip(self):
        """
        Search query can be set, retrieved and cleared.
        """
        popup = SearchPopup()
        try:
            popup.setText("needle")
            self.assertEqual(popup.text(), "needle")
            popup.clear()
            self.assertEqual(popup.text(), "")
        finally:
            popup.close()

    def test_next_and_prev_emit_current_query(self):
        """
        Next and previous actions emit the current query text on button click.
        """
        popup = SearchPopup()
        try:
            next_queries: list[str] = []
            prev_queries: list[str] = []
            popup.nextRequested.connect(next_queries.append)
            popup.prevRequested.connect(prev_queries.append)

            popup.setText("alpha")
            popup._nextButton.click()
            popup._prevButton.click()

            self.assertEqual(next_queries, ["alpha"])
            self.assertEqual(prev_queries, ["alpha"])
        finally:
            popup.close()

    def test_query_edited_emits_current_query(self):
        """
        Query edits emit the updated text for incremental search.
        """
        popup = SearchPopup()
        try:
            edited_queries: list[str] = []
            popup.queryEdited.connect(edited_queries.append)

            popup.setText("alpha")

            self.assertEqual(edited_queries, ["alpha"])
        finally:
            popup.close()

    def test_cancel_emits_and_hides(self):
        """
        Cancel emits cancelRequested and hides the popup menu.
        """
        popup = SearchPopup()
        try:
            cancelled: list[bool] = []
            popup.cancelRequested.connect(lambda: cancelled.append(True))

            popup.show()
            self.application.processEvents()
            self.assertTrue(popup.isVisible())

            popup._cancelButton.click()
            self.application.processEvents()

            self.assertEqual(cancelled, [True])
            self.assertFalse(popup.isVisible())
        finally:
            popup.close()


if __name__ == "__main__":
    unittest.main()
