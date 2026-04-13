"""
Unit tests for GUI settings components.
"""
import os
import tempfile
import unittest
from pathlib import Path

from qassistant.gui.workspaceview import WorkspaceView

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox, QDialogButtonBox, QPushButton, QTextEdit

from qassistant.agent.common import TextContent
from qassistant.gui.settings import Settings, SettingsDlg, SettingsView
from qassistant.gui.widgets import TextContentView


class TestSettings(unittest.TestCase):
    """
    Validate Settings model behavior.
    """

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_copy_from_and_reset(self):
        """
        Copying preserves values and reset restores defaults.
        """
        settings = Settings(model="gpt-4.1", available_models=["gpt-4.1", "gpt-5"])
        other = Settings(model="gpt-5", available_models=["gpt-5", "gpt-4.1", "gpt-5"])

        settings.copyFrom(other)

        self.assertEqual(settings.model, "gpt-5")
        self.assertEqual(settings.available_models, ["gpt-5", "gpt-4.1"])

        settings.reset()

        self.assertEqual(settings.model, "")
        self.assertEqual(settings.available_models, ["gpt-5", "gpt-4.1"])

    def test_workspace_path_persists_via_qsettings(self):
        """
        Setting workspace_path stores it in QSettings and a new Settings instance reads it back.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = Settings()
            settings.workspace_path = tmp_dir

            settings2 = Settings()
            self.assertEqual(settings2.workspace_path, str(Path(tmp_dir)))

    def test_workspace_path_rejects_invalid_directory(self):
        """
        Assigning a non-existent directory does not update the stored path.
        """
        settings = Settings()
        original = settings.workspace_path

        settings.workspace_path = "/this/path/does/not/exist"

        self.assertEqual(settings.workspace_path, original)

    def test_workspace_path_returns_empty_for_stale_path(self):
        """
        If the stored path no longer exists, workspace_path returns empty string.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = Settings()
            settings.workspace_path = tmp_dir

        # tmp_dir is now deleted
        settings2 = Settings()
        self.assertEqual(settings2.workspace_path, "")

    def test_workspace_path_emits_property_changed(self):
        """
        Setting workspace_path fires a property_changed notification.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = Settings()
            received: list[tuple[str, str]] = []
            settings.property_changed.connect(lambda name, value: received.append((name, value)))

            settings.workspace_path = tmp_dir

            self.assertEqual(len(received), 1)
            self.assertEqual(received[0][0], "workspace_path")
            self.assertEqual(received[0][1], tmp_dir)


class TestSettingsWidgets(unittest.TestCase):
    """
    Validate SettingsView and SettingsDlg behavior.
    """

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_settings_view_updates_bound_settings(self):
        """
        Selecting listed models updates the bound Settings object and the combo box is not editable.
        """
        settings = Settings(model="gpt-4.1", available_models=["gpt-4.1", "gpt-5"])
        view = SettingsView(settings)
        combo_box = view.findChild(QComboBox, "modelComboBox")

        self.assertFalse(combo_box.isEditable())
        combo_box.setCurrentText("gpt-5")
        self.application.processEvents()
        self.assertEqual(settings.model, "gpt-5")

        settings.model = "missing-model"
        self.application.processEvents()
        self.assertEqual(combo_box.currentIndex(), -1)
        self.assertEqual(combo_box.currentText(), "")

    def test_dialog_accept_applies_changes(self):
        """
        Accepting the dialog copies the working copy back to the live settings object.
        """
        settings = Settings(model="gpt-4.1", available_models=["gpt-4.1", "gpt-5"])
        dialog = SettingsDlg(settings)
        applied = []
        combo_box = dialog.findChild(QComboBox, "modelComboBox")
        button_box = dialog.findChild(QDialogButtonBox)

        dialog.settingsApplied.connect(lambda: applied.append(True))
        combo_box.setCurrentText("gpt-5")
        button_box.button(QDialogButtonBox.StandardButton.Ok).click()
        self.application.processEvents()

        self.assertEqual(dialog.result(), dialog.DialogCode.Accepted)
        self.assertEqual(settings.model, "gpt-5")
        self.assertEqual(applied, [True])

    def test_dialog_cancel_and_reset_do_not_apply_immediately(self):
        """
        Reset affects only the dialog working copy, and cancel discards pending edits.
        """
        settings = Settings(model="gpt-4.1", available_models=["gpt-4.1", "gpt-5"])
        dialog = SettingsDlg(settings)
        combo_box = dialog.findChild(QComboBox, "modelComboBox")
        reset_button = dialog.findChild(QPushButton, "resetButton")
        button_box = dialog.findChild(QDialogButtonBox)

        combo_box.setCurrentText("gpt-5")
        reset_button.click()
        self.application.processEvents()

        self.assertEqual(combo_box.currentText(), "")
        self.assertEqual(settings.model, "gpt-4.1")

        combo_box.setCurrentText("custom-model")
        self.application.processEvents()
        self.assertEqual(combo_box.currentText(), "")
        button_box.button(QDialogButtonBox.StandardButton.Cancel).click()
        self.application.processEvents()

        self.assertEqual(dialog.result(), dialog.DialogCode.Rejected)
        self.assertEqual(settings.model, "gpt-4.1")


class TestTextContentView(unittest.TestCase):
    """
    Validate text content rendering behavior.
    """

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_markdown_is_rendered_by_default(self):
        """
        Markdown content should render through the rich text document by default.
        """
        view = TextContentView(TextContent(text="**bold**"))
        self.application.processEvents()

        self.assertIsInstance(view, QTextEdit)
        self.assertTrue(view.isReadOnly())
        self.assertEqual(view.verticalScrollBarPolicy(), Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.assertEqual(view.horizontalScrollBarPolicy(), Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.assertIn("font-weight", view.toHtml())
        self.assertIn("bold", view.toPlainText())

    def test_plain_text_does_not_interpret_markdown(self):
        """
        Plain text content should keep markdown markers literal.
        """
        view = TextContentView(TextContent(text="**bold**", format="plain"))
        self.application.processEvents()

        self.assertEqual(view.toPlainText(), "**bold**")

    def test_html_content_is_preserved(self):
        """
        HTML content should be rendered through the HTML API.
        """
        view = TextContentView(TextContent(text="<i>Aborted...</i>", format="html"))
        self.application.processEvents()

        self.assertIn("font-style:italic", view.toHtml().replace(" ", ""))
        self.assertEqual(view.toPlainText(), "Aborted...")


class TestWorkspaceView(unittest.TestCase):
    """
    Validate workspace tree behavior.
    """

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_setWorkspacePath_updates_model_root(self):
        """
        Setting a valid directory updates the model root and emits a change signal.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_path = Path(tmp_dir).resolve()
            view = WorkspaceView()
            received: list[str] = []
            view.workspacePathChanged.connect(received.append)

            updated = view.setWorkspacePath(workspace_path)
            self.application.processEvents()

            self.assertTrue(updated)
            self.assertEqual(Path(view.workspacePath), workspace_path)
            self.assertEqual(Path(view.model.rootPath()), workspace_path)
            self.assertEqual(Path(view.model.filePath(view.treeView.rootIndex())), workspace_path)
            self.assertEqual(received[-1], str(workspace_path))

    def test_setWorkspacePath_rejects_invalid_path(self):
        """
        Invalid directories are ignored and return False.
        """
        view = WorkspaceView()
        original = view.workspacePath

        updated = view.setWorkspacePath(Path("C:/this/path/does/not/exist"))

        self.assertFalse(updated)
        self.assertEqual(view.workspacePath, original)

    def test_fileActivated_emits_selected_path(self):
        """
        Activating an index emits the absolute path from the model.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_path = Path(tmp_dir).resolve()
            test_file = workspace_path / "sample.txt"
            test_file.write_text("hello", encoding="utf-8")

            view = WorkspaceView(workspace_path)
            self.application.processEvents()

            index = view.model.index(str(test_file))
            received: list[str] = []
            view.fileActivated.connect(received.append)

            view._onItemActivated(index)

            self.assertEqual(received, [str(test_file)])
