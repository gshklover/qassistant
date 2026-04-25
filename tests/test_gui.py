"""
Unit tests for GUI settings components.
"""
import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from qassistant.gui.workspaceview import WorkspaceView

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox, QDialogButtonBox, QPushButton, QTextEdit

from qassistant.agent import DEFAULT_MODEL
from qassistant.agent.agent import AgentAPI
from qassistant.agent.common import TextContent
from qassistant.gui.application import MainWindow, SessionListWidget
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
        self.assertEqual(settings.available_models, ["gpt-4.1", "gpt-5"])

        settings.reset()

        self.assertEqual(settings.model, DEFAULT_MODEL)
        self.assertEqual(settings.available_models, ["gpt-4.1", "gpt-5"])

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

    def test_workspace_path_updates_without_observer_api(self):
        """
        workspace_path is updated without requiring observable hooks.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = Settings()
            settings.workspace_path = tmp_dir
            self.assertEqual(settings.workspace_path, tmp_dir)


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

        # The view no longer observes external settings changes automatically.
        settings.model = "missing-model"
        self.application.processEvents()
        self.assertEqual(combo_box.currentText(), "gpt-5")

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


class TestMainWindow(unittest.TestCase):
    """
    Validate main window session side-panel behavior.
    """

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    @staticmethod
    def _createApi():
        """
        Return a real AgentAPI backed by a fully mocked CopilotClient so Qt signals work.
        """
        with patch("qassistant.agent.agent.copilot.CopilotClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.on = MagicMock()
            mock_client.get_state = MagicMock(return_value="connected")
            mock_client.list_models = AsyncMock(return_value=[])
            mock_client.list_sessions = AsyncMock(return_value=[])
            mock_client.delete_session = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client
            return AgentAPI()

    def test_toolbar_sessions_toggle_hides_and_shows_dock(self):
        """
        The first toolbar action toggles the sessions dock hidden state.
        """
        window = MainWindow(api=self._createApi())
        try:
            first_action = window._tool_bar.actions()[0]
            self.assertIs(first_action, window._toggle_session_list_action)
            self.assertFalse(window._session_dock.isHidden())

            window._toggle_session_list_action.trigger()
            self.application.processEvents()
            self.assertTrue(window._session_dock.isHidden())

            window._toggle_session_list_action.trigger()
            self.application.processEvents()
            self.assertFalse(window._session_dock.isHidden())
        finally:
            window.close()


class TestSessionListWidget(unittest.TestCase):
    """
    Validate session list side-panel behavior.
    """

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    @staticmethod
    def _createApi():
        """
        Return a real AgentAPI backed by a fully mocked CopilotClient so Qt signals work.
        """
        with patch("qassistant.agent.agent.copilot.CopilotClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.on = MagicMock()
            mock_client.get_state = MagicMock(return_value="connected")
            mock_client.list_sessions = AsyncMock(return_value=[])
            mock_client.delete_session = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client
            return AgentAPI()

    def test_session_list_widget_emits_expected_signals(self):
        """
        The session list widget emits openSessionRequested on item double-click.
        """
        widget = SessionListWidget(api=self._createApi())
        try:
            opened: list[str] = []
            widget.openSessionRequested.connect(opened.append)

            asyncio.run(widget.loadSessions())
            self.application.processEvents()
            self.assertEqual(widget.count(), 0)

            # Manually inject a session straight into the list widget to drive the signal test.
            from PySide6.QtWidgets import QListWidgetItem
            item = QListWidgetItem("Session A")
            item.setData(widget._list_widget.model().index(0, 0).UserRole if False else 256, "sid-A")
            from PySide6.QtCore import Qt as _Qt
            item.setData(_Qt.ItemDataRole.UserRole, "sid-A")
            widget._list_widget.addItem(item)
            widget._list_widget.itemDoubleClicked.emit(item)

            self.assertEqual(opened, ["sid-A"])
        finally:
            widget.close()
