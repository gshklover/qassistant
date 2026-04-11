"""
Application entry for qassistant GUI.
"""
import asyncio
from contextlib import contextmanager
import sys
from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QAction, Qt
from PySide6.QtWidgets import QApplication, QGridLayout, QMainWindow, QStatusBar, QTabWidget, QToolBar, QWidget
import PySide6.QtAsyncio as QtAsyncio
import qtawesome
import traceback


from ..agent import Agent, AgentEventHandler, Message, Role, TextContent, list_models
from .settings import Settings, SettingsDlg
from .widgets import ChatWidget


@contextmanager
def wait_cursor():
    """
    Override the application cursor with an hourglass while inside the context.
    """
    if QApplication.instance() is None:
        yield
        return

    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()


class _SessionStreamHandler(AgentEventHandler):
    """
    Bridge agent streaming events to the owning SessionWidget.
    """

    def __init__(self, widget: "SessionWidget"):
        self._widget = widget

    async def on_assistant_message_delta(self, delta_content, message_id, interaction_id):
        if delta_content:
            self._widget._onMessageDelta(delta_content)

    async def on_assistant_message(self, content, message_id, interaction_id, reasoning_text, tool_requests):
        if content:
            self._widget._setAssistantMessage(content)

    async def on_assistant_turn_end(self, turn_id):
        self._widget._finalizeResponse()

    async def on_session_idle(self, background_tasks):
        self._widget._onSessionIdle()

    async def on_session_error(self, error_type, message, error, status_code, url):
        details = message or str(error) or "unknown session error"
        self._widget._onSessionError(details)


class SessionWidget(QWidget):
    """
    Widget representing a single agent session.
    """

    workspaceChanged = Signal(str)

    def __init__(self, settings: Settings, parent: QWidget = None):
        super().__init__(parent=parent)

        self._settings = settings
        self._stream_handler = _SessionStreamHandler(self)
        self._agent = Agent(model=settings.model, event_handlers=[self._stream_handler])
        self._chat_widget = ChatWidget(parent=self, sendRequested=self._onSendRequested, stopRequested=self._onStopRequested)
        self._response_message: Message | None = None
        self._response_text: TextContent | None = None
        self._has_delta_content = False
        self._aborted = False
        self._prev_workspace_path = self._agent.workspace_path

        layout = QGridLayout(self)
        layout.addWidget(self._chat_widget, 0, 0)
        layout.setRowStretch(0, 1)
        layout.setColumnStretch(0, 1)
        layout.setContentsMargins(0, 0, 0, 0)

    @property
    def workspacePath(self) -> str:
        """
        Return current workspace path for this session.
        """
        return self._agent.workspace_path

    def _onSendRequested(self, message: str):
        """
        Handle send requests from the UI and route them to the agent.
        """
        # append user message to the chat history:
        self._chat_widget.appendMessage(
            Message(role=Role.USER, content=[TextContent(text=message)])
        )

        # create a new placeholder for assisatant response and run the agent:
        self._response_text = TextContent(text="")
        self._response_message = Message(role=Role.ASSISTANT, content=[self._response_text], complete=False)
        self._chat_widget.appendMessage(self._response_message)
        self._has_delta_content = False
        self._aborted = False
        self._prev_workspace_path = self._agent.workspace_path

        self._chat_widget.busy = True
        task = asyncio.create_task(self.submit(message))
        task.add_done_callback(self._onSubmitDone)

    def _onStopRequested(self):
        """
        Handle stop requests from the UI and signal the agent to stop processing.
        """
        if self._agent.running:
            self._aborted = True
            asyncio.create_task(self._agent.abort())

    async def submit(self, message: str):
        """
        Submit user message to the agent and rely on event callbacks for streamed updates.
        """
        if not self._agent.running:
            await self._agent.start()

        await self._agent.submit(message=message)

    def _onSubmitDone(self, task: asyncio.Task) -> None:
        """
        Handle immediate submission failures.
        """
        try:
            task.result()
        except Exception as exc:
            traceback.print_exc()
            self._onSessionError(str(exc))

    def _onMessageDelta(self, delta: str) -> None:
        """
        Append streamed assistant delta content to the active response message.
        """
        if self._response_message is None or self._response_text is None:
            return

        self._has_delta_content = True
        self._response_text.text += delta
        self._chat_widget.updateMessage(self._response_message)

    def _setAssistantMessage(self, content: str) -> None:
        """
        Apply final assistant message content when no deltas were produced.
        """
        if self._response_message is None or self._response_text is None:
            return

        if not self._has_delta_content:
            self._response_text.text = content
            self._chat_widget.updateMessage(self._response_message)

    def _onSessionError(self, details: str) -> None:
        """
        Render session errors inside the active assistant message.
        """
        if self._response_message is not None and self._response_text is not None:
            if self._response_text.text:
                self._response_text.text += "\n\n"
            self._response_text.text += f"Error: {details}"
        self._finalizeResponse()

    def _onSessionIdle(self) -> None:
        """
        Session is idle; ensure the current response is finalized.
        """
        if self._response_message is not None and not self._response_message.complete:
            self._finalizeResponse()

    def _finalizeResponse(self) -> None:
        """
        Finalize current response and reset busy UI state.
        """
        if self._response_message is None:
            self._chat_widget.busy = False
            return

        if self._aborted and self._response_text is not None and not self._response_text.text:
            self._response_text.text = "<i>Aborted...</i>"
            self._response_text.format = "html"

        self._response_message.complete = True
        self._chat_widget.updateMessage(self._response_message)
        self._chat_widget.busy = False

        if self.workspacePath != self._prev_workspace_path:
            self.workspaceChanged.emit(self.workspacePath)

        self._response_message = None
        self._response_text = None
        self._has_delta_content = False
        self._aborted = False

    def reset(self) -> None:
        """
        Clear the chat history for this session.
        """
        self._chat_widget.clearHistory()
        asyncio.create_task(self._agent.reset())  # NOTE: this is async task
        self.workspaceChanged.emit(self.workspacePath)

    def applySettings(self, settings: Settings) -> None:
        """
        Apply updated application settings to this session.
        """
        self._settings = settings

        if self._agent.model != settings.model:
            self._agent.model = settings.model


class MainWindow(QMainWindow):
    """
    Main window for qassistant GUI. This is a placeholder for the actual UI.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setWindowTitle("qassistant")
        self._settings = Settings()
        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.tabCloseRequested.connect(self._onTabCloseRequested)
        self._tabs.currentChanged.connect(self._onCurrentTabChanged)
        self.setCentralWidget(self._tabs)
        self._tool_bar = QToolBar("Main", self)
        self._tool_bar.setMovable(False)
        self._tool_bar.setIconSize(QSize(32, 32))
        self.addToolBar(self._tool_bar)

        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Workspace: <unknown>")

        self._new_session_action = QAction(qtawesome.icon("mdi6.plus", color='darkgreen'), "New Session", self)
        self._new_session_action.setToolTip("Open a new session tab")
        self._new_session_action.triggered.connect(self.addSessionTab)
        self._tool_bar.addAction(self._new_session_action)

        self._reset_session_action = QAction(qtawesome.icon("mdi6.refresh", color='darkred'), "Reset", self)
        self._reset_session_action.setToolTip("Reset the current session")
        self._reset_session_action.triggered.connect(self._onResetSession)
        self._tool_bar.addAction(self._reset_session_action)

        self._tool_bar.addSeparator()

        self._settings_action = QAction(qtawesome.icon("mdi6.cog-outline", color='#404040'), "Settings", self)
        self._settings_action.setToolTip("Open application settings")
        self._settings_action.triggered.connect(self._onSettingsRequested)
        self._tool_bar.addAction(self._settings_action)

    def _onResetSession(self) -> None:
        """
        Reset the chat history of the currently active session tab.
        """
        widget = self._tabs.currentWidget()
        if isinstance(widget, SessionWidget):
            widget.reset()

    def _onSettingsRequested(self) -> None:
        """
        Resolve model list and open the application settings dialog.
        """
        self._settings_action.setEnabled(False)
        asyncio.create_task(self._prepareSettingsDialog())

    async def _prepareSettingsDialog(self):
        """
        Load available model ids prior to opening the settings dialog.
        """
        try:
            with wait_cursor():
                models = await list_models()
            if not models:
                models = [self._settings.model]
        except Exception:
            traceback.print_exc()
            models = [self._settings.model]

        if self._settings.model and self._settings.model not in models:
            models.insert(0, self._settings.model)

        self._settings.available_models = sorted(models)
        self._settings_action.setEnabled(True)

        dialog = SettingsDlg(self._settings, self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.settingsApplied.connect(self._applySettings)
        dialog.exec()

    def _applySettings(self) -> None:
        """
        Apply persisted settings to the main window and active sessions.
        """
        for index in range(self._tabs.count()):
            widget = self._tabs.widget(index)
            if isinstance(widget, SessionWidget):
                widget.applySettings(self._settings)

    def _onTabCloseRequested(self, index: int) -> None:
        """
        Close the tab at the given index. The last tab cannot be closed.
        """
        if self._tabs.count() > 1:
            widget = self._tabs.widget(index)
            self._tabs.removeTab(index)
            if widget is not None:
                widget.deleteLater()
            self._updateStatusBar()

    def _onCurrentTabChanged(self, _: int) -> None:
        """
        Refresh status bar when active tab changes.
        """
        self._updateStatusBar()

    def _onSessionWorkAreaChanged(self, path: str) -> None:
        """
        Update status bar only for the currently selected session.
        """
        if self.sender() is self._tabs.currentWidget():
            self._status_bar.showMessage(f"Workspace: {path}")

    def _updateStatusBar(self) -> None:
        """
        Display current session work area in the status bar.
        """
        widget = self._tabs.currentWidget()
        if isinstance(widget, SessionWidget):
            self._status_bar.showMessage(f"Workspace: {widget.workspacePath}")
        else:
            self._status_bar.showMessage("Workspace: <unknown>")

    def addSessionTab(self) -> ChatWidget:
        """
        Add a new session tab to the main window.
        """
        n = self._tabs.count() + 1
        chat_widget = SessionWidget(settings=self._settings, parent=self._tabs)
        chat_widget.workspaceChanged.connect(self._onSessionWorkAreaChanged)
        self._tabs.addTab(chat_widget, qtawesome.icon('mdi6.comment-multiple-outline'), f"Session {n}")
        self._tabs.setCurrentWidget(chat_widget)
        self._updateStatusBar()
        return chat_widget

    def currentSessionTab(self) -> ChatWidget:
        """
        Get the current session tab widget. This can be used to route messages to the correct session.
        """
        return self._tabs.currentWidget()


class Application(QApplication):
    """
    Main application class for qassistant GUI.
    """
    def __init__(self):
        super().__init__([])

        if sys.platform == "win32":
            import ctypes
            myappid = 'com.qassistant.app'  # arbitrary string
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        self.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
        self.setApplicationName("qassistant")
        self.setDesktopFileName('qassistant')
        self.setApplicationVersion("0.0.1")
        self.setWindowIcon(qtawesome.icon("mdi6.comment-multiple-outline", size=64))

        self.main_window = MainWindow()
        self.main_window.resize(QSize(800, 600))
        self.main_window.addSessionTab()
        self.main_window.show()


def run_app():
    """
    Run the Qt application. This is intentionally minimal for scaffolding.
    """
    app = Application()
    QtAsyncio.run()
