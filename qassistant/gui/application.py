"""
Application entry for qassistant GUI.
"""
import asyncio
from contextlib import contextmanager
import sys
from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QAction, Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QPushButton, QStatusBar, QTabWidget, QToolBar, \
    QWidget, QSizePolicy
import PySide6.QtAsyncio as QtAsyncio
import qtawesome
import traceback

from ..agent import Agent, AgentEventHandler, Message, Role, TextContent, list_models
from .settings import Settings, SettingsDlg
from .._version import __version__
from .widgets import ChatWidget, UsagePieWidget
from ..agent.common import MessageState, ToolCallContent


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

    async def on_tool_execution_start(self, tool_name, arguments, tool_call_id, interaction_id):
        self._widget._onToolExecutionStart(tool_name, arguments, tool_call_id)
        self._widget._onMessageStateChanged(MessageState.EXECUTING)

    async def on_tool_execution_complete(self, tool_call_id, success, result, error, interaction_id):
        self._widget._onToolExecutionComplete(tool_call_id, success, result, error)
        self._widget._onMessageStateChanged(MessageState.PROCESSING)

    async def on_assistant_message_delta(self, delta_content, message_id, interaction_id):
        if delta_content:
            self._widget._onMessageDelta(delta_content)

    async def on_assistant_message(self, content, message_id, interaction_id, reasoning_text, tool_requests):
        if content:
            self._widget._setAssistantMessage(content)

    async def on_assistant_turn_end(self, turn_id):
        # will be handled by assistant idle
        # self._widget._finalizeResponse()
        pass

    async def on_session_idle(self, background_tasks):
        self._widget._onSessionIdle()

    async def on_session_error(self, error_type, message, error, status_code, url):
        details = message or str(error) or "unknown session error"
        self._widget._onSessionError(details)

    async def on_session_usage(self, usage_percentage):
        self._widget._onSessionUsage(usage_percentage)


class SessionWidget(QWidget):
    """
    Widget representing a single agent session.
    """

    workspaceChanged = Signal(str)
    usageChanged = Signal(float)

    def __init__(self, settings: Settings, parent: QWidget = None, workspace_path: str = ""):
        super().__init__(parent=parent)

        self._settings = settings
        self._stream_handler = _SessionStreamHandler(self)
        self._agent = Agent(model=settings.model, event_handlers=[self._stream_handler], workspace_path=workspace_path)
        self._chat_widget = ChatWidget(parent=self, sendRequested=self._onSendRequested, stopRequested=self._onStopRequested)
        self._response_message: Message | None = None
        self._tool_call_content_map: dict[str, ToolCallContent] = {}
        self._has_delta_content = False
        self._aborted = False

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

    @property
    def usage(self) -> float:
        """
        Return the current context window usage percentage for this session.
        """
        return self._agent.usage

    def _onSendRequested(self, message: str):
        """
        Handle send requests from the UI and route them to the agent.
        """
        # append user message to the chat history:
        self._chat_widget.appendMessage(
            Message(role=Role.USER, content=[TextContent(text=message)])
        )

        # create a new placeholder for assisatant response and run the agent:
        self._response_message = Message(role=Role.ASSISTANT, content=[], state=MessageState.PROCESSING)
        self._chat_widget.appendMessage(self._response_message)
        self._has_delta_content = False
        self._aborted = False

        self._chat_widget.busy = True
        asyncio.create_task(self.submit(message))

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

    def _onMessageStateChanged(self, state: MessageState) -> None:
        """
        Update the state of the active response message and refresh the chat view.
        """
        if self._response_message is None:
            return

        self._response_message.state = state
        self._chat_widget.updateMessage(self._response_message)

    def _onMessageDelta(self, delta: str) -> None:
        """
        Append streamed assistant delta content to the active response message.
        """
        if self._response_message is None:
            return

        response_text = self._ensureTrailingTextContent()
        self._has_delta_content = True
        response_text.text += delta
        self._chat_widget.updateMessage(self._response_message)

    def _setAssistantMessage(self, content: str) -> None:
        """
        Apply final assistant message content when no deltas were produced.
        """
        if self._response_message is None:
            return

        if not self._has_delta_content:
            response_text = self._ensureTrailingTextContent()
            response_text.text = content
            self._chat_widget.updateMessage(self._response_message)

    def _ensureTrailingTextContent(self) -> TextContent:
        """
        Return the trailing text content block, creating one at the end if needed.
        """
        if self._response_message is None:
            raise RuntimeError("No active response message")

        last_content = self._response_message.content[-1] if self._response_message.content else None
        if isinstance(last_content, TextContent):
            return last_content

        response_text = TextContent(text="")
        self._response_message.append(response_text)
        return response_text

    def _onToolExecutionStart(self, tool_name: str | None, arguments, tool_call_id: str | None):
        """
        Append a tool call content block when tool execution starts.
        """
        if self._response_message is None:
            return

        tool_call_content = ToolCallContent(
            tool_name=tool_name or "",
            arguments=str(arguments) if arguments is not None else "",
            result="<running>",
        )
        self._response_message.append(tool_call_content)
        if tool_call_id:
            self._tool_call_content_map[tool_call_id] = tool_call_content
        # Next streamed delta should become a new text chunk after this tool call.
        self._chat_widget.updateMessage(self._response_message)

    def _onToolExecutionComplete(self, tool_call_id: str | None, success: bool | None, result, error):
        """
        Update tool call content with completion result.
        """
        if self._response_message is None:
            return

        tool_call_content = self._tool_call_content_map.get(tool_call_id or "")
        if tool_call_content is None:
            tool_call_content = ToolCallContent(tool_name="", arguments="", result="")
            self._response_message.append(tool_call_content)

        if success:
            tool_call_content.result = str(result) if result is not None else ""
        else:
            tool_call_content.result = str(error) if error is not None else "<failed>"

        self._chat_widget.updateMessage(self._response_message)

    def _onSessionError(self, details: str) -> None:
        """
        Render session errors inside the active assistant message.
        """
        if self._response_message is not None:
            response_text = self._ensureTrailingTextContent()
            if response_text.text:
                response_text.text += "\n\n"
            response_text.text += f"Error: {details}"
        self._finalizeResponse()

    def _onSessionUsage(self, usage_percentage: float):
        """
        Forward session usage percentage to listeners.
        """
        self.usageChanged.emit(usage_percentage)

    def _onSessionIdle(self) -> None:
        """
        Session is idle; ensure the current response is finalized.
        """
        if self._response_message is not None:
            self._finalizeResponse()

    def _finalizeResponse(self) -> None:
        """
        Finalize current response and reset busy UI state.
        """
        if self._response_message is None:
            self._chat_widget.busy = False
            return

        if self._aborted:
            response_text = self._ensureTrailingTextContent()
            response_text.text = "<i>Aborted...</i>"
            response_text.format = "html"

        self._response_message.state = MessageState.COMPLETE
        self._chat_widget.updateMessage(self._response_message)
        self._chat_widget.busy = False

        self._response_message = None
        self._tool_call_content_map.clear()
        self._has_delta_content = False
        self._aborted = False

    def reset(self) -> None:
        """
        Clear the chat history for this session.
        """
        self._chat_widget.clearHistory()
        asyncio.create_task(self._agent.reset())  # NOTE: this is async task
        self._tool_call_content_map.clear()
        self.workspaceChanged.emit(self.workspacePath)

    def applySettings(self, settings: Settings) -> None:
        """
        Apply updated application settings to this session.
        """
        self._settings = settings

        if self._agent.model != settings.model:
            self._agent.model = settings.model

    def setWorkspacePath(self, path: str):
        """
        Update the agent workspace path and notify listeners.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._setWorkspacePathAsync(path))
        except RuntimeError:
            asyncio.run(self._setWorkspacePathAsync(path))

    async def _setWorkspacePathAsync(self, path: str):
        """
        Async helper for applying workspace updates.
        """
        await self._agent.set_workspace(path)
        self.workspaceChanged.emit(self.workspacePath)


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

        spacer = QWidget(self._tool_bar)
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._tool_bar.addWidget(spacer)

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

        spacer = QWidget(self._tool_bar)
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._tool_bar.addWidget(spacer)

        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)
        self._status_bar.setSizeGripEnabled(False)

        self._status_widget = QWidget(self._status_bar)
        status_layout = QHBoxLayout(self._status_widget)
        status_layout.setContentsMargins(1, 1, 1, 1)
        status_layout.setSpacing(3)

        self._workspace_button = QPushButton(
            parent=self._status_widget,
            flat=True,
            icon=qtawesome.icon("mdi6.folder-outline", color="#505050"),
            toolTip="Select workspace directory",
            clicked=self._onSelectWorkspaceRequested,
        )
        self._workspace_button.setStyleSheet(
            "QPushButton { background-color: transparent; border: none; padding: 0px; }"
            "QPushButton:hover { background-color: transparent; border: none; }"
            "QPushButton:pressed { background-color: transparent; border: none; }"
        )
        status_layout.addWidget(self._workspace_button)

        self._workspace_status_label = QLabel(self._status_widget)
        self._workspace_status_label.setMinimumWidth(0)
        status_layout.addWidget(self._workspace_status_label, 1)

        self._usage_indicator = UsagePieWidget(parent=self._status_widget)
        status_layout.addWidget(self._usage_indicator)

        self._status_bar.addWidget(self._status_widget, 1)

        self._setWorkspaceStatusText("Workspace: <unknown>")

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
        Update status bar only for the currently selected session and persist the path.
        """
        if self.sender() is self._tabs.currentWidget():
            self._setWorkspaceStatusText(f"Workspace: {path}")
        if path:
            self._settings.workspace_path = path

    def _onSessionUsageChanged(self, usage_percentage: float):
        """
        Update the usage pie indicator in the status bar when the active session reports context window utilization.
        """
        if self.sender() is self._tabs.currentWidget():
            self._usage_indicator.percentage = usage_percentage

    def _onSelectWorkspaceRequested(self):
        """
        Prompt for a workspace directory and apply it to the active session.
        """
        widget = self._tabs.currentWidget()
        if not isinstance(widget, SessionWidget):
            return

        selected_path = QFileDialog.getExistingDirectory(
            self,
            "Select Workspace Directory",
            widget.workspacePath,
        )
        if not selected_path:
            return

        widget.setWorkspacePath(selected_path)

    def _updateStatusBar(self) -> None:
        """
        Display current session work area and usage in the status bar.
        """
        widget = self._tabs.currentWidget()
        if isinstance(widget, SessionWidget):
            self._setWorkspaceStatusText(f"Workspace: {widget.workspacePath}")
            self._usage_indicator.percentage = widget.usage
        else:
            self._setWorkspaceStatusText("Workspace: <unknown>")
            self._usage_indicator.percentage = 0.0

    def _setWorkspaceStatusText(self, text: str) -> None:
        """
        Update status text rendered by the custom status widget.
        """
        self._workspace_status_label.setText(text)
        self._workspace_status_label.setToolTip(text)

    def addSessionTab(self) -> ChatWidget:
        """
        Add a new session tab to the main window.
        """
        n = self._tabs.count() + 1
        stored_path = self._settings.workspace_path
        chat_widget = SessionWidget(settings=self._settings, parent=self._tabs, workspace_path=stored_path)
        chat_widget.workspaceChanged.connect(self._onSessionWorkAreaChanged)
        chat_widget.usageChanged.connect(self._onSessionUsageChanged)
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

        self.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
        self.setApplicationName("qassistant")
        self.setDesktopFileName('qassistant')
        self.setApplicationVersion(__version__)
        self.setWindowIcon(qtawesome.icon("mdi6.comment-multiple-outline", size=64))

        self.main_window = MainWindow()
        self.main_window.resize(QSize(800, 600))
        self.main_window.addSessionTab()
        self.main_window.show()
        self.main_window.raise_()


def run_app():
    """
    Run the Qt application. This is intentionally minimal for scaffolding.
    """
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('com.qassistant.app')

    app = Application()
    QtAsyncio.run()
