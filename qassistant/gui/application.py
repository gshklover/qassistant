"""
Application entry for qassistant GUI.
"""

import asyncio
import sys
import traceback
from contextlib import contextmanager
from typing import Optional

import PySide6.QtAsyncio as QtAsyncio
import qtawesome
from PySide6.QtCore import QSize, Signal, QTimer
from PySide6.QtGui import QAction, Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .._version import __version__
from ..agent import AgentAPI, CustomAgentConfig, SessionEventHandler, Message, Role, Session, TextContent, load_agents
from ..agent.common import MessageState, ToolCallContent
from .settings import Settings, SettingsDlg
from .widgets import ChatWidget, UsagePieWidget


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


class _SessionStreamHandler(SessionEventHandler):
    """
    Bridge agent streaming events to the owning SessionWidget.
    """

    def __init__(self, widget: "SessionWidget"):
        self._widget = widget

    async def on_tool_execution_start(
        self, tool_name, arguments, tool_call_id, interaction_id
    ):
        self._widget._onToolExecutionStart(tool_name, arguments, tool_call_id)
        self._widget._onMessageStateChanged(MessageState.EXECUTING)

    async def on_tool_execution_complete(
        self, tool_call_id, success, result, error, interaction_id
    ):
        self._widget._onToolExecutionComplete(tool_call_id, success, result, error)
        self._widget._onMessageStateChanged(MessageState.PROCESSING)

    async def on_assistant_message_delta(
        self, delta_content, message_id, interaction_id
    ):
        if delta_content:
            self._widget._onMessageDelta(delta_content)

    async def on_assistant_message(
        self, content, message_id, interaction_id, reasoning_text, tool_requests
    ):
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

    def __init__(
        self,
        settings: Settings,
        parent: Optional[QWidget] = None,
        workspace_path: str = "",
        session_id: str = "",
    ):
        super().__init__(parent=parent)

        self._settings = settings
        self._stream_handler = _SessionStreamHandler(self)
        self._agent = Session(
            model=settings.model,
            event_handlers=[self._stream_handler],
            workspace_path=workspace_path,
        )
        self._chat_widget = ChatWidget(
            parent=self,
            sendRequested=self._onSendRequested,
            stopRequested=self._onStopRequested,
        )
        self._response_message: Message | None = None
        self._session_id = session_id
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

    @property
    def sessionId(self) -> str:
        """
        Return bound session id for resumed sessions, if any.
        """
        return self._session_id

    def setAgents(self, agents: list[CustomAgentConfig], agent: str):
        """
        Apply the specified list of custom agents and current agent selection.
        """
        asyncio.create_task(self._agent.set_agents(agents, agent))

    def _onSendRequested(self, message: str):
        """
        Handle send requests from the UI and route them to the agent.
        """
        # append user message to the chat history:
        self._chat_widget.appendMessage(
            Message(role=Role.USER, content=[TextContent(text=message)])
        )

        # create a new placeholder for assisatant response and run the agent:
        self._response_message = Message(
            role=Role.ASSISTANT, content=[], state=MessageState.PROCESSING
        )
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
            await self._agent.start(session_id=self._session_id or None)
        try:
            await self._agent.submit(message=message)  # may throw if session was killed
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._finalizeResponse()

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

        last_content = (
            self._response_message.content[-1]
            if self._response_message.content
            else None
        )
        if isinstance(last_content, TextContent):
            return last_content

        response_text = TextContent(text="")
        self._response_message.append(response_text)
        return response_text

    def _onToolExecutionStart(
        self, tool_name: str | None, arguments, tool_call_id: str | None
    ):
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

    def _onToolExecutionComplete(
        self, tool_call_id: str | None, success: bool | None, result, error
    ):
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
        asyncio.create_task(self._resetAsync())
        self._tool_call_content_map.clear()
        self.workspaceChanged.emit(self.workspacePath)

    async def _resetAsync(self):
        """
        Abort any active turn before resetting the session.
        """
        if self._agent.running:  # this is not checking if there is current turn active, it check is the session was started
            await self._agent.abort()
        await self._agent.reset()

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


class SessionListWidget(QWidget):
    """
    Side-panel widget used to manage and select session tabs.
    Reflects the sessions available via AgentAPI and uses its Qt signals for live updates.
    """

    openSessionRequested = Signal(str)

    def __init__(self, api: AgentAPI, parent: Optional[QWidget] = None):
        super().__init__(parent=parent)
        self.setObjectName("sessionListWidget")
        self._api = api
        self._sessions: dict[str, Any] = {}

        api.sessionCreated.connect(self._onApiSessionCreated)
        api.sessionDeleted.connect(self._onApiSessionDeleted)
        api.sessionUpdated.connect(self._onApiSessionUpdated)

        self._delete_button = QPushButton(
            qtawesome.icon("mdi6.delete-outline", color="darkred"), "", self
        )
        self._delete_button.clicked.connect(self._onDeleteSessionClicked)
        self._delete_button.setToolTip("Delete the selected session")
        self._list_widget = QListWidget(parent=self)
        self._list_widget.setObjectName("sessionListItems")
        self._list_widget.setAlternatingRowColors(True)
        self._list_widget.currentRowChanged.connect(self._onCurrentRowChanged)
        self._list_widget.itemDoubleClicked.connect(self._onItemDoubleClicked)

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addStretch()
        button_layout.addWidget(self._delete_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        layout.addLayout(button_layout)
        layout.addWidget(self._list_widget)

        self._updateDeleteButton()

        # Load known sessions once the Qt event loop starts.
        QTimer.singleShot(0, self._scheduleLoadSessions)

    def _scheduleLoadSessions(self):
        """
        Schedule asynchronous loading of existing sessions if an event loop is available.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.loadSessions())
        except RuntimeError:
            pass

    async def loadSessions(self):
        """
        Load existing sessions from the Copilot API and refresh the list.
        """
        try:
            sessions = await self._api.list_sessions()
        except Exception:
            traceback.print_exc()
            return

        self._sessions = {}
        self._list_widget.blockSignals(True)
        self._list_widget.clear()

        icon = qtawesome.icon("mdi6.comment-multiple-outline")
        for session in sessions:
            session_id = session.sessionId
            summary = session.summary or ""

            if '\n' in summary:
                summary = summary.split('\n', 1)[0] + "..."  # use only the first line of the summary for display

            self._sessions[session_id] = session
            item = QListWidgetItem(icon, str(summary))
            item.setData(Qt.ItemDataRole.UserRole, session_id)
            self._list_widget.addItem(item)

        self._list_widget.blockSignals(False)
        self._updateDeleteButton()

    def indexForSessionId(self, session_id: str) -> int:
        """
        Return the current row index for the given session id, or -1 if not found.
        """
        for i in range(self._list_widget.count()):
            item = self._list_widget.item(i)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == session_id:
                return i
        return -1

    def removeSessionById(self, session_id: str):
        """
        Remove a session entry by its id.
        """
        index = self.indexForSessionId(session_id)
        if index >= 0:
            self._list_widget.takeItem(index)
            self._sessions.pop(session_id, None)
            self._updateDeleteButton()

    def sessionTitle(self, session_id: str) -> str:
        """
        Return display title for a known session id.
        """
        index = self.indexForSessionId(session_id)
        if index >= 0:
            item = self._list_widget.item(index)
            if item is not None:
                return item.text()
        return session_id

    def count(self) -> int:
        """
        Return the number of listed sessions.
        """
        return self._list_widget.count()

    def item(self, index: int) -> QListWidgetItem | None:
        """
        Return a session list item by index.
        """
        return self._list_widget.item(index)

    def currentRow(self) -> int:
        """
        Return the currently selected session row.
        """
        return self._list_widget.currentRow()

    def _onCurrentRowChanged(self, index: int):
        """
        Refresh state when selection changes.
        """
        self._updateDeleteButton()

    def _onItemDoubleClicked(self, item: QListWidgetItem) -> None:
        """
        Open the specific session item that was double-clicked.
        """
        session_id = item.data(Qt.ItemDataRole.UserRole)
        if session_id:
            self.openSessionRequested.emit(str(session_id))

    def _onDeleteSessionClicked(self):
        """
        Delete the currently selected session via the API.
        """
        index = self._list_widget.currentRow()
        if index < 0:
            return

        item = self._list_widget.item(index)
        if item is None:
            return

        session_id = item.data(Qt.ItemDataRole.UserRole)
        if not session_id:
            return

        # Remove from UI immediately; signal handler will clean up if needed.
        self._list_widget.takeItem(index)
        self._sessions.pop(session_id, None)
        self._updateDeleteButton()
        asyncio.create_task(self._asyncDeleteSession(session_id))

    async def _asyncDeleteSession(self, session_id: str):
        """
        Call the API to delete the specified session.
        """
        try:
            await self._api.delete_session(session_id)
        except Exception:
            traceback.print_exc()

    def _onApiSessionCreated(self, session_id: str):
        """
        Reload the session list when a new session is created.
        """
        self._scheduleLoadSessions()

    def _onApiSessionDeleted(self, session_id: str):
        """
        Remove the session from the list when it is deleted externally.
        """
        self.removeSessionById(session_id)

    def _onApiSessionUpdated(self, session_id: str):
        """
        Reload the session list when a session is updated.
        """
        self._scheduleLoadSessions()

    def _updateDeleteButton(self):
        """
        Enable deletion only when a session is selected and more than one exists.
        """
        current_row = self._list_widget.currentRow()
        can_delete = self._list_widget.count() > 0 and current_row >= 0
        self._delete_button.setEnabled(can_delete)


class MainWindow(QMainWindow):
    """
    Main window for qassistant GUI. This is a placeholder for the actual UI.
    """

    def __init__(self, api: AgentAPI, **kwargs):
        super().__init__(**kwargs)
        self.setWindowTitle("qassistant")
        self._api = api
        self._settings = Settings()
        self._tab_session_ids: list[str] = []
        self._session_list_widget = SessionListWidget(api=self._api, parent=self)
        self._session_list_widget.openSessionRequested.connect(self._onSessionOpenRequested)
        api.sessionDeleted.connect(self._onApiSessionDeleted)
        self._session_dock = QDockWidget("Sessions", self)
        self._session_dock.setObjectName("sessionDockWidget")
        self._session_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self._session_dock.setWidget(self._session_list_widget)
        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.tabCloseRequested.connect(self._onTabCloseRequested)
        self._tabs.currentChanged.connect(self._onCurrentTabChanged)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._session_dock)
        self.setCentralWidget(self._tabs)
        self._tool_bar = QToolBar("Main", self)
        self._tool_bar.setMovable(False)
        self._tool_bar.setIconSize(QSize(32, 32))
        self.addToolBar(self._tool_bar)

        self._toggle_session_list_action = QAction(
            qtawesome.icon("mdi6.dock-left", color="#404040"),
            "Sessions",
            self,
            checkable=True,
            checked=True,
            toolTip="Show or hide the sessions side panel",
        )
        self._toggle_session_list_action.toggled.connect(self._onSessionDockToggleRequested)
        self._tool_bar.addAction(self._toggle_session_list_action)
        self._session_dock.visibilityChanged.connect(self._onSessionDockVisibilityChanged)

        spacer = QWidget(self._tool_bar)
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._tool_bar.addWidget(spacer)

        self._new_session_action = QAction(
            qtawesome.icon("mdi6.plus", color="darkgreen"), "New Session", self
        )
        self._new_session_action.setToolTip("Open a new session tab")
        self._new_session_action.triggered.connect(self.addSessionTab)
        self._tool_bar.addAction(self._new_session_action)

        self._reset_session_action = QAction(
            qtawesome.icon("mdi6.refresh", color="darkred"), "Reset", self
        )
        self._reset_session_action.setToolTip("Reset the current session")
        self._reset_session_action.triggered.connect(self._onResetSession)
        self._tool_bar.addAction(self._reset_session_action)

        self._tool_bar.addSeparator()

        self._settings_action = QAction(
            qtawesome.icon("mdi6.cog-outline", color="#404040"), "Settings", self
        )
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

        # agent selection:
        self._agent_combo = QComboBox(
            parent=self._status_widget,
            editable=False,
            toolTip="Select agent",
        )
        self._agent_combo.setMinimumWidth(120)
        self._agent_combo.mousePressEvent = self._onAgentComboClicked
        self._agent_combo.currentTextChanged.connect(self._onAgentSelectionChanged)
        self._agents: list[CustomAgentConfig] = []
        status_layout.addWidget(self._agent_combo)

        # usage indicator:
        self._usage_indicator = UsagePieWidget(parent=self._status_widget)
        status_layout.addWidget(self._usage_indicator)

        self._status_bar.addWidget(self._status_widget, 1)

        self._setWorkspaceStatusText("Workspace: <unknown>")
        self._loadAgents()

    def _onSessionDockToggleRequested(self, checked: bool) -> None:
        """
        Toggle session side panel hidden state from the toolbar button.
        """
        self._session_dock.setHidden(not checked)

    def _onSessionDockVisibilityChanged(self, _: bool) -> None:
        """
        Keep the toolbar toggle checked state synchronized with dock hidden state.
        """
        checked = not self._session_dock.isHidden()
        if self._toggle_session_list_action.isChecked() != checked:
            self._toggle_session_list_action.blockSignals(True)
            self._toggle_session_list_action.setChecked(checked)
            self._toggle_session_list_action.blockSignals(False)

    def _onResetSession(self) -> None:
        """
        Reset the chat history of the currently active session tab.
        """
        widget = self._tabs.currentWidget()
        if isinstance(widget, SessionWidget):
            widget.reset()

    def _onAgentComboClicked(self, event):
        """
        Refresh agent list before showing the dropdown.
        """
        self._loadAgents()
        QComboBox.mousePressEvent(self._agent_combo, event)

    def _loadAgents(self):
        """
        Load available agent definitions and populate the agent combo box.
        """
        try:
            self._agents = load_agents()
        except Exception:
            traceback.print_exc()
            self._agents = []

        current = self._agent_combo.currentText()
        self._agent_combo.blockSignals(True)
        self._agent_combo.clear()
        self._agent_combo.addItem("(default)")
        for agent in self._agents:
            name = agent.get("name", "unnamed")
            icon_name = agent.get("icon", "")
            if icon_name:
                try:
                    self._agent_combo.addItem(qtawesome.icon(icon_name), name)
                except Exception:
                    self._agent_combo.addItem(name)
            else:
                self._agent_combo.addItem(name)

        index = self._agent_combo.findText(current)
        if index >= 0:
            self._agent_combo.setCurrentIndex(index)
        self._agent_combo.blockSignals(False)

    def _onAgentSelectionChanged(self, text: str):
        """
        Apply agent selection change to the active session.
        """
        agent_name = "" if text == "(default)" else text
        widget = self._tabs.currentWidget()
        if isinstance(widget, SessionWidget):
            widget.setAgents(self._agents, agent_name)

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
                models = [model.id for model in await self._api.list_models()]
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
        if self._tabs.count() <= 1:
            return

        self._removeTab(index)

    def _removeTab(self, index: int):
        """
        Remove a tab at the specified index and keep session id mapping aligned.
        """
        widget = self._tabs.widget(index)
        self._tabs.removeTab(index)
        if 0 <= index < len(self._tab_session_ids):
            self._tab_session_ids.pop(index)
        if widget is not None:
            widget.deleteLater()
        self._updateStatusBar()

    def _onCurrentTabChanged(self, _: int) -> None:
        """
        Refresh status bar when active tab changes.
        """
        self._updateStatusBar()

    def _onSessionOpenRequested(self, session_id: str) -> None:
        """
        Open or focus a tab bound to the selected session id.
        """
        try:
            index = self._tab_session_ids.index(session_id)
        except ValueError:
            title = self._session_list_widget.sessionTitle(session_id)
            self.addSessionTab(session_id=session_id, title=title)
            return

        if index != self._tabs.currentIndex():
            self._tabs.setCurrentIndex(index)

    def _onApiSessionDeleted(self, session_id: str) -> None:
        """
        Close any open tab whose session was deleted via the API.
        """
        try:
            index = self._tab_session_ids.index(session_id)
        except ValueError:
            return
        self._removeTab(index)

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

    def addSessionTab(self, session_id: str = "", title: str = "") -> ChatWidget:
        """
        Add a new session tab to the main window.
        """
        n = self._tabs.count() + 1
        tab_title = title or f"Session {n}"
        stored_path = self._settings.workspace_path
        chat_widget = SessionWidget(
            settings=self._settings,
            parent=self._tabs,
            workspace_path=stored_path,
            session_id=session_id,
        )
        chat_widget.workspaceChanged.connect(self._onSessionWorkAreaChanged)
        chat_widget.usageChanged.connect(self._onSessionUsageChanged)
        self._tabs.addTab(
            chat_widget,
            qtawesome.icon("mdi6.comment-multiple-outline"),
            tab_title,
        )
        self._tab_session_ids.append(session_id)
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
        self.setDesktopFileName("qassistant")
        self.setApplicationVersion(__version__)
        self.setWindowIcon(qtawesome.icon("mdi6.comment-multiple-outline", size=64))

        self._api = AgentAPI()
        self.main_window = MainWindow(api=self._api)
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

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "com.qassistant.app"
        )

    app = Application()  # noqa
    QtAsyncio.run()
