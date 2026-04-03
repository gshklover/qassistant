"""
Application entry for qassistant GUI.
"""
import asyncio
from contextlib import contextmanager
from PySide6.QtCore import QSize
from PySide6.QtGui import QAction, Qt
from PySide6.QtWidgets import QApplication, QGridLayout, QMainWindow, QTabWidget, QToolBar, QWidget
import PySide6.QtAsyncio as QtAsyncio
import qtawesome
import traceback


from ..agent import Agent, Message, Role, TextContent, list_models
from .settings import Settings, SettingsDlg
from .widgets import ChatWidget


@contextmanager
def WaitCursor():
    """
    Override the application cursor with an hourglass while inside the context.
    """
    if QApplication.instance() is None:
        yield
        return

    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()


class SessionWidget(QWidget):
    """
    Widget representing a single agent session. 
    """
    def __init__(self, settings: Settings, parent: QWidget = None):
        super().__init__(parent=parent)

        self._settings = settings
        self._agent = Agent(model=settings.model)
        self._chat_widget = ChatWidget(parent=self, sendRequested=self._onSendRequested, stopRequested=self._onStopRequested)
        
        layout = QGridLayout(self)
        layout.addWidget(self._chat_widget, 0, 0)
        layout.setRowStretch(0, 1)
        layout.setColumnStretch(0, 1)
        layout.setContentsMargins(0, 0, 0, 0)

    def _onSendRequested(self, message: str):
        """
        Handle send requests from the UI and route them to the agent.
        """
        # append user message to the chat history:
        self._chat_widget.appendMessage(
            Message(role=Role.USER, content=[TextContent(text=message)])
        )
        
        # create a new placeholder for assisatant response and run the agent:
        response_message = Message(role=Role.ASSISTANT, content=[], complete=False)
        self._chat_widget.appendMessage(response_message)

        self._chat_widget.busy = True
        asyncio.create_task(self._processRequest(message, response_message))

    def _onStopRequested(self):
        """
        Handle stop requests from the UI and signal the agent to stop processing.
        """
        if self._agent.running:
            asyncio.create_task(self._agent.abort())

    async def _processRequest(self, message: str, response_message: Message):
        """
        Handle the agent response and update the UI in real-time.
        """
        if not self._agent.running:
            await self._agent.start()

        try:
            response = await self._agent.send(message=message)
        except Exception as exc:
            # error handling:
            traceback.print_exc()
            response_message.content.append(TextContent(text=f"Error: {exc}"))
        else:
            if response is None:
                # aborted response:
                response_message.content.append(TextContent(text="<i>Aborted...</i>", format='html'))
            else:
                # regular response:
                response_message.content.append(TextContent(text=response.data.content))

        response_message.complete = True
        self._chat_widget.updateMessage(response_message)
        self._chat_widget.busy = False

    def reset(self) -> None:
        """
        Clear the chat history for this session.
        """
        self._chat_widget.clearHistory()
        asyncio.create_task(self._agent.reset())  # NOTE: this is async task

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
        self.setCentralWidget(self._tabs)
        self._tool_bar = QToolBar("Main", self)
        self._tool_bar.setMovable(False)
        self._tool_bar.setIconSize(QSize(32, 32))
        self.addToolBar(self._tool_bar)

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
            with WaitCursor():
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

    def addSessionTab(self) -> ChatWidget:
        """
        Add a new session tab to the main window.
        """
        n = self._tabs.count() + 1
        chat_widget = SessionWidget(settings=self._settings, parent=self._tabs)
        self._tabs.addTab(chat_widget, qtawesome.icon('mdi6.comment-multiple-outline'), f"Session {n}")
        self._tabs.setCurrentWidget(chat_widget)
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

        self.setApplicationName("qassistant")
        self.setDesktopFileName('qassistant')
        self.setApplicationVersion("0.0.1")
        self.setWindowIcon(qtawesome.icon("mdi6.comment-multiple-outline"))

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
