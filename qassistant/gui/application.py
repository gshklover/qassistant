"""
Application entry for qassistant GUI.
"""
import asyncio
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QGridLayout, QMainWindow, QTabWidget, QWidget, QPushButton
import PySide6.QtAsyncio as QtAsyncio
import qtawesome
import traceback


from ..agent import Agent, Message, Role, TextContent
from .widgets import ChatWidget


class SessionWidget(QWidget):
    """
    Widget representing a single agent session. 
    """
    def __init__(self, parent: QWidget = None):
        super().__init__(parent=parent)

        self._agent = Agent()
        self._chat_widget = ChatWidget(parent=self, sendRequested=self._onSendRequested)
        
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

        asyncio.create_task(self._processRequest(message, response_message))

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
            # regular response:
            response_message.content.append(TextContent(text=response.data.content))

        response_message.complete = True
        self._chat_widget.updateMessage(response_message)


class MainWindow(QMainWindow):
    """
    Main window for qassistant GUI. This is a placeholder for the actual UI.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setWindowTitle("qassistant")
        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.tabCloseRequested.connect(self._onTabCloseRequested)
        self.setCentralWidget(self._tabs)

        # "+" button pinned to the right side of the tab bar
        self._add_tab_btn = QPushButton(qtawesome.icon("mdi6.plus"), "", self._tabs)
        self._add_tab_btn.setFlat(True)
        self._add_tab_btn.setStyleSheet("QPushButton { padding: 2px; }")
        self._add_tab_btn.setToolTip("New session")
        self._add_tab_btn.clicked.connect(self.addSessionTab)
        self._tabs.setCornerWidget(self._add_tab_btn)

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
        chat_widget = SessionWidget(parent=self._tabs)
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
        self.setDesktopFileName("qassistant")  # TODO: need to create a .desktop file with an icon path for this to work properly
        self.setApplicationVersion("0.0.1")
        self.setWindowIcon(qtawesome.icon("mdi6.comment-multiple-outline"))

        self.main_window = MainWindow()
        self.main_window.resize(800, 600)
        self.main_window.addSessionTab()
        self.main_window.show()


def run_app():
    """
    Run the Qt application. This is intentionally minimal for scaffolding.
    """
    app = Application()
    QtAsyncio.run()
