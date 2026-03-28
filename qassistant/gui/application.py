"""
Application entry for qassistant GUI.
"""
import asyncio
import os
from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QGridLayout, QMainWindow, QTabWidget, QWidget, QPushButton, QHBoxLayout, QMenu, QToolButton
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

    def reset(self) -> None:
        """
        Clear the chat history for this session.
        """
        self._chat_widget.clearHistory()
        asyncio.create_task(self._agent.reset())  # NOTE: this is async task


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

        # Corner widget: menu button + "+" button in the top-right of the tab bar
        corner = QWidget()
        corner_layout = QHBoxLayout(corner)
        corner_layout.setContentsMargins(0, 0, 0, 0)
        corner_layout.setSpacing(2)

        # Context menu button
        self._menu_btn = QToolButton(corner)
        self._menu_btn.setIcon(qtawesome.icon("mdi6.menu"))
        self._menu_btn.setAutoRaise(True)
        self._menu_btn.setToolTip("Options...")
        self._menu_btn.setPopupMode(QToolButton.InstantPopup)
        self._menu_btn.setStyleSheet("QPushButton { padding: 2px; } QToolButton::menu-indicator { image: none; }")
        session_menu = QMenu(self._menu_btn)
        session_menu.addAction(qtawesome.icon("mdi6.refresh"), "Reset").triggered.connect(self._onResetSession)
        self._menu_btn.setMenu(session_menu)

        # New session button
        self._add_tab_btn = QPushButton(qtawesome.icon("mdi6.plus"), "", corner)
        self._add_tab_btn.setFlat(True)
        self._add_tab_btn.setStyleSheet("QPushButton { padding: 2px; }")
        self._add_tab_btn.setToolTip("New session")
        self._add_tab_btn.clicked.connect(self.addSessionTab)

        corner_layout.addWidget(self._menu_btn)
        corner_layout.addWidget(self._add_tab_btn)
        self._tabs.setCornerWidget(corner)

    def _onResetSession(self) -> None:
        """
        Reset the chat history of the currently active session tab.
        """
        widget = self._tabs.currentWidget()
        if isinstance(widget, SessionWidget):
            widget.reset()

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
        self.setDesktopFileName('qassistant')
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
