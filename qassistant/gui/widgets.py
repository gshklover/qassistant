"""
QAssistant GUI widgets: content views, chat history and chat widget.
"""
import math
from PySide6.QtCore import Qt, Signal, QSize, QTimer
from PySide6.QtGui import QPixmap, QFont, QFontMetrics, QPainter, QConicalGradient, QColor, QPen
from PySide6.QtWidgets import (
    QWidget,
    QTextEdit,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QSizePolicy,
    QGridLayout,
    QGroupBox,
    QSpacerItem,
)
import qtawesome
from typing import Callable, Any


from ..agent.common import Content, CodeContent, ImageContent, Message, TableContent, TextContent, SectionContent


class Spinner(QWidget):
    """
    A rotating gradient-ring spinner widget.

    Renders a counter-clockwise-rotating conical gradient ring using ``paintEvent``.
    Call ``start()`` / ``stop()`` to control the animation.

    Parameters
    ----------
    parent:
        Optional parent widget.
    size:
        Diameter of the spinner in pixels (default 40).
    interval:
        Timer interval in milliseconds between animation frames (default 30).
    ring_width:
        Thickness of the ring in pixels (default 4).
    color:
        The solid colour at the leading tip of the gradient (default ``#5080ff``).
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        size: int = 40,
        interval: int = 30,
        ring_width: int = 4,
        color: str = "#5080ff",
    ) -> None:
        super().__init__(parent)
        self._angle: float = 0.0
        self._ring_width = ring_width
        self._color = QColor(color)
        self.setFixedSize(QSize(size, size))
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self._timer = QTimer(self)
        self._timer.setInterval(interval)
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        """
        Start the spinning animation.
        """
        self._timer.start()

    def stop(self) -> None:
        """
        Stop the spinning animation and hide the widget.
        """
        self._timer.stop()

    def _tick(self) -> None:
        """
        Called on timer to update the animation
        """
        self._angle = (self._angle + 6.0) % 360.0
        self.update()

    def paintEvent(self, event) -> None:  # pragma: no cover - UI
        """
        Custom paint event to draw the rotating gradient ring.
        """
        size = min(self.width(), self.height())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Centre the drawing square
        x_off = (self.width() - size) // 2
        y_off = (self.height() - size) // 2
        painter.translate(x_off + size / 2, y_off + size / 2)

        # Rotate so the tip of the gradient leads the counter-clockwise motion
        painter.rotate(self._angle)

        # Conical gradient: tip colour at 0°, fading to transparent at 359°
        gradient = QConicalGradient(0, 0, 0)
        tip = QColor(self._color)
        tip.setAlpha(255)
        fade = QColor(self._color)
        fade.setAlpha(0)
        gradient.setColorAt(0.0, tip)
        gradient.setColorAt(1.0, fade)

        pen_width = self._ring_width
        pen = QPen(gradient, pen_width)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        margin = pen_width / 2 + 1
        rect_size = size - 2 * margin
        painter.drawEllipse(
            int(-rect_size / 2),
            int(-rect_size / 2),
            int(rect_size),
            int(rect_size),
        )
        painter.end()


class ContentView:
    """
    Base interface for rendering content objects.
    """
    def __init__(self, **kwargs) -> None:
        super(ContentView, self).__init__()

    def updateContent(self, content: Any) -> None:  # pragma: no cover - UI
        """
        Populate the view from a content object.
        """
        raise NotImplementedError


class TextContentView(QLabel, ContentView):
    """
    View for `TextContent` using a read-only QTextEdit for wrapping.
    """

    def __init__(self, content: TextContent = None, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        if content:
            self.updateContent(content)

    def updateContent(self, content: TextContent) -> None:
        """
        Update the view with new content.
        """
        self.setText(content.text)


class CodeContentView(QScrollArea, ContentView):
    """
    Simple code view. Syntax highlighting can be added later.

    Features:
    - Horizontal scrolling enabled (X axis scrollable), vertical size fixed to a
      limited number of lines.
    - Narrow horizontal scrollbar without corner buttons and with a rounded slider.
    """

    MAX_VISIBLE_LINES = 10

    def __init__(self, content: CodeContent = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = QLabel()
        self._text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # Use a monospace font for code
        font = QFont("Monospace")
        font.setStyleHint(QFont.TypeWriter)
        self._text.setFont(font)
        self._text.setWordWrap(False)
        self._text.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._text.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.setWidget(self._text)
        # We will control the label size explicitly so the scroll area can show
        # a horizontal scrollbar when needed.
        self.setWidgetResizable(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Narrow, rounded scrollbar without corner buttons
        self.setStyleSheet("""
            QScrollBar:horizontal { height:10px; background: transparent; margin: 0px; }
            QScrollBar::handle:horizontal { background: #9d9d9d; min-width: 20px; border-radius: 5px; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; height: 0px; }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
        """)

        if content:
            self.updateContent(content)

    def updateContent(self, content: CodeContent) -> None:
        """
        Update the content beeing displayed by the view.
        """
        code = content.code or ""
        # Put plain code into the label. QLabel will show newlines.
        self._text.setText(code)

        # Measure text to ensure horizontal scrolling for long lines and compute
        # a fixed height based on MAX_VISIBLE_LINES.
        fm = QFontMetrics(self._text.font())
        lines = code.splitlines() or [""]
        max_line_width = 0
        for line in lines:
            w = fm.horizontalAdvance(line)
            if w > max_line_width:
                max_line_width = w

        # add horizontal padding so the handle isn't tight to text
        padding = 12
        self._text.setMinimumWidth(max_line_width + padding)

        visible_lines = min(len(lines), self.MAX_VISIBLE_LINES)
        height = fm.lineSpacing() * visible_lines + padding
        # Fix the height of the scroll area so vertical scrolling is disabled
        # and the widget appears as a horizontally scrollable code strip.
        self.setFixedHeight(height)


class ImageContentView(QLabel, ContentView):
    """
    View for image content.
    """

    def __init__(self, content: ImageContent = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if content:
            self.updateContent(content)

    def updateContent(self, content: ImageContent) -> None:
        """
        Load image from content and display it. If loading fails, show alt text or placeholder.
        """
        pix = QPixmap(content.path)
        if not pix.isNull():
            self.setPixmap(pix.scaledToWidth(600, Qt.SmoothTransformation))
        else:
            self.setText("[image not available]")


class TableContentView(QTableWidget, ContentView):
    """
    View for table content using QTableWidget.
    """

    def __init__(self, content: TableContent = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if content:
            self.updateContent(content)

    def updateContent(self, content: TableContent) -> None:
        """
        Update content with the specified table data.
        """
        data = content.table
        if not data.shape[0] or not data.shape[1]:
            self.setRowCount(0)
            self.setColumnCount(0)
            return

        self.setRowCount(data.shape[0])
        self.setColumnCount(data.shape[1])
        self.setHorizontalHeaderLabels(data.columns.astype(str).tolist())
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                item = QTableWidgetItem(str(data.iat[i, j]))
                self.setItem(i, j, item)


class SectionContentView(QGroupBox, ContentView):
    """
    Grouped section view that renders nested content.
    """

    def __init__(self, content: SectionContent = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        if content:
            self.updateContent(content)

    def updateContent(self, content: SectionContent):
        """
        Update the section view with title and nested contents.
        """
        self.setTitle(content.title)
        layout = self.layout()

        # Clear existing
        while layout.count():
            w = layout.takeAt(0).widget()
            if w is not None:
                w.setParent(None)

        for child in content.contents:
            view = _make_view(child)
            if view:
                view.updateContent(child)
                layout.addWidget(view)


def _make_view(content: Content, parent: QWidget = None) -> ContentView | None:
    """
    Factory to create a ContentView for given content object.
    """
    if isinstance(content, TextContent):
        return TextContentView(content=content, parent=parent)
    if isinstance(content, CodeContent):
        return CodeContentView(content=content, parent=parent)
    if isinstance(content, ImageContent):
        return ImageContentView(content=content, parent=parent)
    if isinstance(content, TableContent):
        return TableContentView(content=content, parent=parent)
    if isinstance(content, SectionContent):
        return SectionContentView(content=content, parent=parent)
    return None


_ROLE_STYLES = {
    "user": "background-color: #e8e8ff; border-radius: 10px;",
    "assistant": "background-color: #e0e0e0; border-radius: 10px;",
}
_ROLE_STYLE_DEFAULT = "background-color: #444444; border-radius: 10px;"


class ChatMessageView(QWidget):
    """
    Widget to render a single chat message composed of multiple content parts.
    Applies a background colour and rounded corners based on the message role.
    """

    def __init__(self, parent: QWidget | None = None, message: Message = None) -> None:
        super().__init__(parent)
        self._role: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Enable stylesheet-based background painting on a QWidget
        self.setAttribute(Qt.WA_StyledBackground, True)

        # Spinner shown while the message is incomplete (streaming)
        self._spinner = Spinner(parent=self, size=24)
        layout.addWidget(self._spinner, alignment=Qt.AlignLeft)
        self._spinner.hide()

        if message:
            self.updateContent(message)

    def _applyRoleStyle(self, role: str) -> None:
        style = _ROLE_STYLES.get(role, _ROLE_STYLE_DEFAULT)
        self.setStyleSheet(f"ChatMessageView {{ {style} }}")

    def role(self) -> str | None:
        return self._role

    def updateContent(self, message: Message) -> None:
        """
        Replace current contents with the content parts of the given message.
        """
        self._role = message.role
        self._applyRoleStyle(message.role)

        layout = self.layout()
        # clear existing widgets except the spinner
        for i in reversed(range(layout.count())):
            w = layout.itemAt(i).widget()
            if w is not None and w is not self._spinner:
                layout.takeAt(i)
                w.setParent(None)

        for c in message.content:
            view = _make_view(c)
            if view:
                view.updateContent(c)
                layout.insertWidget(layout.indexOf(self._spinner), view)

        # Show spinner (at the bottom) when the message is still streaming
        if message.complete:
            self._spinner.stop()
            self._spinner.hide()
        else:
            self._spinner.show()
            self._spinner.start()

    def appendContent(self, content: Any) -> None:
        """
        Append content incrementally (useful for streaming).
        """
        view = _make_view(content)
        if view:
            view.updateContent(content)
            # Insert before the spinner so it stays at the bottom
            self.layout().insertWidget(self.layout().indexOf(self._spinner), view)


class ChatHistoryView(QScrollArea):
    """
    Scrollable list of chat messages.
    """
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        container = QWidget()
        self._container_layout = QVBoxLayout(container)
        self._container_layout.setAlignment(Qt.AlignTop)
        container.setLayout(self._container_layout)
        self.setWidget(container)
        self._auto_scroll = True

    def appendMessage(self, message_view: ChatMessageView) -> None:
        """
        Add a ChatMessageView to the history, aligned by role:
        user → right, assistant (and others) → left.
        """
        row = QHBoxLayout()
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(0)
        # Limit bubble to 80% of available width
        message_view.setMaximumWidth(9999)  # reset any previous constraint
        spacer = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        if message_view.role() == "user":
            row.addSpacerItem(spacer)
            row.addWidget(message_view, 4)
        else:
            row.addWidget(message_view, 4)
            row.addSpacerItem(spacer)

        # Wrap the row layout in a QWidget so it can live in the container layout
        row_widget = QWidget()
        row_widget.setLayout(row)
        self._container_layout.addWidget(row_widget)
        if self._auto_scroll:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def clear(self) -> None:
        """
        Remove all messages from history.
        """
        while self._container_layout.count():
            w = self._container_layout.takeAt(0).widget()
            if w is not None:
                w.setParent(None)

    def setAutoScroll(self, enabled: bool) -> None:
        """
        Enable or disable auto-scrolling when new messages are added.
        """
        self._auto_scroll = bool(enabled)


class _SendTextEdit(QTextEdit):
    """
    Internal QTextEdit subclass that emits `submitted` on Enter and inserts a newline on Ctrl+Enter.
    """

    submitted = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

    def keyPressEvent(self, event):  # pragma: no cover - UI
        """
        Intercept Enter key presses to emit `submitted` or insert a newline.
        """
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ControlModifier:
                # Ctrl+Enter → insert a literal newline
                self.insertPlainText("\n")
            else:
                # Plain Enter → submit
                self.submitted.emit()
            return
        super().keyPressEvent(event)


class EditBox(QWidget):
    """
    Composite widget containing a QTextEdit and a send button.

    Emits `sendRequested` when the user requests a send (button or Ctrl+Enter).
    """

    sendRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None, sendRequested: Callable = None) -> None:
        super().__init__(parent)

        # Internal multi-line edit with Enter-to-send handling
        self._edit = _SendTextEdit(parent=self)
        self._edit.setPlaceholderText("Type your message here...")
        self._edit.submitted.connect(self._onSendRequested)

        # Set height to roughly 4 lines
        fm = self._edit.fontMetrics()
        approx_height = fm.lineSpacing() * 4 + 12
        self._edit.setFixedHeight(approx_height)

        # Send button inside the edit box widget
        self._send_btn = QPushButton(parent=self, enabled=False, clicked=self._onSendRequested)
        self._send_btn.setFlat(True)
        self._send_btn.setIconSize(QSize(48, 48))
        self._send_btn.setIcon(qtawesome.icon("mdi6.send", active="mdi6.send", color="#808080", disabled="mdi6.robot"))

        # Layout the edit and the button next to each other
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._edit, 0, 0)
        layout.addWidget(self._send_btn, 0, 1)

        # Update button state when text changes
        self._edit.textChanged.connect(self._onTextChanged)

        if sendRequested is not None:
            self.sendRequested.connect(sendRequested)

    def _onTextChanged(self):
        """
        Triggered when the text in the edit box changes.
        """
        active = len(self.getText()) > 0
        self._send_btn.setEnabled(active)

    def _onSendRequested(self):
        """
        Emit sendRequested only if the edit box contains non-empty text.
        """
        text = self.getText()
        if len(text) > 0:
            self.sendRequested.emit(text)

    def getText(self) -> str:
        """
        Return the trimmed text content.
        """
        return self._edit.toPlainText().strip()

    def clearText(self):
        """
        Clear the edit box content.
        """
        self._edit.clear()


class ChatWidget(QWidget):
    """
    Composite widget exposing chat history and input box with send button.
    """

    sendRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None, sendRequested: Callable[[str], None] = None) -> None:
        super().__init__(parent)
        self._messageHistory = ChatHistoryView()
        self._editBox = EditBox(sendRequested=self._onSendRequested)
        
        layout = QGridLayout(self)        
        layout.addWidget(self._messageHistory, 0, 0, 1, 2)
        layout.addWidget(self._editBox, 1, 0, 1, 2)
        layout.setRowStretch(0, 1)

        if sendRequested is not None:
            self.sendRequested.connect(sendRequested)

    def appendMessage(self, message: Message) -> ChatMessageView:
        """
        Create a ChatMessageView for the given message and add it to the history.
        """
        view = ChatMessageView(message=message)
        self._messageHistory.appendMessage(view)
        return view

    def updateMessage(self, message: Message) -> None:
        """
        Update an existing ChatMessageView with new message content. 
        Useful for streaming updates.
        """
        for view in self._messageHistory.findChildren(ChatMessageView):
            if view.role() == message.role:
                view.updateContent(message)

    def _onSendRequested(self, text: str) -> None:
        """
        Emit sendRequested with the current edit box text and clear it.
        """
        self._editBox.clearText()
        self.sendRequested.emit(text)
