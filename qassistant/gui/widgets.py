"""
QAssistant GUI widgets: content views, chat history and chat widget.
"""
from PySide6.QtCore import QPoint, Qt, Signal, QSize, QTimer
from PySide6.QtGui import QKeyEvent, QPixmap, QFont, QFontMetrics, QPainter, QConicalGradient, QColor, QPen, QTextCursor
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
from pathlib import Path
import os
import qtawesome
from typing import Callable, Any, TypeVar, Generic


from ..agent.common import Content, CodeContent, ImageContent, Message, TableContent, TextContent, SectionContent


ContentT = TypeVar("ContentT", bound=Content)  # generic type variable for ContentView


def drawSpinner(
    painter: QPainter,
    size: int,
    width: int,
    angle: float,
    color: QColor = QColor("#5080ff"),
) -> None:
    """
    Draw a rotating conical-gradient ring spinner.
    """
    # Rotate so the tip of the gradient leads the counter-clockwise motion
    painter.rotate(angle)

    # Conical gradient: tip colour at 0°, fading to transparent at 359°
    gradient = QConicalGradient(0, 0, 0)
    tip = QColor(color)
    tip.setAlpha(255)
    fade = QColor(color)
    fade.setAlpha(0)
    gradient.setColorAt(0.0, tip)
    gradient.setColorAt(1.0, fade)

    pen = QPen(gradient, width)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)

    margin = width / 2 + 1
    rect_size = size - 2 * margin
    painter.drawEllipse(
        int(-rect_size / 2),
        int(-rect_size / 2),
        int(rect_size),
        int(rect_size),
    )


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

        drawSpinner(
            painter=painter,
            size=size,
            width=self._ring_width,
            angle=self._angle,
            color=self._color,
        )
        painter.end()


class ContentView(Generic[ContentT]):
    """
    Base interface for rendering content objects.
    """
    def __init__(self, **kwargs) -> None:
        super(ContentView, self).__init__()
        self._content = None

    def updateContent(self, content: ContentT) -> None:  # pragma: no cover - UI
        """
        Populate the view from a content object and register for change notifications.
        """
        if self._content is not None:
            self._content.property_changed.disconnect(self._onContentChanged)
        
        self._content = content

        if self._content is not None:
            self._content.property_changed.connect(self._onContentChanged)
            self._onUpdateContent(self._content)

    def _onContentChanged(self, prop_name: str, prop_value: Any) -> None:
        """
        Handle content property changes by refreshing the view.
        """
        if self._content is not None:
            self._onUpdateContent(self._content)

    def _onUpdateContent(self, content: ContentT) -> None:
        """
        Override in subclasses to update the view when content changes.
        """
        raise NotImplementedError()


class TextContentView(QLabel, ContentView):
    """
    View for `TextContent` using a read-only QTextEdit for wrapping.
    """

    def __init__(self, content: TextContent = None, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setWordWrap(True)
        if content:
            self.updateContent(content)

    def _onUpdateContent(self, content: TextContent) -> None:
        """
        Update the view with new content
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

    def _onUpdateContent(self, content: CodeContent) -> None:
        """
        Update the view from specified content
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

    def _onUpdateContent(self, content: ImageContent) -> None:
        """
        Load image from content and display it.
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

    def _onUpdateContent(self, content: TableContent) -> None:
        """
        Update content with the specified table data
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

    def _onUpdateContent(self, content: SectionContent):
        """
        Update the section view with title and nested contents
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


_ROLE_ICONS = {
    "user": "mdi6.face-man-outline",
    "assistant": "mdi6.robot-outline",
}
_ROLE_ICON_SIZE = 28


class ChatMessageView(QWidget):
    """
    Widget to render a single chat message composed of multiple content parts.
    Applies a background colour and rounded corners based on the message role.
    """

    stopRequested = Signal(object)

    def __init__(self, parent: QWidget | None = None, message: Message = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Content column: holds all content views and the spinner
        self._content_widget = QWidget()
        self._content_widget.setAttribute(Qt.WA_StyledBackground, True)
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(10, 8, 10, 8)
        self._content_layout.setSpacing(4)

        # Spinner shown while the message is incomplete (streaming)
        self._spinner = Spinner(parent=self._content_widget, size=24)
        self._content_layout.addWidget(self._spinner, alignment=Qt.AlignLeft)
        self._spinner.hide()

        # Role icon on the right
        self._icon_label = QLabel()
        self._icon_label.setFixedSize(_ROLE_ICON_SIZE, _ROLE_ICON_SIZE)
        self._icon_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        # Outer layout: content + icon side by side
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 2)
        outer.setSpacing(6)
        outer.addWidget(self._icon_label, 0, Qt.AlignTop)
        outer.addWidget(self._content_widget, 1)

        self._message = message
        if message:
            self.updateContent(message)

    @property
    def message(self) -> Message | None:
        """
        Return the Message object associated with this view, if any.
        """
        return self._message

    def role(self) -> str | None:
        """
        Return message role
        """
        return self._message.role if self._message else None

    def _applyRoleStyle(self, role: str) -> None:
        """
        Apply background color and icon based on the message role.
        """
        style = _ROLE_STYLES.get(role, _ROLE_STYLE_DEFAULT)
        self._content_widget.setStyleSheet(f"QWidget {{ {style} }}")
        icon_name = _ROLE_ICONS.get(role)
        if icon_name:
            pix = qtawesome.icon(icon_name).pixmap(_ROLE_ICON_SIZE, _ROLE_ICON_SIZE)
            self._icon_label.setPixmap(pix)
        else:
            self._icon_label.clear()

    def updateContent(self, message: Message) -> None:
        """
        Replace current contents with the content parts of the given message.
        """
        self._applyRoleStyle(message.role)

        layout = self._content_layout
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
            self._content_layout.insertWidget(self._content_layout.indexOf(self._spinner), view)

    def _onStopRequested(self) -> None:
        """
        Emit a stop request for this message.
        """
        if self._message is not None:
            self.stopRequested.emit(self._message)


class ChatHistoryView(QScrollArea):
    """
    Scrollable list of chat messages.
    """

    resetRequested = Signal()
    stopRequested = Signal(object)

    def __init__(self, parent: QWidget | None = None, stopRequested: Callable[[object], None] = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._container_layout = QVBoxLayout(container)
        self._container_layout.setAlignment(Qt.AlignTop)
        container.setLayout(self._container_layout)
        self.setWidget(container)
        
        if stopRequested is not None:
            self.stopRequested.connect(stopRequested)

    def appendMessage(self, message_view: ChatMessageView) -> None:
        """
        Add a ChatMessageView to the history, aligned by role:
        user → right, assistant (and others) → left.
        """
        row = QHBoxLayout()
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(0)
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
        message_view.stopRequested.connect(self.stopRequested.emit)
        # auto-scroll:
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def clear(self) -> None:
        """
        Remove all messages from history.
        """
        while self._container_layout.count():
            w = self._container_layout.takeAt(0).widget()
            if w is not None:
                w.setParent(None)


class TextAutoCompleter:
    """
    Base text auto-completer interface
    """
    def __init__(self):
        pass

    def complete(self, text: str) -> list[str]:
        """
        Return a list of completion suggestions for the given text.
        """
        return []


class PathAutoCompleter(TextAutoCompleter):
    """
    Auto-completer for filesystem paths. Can be attached to a QTextEdit to provide path suggestions.
    """
    def __init__(self):
        super().__init__()

    def complete(self, text: str) -> list[str]:
        """
        Return path completion candidates for the provided path fragment.
        """
        full_text = text or ""
        if not full_text:
            return []

        delimiters = set(" \t\r\n\"'`()[]{}<>,;|&")
        start = len(full_text)
        while start > 0 and full_text[start - 1] not in delimiters:
            start -= 1

        prefix_text = full_text[:start]
        fragment = full_text[start:]
        if not fragment:
            return []

        has_separator = "/" in fragment
        if has_separator:
            base_part, name_part = fragment.rsplit("/", 1)
            if fragment.startswith("/") and base_part == "":
                base_part = "/"
        else:
            base_part, name_part = ".", fragment

        search_dir = Path(base_part).expanduser()
        if not search_dir.is_absolute():
            search_dir = Path.cwd() / search_dir

        try:
            entries = sorted(search_dir.iterdir(), key=lambda p: p.name)
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return []

        completions: list[str] = []
        for entry in entries:
            if not entry.name.startswith(name_part):
                continue

            if has_separator:
                if base_part == "/":
                    candidate = f"/{entry.name}"
                else:
                    candidate = f"{base_part}/{entry.name}"
            else:
                candidate = entry.name

            if entry.is_dir():
                candidate += "/"
            completions.append(candidate)

        if not completions:
            return []

        if len(completions) == 1:
            return [f"{prefix_text}{completions[0]}"]

        common = os.path.commonprefix(completions)
        if len(common) > len(fragment):
            return [f"{prefix_text}{common}"]
        return []


class _TextEdit(QTextEdit):
    """
    Internal QTextEdit subclass that emits `submitted` on Enter and inserts a newline on Ctrl/Shift+Enter.
    Has auto-completion support that can be used to auto-complete file paths, code snippets, etc...
    """

    submitted = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._completer = PathAutoCompleter()
        self._completion_active = False

    def keyPressEvent(self, event: QKeyEvent):  # pragma: no cover - UI
        """
        Intercept Enter key presses to emit `submitted` or insert a newline.
        """
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ControlModifier or event.modifiers() & Qt.ShiftModifier:
                # Ctrl+Enter → insert a literal newline
                self.insertPlainText("\n")
            else:
                # Plain Enter → submit
                self.submitted.emit()
            return

        if event.key() == Qt.Key_Tab and self._completion_active and self.textCursor().hasSelection():
            self._acceptCompletion()
            return

        # let user delete the selection without triggering new completion immediately:
        if event.key() == Qt.Key_Backspace or event.key() == Qt.Key_Delete:
            super().keyPressEvent(event)
            self._completion_active = False
            return

        super().keyPressEvent(event)

        if event.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier):
            self._completion_active = False
            return

        if event.text():
            self._updateCompletion()
        else:
            self._completion_active = False

    def _currentPathFragment(self) -> tuple[str, int, int]:
        """
        Return the current path-like fragment around cursor and its text range.
        """
        cursor = self.textCursor()
        end = cursor.position()
        text = self.toPlainText()

        delimiters = set(" \t\r\n\"'`()[]{}<>,;|&")
        start = end
        while start > 0 and text[start - 1] not in delimiters:
            start -= 1

        return text[start:end], start, end

    def _updateCompletion(self) -> None:
        """
        Compute and display inline completion as selected text.
        """
        fragment, start, end = self._currentPathFragment()
        if not fragment:
            self._completion_active = False
            return

        candidates = self._completer.complete(fragment)
        if not candidates:
            self._completion_active = False
            return

        suggestion = candidates[0]
        if suggestion == fragment or not suggestion.startswith(fragment):
            self._completion_active = False
            return

        suffix = suggestion[len(fragment):]
        if not suffix:
            self._completion_active = False
            return

        cursor = self.textCursor()
        cursor.setPosition(end)
        cursor.insertText(suffix)
        selection_end = cursor.position()
        cursor.setPosition(end)
        cursor.setPosition(selection_end, QTextCursor.KeepAnchor)
        self.setTextCursor(cursor)
        self._completion_active = True

    def _acceptCompletion(self) -> None:
        """
        Accept currently selected inline completion text.
        """
        cursor = self.textCursor()
        if not cursor.hasSelection():
            self._completion_active = False
            return

        cursor.setPosition(cursor.selectionEnd())
        self.setTextCursor(cursor)
        self._completion_active = False


class SpinnerButton(QPushButton):
    """
    Button that can render an animated spinner ring while in busy state.
    """

    def __init__(self, parent: QWidget | None = None, color: str = "#808080") -> None:
        super().__init__(parent=parent)
        self._busy = False
        self._angle = 0.0
        self._ring_width = 4
        self._color = color
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._onTick)

    @property
    def busy(self) -> bool:
        """
        Return busy state for this button.
        """
        return self._busy

    @busy.setter
    def busy(self, value: bool):
        """
        Toggle busy animation around the button icon.
        """
        self._busy = value
        if value:
            self._timer.start()
        else:
            self._timer.stop()
            self._angle = 0.0
        self.update()

    def _onTick(self) -> None:
        """
        Advance spinner angle and repaint.
        """
        self._angle = (self._angle + 6.0) % 360.0
        self.update()

    def paintEvent(self, event) -> None:  # pragma: no cover - UI
        """
        Paint button and, when busy, a spinner ring around its icon.
        """
        super().paintEvent(event)
        if not self._busy:
            return

        size = min(self.width(), self.height()) - 4
        if size <= 8:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.rect().center() + QPoint(1, 1))
        drawSpinner(
            painter=painter,
            size=size,
            width=self._ring_width,
            angle=self._angle,
            color=QColor(self._color),
        )
        painter.end()


class EditBox(QWidget):
    """
    Composite widget containing a QTextEdit and a send button.

    Emits `sendRequested` when the user requests a send (button or Ctrl+Enter).
    Emits `stopRequested` when the stop button is clicked while busy.
    """

    sendRequested = Signal(str)
    stopRequested = Signal()

    def __init__(self, parent: QWidget | None = None, sendRequested: Callable = None) -> None:
        super().__init__(parent)
        self._busy = False

        # Internal multi-line edit with Enter-to-send handling
        self._edit = _TextEdit(parent=self)
        self._edit.setPlaceholderText("Type your message here...")
        self._edit.submitted.connect(self._onButtonClicked)

        # Set height to roughly 4 lines
        fm = self._edit.fontMetrics()
        approx_height = fm.lineSpacing() * 4 + 12
        self._edit.setFixedHeight(approx_height)

        # Single button: shows send icon normally, stop icon when busy
        self._send_btn = SpinnerButton(parent=self)
        self._send_btn.setEnabled(False)
        self._send_btn.setFlat(True)
        self._send_btn.setIconSize(QSize(48, 48))
        self._send_btn.setIcon(qtawesome.icon("mdi6.send", active="mdi6.send", color="#808080", disabled="mdi6.comment-multiple-outline"))
        self._send_btn.clicked.connect(self._onButtonClicked)

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
        self._send_btn.setEnabled(self._busy or len(self.getText()) > 0)

    @property
    def busy(self) -> bool:
        """
        Return True if the widget is in busy state.
        """
        return self._busy

    @busy.setter
    def busy(self, value: bool):
        """
        Set the busy state. When True, the button shows a stop icon and emits ``stopRequested`` on click.
        """
        self._busy = value
        if value:
            self._send_btn.setIcon(qtawesome.icon("mdi6.stop", color="#808080"))
            self._send_btn.setEnabled(True)
            self._send_btn.busy = True
        else:
            self._send_btn.setIcon(qtawesome.icon("mdi6.send", active="mdi6.send", color="#808080", disabled="mdi6.comment-multiple-outline"))
            self._send_btn.setEnabled(len(self.getText()) > 0)
            self._send_btn.busy = False

    def _onButtonClicked(self):
        """
        Emit stopRequested when busy, or sendRequested when text is available.
        """
        if self._busy:
            self.stopRequested.emit()
        else:
            text = self.getText()
            if text:
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
    stopRequested = Signal(object)

    def __init__(self, parent: QWidget | None = None, sendRequested: Callable[[str], None] = None, 
                 stopRequested: Callable[[object], None] = None) -> None:
        super().__init__(parent)
        self._messageHistory = ChatHistoryView(stopRequested=self.stopRequested.emit)
        self._editBox = EditBox(sendRequested=self._onSendRequested)
        self._editBox.stopRequested.connect(lambda: self.stopRequested.emit(None))

        layout = QGridLayout(self)        
        layout.addWidget(self._messageHistory, 0, 0, 1, 2)
        layout.addWidget(self._editBox, 1, 0, 1, 2)
        layout.setRowStretch(0, 1)

        if sendRequested is not None:
            self.sendRequested.connect(sendRequested)

        if stopRequested is not None:
            self.stopRequested.connect(stopRequested)

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
            if view.role() == message.role and view.message is message:
                view.updateContent(message)

    def removeMessage(self, message: Message) -> None:
        """
        Remove the ChatMessageView associated with the given message from the history.
        """
        for view in self._messageHistory.findChildren(ChatMessageView):
            if view.role() == message.role and view.message is message:
                view.setParent(None)
                break

    def _onSendRequested(self, text: str) -> None:
        """
        Emit sendRequested with the current edit box text and clear it.
        """
        self._editBox.clearText()
        self.sendRequested.emit(text)

    @property
    def busy(self) -> bool:
        """
        Return True if the widget is in busy state.
        """
        return self._editBox.busy

    @busy.setter
    def busy(self, value: bool):
        """
        Set the busy state. When True, the submit button is replaced by a spinner and stop button.
        Clicking stop emits ``stopRequested``. While busy, ``sendRequested`` is not emitted.
        """
        self._editBox.busy = value

    def clearHistory(self) -> None:
        """
        Clear all messages from the chat history.
        """
        self._messageHistory.clear()
