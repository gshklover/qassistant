"""
GUI utilities for the application.
"""
from contextlib import contextmanager
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget


@contextmanager
def wait_cursor(widget: QWidget | None = None):
    """
    Override the application cursor with an hourglass while inside the context.
    """
    if QApplication.instance() is None:
        yield
        return

    # set cursor:
    cursor = None
    enabled = True
    if widget is not None:
        cursor = widget.cursor()
        enabled = widget.isEnabled()
        widget.setCursor(Qt.CursorShape.WaitCursor)
        widget.setDisabled(True)
    else:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    try:
        yield
    finally:
        # reset the cursor:
        if widget is None:
            QApplication.restoreOverrideCursor()
        else:
            try:
                if cursor is not None:
                    widget.setCursor(cursor)
                else:
                    widget.unsetCursor()
                widget.setEnabled(enabled)
            except:  # noqa
                # in case the widget was deleted while yielding
                pass
