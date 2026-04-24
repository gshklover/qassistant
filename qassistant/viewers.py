"""
GUI viewers for different types of content.
"""
from __future__ import annotations
import pandas
from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QTableView, QWidget, QFileDialog
import shiboken6
import webbrowser


_refs = []


def _add_ref(widget: QWidget):
    """
    Add a reference to the widget to prevent it from being garbage collected.
    """
    global _refs
    _refs = [ref for ref in _refs if shiboken6.isValid(ref)]  # clean up invalid references
    _refs.append(widget)


def _to_standard_model(data: pandas.DataFrame) -> QStandardItemModel:
    """
    Convert a DataFrame to a QStandardItemModel for display in a QTableView.
    """
    model = QStandardItemModel()
    model.setRowCount(data.shape[0])
    model.setColumnCount(data.shape[1])
    model.setHorizontalHeaderLabels(data.columns.tolist())
    
    for col_idx, col_name in enumerate(data.columns):
        for row_idx, value in enumerate(data[col_name].values):
            item = QStandardItem(value)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)  # make cells read-only
            model.setItem(row_idx, col_idx, item)
    
    return model


def select_file(extensions: list[str] = None) -> str | None:
    """
    Open a file dialog to select a file with the given extensions.
    Returns the selected file path or None if no file was selected.
    """
    filter_str = "Files (" + " ".join(f"*.{ext}" for ext in extensions) + ")" if extensions else "All Files (*)"
    file_path, _ = QFileDialog.getOpenFileName(None, "Select File", "", filter_str)
    return file_path if file_path else None


def display_html(path: str):
    """
    Display specified HTML file in a new window.
    """
    webbrowser.open(path)


def display_table(df: pandas.DataFrame):
    """
    Display a DataFrame as an interactive table in the default web browser.
    """
    table_view = QTableView()
    model = _to_standard_model(df)
    table_view.setModel(model)
    table_view.show()
    _add_ref(table_view)
