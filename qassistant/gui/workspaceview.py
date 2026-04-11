"""
Workspace viewer using QTreeView and QFileSystemModel
"""

from pathlib import Path

from PySide6.QtCore import QDir, Qt, Signal
from PySide6.QtWidgets import QFileSystemModel, QGridLayout, QTreeView, QWidget


class WorkspaceView(QWidget):
	"""
	Tree view widget that displays the current workspace directory.
	"""

	fileActivated = Signal(str)
	workspacePathChanged = Signal(str)

	def __init__(self, workspacePath: str | Path | None = None, parent: QWidget | None = None):
		super().__init__(parent)
		self._workspacePath = Path.cwd()

		self._model = QFileSystemModel(self)
		self._model.setFilter(QDir.Filter.AllEntries | QDir.Filter.NoDotAndDotDot)
		self._model.setReadOnly(True)

		self._treeView = QTreeView(self)
		self._treeView.setObjectName("workspaceTreeView")
		self._treeView.setModel(self._model)
		self._treeView.setSortingEnabled(True)
		self._treeView.sortByColumn(0, Qt.SortOrder.AscendingOrder)
		self._treeView.setAlternatingRowColors(True)
		self._treeView.setAnimated(True)
		self._treeView.setHeaderHidden(False)
		self._treeView.setColumnHidden(1, True)
		self._treeView.setColumnHidden(2, True)
		self._treeView.setColumnHidden(3, True)
		self._treeView.activated.connect(self._onItemActivated)
		self._treeView.doubleClicked.connect(self._onItemActivated)

		layout = QGridLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.addWidget(self._treeView, 0, 0)

		self.setWorkspacePath(workspacePath or Path.cwd())

	@property
	def workspacePath(self) -> str:
		"""
		Return the currently configured workspace path.
		"""
		return str(self._workspacePath)

	@property
	def model(self) -> QFileSystemModel:
		"""
		Return the underlying file system model.
		"""
		return self._model

	@property
	def treeView(self) -> QTreeView:
		"""
		Return the tree view used to render workspace files.
		"""
		return self._treeView

	def setWorkspacePath(self, workspacePath: str | Path) -> bool:
		"""
		Point the view to a directory. Returns ``True`` when accepted.
		"""
		candidate = Path(workspacePath).expanduser().resolve()
		if not candidate.exists() or not candidate.is_dir():
			return False

		self._workspacePath = candidate
		rootIndex = self._model.setRootPath(str(candidate))
		self._treeView.setRootIndex(rootIndex)
		self.workspacePathChanged.emit(str(candidate))
		return True

	def _onItemActivated(self, index):
		"""
		Emit the absolute file path for the activated tree item.
		"""
		path = self._model.filePath(index)
		if path:
			self.fileActivated.emit(path)

