"""
Settings model and editor widgets for the qassistant GUI.
"""
from PySide6.QtCore import QSignalBlocker, Signal as QtSignal
from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget

from ..agent.common import IObservable


class Settings(IObservable):
	"""
	Observable application settings.
	"""

	DEFAULT_MODEL = ""

	def __init__(self, model: str = DEFAULT_MODEL, available_models: list[str] | None = None):
		super().__init__()
		self._model = self.DEFAULT_MODEL
		self._available_models: list[str] = []
		self.available_models = available_models or []
		self.model = model

	@property
	def model(self) -> str:
		"""
		Selected model identifier.
		"""
		return self._model

	@model.setter
	def model(self, value: str) -> None:
		normalized_value = value or ""
		if self._model == normalized_value:
			return

		self._model = normalized_value
		self._on_property_changed("model", self._model)

	@property
	def available_models(self) -> list[str]:
		"""
		Known model identifiers available to the UI.
		"""
		return list(self._available_models)

	@available_models.setter
	def available_models(self, values: list[str]) -> None:
		normalized_values: list[str] = []
		for value in values:
			if value and value not in normalized_values:
				normalized_values.append(value)

		if self._available_models == normalized_values:
			return

		self._available_models = normalized_values
		self._on_property_changed("available_models", self.available_models)

	def clone(self) -> "Settings":
		"""
		Create a detached copy of the settings object.
		"""
		return Settings(model=self.model, available_models=self.available_models)

	def copyFrom(self, settings: "Settings"):
		"""
		Replace current values with values from another settings object.
		"""
		self.available_models = settings.available_models
		self.model = settings.model

	def reset(self):
		"""
		Restore setting values to their defaults.
		"""
		self.model = self.DEFAULT_MODEL


class SettingsView(QWidget):
	"""
	Widget that edits a Settings object.
	"""

	def __init__(self, settings: Settings | None = None, parent: QWidget | None = None):
		super().__init__(parent)
		self._settings: Settings | None = None
		self._model_combo_box = QComboBox(self)
		self._model_combo_box.setObjectName("modelComboBox")
		self._model_combo_box.setEditable(True)
		self._model_combo_box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
		self._model_combo_box.currentTextChanged.connect(self._onModelChanged)

		line_edit = self._model_combo_box.lineEdit()
		if isinstance(line_edit, QLineEdit):
			line_edit.setPlaceholderText("Enter model name")

		layout = QFormLayout(self)
		layout.addRow("Model", self._model_combo_box)

		if settings is not None:
			self.setSettings(settings)

	def setSettings(self, settings: Settings | None):
		"""
		Attach the view to a Settings object.
		"""
		if self._settings is settings:
			return

		if self._settings is not None:
			self._settings.property_changed.disconnect(self._onSettingsChanged)

		self._settings = settings

		if self._settings is not None:
			self._settings.property_changed.connect(self._onSettingsChanged)

		self._syncFromSettings()

	def _onSettingsChanged(self, property_name: str, value):
		"""
		Refresh the UI after model changes.
		"""
		if property_name in {"model", "available_models"}:
			self._syncFromSettings()

	def _syncFromSettings(self):
		"""
		Update widget state from the bound settings object.
		"""
		blocker = QSignalBlocker(self._model_combo_box)
		try:
			self._model_combo_box.clear()

			if self._settings is None:
				self._model_combo_box.setCurrentText("")
				return

			self._model_combo_box.addItems(self._settings.available_models)
			self._model_combo_box.setCurrentText(self._settings.model)
		finally:
			del blocker

	def _onModelChanged(self, value: str):
		"""
		Persist user edits back to the bound settings object.
		"""
		if self._settings is not None:
			self._settings.model = value


class SettingsDlg(QDialog):
	"""
	Dialog hosting a SettingsView with apply and reset controls.
	"""

	settingsApplied = QtSignal()

	def __init__(self, settings: Settings, parent: QWidget | None = None):
		super().__init__(parent)
		self._settings = settings
		self._working_settings = settings.clone()
		self._settings_view = SettingsView(self._working_settings, self)
		self._button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
		self._reset_button = self._button_box.addButton("Reset", QDialogButtonBox.ButtonRole.ResetRole)
		self._reset_button.setObjectName("resetButton")

		self.setWindowTitle("Settings")
		self._button_box.accepted.connect(self._onAccepted)
		self._button_box.rejected.connect(self.reject)
		self._reset_button.clicked.connect(self._onResetClicked)

		layout = QVBoxLayout(self)
		layout.addWidget(self._settings_view)
		layout.addWidget(self._button_box)

	def _onAccepted(self):
		"""
		Apply dialog edits to the live settings object.
		"""
		self._settings.copyFrom(self._working_settings)
		self.settingsApplied.emit()
		self.accept()

	def _onResetClicked(self):
		"""
		Restore default values in the working copy.
		"""
		self._working_settings.reset()
