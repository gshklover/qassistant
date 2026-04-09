"""
Unit tests for GUI settings components.
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QComboBox, QDialogButtonBox, QPushButton

from qassistant.gui.settings import Settings, SettingsDlg, SettingsView


class TestSettings(unittest.TestCase):
    """
    Validate Settings model behavior.
    """

    def test_copy_from_and_reset(self):
        """
        Copying preserves values and reset restores defaults.
        """
        settings = Settings(model="gpt-4.1", available_models=["gpt-4.1", "gpt-5"])
        other = Settings(model="gpt-5", available_models=["gpt-5", "gpt-4.1", "gpt-5"])

        settings.copyFrom(other)

        self.assertEqual(settings.model, "gpt-5")
        self.assertEqual(settings.available_models, ["gpt-5", "gpt-4.1"])

        settings.reset()

        self.assertEqual(settings.model, "")
        self.assertEqual(settings.available_models, ["gpt-5", "gpt-4.1"])


class TestSettingsWidgets(unittest.TestCase):
    """
    Validate SettingsView and SettingsDlg behavior.
    """

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_settings_view_updates_bound_settings(self):
        """
        Editing the combo box updates the bound Settings object and vice versa.
        """
        settings = Settings(model="gpt-4.1", available_models=["gpt-4.1", "gpt-5"])
        view = SettingsView(settings)
        combo_box = view.findChild(QComboBox, "modelComboBox")

        combo_box.setCurrentText("gpt-5")
        self.application.processEvents()
        self.assertEqual(settings.model, "gpt-5")

        settings.model = "custom-model"
        self.application.processEvents()
        self.assertEqual(combo_box.currentText(), "custom-model")

    def test_dialog_accept_applies_changes(self):
        """
        Accepting the dialog copies the working copy back to the live settings object.
        """
        settings = Settings(model="gpt-4.1", available_models=["gpt-4.1", "gpt-5"])
        dialog = SettingsDlg(settings)
        applied = []
        combo_box = dialog.findChild(QComboBox, "modelComboBox")
        button_box = dialog.findChild(QDialogButtonBox)

        dialog.settingsApplied.connect(lambda: applied.append(True))
        combo_box.setCurrentText("gpt-5")
        button_box.button(QDialogButtonBox.StandardButton.Ok).click()
        self.application.processEvents()

        self.assertEqual(dialog.result(), dialog.DialogCode.Accepted)
        self.assertEqual(settings.model, "gpt-5")
        self.assertEqual(applied, [True])

    def test_dialog_cancel_and_reset_do_not_apply_immediately(self):
        """
        Reset affects only the dialog working copy, and cancel discards pending edits.
        """
        settings = Settings(model="gpt-4.1", available_models=["gpt-4.1", "gpt-5"])
        dialog = SettingsDlg(settings)
        combo_box = dialog.findChild(QComboBox, "modelComboBox")
        reset_button = dialog.findChild(QPushButton, "resetButton")
        button_box = dialog.findChild(QDialogButtonBox)

        combo_box.setCurrentText("gpt-5")
        reset_button.click()
        self.application.processEvents()

        self.assertEqual(combo_box.currentText(), "")
        self.assertEqual(settings.model, "gpt-4.1")

        combo_box.setCurrentText("custom-model")
        button_box.button(QDialogButtonBox.StandardButton.Cancel).click()
        self.application.processEvents()

        self.assertEqual(dialog.result(), dialog.DialogCode.Rejected)
        self.assertEqual(settings.model, "gpt-4.1")