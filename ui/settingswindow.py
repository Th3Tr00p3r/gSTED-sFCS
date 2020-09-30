# -*- coding: utf-8 -*-

"""
Module implementing SettingsWindow.
"""

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget

from .Ui_settingswindow import Ui_Settings


class SettingsWindow(QWidget, Ui_Settings):
    """
    Class documentation goes here.
    """
    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget
        @type QWidget
        """
        super(SettingsWindow, self).__init__(parent)
        self.setupUi(self)
    
    @pyqtSlot()
    def on_saveButton_released(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        raise NotImplementedError
