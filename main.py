# -*- coding: utf-8 -*-
"""
Main module.
App object is instantiated which in turn loads the UI.

"""

import sys

from PyQt5.QtWidgets import QApplication

import gui.icons  # this generates an icon resource file (see gui.py) as well as paths to all icons (see logic.py) # NOQA
from logic.app import App

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = App()
    app.exec_()
