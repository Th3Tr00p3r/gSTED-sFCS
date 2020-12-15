# -*- coding: utf-8 -*-
"""
Main module.
App object is instantiated which in turn loads the UI.

"""

import asyncio
import sys

from asyncqt import QEventLoop
from PyQt5.QtWidgets import QApplication

import gui.icons  # this generates an icon resource file (see gui.py) as well as paths to all icons (see logic.py) # NOQA
from logic.app import App

if __name__ == "__main__":
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    ui = App(loop)
    app.exec_()

    with loop:
        sys.exit(loop.run_forever())
