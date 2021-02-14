# -*- coding: utf-8 -*-
"""
Main module.
App object is instantiated which in turn loads the UI.
"""

import asyncio
import sys

from PyQt5.QtWidgets import QApplication
from qasync import QEventLoop

from logic.app import App

if __name__ == "__main__":
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    App(loop)
    app.exec_()

    with loop:
        sys.exit(loop.run_forever())
