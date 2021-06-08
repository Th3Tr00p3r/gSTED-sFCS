"""
Main module.
App object is instantiated which in turn loads the UI.
"""

import asyncio
import sys

from PyQt5.QtWidgets import QApplication
from qasync import QEventLoop

if __name__ == "__main__":
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    # here and not before, otherwise QIcon on constants gets stuck
    from logic.app import App

    App(loop)
    app.exec_()

    with loop:
        sys.exit(loop.run_forever())
