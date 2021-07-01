"""
Main module.
App object is instantiated which in turn loads the UI.
"""

import asyncio
import sys

from PyQt5.QtWidgets import QApplication
from qasync import QEventLoop

if __name__ == "__main__":

    print("Initializing Application.")
    print("Importing App object...")
    from logic.app import App

    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    App(loop)
    app.exec_()

    with loop:
        sys.exit(loop.run_forever())
