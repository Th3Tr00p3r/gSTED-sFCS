"""
Main module.
App object is instantiated which in turn loads the UI.
"""

import asyncio
import sys

from PyQt5.QtWidgets import QApplication
from qasync import QEventLoop

if __name__ == "__main__":

    print("\nInitializing Application:")
    print("    Importing Modules...", end=" ")
    from logic.app import App

    print("Done.")

    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    App(loop)

    with loop:
        sys.exit(loop.run_forever())
