import sys
from PyQt5.QtWidgets import QApplication
from implementation.logic import App
import gui.icons # this generates an icon resource file (see gui.py) as well as paths to all icons (see logic.py) # NOQA

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = App()
    app.exec_()
    ui = None
    app = None
