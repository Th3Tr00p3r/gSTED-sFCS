import sys
from PyQt5.QtWidgets import QApplication
from implementation.logic import App
import gui.icons # NOQA

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = App()
    app.exec_()
#    ui.win['main'].destroy()
    ui = None
    app = None
