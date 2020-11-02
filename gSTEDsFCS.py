from PyQt5.QtWidgets import QApplication
from gui.gui import MainWindow
import implementation.constants as const
import gui.icons # NOQA

if __name__ == "__main__":
    import sys
    
    current_exit_code = const.EXIT_CODE_REBOOT
    while current_exit_code == const.EXIT_CODE_REBOOT:
        app = QApplication(sys.argv)
        ui = MainWindow()
        ui.show()
        current_exit_code = app.exec_()
        ui.destroy()
        ui = None
        app = None
#        print(current_exit_code)
