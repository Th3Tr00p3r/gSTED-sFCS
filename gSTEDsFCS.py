from PyQt5.QtWidgets import QApplication
from ui.mainwindow import MainWindow
import implementation.constants as const

if __name__ == "__main__":
    import sys
    current_exit_code = const.EXIT_CODE_REBOOT
    while current_exit_code == const.EXIT_CODE_REBOOT:
        app = QApplication(sys.argv)
        ui = MainWindow()
        ui.show()
        current_exit_code = app.exec_()
        app = None
#        print(current_exit_code)
