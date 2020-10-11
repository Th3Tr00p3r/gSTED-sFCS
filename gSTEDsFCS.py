from PyQt5.QtWidgets import QApplication
from gui.mainwindow import MainWindow
import implementation.constants as const
import icons
import pyqt5ac

if __name__ == "__main__":
    import sys
    
#    pyqt5ac.main(rccOptions='', uicOptions='--from-imports',
#                       force=False, initPackage=True, config='',
#                       ioPaths=[['gui/*.ui', 'gui/%%FILENAME%%_ui.py'],
#                       ['icons/*.qrc', 'icons/%%FILENAME%%_rc.py']])
    
    current_exit_code = const.EXIT_CODE_REBOOT
    while current_exit_code == const.EXIT_CODE_REBOOT:
        app = QApplication(sys.argv)
        ui = MainWindow()
        ui.show()
        current_exit_code = app.exec_()
        app = None
#        print(current_exit_code)
