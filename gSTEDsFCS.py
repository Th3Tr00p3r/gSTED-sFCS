from PyQt5.QtWidgets import QApplication
from gui.gui import MainWindow
import implementation.constants as const
import gui.icons # NOQA
#import pyqt5ac

if __name__ == "__main__":
    import sys
    
#    pyqt5ac.main(rccOptions='', force=False, initPackage=True, config='',
#                       ioPaths=['gui/icons/*.qrc', 'gui/icons/%%FILENAME%%_rc.py'])
    
    current_exit_code = const.EXIT_CODE_REBOOT
    while current_exit_code == const.EXIT_CODE_REBOOT:
        app = QApplication(sys.argv)
        ui = MainWindow()
        ui.show()
        current_exit_code = app.exec_()
        app = None
#        print(current_exit_code)
