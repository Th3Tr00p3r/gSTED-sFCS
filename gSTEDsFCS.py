from PyQt5.QtWidgets import QApplication
from implementation.logic import App
import implementation.constants as const
import gui.icons # NOQA

if __name__ == "__main__":
    import sys
    
    current_exit_code = const.EXIT_CODE_REBOOT
    while current_exit_code == const.EXIT_CODE_REBOOT:
        app = QApplication(sys.argv)
        ui = App()
#        ui.show()
        current_exit_code = app.exec_()
#        ui.win['main'].destroy()
        ui = None
        app = None
#        print(current_exit_code)
# TODO: restart not working in current configuration
