import os
import sys
from PyQt5.QtWidgets import (QWidget, QLineEdit, QSpinBox,
                                              QDoubleSpinBox, QMessageBox, QApplication)
from PyQt5.QtCore import QTimer
import implementation.constants as const

class Measurement():
    
    # class attributes
    _timer = QTimer()
    
    def __init__(self, type, duration_spinbox, prog_bar=None):
        # instance attributes
        self.type = type
        self.duration_spinbox = duration_spinbox
        self.prog_bar = prog_bar
        self._timer.timeout.connect(self.__timer_timeout)
        self.time_passed = 0
        
    # public methods
    def start(self):
        self._timer.start(1000)
        
    def stop(self):
        self._timer.stop()
        if self.prog_bar:
            self.prog_bar.setValue(0)
    
    # private methods
    def __timer_timeout(self):
#        from PyQt5.QtMultimedia import QSound
        if self.time_passed < self.duration_spinbox.value():
            self.time_passed += 1
#            if self.time_passed == self.duration_spinbox.value():
#                QSound.play(const.MEAS_COMPLETE_SOUND);
        else:
            self.time_passed = 1
        if self.prog_bar:
            prog = self.time_passed / self.duration_spinbox.value() * 100
            self.prog_bar.setValue(prog)          

class Qt2csv():
    # public methods
    def write_csv(window, filepath):
        '''
        Write all QLineEdit, QspinBox and QdoubleSpinBox of settings window to 'filepath' (csv).
        Show 'filepath' in 'settingsFileName' QLineEdit.
        '''
        import csv
        if filepath:
            window.frame.findChild(QWidget, 'settingsFileName').setText(filepath)
            with open(filepath, 'w') as stream:
                #print("saving", filepath)
                writer = csv.writer(stream)

                # get all names of fields in settings window (for file saving/loading)
                l1 = window.frame.findChildren(QLineEdit)
                l2 = window.frame.findChildren(QSpinBox)
                l3 = window.frame.findChildren(QDoubleSpinBox)
                field_names = [w.objectName() for w in (l1 + l2 + l3) if (not w.objectName() == 'qt_spinbox_lineedit') and (not w.objectName() == 'settingsFileName')]
                #print(fieldNames)
                for i in range(len(field_names)):
                    widget = window.frame.findChild(QWidget, field_names[i])
                    if hasattr(widget, 'value'): # spinner
                        rowdata = [field_names[i],  window.frame.findChild(QWidget, field_names[i]).value()]
                    else: # line edit
                        rowdata = [field_names[i],  window.frame.findChild(QWidget, field_names[i]).text()]
                    writer.writerow(rowdata)

    def read_csv(window, filepath):
        '''
        Read 'filepath' (csv) and write to matching QLineEdit, QspinBox and QdoubleSpinBox of settings window.
        Show 'filepath' in 'settingsFileName' QLineEdit.
        '''
        import pandas as pd
        if filepath:
            try:
                df = pd.read_csv(filepath, header=None, delimiter=',', keep_default_na=False, error_bad_lines=False)
                window.frame.findChild(QWidget, 'settingsFileName').setText(filepath)
                for i in range(len(df)):
                    widget = window.frame.findChild(QWidget, df.iloc[i, 0])
                    if not widget == 'nullptr':
                        if hasattr(widget, 'value'): # spinner
                            widget.setValue(float(df.iloc[i, 1]))
                        else: # line edit
                            widget.setText(df.iloc[i, 1])
            except Exception: # handeling missing default settings file
                error_txt = ('Default settings file "default_settings.csv"'
                                   'not found in:\n\n' +
                                   filepath +
                                   '.\n\nUsing standard settings.\n'
                                   'To avoid this error, please save some settings as "default_settings.csv".')
                Error(sys.exc_info(), error_txt=error_txt).display()

class UserDialog():
    
    def __init__(self,
                       msg_icon=QMessageBox.NoIcon,
                       msg_title=None,
                       msg_text=None):
        self._msg_box = QMessageBox()
        self._msg_box.setIcon(msg_icon)
        self._msg_box.setWindowTitle(msg_title)
        self._msg_box.setText(msg_text)
    
    def set_buttons(self, std_bttn_list):
        for std_bttn in std_bttn_list:
            self._msg_box.addButton(getattr(QMessageBox, std_bttn))
        
    def display(self):
        return self._msg_box.exec_()
    
class Error(UserDialog):
    
    def __init__(self, error_info, error_txt=None):
        self.exc_type, _, self.tb = error_info
        self.error_type = self.exc_type.__name__
        self.fname = os.path.split(self.tb.tb_frame.f_code.co_filename)[1]
        self.lineno = self.tb.tb_lineno
        super().__init__(msg_icon=QMessageBox.Critical,
                               msg_title=('Error: {} ({}' +
                                               ', line: {})').format(self.error_type,
                                                                                self.fname,
                                                                                self.lineno),
                               msg_text=error_txt)

class Question(UserDialog):
    def __init__(self, q_title, q_txt):
        super().__init__(msg_icon=QMessageBox.Question,
                               msg_title=(q_title),
                               msg_text=q_txt)
        self.set_buttons(['Yes', 'No'])
        self._msg_box.setDefaultButton(QMessageBox.No)

def clean_up_app(main_window):
    '''
    Disconnect from all devices safely
    and close secondary windows
    (before closing/restarting application)
    '''
    main_window.settings_win.reject()
    if hasattr(main_window, 'camera1_win'):
        main_window.camera1_win.reject()
        
def exit_app(main_window, event):
    '''
    Clean up and exit, ask first.
    '''
    pressed = Question(q_title='Quitting Program', q_txt='Are you sure you want to quit?').display()
    if pressed == QMessageBox.Yes:
        clean_up_app(main_window)
        main_window.close()
    else:
        event.ignore()
        
def restart_app(main_window):
    '''
    Clean up and restart, ask first.
    '''
    pressed = Question(q_title='Restarting Program',
                               q_txt=('Are you sure you want ' +
                                         'to restart the program?')).display()
    if pressed == QMessageBox.Yes:
        clean_up_app(main_window)
        QApplication.exit(const.EXIT_CODE_REBOOT);
