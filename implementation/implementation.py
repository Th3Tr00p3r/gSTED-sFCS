import os
import sys
import csv
import pandas as pd
from PyQt5.QtWidgets import (QWidget, QLineEdit, QSpinBox,
                                              QDoubleSpinBox, QMessageBox)
from PyQt5.QtCore import QTimer

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
        if self.time_passed < self.duration_spinbox.value():
            self.time_passed += 1
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
                error_text = ('Default settings file "default_settings.csv"'
                                   'not found in:\n\n' +
                                   filepath +
                                   '.\n\nUsing standard settings.\n'
                                   'To avoid this error, please save some settings as "default_settings.csv".')
                Error(sys.exc_info(), error_text=error_text).display()

class UserDialog():
    
    def __init__(self,
                       msg_icon=QMessageBox.NoIcon,
                       msg_title=None,
                       msg_text=None):
        self._msgBox = QMessageBox()
        self._msgBox.setIcon(msg_icon)
        self._msgBox.setWindowTitle(msg_title)
        self._msgBox.setText(msg_text)
    
    def display(self):
        self._msgBox.exec_()
    
class Error(UserDialog):
    
    def __init__(self, error_info, error_text=None):
        self.exc_type, _, self.tb = error_info
        self.error_type = self.exc_type.__name__
        self.fname = os.path.split(self.tb.tb_frame.f_code.co_filename)[1]
        self.lineno = self.tb.tb_lineno
        super().__init__(msg_icon=QMessageBox.Critical,
                               msg_title=('Error: {} ({}' +
                                               ', line: {})').format(self.error_type,
                                                                                self.fname,
                                                                                self.lineno),
                               msg_text=error_text)

class Note(UserDialog):
    pass
    
def Exit(main_window):
    '''
    Clean up and exit
    '''
    main_window.settings_win.reject()
    if hasattr(main_window, 'camera1_win'):
        main_window.camera1_win.reject()
    # finally
    main_window.close()
