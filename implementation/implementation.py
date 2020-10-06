import csv
import pandas as pd
from PyQt5.QtWidgets import (QWidget, QLineEdit, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import QTimer

class Measurement():
    
    timer = QTimer()
    time_passed = 0
    
    def __init__(self, type, duration_spinbox, prog_bar=None):
        self.type = type
        self.duration_spinbox = duration_spinbox
        self.prog_bar = prog_bar
        self.timer.timeout.connect(self.__timer_timeout)
        
    # public methods
    def start(self):
        self.timer.start(1000)
        
    def stop(self):
        self.timer.stop()
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
        if filepath:
            window.frame.findChild(QWidget, 'settingsFileName').setText(filepath)
            with open(filepath, 'w') as stream:
                print("saving", filepath)
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
        if filepath:
            df = pd.read_csv(filepath, header=None, delimiter=',', keep_default_na=False, error_bad_lines=False)
            window.frame.findChild(QWidget, 'settingsFileName').setText(filepath)
            for i in range(len(df)):
                widget = window.frame.findChild(QWidget, df.iloc[i, 0])
                if not widget == 'nullptr':
                    if hasattr(widget, 'value'): # spinner
                        widget.setValue(float(df.iloc[i, 1]))
                    else: # line edit
                        widget.setText(df.iloc[i, 1])

class ErrorMessage():
    pass
