import csv
import pandas as pd
from PyQt5.QtWidgets import (QWidget, QLineEdit, QSpinBox, QDoubleSpinBox)

class Qt2csv():

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
