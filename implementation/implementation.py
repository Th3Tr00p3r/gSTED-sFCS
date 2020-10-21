import os
import sys
from PyQt5.QtWidgets import (QWidget, QLineEdit, QSpinBox,
                                              QDoubleSpinBox, QMessageBox, QApplication)
from PyQt5.QtCore import QTimer
import implementation.constants as const
import drivers
    
class Measurement():
    
    # class attributes
    __timer = QTimer()
    
    def __init__(self, type, duration_spinbox, prog_bar=None):
        # instance attributes
        self.type = type
        self.duration_spinbox = duration_spinbox
        self.prog_bar = prog_bar
        self.__timer.timeout.connect(self.___timer_timeout)
        self.time_passed = 0
        
    # public methods
    def start(self):
        self.__timer.start(1000)
        
    def stop(self):
        self.__timer.stop()
        if self.prog_bar:
            self.prog_bar.setValue(0)
    
    # private methods
    def ___timer_timeout(self):
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
            except: # handeling missing default settings file
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
                       msg_text=None,
                       msg_inf=None):
        self._msg_box = QMessageBox()
        self._msg_box.setIcon(msg_icon)
        self._msg_box.setWindowTitle(msg_title)
        self._msg_box.setText(msg_text)
        self._msg_box.setInformativeText(msg_inf)
    
    def set_buttons(self, std_bttn_list):
        for std_bttn in std_bttn_list:
            self._msg_box.addButton(getattr(QMessageBox, std_bttn))
        
    def display(self):
        return self._msg_box.exec_()
    
class Error(UserDialog):
    
    def __init__(self, error_info, error_txt='Error occured. See details in title.'):
        self.exc_type, _, self.tb = error_info
        self.error_type = self.exc_type.__name__
        self.fname = os.path.split(self.tb.tb_frame.f_code.co_filename)[1]
        self.lineno = self.tb.tb_lineno
        super().__init__(msg_icon=QMessageBox.Critical,
                               msg_title='Error',
                               msg_text=error_txt,
                               msg_inf=('{}, {}' +
                                           ', line: {}').format(self.error_type,
                                                                            self.fname,
                                                                            self.lineno))

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
    main_window.errors_win.reject()
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

class LogDockImp():
    
    def ready_dock(self):
        '''
        ready and show the log window
        '''
        try:
            # load today's log file, add "program started" etc.
            pass
        except:
            error_txt = ('')
            Error(sys.exc_info(), error_txt=error_txt).display()

class CamWinImp():
    
    # class attributes
    __video_timer = QTimer()
    
    def __init__(self):
        # instance attributes
        pass
        
    # public methods
    def ready_window(self):
        '''
        ready the gui for camera view, connect to camera and show window
        '''
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        # add matplotlib-ready widget (canvas) for showing camera output
        try:
            self.figure = plt.figure()
            self.canvas = FigureCanvas(self.figure)
            self.gridLayout.addWidget(self.canvas, 0, 1)
            
            # initialize camera
            self.cam = drivers.Camera() # instantiate camera object
            self.cam.open() # connect to first available camera
            
            # show window
            self.show()
            self.activateWindow()
        except:
            error_txt = ('No cameras appear to be connected.')
            Error(sys.exc_info(), error_txt=error_txt).display()

    def start_stop_video(self):
        try:
            #Turn On
            if self.videoButton.text() == 'Start Video':
                self.videoButton.setStyleSheet("background-color: rgb(225, 245, 225); color: black;")
                self.videoButton.setText('Video ON')
                self.cam.start_live_video()
                self.__video_timer.timeout.connect(self.__video_timeout)
                self.__video_timer.start(50) # video refresh period (ms)
            #Turn Off
            else:
                self.videoButton.setStyleSheet("background-color: rgb(225, 225, 225); color: black;")
                self.videoButton.setText('Start Video')
                self.cam.stop_live_video()
                self.__video_timer.stop()
        except:
            error_txt = ('Camera disconnected.' + '\n' +
                             'Reconnect and re-open the camera window.')
            Error(sys.exc_info(), error_txt).display()
            self.on_rejected()

    def shoot(self):
        try:
            img = self.cam.grab_image()
            self.__imshow(img)
        except:
            error_txt = ('Camera disconnected.' + '\n' +
                             'Reconnect and re-open the camera window.')
            Error(sys.exc_info(), error_txt).display()
            self.on_rejected()

    def clean_up(self):
        '''
        clean up before closing window
        '''
        if hasattr(self, '__video_timer'):
            self.videoButton.setStyleSheet("background-color: rgb(225, 225, 225); color: black;")
            self.videoButton.setText('Start Video')
            self.__video_timer.stop()
        self.cam.close()
        self.close()
    
    # private methods
    def __imshow(self, img):
        '''
        Plot image
        '''
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(img)
        self.canvas.draw()
    
    def __video_timeout(self):
        '''
        '''
        try:
            img = self.cam.grab_image()
            self.__imshow(img)
        except:
            error_txt = ('Camera disconnected.' + '\n' +
                             'Reconnect and re-open the camera window.')
            Error(sys.exc_info(), error_txt).display()
            self.on_rejected()
