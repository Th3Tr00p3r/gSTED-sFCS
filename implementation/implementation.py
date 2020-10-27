import os
import sys
#import time
from PyQt5.QtWidgets import (QWidget, QLineEdit, QSpinBox,
                                              QDoubleSpinBox, QMessageBox, QApplication, 
                                              QFileDialog
                                              )
from PyQt5.QtCore import QTimer
import implementation.constants as const
import drivers

import pyvisa as visa
    
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

class SettingsWin():
    
    def clean_up(self):
        # TODO: add check to see if changes were made, if not, don't ask user
        pressed = Question('Keep changes if made? ' +
                                  '(otherwise, revert to last loaded settings file.)'
                                  ).display()
        if pressed == QMessageBox.No:
            self.read_csv(self.settingsFileName.text())

    # public methods
    def write_csv(self):
        '''
        Write all QLineEdit, QspinBox and QdoubleSpinBox of settings window to 'filepath' (csv).
        Show 'filepath' in 'settingsFileName' QLineEdit.
        '''
        filepath, _ = QFileDialog.getSaveFileName(self,
                                                                 'Save Settings',
                                                                 const.SETTINGS_FOLDER_PATH,
                                                                 "CSV Files(*.csv *.txt)")
        import csv
        if filepath:
            self.frame.findChild(QWidget, 'settingsFileName').setText(filepath)
            with open(filepath, 'w') as stream:
                #print("saving", filepath)
                writer = csv.writer(stream)

                # get all names of fields in settings window (for file saving/loading)
                l1 = self.frame.findChildren(QLineEdit)
                l2 = self.frame.findChildren(QSpinBox)
                l3 = self.frame.findChildren(QDoubleSpinBox)
                field_names = [w.objectName() for w in (l1 + l2 + l3) if (not w.objectName() == 'qt_spinbox_lineedit') and (not w.objectName() == 'settingsFileName')]
                #print(fieldNames)
                for i in range(len(field_names)):
                    widget = self.frame.findChild(QWidget, field_names[i])
                    if hasattr(widget, 'value'): # spinner
                        rowdata = [field_names[i],  self.frame.findChild(QWidget, field_names[i]).value()]
                    else: # line edit
                        rowdata = [field_names[i],  self.frame.findChild(QWidget, field_names[i]).text()]
                    writer.writerow(rowdata)

    def read_csv(self, filepath=''):
        '''
        Read 'filepath' (csv) and write to matching QLineEdit, QspinBox and QdoubleSpinBox of settings window.
        Show 'filepath' in 'settingsFileName' QLineEdit.
        '''
        # TODO: return to try/catch after fixing Error() so that the full stack is shown (hide/show details)
        if not filepath:
            filepath, _ = QFileDialog.getOpenFileName(self,
                                                                     "Load Settings",
                                                                     const.SETTINGS_FOLDER_PATH,
                                                                     "CSV Files(*.csv *.txt)")
        import pandas as pd
        if filepath:
            try:
                 df = pd.read_csv(filepath, header=None, delimiter=',', keep_default_na=False, error_bad_lines=False)
                 self.frame.findChild(QWidget, 'settingsFileName').setText(filepath)
                 for i in range(len(df)):
                    widget = self.frame.findChild(QWidget, df.iloc[i, 0])
                    if not widget == 'nullptr':
                        if hasattr(widget, 'value'): # spinner
                            widget.setValue(float(df.iloc[i, 1]))
                        else: # line edit
                            widget.setText(df.iloc[i, 1])
            except: # handeling missing default settings file
                error_txt = ('Default settings file "default_settings.csv"'
                                   'not found in:\n\n' +
                                   filepath +
                                   '.\n\nUsing standard settings.\n' +
                                   'To avoid this error, please save' +
                                   'some settings as "default_settings.csv".')
                Error(sys.exc_info(), error_txt=error_txt).display()

        else:
            error_txt = ('File path not supplied.')
            Error(error_txt=error_txt).display()
            
class UserDialog():
    
    def __init__(self,
                       msg_icon=QMessageBox.NoIcon,
                       msg_title=None,
                       msg_text=None,
                       msg_det=None):
        self._msg_box = QMessageBox()
        self._msg_box.setIcon(msg_icon)
        self._msg_box.setWindowTitle(msg_title)
        self._msg_box.setText(msg_text)
        self._msg_box.setDetailedText(msg_det)
    
    def set_buttons(self, std_bttn_list):
        for std_bttn in std_bttn_list:
            self._msg_box.addButton(getattr(QMessageBox, std_bttn))
        
    def display(self):
        return self._msg_box.exec_()
    
class Error(UserDialog):
    
    def __init__(self, error_info='', error_txt='Error occured.'):
        if error_info:
            self.exc_type, _, self.tb = error_info
            self.error_type = self.exc_type.__name__
            self.fname = os.path.split(self.tb.tb_frame.f_code.co_filename)[1]
            self.lineno = self.tb.tb_lineno
            super().__init__(msg_icon=QMessageBox.Critical,
                                   msg_title='Error',
                                   msg_text=error_txt,
                                   msg_det=('{}, {}' + ', line: {}').format(self.error_type,
                                                                                                 self.fname,
                                                                                                 self.lineno)
                                  )
        else:
            super().__init__(msg_icon=QMessageBox.Critical,
                                   msg_title='Error',
                                   msg_text=error_txt,
                                  )

class Question(UserDialog):
    def __init__(self, q_txt,  q_title='User Input Needed'):
        super().__init__(msg_icon=QMessageBox.Question,
                               msg_title=q_title,
                               msg_text=q_txt
                               )
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
    if hasattr(main_window, 'cam_win'):
        main_window.cam_win.reject()
        
def exit_app(main_window, event):
    '''
    Clean up and exit, ask first.
    '''
    pressed = Question(q_txt='Are you sure you want to quit?', q_title='Quitting Program').display()
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

class LogDock():
    
    def __init__(self):
        '''
        ready and show the log window
        '''
        try:
            # load today's log file, add "program started" etc.
            print('Log Dock Initialized') # TEST
            pass
        except:
            error_txt = ('')
            Error(sys.exc_info(), error_txt=error_txt).display()

class CamWin():
    
    # class attributes
    __video_timer = QTimer()
    
    def __init__(self, cam_gui):
        self.gui = cam_gui
        self.ready_window()
        self.init_cam()
        pass

    def clean_up(self):
        '''
        clean up before closing window
        '''
        if hasattr(self, '__video_timer'):
            self.videoButton.setStyleSheet("background-color: rgb(225, 225, 225); color: black;")
            self.videoButton.setText('Start Video')
            self.__video_timer.stop()
        self.cam.close()
        return None
#        self.gui.close()

    # public methods
    def ready_window(self):
        '''
        ready the gui for camera view, connect to camera and show window
        '''
        #TODO: fix Error to show full stack, then return the try/except
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        
        # add matplotlib-ready widget (canvas) for showing camera output
        self.gui.figure = plt.figure()
        self.gui.canvas = FigureCanvas(self.gui.figure)
        self.gui.gridLayout.addWidget(self.gui.canvas, 0, 1)
        
        # show window
        self.gui.show()
        self.gui.activateWindow()

#        try:
#        except:
#            error_txt = ('No cameras appear to be connected.')
#            Error(sys.exc_info(), error_txt=error_txt).display()

    def init_cam(self):
        # initialize camera
        self.cam = drivers.Camera(reopen_policy='new') # instantiate camera object
        self.cam.open() # connect to first available camera
    
    def start_stop_video(self):
        #TODO: fix Error to show full stack, then return the try/except
        #Turn On
        if self.gui.videoButton.text() == 'Start Video':
            self.gui.videoButton.setStyleSheet("background-color: rgb(225, 245, 225); color: black;")
            self.gui.videoButton.setText('Video ON')
            self.cam.start_live_video()
            self.__video_timer.timeout.connect(self.__video_timeout)
            self.__video_timer.start(50) # video refresh period (ms)
        #Turn Off
        else:
            self.gui.videoButton.setStyleSheet("background-color: rgb(225, 225, 225); color: black;")
            self.gui.videoButton.setText('Start Video')
            self.cam.stop_live_video()
            self.__video_timer.stop()
#        try:
#        except:
#            error_txt = ('Camera disconnected.' + '\n' +
#                             'Reconnect and re-open the camera window.')
#            Error(sys.exc_info(), error_txt).display()
#            self.on_rejected()

    def shoot(self):
        #TODO: fix Error to show full stack, then return the try/except
        img = self.cam.grab_image()
        self.__imshow(img)
#        try:
#        except:
#            error_txt = ('Camera disconnected.' + '\n' +
#                             'Reconnect and re-open the camera window.')
#            Error(sys.exc_info(), error_txt).display()
#            self.clean_up()
    
    # private methods
    def __imshow(self, img):
        '''
        Plot image
        '''
        self.gui.figure.clear()
        ax = self.gui.figure.add_subplot(111)
        ax.imshow(img)
        self.gui.canvas.draw()
    
    def __video_timeout(self):
        #TODO: fix Error to show full stack, then return the try/except
        img = self.cam.grab_image()
        self.__imshow(img)
#        try:
#        except:
#            error_txt = ('Camera disconnected.' + '\n' +
#                             'Reconnect and re-open the camera window.')
#            Error(sys.exc_info(), error_txt).display()
#            self.clean_up()

class StepperStage():
    
    def __init__(self,  rsrc_alias):
        self.rm = visa.ResourceManager()
        self.rsrc = self.rm.open_resource(rsrc_alias)
#        print(rsrc_alias,  self.rsrc) # TEST

    def clean_up(self):
        self.rsrc.close()
        return None
    
    def move(self, dir=None,  steps=None):
        cmd_dict = {'UP': (lambda steps: 'my ' + str(-steps)),
                          'DOWN': (lambda steps: 'my ' + str(steps)),
                          'LEFT': (lambda steps: 'mx ' + str(steps)),
                          'RIGHT': (lambda steps: 'mx ' + str(-steps))
                        }
        self.rsrc.write(cmd_dict[dir](steps))
    
    def release(self):
        cmnd = 'ryx '
        self.rsrc.write(cmnd)
