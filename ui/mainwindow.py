"""
GUI Module.
"""

import drivers
import implementation.implementation as imp

# for cameras (to be moved to seperate module for camera class
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtCore import pyqtSlot,  QTimer, QDir
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QMessageBox, QDialog)
from PyQt5.QtGui import QIcon

from .Ui_mainwindow import Ui_MainWindow
from .Ui_settingswindow import Ui_Settings
from .Ui_camerawindow import Ui_Camera

class SettingsWindow(QDialog, Ui_Settings):
    """
    This "window" is a QWidget. If it has no parent, it 
    will appear as a free-floating window as we want.
    """
    def __init__(self,  parent=None):
        super(SettingsWindow,  self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('Settings')
        
        # load default settings
        try:
            def_filepath = QDir.currentPath() + '/settings/default_settings.csv'
            imp.Qt2csv.read_csv(self, def_filepath)
        except FileNotFoundError:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText('Default settings file "default_settings.csv"'
                                'not found in:\n\n' +
                                def_filepath +
                                '.\n\nUsing standard settings.\n'
                                'To avoid this error, please save some settings as "default_settings.csv".')
            msgBox.setWindowTitle('Error: File Not Found')
            msgBox.exec_()

    @pyqtSlot()
    def on_saveButton_released(self):
        """
        Save settings as .csv
        """
        # TODO: add all forms in main window too
        filepath, _ = QFileDialog.getSaveFileName(self, 'Save Settings', QDir.currentPath() + "\settings\\", "CSV Files(*.csv *.txt)")
        imp.Qt2csv.write_csv(self, filepath)
        
    @pyqtSlot()
    def on_loadButton_released(self):
        """
        load settings .csv file
        """
        # TODO: add all forms in main window too
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Settings", QDir.currentPath() + "\settings\\", "CSV Files(*.csv *.txt)")
        imp.Qt2csv.read_csv(self, filepath)

class CameraWindow(QDialog, Ui_Camera):
    """
    This "window" is a QWidget. If it has no parent, it 
    will appear as a free-floating window as we want.
    """
        
    def __init__(self,  parent=None):
        super(CameraWindow,  self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('Camera')
        
        # add matplotlib-ready widget (canvas) for showing camera output
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.gridLayout.addWidget(self.canvas, 0, 1)
        
        # initialize camera
        self.cam = drivers.Camera() # instantiate camera object
        self.cam.open() # connect to first available camera
    
    def reject(self):
        '''
        On pressing the 'X' button (closing the dialog, but really only hiding it)
        '''
        # clean up
        if hasattr(self, 'video_timer'):
            self.videoButton.setStyleSheet("background-color: rgb(225, 225, 225); color: black;")
            self.videoButton.setText('Start Video')
            self.cam.stop_live_video()
            self.video_timer.stop()
        self.hide()

    def imshow(self, img):
        '''
        Plot image
        '''
        # instead of ax.hold(False)
        self.figure.clear()

        # create an axis
        ax = self.figure.add_subplot(111)

        # plot data
        ax.imshow(img)

        # refresh canvas
        self.canvas.draw()
    
    def video_timeout(self):
        img = self.cam.grab_image()
        self.imshow(img)

    @pyqtSlot()
    def on_shootButton_released(self):
        """
        Slot documentation goes here.
        """
        # take a single image
        img = self.cam.grab_image()
        self.imshow(img)

    @pyqtSlot()
    def on_videoButton_released(self):
        """
        Slot documentation goes here.
        """
        # TODO:
        #Turn On
        if self.videoButton.text() == 'Start Video':
            self.videoButton.setStyleSheet("background-color: rgb(225, 245, 225); color: black;")
            self.videoButton.setText('Video ON')
            self.cam.start_live_video()
            self.video_timer = QTimer()
            self.video_timer.timeout.connect(self.video_timeout)
            self.video_timer.start(50) # video refresh period (ms)
        #Turn Off
        else:
            self.videoButton.setStyleSheet("background-color: rgb(225, 225, 225); color: black;")
            self.videoButton.setText('Start Video')
            self.cam.stop_live_video()
            self.video_timer.stop()

class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Class documentation goes here.
    """
    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget
        @type QWidget
        """
        # general window settings
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('gSTED-sFCS Measurement Program')
        
        # set up main timeout event
        self.timer = QTimer()
        self.timer.timeout.connect(self.timeout)
        self.timer.start(10)
        
        # define additinal windows
        self.settings_win = SettingsWindow()
        self.camera_win = CameraWindow()
        
        # intialize buttons
        self.actionLaser_Control.setChecked(True)
        self.actionStepper_Stage_Control.setChecked(True)
        
    def closeEvent(self, event):
        # clean up
        if hasattr(self, 'camera'):
            self.camera_win.cam.close()

    def timeout(self):
        if self.depTemp.value() < 52:
            self.depTemp.setStyleSheet("background-color: rgb(255, 0, 0); color: white;")
        else:
            self.depTemp.setStyleSheet("background-color: white; color: black;")

    @pyqtSlot()
    def on_startFcsMeasurementButton_released(self):
        """
        Begin FCS Measurement.
        """
        # TODO: 
        if self.startFcsMeasurementButton.text() == 'Start \nMeasurement':
            self.measurement = imp.Measurement('FCS',  self.measFCSDurationSpinBox.value(), self.FCSprogressBar)
            self.measurement.start()
            self.startFcsMeasurementButton.setText('Stop \nMeasurement')
        else: # off state
            self.measurement.stop()
            self.startFcsMeasurementButton.setText('Start \nMeasurement')

    @pyqtSlot()
    def on_excOnButton_released(self):
        """
        Slot documentation goes here.
        """
        # TODO: 
        # turning on
        if self.excOnButton.text() == 'Laser \nOFF':
            self.excOnButton.setStyleSheet("background-color: rgb(105, 105, 255); color: white;")
            self.excOnButton.setText('Laser \nON')
            self.ledExc.setIcon(QIcon('./icons/myIcons/led_blue.png')) 
        # turning off
        else:
            self.excOnButton.setStyleSheet("background-color: rgb(225, 225, 225); color: black;")
            self.excOnButton.setText('Laser \nOFF')
            self.ledExc.setIcon(QIcon('./icons/myIcons/led_off.png')) 

    @pyqtSlot()
    def on_depEmissionOn_released(self):
        """
        Slot documentation goes here.
        """
        # TODO: 
        if self.depEmissionOn.text() == 'Laser \nOFF':
            self.depEmissionOn.setStyleSheet("background-color: rgb(255, 222, 155); color: black;")
            self.depEmissionOn.setText('Laser \nON')
            self.ledDep.setIcon(QIcon('./icons/myIcons/led_orange.png')) 
        else:
            self.depEmissionOn.setStyleSheet("background-color: rgb(225, 225, 225); color: black;")
            self.depEmissionOn.setText('Laser \nOFF')
            self.ledDep.setIcon(QIcon('./icons/myIcons/led_off.png')) 
    
    @pyqtSlot()
    def on_depShutterOn_released(self):
        """
        Slot documentation goes here.
        """
        # TODO: 
        if self.depShutterOn.text() == 'Shutter \nClosed':
            self.depShutterOn.setStyleSheet("background-color: rgb(255, 222, 155); color: black;")
            self.depShutterOn.setText('Shutter \nOpen')
            self.ledShutter.setIcon(QIcon('./icons/myIcons/led_green.png')) 
        else:
            self.depShutterOn.setStyleSheet("background-color: rgb(225, 225, 225); color: black;")
            self.depShutterOn.setText('Shutter \nClosed')
            self.ledShutter.setIcon(QIcon('./icons/myIcons/led_off.png')) 

    @pyqtSlot()
    def on_stageOn_released(self):
        """
        Slot documentation goes here.
        """
        # TODO:
        #Turn On
        if not self.stageRelease.isEnabled():
            self.stageOn.setIcon(QIcon('./icons/myIcons/led_green.png')) 
            self.stageUp.setEnabled(True)
            self.stageDown.setEnabled(True)
            self.stageLeft.setEnabled(True)
            self.stageRight.setEnabled(True)
            self.stageRelease.setEnabled(True)
        #Turn Off
        else:
            self.stageOn.setIcon(QIcon('./icons/myIcons/led_off.png')) 
            self.stageUp.setEnabled(False)
            self.stageDown.setEnabled(False)
            self.stageLeft.setEnabled(False)
            self.stageRight.setEnabled(False)
            self.stageRelease.setEnabled(False)
    
    @pyqtSlot(int)
    def on_depSetCombox_currentIndexChanged(self, index):
        """
        Slot documentation goes here.
        
        @param index DESCRIPTION
        @type int
        """
        # TODO:
        self.depModeStacked.setCurrentIndex(index)
    
    @pyqtSlot(int)
    def on_solScanTypeCombox_currentIndexChanged(self, index):
        """
        Change stacked widget 'solScanParamsStacked' index according to index of the combo box 'solScanTypeCombox'.
        
        @param index - the index of the combo box 'solScanTypeCombox'
        @type int
        """
        self.solScanParamsStacked.setCurrentIndex(index)
    
    @pyqtSlot()
    def on_actionSettings_triggered(self):
        """
        Slot documentation goes here.
        """
        # TODO:
        self.settings_win.show()
        self.settings_win.activateWindow()
    
    @pyqtSlot(bool)
    def on_actionLaser_Control_toggled(self, p0):
        """
        Slot documentation goes here.
        
        @param p0 DESCRIPTION
        @type bool
        """
        # TODO:
        if p0:
            self.laserDock.setVisible(True)
        else:
            self.laserDock.setVisible(False)
    
    @pyqtSlot(bool)
    def on_actionStepper_Stage_Control_toggled(self, p0):
        """
        Slot documentation goes here.
        
        @param p0 DESCRIPTION
        @type bool
        """
        # TODO:
        if p0:
            self.stepperDock.setVisible(True)
        else:
            self.stepperDock.setVisible(False)
    
    @pyqtSlot()
    def on_actionCamera_Control_triggered(self):
        """
        Slot documentation goes here.
        
        @param p0 DESCRIPTION
        @type bool
        """
        # TODO:
        self.camera_win.show()
        self.camera_win.activateWindow()
        
    @pyqtSlot()
    def on_ledExc_clicked(self):
        """
        Slot documentation goes here.
        """
        self.laserDock.setVisible(True)
        p0 = self.actionLaser_Control.isChecked()
        if not p0:
            self.actionLaser_Control.setChecked(not p0)
        
    @pyqtSlot()
    def on_ledDep_clicked(self):
        """
        Slot documentation goes here.
        """
        self.laserDock.setVisible(True)
        p0 = self.actionLaser_Control.isChecked()
        if not p0:
            self.actionLaser_Control.setChecked(not p0)

    @pyqtSlot()
    def on_ledShutter_clicked(self):
        """
        Slot documentation goes here.
        """
        self.laserDock.setVisible(True)
        p0 = self.actionLaser_Control.isChecked()
        if not p0:
            self.actionLaser_Control.setChecked(not p0)
            
    @pyqtSlot()
    def on_ledErrors_clicked(self):
        """
        Slot documentation goes here.
        """
        self.tabWidget.setCurrentIndex(3) #error tab
