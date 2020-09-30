# -*- coding: utf-8 -*-

"""
Module implementing MainWindow.
"""
import csv
import pandas as pd

from PyQt5.QtCore import pyqtSlot,  QTimer, QDir
from PyQt5.QtWidgets import QWidget, QMainWindow, QFileDialog, QLineEdit, QSpinBox, QDoubleSpinBox
from PyQt5.QtGui import QIcon

from .Ui_mainwindow import Ui_MainWindow
from .Ui_settingswindow import Ui_Settings
from .Ui_cameraswindow import Ui_Cameras

class SettingsWindow(QMainWindow, Ui_Settings):
    """
    This "window" is a QWidget. If it has no parent, it 
    will appear as a free-floating window as we want.
    """
    def __init__(self,  parent=None):
        super(SettingsWindow,  self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('Settings')
    
    def writeCsv(self):
        filepath, _ = QFileDialog.getSaveFileName(self, 'Save Settings', QDir.currentPath() + "\settings\\", "CSV Files(*.csv *.txt)")
        if filepath:
            self.frame.findChild(QWidget, 'settingsFileName').setText(filepath)
            with open(filepath, 'w') as stream:
                print("saving", filepath)
                writer = csv.writer(stream)
                
                # get all names of fields in settings window (for file saving/loading)
                l1 = self.frame.findChildren(QLineEdit)
                l2 = self.frame.findChildren(QSpinBox)
                l3 = self.frame.findChildren(QDoubleSpinBox)
                fieldNames = [w.objectName() for w in (l1 + l2 + l3) if (not w.objectName() == 'qt_spinbox_lineedit') and (not w.objectName() == 'settingsFileName')]
                #print(fieldNames)
                for i in range(len(fieldNames)):
                    widget = self.frame.findChild(QWidget, fieldNames[i])
                    if hasattr(widget, 'value'): # spinner
                        rowdata = [fieldNames[i],  self.frame.findChild(QWidget, fieldNames[i]).value()]
                    else: # line edit
                        rowdata = [fieldNames[i],  self.frame.findChild(QWidget, fieldNames[i]).text()]
                    writer.writerow(rowdata)
                    
    def readCsv(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Settings", QDir.currentPath() + "\settings\\", "CSV Files(*.csv *.txt)")
        if filepath:
            self.frame.findChild(QWidget, 'settingsFileName').setText(filepath)
            df = pd.read_csv(filepath, header=None, delimiter=',', keep_default_na=False, error_bad_lines=False)
            for i in range(len(df)):
                widget = self.frame.findChild(QWidget, df.iloc[i, 0])
                if not widget == 'nullptr':
                    if hasattr(widget, 'value'): # spinner
                        widget.setValue(float(df.iloc[i, 1]))
                    else: # line edit
                        widget.setText(df.iloc[i, 1])

    @pyqtSlot()
    def on_saveButton_released(self):
        """
        Slot documentation goes here.
        """
        # TODO:
        self.writeCsv()
        
    @pyqtSlot()
    def on_loadButton_released(self):
        """
        Slot documentation goes here.
        """
        # TODO:
        self.readCsv()

class CamerasWindow(QMainWindow, Ui_Cameras):
    """
    This "window" is a QWidget. If it has no parent, it 
    will appear as a free-floating window as we want.
    """
    def __init__(self,  parent=None):
        super(CamerasWindow,  self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('Cameras')
        
    @pyqtSlot()
    def on_cam1VideoModeOn_released(self):
        """
        Slot documentation goes here.
        """
        # TODO:
        #Turn On
        if self.cam1VideoModeOn.text() == 'Start Video':
            self.cam1VideoModeOn.setStyleSheet("background-color: rgb(225, 245, 225); color: black;")
            self.cam1VideoModeOn.setText('Video ON')
        #Turn Off
        else:
            self.cam1VideoModeOn.setStyleSheet("background-color: rgb(225, 225, 225); color: black;")
            self.cam1VideoModeOn.setText('Start Video')
    
    @pyqtSlot()
    def on_cam2VideoModeOn_released(self):
        """
        Slot documentation goes here.
        """
        # TODO:
        #Turn On
        if self.cam2VideoModeOn.text() == 'Start Video':
            self.cam2VideoModeOn.setStyleSheet("background-color: rgb(225, 245, 225); color: black;")
            self.cam2VideoModeOn.setText('Video ON')
        #Turn Off
        else:
            self.cam2VideoModeOn.setStyleSheet("background-color: rgb(225, 225, 225); color: black;")
            self.cam2VideoModeOn.setText('Start Video')

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
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('gSTED-sFCS Measurement Program')

        self.timer = QTimer()
        self.timer.timeout.connect(self.timeout)
        self.timer.start(10)
        
        self.settings = SettingsWindow()
        self.cameras = CamerasWindow()
        
        self.actionLaser_Control.setChecked(True)
        self.actionStepper_Stage_Control.setChecked(True)
        self.actionCamera_Control.setChecked(False)

    def timeout(self):
        if self.depTemp.value() < 52:
            self.depTemp.setStyleSheet("background-color: rgb(255, 0, 0); color: white;")
        else:
            self.depTemp.setStyleSheet("background-color: white; color: black;")

    @pyqtSlot()
    def on_startFcsMeasurementButton_released(self):
        """
        Slot documentation goes here.
        """
        # TODO: 
        if self.startFcsMeasurementButton.isChecked():
            self.startFcsMeasurementButton.setText('Stop \nMeasurement')
            on_state = 1
        else:
            self.startFcsMeasurementButton.setText('Start \nMeasurement')
            on_state = 0

        if on_state:
            self.FCStimer = QTimer()
            self.FCStimer.timeout.connect(self.FCSmeasTimer)
            self.FCStimer.start(1000)
        else: # off state
            self.FCStimer.stop()
            self.FCSprogressBar.setValue(0)

    def FCSmeasTimer(self):
        prog = self.FCSprogressBar.value() + (100 / self.measTimeSpinBox.value())
        if prog > 100:
            prog = 100 / self.measTimeSpinBox.value() 
        self.FCSprogressBar.setValue(prog)            
    
    @pyqtSlot()
    def on_excOnButton_released(self):
        """
        Slot documentation goes here.
        """
        # TODO: 
        # turning on
        if self.excOnButton.text() == 'Laser \nOFF':
            #self.excOnButton.setStyleSheet("background-color: rgb(105, 105, 255); color: white;")
            self.excOnButton.setText('Laser \nON')
            self.ledExc.setIcon(QIcon('./icons/myIcons/led_blue.png')) 
        # turning off
        else:
            #self.excOnButton.setStyleSheet("background-color: rgb(225, 225, 225); color: black;")
            self.excOnButton.setText('Laser \nOFF')
            self.ledExc.setIcon(QIcon('./icons/myIcons/led_off.png')) 

    @pyqtSlot()
    def on_depEmissionOn_released(self):
        """
        Slot documentation goes here.
        """
        # TODO: 
        if self.depEmissionOn.text() == 'Laser \nOFF':
            #self.depEmissionOn.setStyleSheet("background-color: rgb(255, 222, 155); color: black;")
            self.depEmissionOn.setText('Laser \nON')
            self.ledDep.setIcon(QIcon('./icons/myIcons/led_orange.png')) 
        else:
            #self.depEmissionOn.setStyleSheet("background-color: rgb(225, 225, 225); color: black;")
            self.depEmissionOn.setText('Laser \nOFF')
            self.ledDep.setIcon(QIcon('./icons/myIcons/led_off.png')) 
    
    @pyqtSlot()
    def on_depShutterOn_released(self):
        """
        Slot documentation goes here.
        """
        # TODO: 
        if self.depShutterOn.text() == 'Shutter \nClosed':
            #self.depShutterOn.setStyleSheet("background-color: rgb(255, 222, 155); color: black;")
            self.depShutterOn.setText('Shutter \nOpen')
            self.ledShutter.setIcon(QIcon('./icons/myIcons/led_green.png')) 
        else:
            #self.depShutterOn.setStyleSheet("background-color: rgb(225, 225, 225); color: black;")
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
        Slot documentation goes here.
        
        @param index DESCRIPTION
        @type int
        """
        # TODO: not implemented yet
        self.solScanParamsStacked.setCurrentIndex(index)
    
    @pyqtSlot()
    def on_actionSettings_triggered(self):
        """
        Slot documentation goes here.
        """
        # TODO:
        self.settings.show()
        self.settings.activateWindow()
    
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
        self.cameras.show()
        self.cameras.activateWindow()
        
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
