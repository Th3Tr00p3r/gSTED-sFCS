"""
GUI Module.
"""
import gui.icons.icon_paths as icon
import implementation.logic as logic
import implementation.drivers as drivers
import implementation.constants as const

from PyQt5.QtCore import pyqtSlot,  QTimer
from PyQt5.QtWidgets import (QMainWindow, QDialog)
from PyQt5.QtGui import QIcon
from PyQt5 import uic

# resource files
from gui.icons import icons_rc # for initial icons # NOQA

class MainWindow(QMainWindow):
    """
    Class documentation goes here.
    """
    def __init__(self, parent=None):
        """
        Constructor

        @param parent reference to the parent widget
        @type QWidget
        """
        # TODO: move to logic

        # general window settings
        super(MainWindow, self).__init__(parent)
        uic.loadUi(const.MAINWINDOW_UI_PATH, self)

#        # check conected devices
#        self.instrument_names = [param_set._dict['classname'] for param_set in list_instruments()]

        # set up main timeout event
        self.timer = QTimer()
        self.timer.timeout.connect(self.timeout)
        self.timer.start(10)

        # define additinal windows
        # TODO: keep all windows in same place '.windows' (e.g. for closing together)
        self.settings_win = SettingsWindow()
        self.errors_win = ErrorsWindow()

        # intialize buttons
        self.actionLaser_Control.setChecked(True)
        self.actionStepper_Stage_Control.setChecked(True)
        self.stageButtonsGroup.setEnabled(False)
        self.actionLog.setChecked(True)
        
        # initialize Log Dock
        self.log_dock = logic.LogDock()

        #connect signals and slots
        self.ledExc.clicked.connect(self.show_laser_dock)
        self.ledDep.clicked.connect(self.show_laser_dock)
        self.ledShutter.clicked.connect(self.show_laser_dock)
        self.actionRestart.triggered.connect(self.restart)

    def closeEvent(self, event):
        logic.exit_app(self, event)

    def restart(self):
        logic.restart_app(self)

    def timeout(self):
        '''
        '''
        # TODO: move to implementation
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
            self.measurement = logic.Measurement('FCS',  self.measFCSDurationSpinBox, self.FCSprogressBar)
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
            self.excOnButton.setText('Laser \nON')
            self.excOnButton.setIcon(QIcon(icon.SWITCH_ON))
            self.ledExc.setIcon(QIcon(icon.LED_BLUE))
        # turning off
        else:
            self.excOnButton.setText('Laser \nOFF')
            self.excOnButton.setIcon(QIcon(icon.SWITCH_OFF))
            self.ledExc.setIcon(QIcon(icon.LED_OFF)) 

    @pyqtSlot()
    def on_depEmissionOn_released(self):
        """
        Slot documentation goes here.
        """
        # TODO: 
        if self.depEmissionOn.text() == 'Laser \nOFF':
            self.depEmissionOn.setText('Laser \nON')
            self.depEmissionOn.setIcon(QIcon(icon.SWITCH_ON))
            self.ledDep.setIcon(QIcon(icon.LED_ORANGE)) 
        else:
            self.depEmissionOn.setText('Laser \nOFF')
            self.depEmissionOn.setIcon(QIcon(icon.SWITCH_OFF))
            self.ledDep.setIcon(QIcon(icon.LED_OFF)) 

    @pyqtSlot()
    def on_depShutterOn_released(self):
        """
        Slot documentation goes here.
        """
        # TODO: 
        if self.depShutterOn.text() == 'Shutter \nClosed':
            self.depShutterOn.setText('Shutter \nOpen')
            self.depShutterOn.setIcon(QIcon(icon.SWITCH_ON))
            self.ledShutter.setIcon(QIcon(icon.LED_GREEN)) 
        else:
            self.depShutterOn.setText('Shutter \nClosed')
            self.depShutterOn.setIcon(QIcon(icon.SWITCH_OFF))
            self.ledShutter.setIcon(QIcon(icon.LED_OFF)) 

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
        Show settings window
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
        Show/hide stepper stage control dock

        @param p0 DESCRIPTION
        @type bool
        """
        # TODO:
        self.stepperDock.setVisible(p0)

    @pyqtSlot()
    def on_actionCamera_Control_triggered(self):
        """
        Instantiate 'CameraWindow' object and show it
        """
        # TODO: add support for 2nd camera
        self.cam_win = CameraWindow()

    @pyqtSlot(bool)
    def on_actionLog_toggled(self, p0):
        """
        Show/hide log dock

        @param p0 DESCRIPTION
        @type bool
        """
        # TODO:
        self.logDock.setVisible(p0)

    @pyqtSlot()
    def show_laser_dock(self):
        """
        Make the laser dock visible (convenience)
        """
        self.laserDock.setVisible(True)
        p0 = self.actionLaser_Control.isChecked()
        if not p0:
            self.actionLaser_Control.setChecked(not p0)

    @pyqtSlot()
    def on_ledErrors_clicked(self):
        """
        Show errors window
        """
        # TODO:
        self.errors_win.show()
        self.errors_win.activateWindow()

    #----------------------------------------------------------------------------
    # Stepper Stage Dock
    #----------------------------------------------------------------------------
    
    @pyqtSlot()
    def on_stageOn_released(self):
        """
        Slot documentation goes here.
        """
        # TODO:
        #Turn On
        if not self.stageButtonsGroup.isEnabled():
            self.stageOn.setIcon(QIcon(icon.SWITCH_ON))
            self.stageButtonsGroup.setEnabled(True)
            self.stage = drivers.StepperStage(rsrc_alias=self.settings_win.arduinoChan.text())
            
        #Turn Off
        else:
            self.stageOn.setIcon(QIcon(icon.SWITCH_OFF))
            self.stageButtonsGroup.setEnabled(False)
            self.stage = self.stage.clean_up()
            
    @pyqtSlot()
    def on_stageUp_released(self):
        """
        Slot documentation goes here.
        """
        # TODO:
        self.stage.move(dir='UP', steps=self.stageSteps.value())

    @pyqtSlot()
    def on_stageDown_released(self):
        """
        Slot documentation goes here.
        """
        # TODO:
        self.stage.move(dir='DOWN', steps=self.stageSteps.value())

    @pyqtSlot()
    def on_stageLeft_released(self):
        """
        Slot documentation goes here.
        """
        # TODO:
        self.stage.move(dir='LEFT', steps=self.stageSteps.value())

    @pyqtSlot()
    def on_stageRight_released(self):
        """
        Slot documentation goes here.
        """
        # TODO:
        self.stage.move(dir='RIGHT', steps=self.stageSteps.value())

    @pyqtSlot()
    def on_stageRelease_released(self):
        """
        Slot documentation goes here.
        """
        # TODO:
        self.stage.release()

class ErrorsWindow(QDialog):
    """
    This "window" is a QWidget. If it has no parent, it 
    will appear as a free-floating window as we want.
    """
    # TODO: when error occures, change appropriate list items background to red, font to white
    def __init__(self,  parent=None):
        super(ErrorsWindow,  self).__init__(parent)
        uic.loadUi(const.ERRORSWINDOW_UI_PATH, self)

    @pyqtSlot(int)
    def on_errorSelectList_currentRowChanged(self, index):
        self.errorDetailsStacked.setCurrentIndex(index)

class SettingsWindow(QDialog, logic.SettingsWin):
    """
    This "window" is a QWidget. If it has no parent, it 
    will appear as a free-floating window as we want.
    """
    def __init__(self,  parent=None):
        super(SettingsWindow,  self).__init__(parent)
        uic.loadUi(const.SETTINGSWINDOW_UI_PATH, self)
        
        self.read_csv(const.DEFAULT_SETTINGS_FILE_PATH)
    
    def closeEvent(self, event):
        print('Settings window closed (closeEvent)') # TEST
        self.clean_up()

    @pyqtSlot()
    def on_saveButton_released(self):
        """
        Save settings as .csv
        """
        # TODO: add all forms in main window too
        self.write_csv()

    @pyqtSlot()
    def on_loadButton_released(self):
        """
        load settings .csv file
        """
        # TODO: add all forms in main window too
        self.read_csv()

class CameraWindow(QDialog):
    """
    documentation
    """
    def __init__(self,  parent=None):
        super(CameraWindow,  self).__init__(parent)
        uic.loadUi(const.CAMERAWINDOW_UI_PATH, self)
        self.imp = logic.CamWin(self)

    def closeEvent(self, event):
        '''
        cleaning up the camera driver, closing the camera window
        and setting CameraWindow.imp to ''None''
        '''
#        print('Camera window closed (closeEvent)') # TEST
        self.imp = self.imp.clean_up()

    @pyqtSlot()
    def on_shootButton_released(self):
        """
        Slot documentation goes here.
        """
        self.imp.shoot()

    @pyqtSlot()
    def on_videoButton_released(self):
        """
        Slot documentation goes here.
        """
        self.imp.start_stop_video()
