'''
GUI Module.
'''

import implementation.constants as const
import implementation.logic as logic

from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtWidgets import (QMainWindow, QDialog, QWidget)
from PyQt5 import uic

from gui.icons import icons_rc # for initial icons loadout # NOQA

class MainWin(QMainWindow):
    
    """Class documentation goes here."""
    
    def __init__(self, app, parent=None):
        super(MainWin, self).__init__(parent)
        uic.loadUi(const.MAINWINDOW_UI_PATH, self)
        self.imp = logic.MainWin(self, app)

    def closeEvent(self, event):
        
        self.imp.close(event)

    @pyqtSlot()
    def on_startFcsMeasurementButton_released(self):
        
        '''Begin FCS Measurement.'''
        
        self.imp.toggle_FCS_meas()

    @pyqtSlot()
    def on_excOnButton_released(self):
        
        '''Turn excitation laser On/Off'''
        
        self.imp.dvc_toggle('EXC_LASER')

    @pyqtSlot()
    def on_depEmissionOn_released(self):
        
        '''Turn depletion laser On/Off'''
        
        self.imp.dvc_toggle('DEP_LASER')

    @pyqtSlot()
    def on_depShutterOn_released(self):
        
        '''Turn depletion physical shutter On/Off'''
        
        self.imp.dvc_toggle('DEP_SHUTTER')

    @pyqtSlot()
    def on_powModeRadio_released(self):
        
        '''Switch between power/current depletion laser settings'''
        
        self.depModeStacked.setCurrentIndex(1)
    
    @pyqtSlot()
    def on_currModeRadio_released(self):
        
        '''Switch between power/current depletion laser settings'''
        
        self.depModeStacked.setCurrentIndex(0)
        
    @pyqtSlot()
    def on_depApplySettings_released(self):
        
        '''Apply current/power mode and value'''
        # TODO: causes fatal error when pressed if there is DEP ERROR, fix it
        self.imp.dep_sett_apply()

    @pyqtSlot(int)
    def on_solScanTypeCombox_currentIndexChanged(self, index):
        '''
        Change stacked widget 'solScanParamsStacked' index
        according to index of the combo box 'solScanTypeCombox'.

        @param index - the index of the combo box 'solScanTypeCombox'
        @type int
        '''
        
        self.solScanParamsStacked.setCurrentIndex(index)

    @pyqtSlot()
    def on_actionSettings_triggered(self):
        
        '''Show settings window'''
        
        self.imp.open_settwin()

    @pyqtSlot(bool)
    def on_actionLaser_Control_toggled(self, p0):
        
        '''Show/hide stepper laser control dock'''
        
        self.laserDock.setVisible(p0)

    @pyqtSlot(bool)
    def on_actionStepper_Stage_Control_toggled(self, p0):
        
        '''Show/hide stepper stage control dock'''
        
        self.stepperDock.setVisible(p0)

    @pyqtSlot()
    def on_actionCamera_Control_triggered(self):
        
        '''Instantiate 'CameraWindow' object and show it'''
        # TODO: add support for 2nd camera
        
        self.imp.open_camwin()

    @pyqtSlot(bool)
    def on_actionLog_toggled(self, p0):
        
        '''Show/hide log dock'''
        
        self.logDock.setVisible(p0)

    @pyqtSlot()
    def on_ledErrors_clicked(self):
        
        '''Show errors window'''
        # TODO:
        
        self.imp.open_errwin()
        
    @pyqtSlot(int)
    def on_countsAvgSlider_valueChanged(self, val):
        
        self.imp.cnts_avg_sldr_changed(val)

    #----------------------------------------------------------------------------
    # Stepper Stage Dock
    #----------------------------------------------------------------------------
    
    @pyqtSlot()
    def on_stageOn_released(self):

        is_on = self.imp.dvc_toggle('STAGE')
        self.stageButtonsGroup.setEnabled(is_on)

    @pyqtSlot()
    def on_stageUp_released(self):

        self.imp.move_stage(dir='UP', steps=self.stageSteps.value())

    @pyqtSlot()
    def on_stageDown_released(self):

        self.imp.move_stage(dir='DOWN', steps=self.stageSteps.value())
    
    @pyqtSlot()
    def on_stageLeft_released(self):

        self.imp.move_stage(dir='LEFT', steps=self.stageSteps.value())

    @pyqtSlot()
    def on_stageRight_released(self):

        self.imp.move_stage(dir='RIGHT', steps=self.stageSteps.value())

    @pyqtSlot()
    def on_stageRelease_released(self):
 
        self.imp.release_stage()

class ErrWin(QDialog):
    
    '''Documentation'''
    # TODO: when error occures, change appropriate list items background to red, font to white
    
    def __init__(self, app, parent=None):
        super(ErrWin,  self).__init__(parent)
        uic.loadUi(const.ERRORSWINDOW_UI_PATH, self)
        self.imp = logic.ErrWin(self, app)
    
    @pyqtSlot(int)
    def on_errorSelectList_currentRowChanged(self, index):
        self.errorDetailsStacked.setCurrentIndex(index)

class SettWin(QDialog):
    
    '''Documentation'''
    
    def __init__(self, app, parent=None):
        super(SettWin,  self).__init__(parent)
        uic.loadUi(const.SETTINGSWINDOW_UI_PATH, self)
        self.imp = logic.SettWin(self, app)
    
    def closeEvent(self, event):
    
        self.imp.clean_up()

    @pyqtSlot()
    def on_saveButton_released(self):
        
        '''Save settings as .csv'''
        # TODO: add all forms in main window too
        
        self.imp.write_csv()

    @pyqtSlot()
    def on_loadButton_released(self):
        
        '''load settings .csv file'''
        # TODO: add all forms in main window too
        
        self.imp.read_csv()

class CamWin(QWidget):

    def __init__(self, app, parent=None):
        
        super(CamWin,  self).__init__(parent, Qt.WindowStaysOnTopHint)
        uic.loadUi(const.CAMERAWINDOW_UI_PATH, self)
        self.move(30, 180)
        self.imp = logic.CamWin(self, app)

    def closeEvent(self, event):
        
        self.imp.clean_up()

    @pyqtSlot()
    def on_shootButton_released(self):

        self.imp.shoot()

    @pyqtSlot()
    def on_videoButton_released(self):
        
        self.imp.toggle_video(True) if (self.videoButton.text() == 'Start Video') else self.imp.toggle_video(False)
