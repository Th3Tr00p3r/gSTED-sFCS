"""
GUI Module.
"""
import implementation.logic as logic
import implementation.constants as const

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (QMainWindow, QDialog)
from PyQt5 import uic

# resource files
from gui.icons import icons_rc # for initial icons # NOQA

class MainWindow(QMainWindow):
    """
    Class documentation goes here.
    """
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        uic.loadUi(const.MAINWINDOW_UI_PATH, self)
        self.imp = logic.MainWin(self)
        # initialize other windows
        self.windows = {}
        self.windows['settings'] = SettingsWindow()
        self.windows['errors'] = ErrorsWindow()
        self.windows['cameras'] = CameraWindow()

    def closeEvent(self, event):
        self.imp.exit_app(event)

    @pyqtSlot()
    def on_startFcsMeasurementButton_released(self):
        '''
        Begin FCS Measurement.
        '''
        self.imp.start_FCS_meas()

    @pyqtSlot()
    def on_excOnButton_released(self):
        '''
        Turn excitation laser On/Off
        '''
        
        self.imp.exc_emission_toggle()

    @pyqtSlot()
    def on_depEmissionOn_released(self):
        '''
        Turn depletion laser On/Off
        '''
        
        self.imp.dep_emission_toggle()

    @pyqtSlot()
    def on_depShutterOn_released(self):
        '''
        Turn depletion physical shutter On/Off
        '''
        
        self.imp.dep_shutter_toggle()

    @pyqtSlot(int)
    def on_depSetCombox_currentIndexChanged(self, index):
        '''
        Switch between power/current depletion laser settings
        '''
        
        self.depModeStacked.setCurrentIndex(index)

    @pyqtSlot(int)
    def on_solScanTypeCombox_currentIndexChanged(self, index):
        '''
        Change stacked widget 'solScanParamsStacked' index according to index of the combo box 'solScanTypeCombox'.

        @param index - the index of the combo box 'solScanTypeCombox'
        @type int
        '''
        
        self.solScanParamsStacked.setCurrentIndex(index)

    @pyqtSlot()
    def on_actionSettings_triggered(self):
        '''
        Show settings window
        '''
        
        self.windows['settings'].show()
        self.windows['settings'].activateWindow()

    @pyqtSlot(bool)
    def on_actionLaser_Control_toggled(self, p0):
        '''
        Show/hide stepper laser control dock
        '''
        self.laserDock.setVisible(p0)

    @pyqtSlot(bool)
    def on_actionStepper_Stage_Control_toggled(self, p0):
        '''
        Show/hide stepper stage control dock
        '''
        self.stepperDock.setVisible(p0)

    @pyqtSlot()
    def on_actionCamera_Control_triggered(self):
        '''
        Instantiate 'CameraWindow' object and show it
        '''
        # TODO: add support for 2nd camera
        self.windows['cameras'].show()
        self.windows['cameras'].activateWindow()
        self.windows['cameras'].imp.init_cam()

    @pyqtSlot(bool)
    def on_actionLog_toggled(self, p0):
        '''
        Show/hide log dock
        '''
        self.logDock.setVisible(p0)

    @pyqtSlot()
    def on_ledErrors_clicked(self):
        '''
        Show errors window
        '''
        # TODO:
        self.windows['errors'].show()
        self.windows['errors'].activateWindow()

    #----------------------------------------------------------------------------
    # Stepper Stage Dock
    #----------------------------------------------------------------------------
    
    @pyqtSlot()
    def on_stageOn_released(self):

        self.imp.stage_toggle()

    @pyqtSlot()
    def on_stageUp_released(self):

        self.imp.stage.move(dir='UP', steps=self.stageSteps.value())

    @pyqtSlot()
    def on_stageDown_released(self):

        self.imp.stage.move(dir='DOWN', steps=self.stageSteps.value())

    @pyqtSlot()
    def on_stageLeft_released(self):

        self.imp.stage.move(dir='LEFT', steps=self.stageSteps.value())

    @pyqtSlot()
    def on_stageRight_released(self):

        self.imp.stage.move(dir='RIGHT', steps=self.stageSteps.value())

    @pyqtSlot()
    def on_stageRelease_released(self):
 
        self.imp.stage.release()

class ErrorsWindow(QDialog):
    '''
    This "window" is a QWidget. If it has no parent, it 
    will appear as a free-floating window as we want.
    '''
    # TODO: when error occures, change appropriate list items background to red, font to white
    def __init__(self,  parent=None):
        super(ErrorsWindow,  self).__init__(parent)
        uic.loadUi(const.ERRORSWINDOW_UI_PATH, self)

    @pyqtSlot(int)
    def on_errorSelectList_currentRowChanged(self, index):
        self.errorDetailsStacked.setCurrentIndex(index)

class SettingsWindow(QDialog):
    '''
    This "window" is a QWidget. If it has no parent, it 
    will appear as a free-floating window as we want.
    '''
    def __init__(self,  parent=None):
        super(SettingsWindow,  self).__init__(parent)
        uic.loadUi(const.SETTINGSWINDOW_UI_PATH, self)
        self.imp = logic.SettingsWin(self)
        self.imp.read_csv(const.DEFAULT_SETTINGS_FILE_PATH)
    
    def closeEvent(self, event):
        print('Settings window closed (closeEvent)') # TEST
        self.imp.clean_up()

    @pyqtSlot()
    def on_saveButton_released(self):
        '''
        Save settings as .csv
        '''
        # TODO: add all forms in main window too
        self.imp.write_csv()

    @pyqtSlot()
    def on_loadButton_released(self):
        '''
        load settings .csv file
        '''
        # TODO: add all forms in main window too
        self.imp.read_csv()

class CameraWindow(QDialog):
    '''
    documentation
    '''
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
        self.imp.clean_up()

    @pyqtSlot()
    def on_shootButton_released(self):
        '''
        Slot documentation goes here.
        '''
        self.imp.shoot()

    @pyqtSlot()
    def on_videoButton_released(self):
        '''
        Slot documentation goes here.
        '''
        self.imp.toggle_video()
