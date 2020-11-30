'''
Measurements Module
'''

class Measurement():
    
    # TODO: move timer to timeout_loop. this would mean all updates will be made from there too (progress bar etc.)
    
    def __init__(self, app, type=None,
                       duration_spinner=None,
                       prog_bar=None,
                       log=None):
                           
        # instance attributes
        self._app = app
        self.type = type
        self.duration_spinner = duration_spinner
        self.prog_bar = prog_bar
        self.log = log
        self.time_passed = 0
        self.update_ready = False
        
    def start(self):
        
        self._app.dvc_dict['TDC'].toggle(True)
        self.log.update(F"{self.type} measurement started")
        
    def stop(self):
        
        self.type = None
        
        self._app.dvc_dict['TDC'].toggle(False)
            
        if self.prog_bar:
            self.prog_bar.setValue(0)
            
        self.log.update(F"{self.type} measurement stopped")
    
    def toggle_update_ready(self, bool):
        
        self.update_ready = bool
    
    def disp_ACF(self):
        
        print(F"Measurement Finished:\n"
                F"Full Data = {self._app.dvc_dict['UM232'].data}\n"
                F"Total Bytes = {self._app.dvc_dict['UM232'].tot_bytes}\n")