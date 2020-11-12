import os
from datetime import datetime
import implementation.constants as const

class Log():
    
    def __init__(self, gui, dir_path):
        
        self.gui = gui
        self.dir_path = dir_path
        try:
            os.mkdir(self.dir_path)
        except:
            pass
        date_str = datetime.now().strftime("%d_%m_%Y")
        self.file_path = self.dir_path + date_str + '.csv'
        self.update('Application Started')
    
    def update(self, log_line, tag='always'):
        
        # add line to log file
        if tag in const.VERBOSITY:
            with open(self.file_path, 'a+') as file:
                time_str = datetime.now().strftime("%H:%M:%S")
                file.write(time_str + ' ' + log_line + '\n')
            # read log file to log dock
            self.read_log()
            # read last line
            self.gui.lastActionLineEdit.setText(self.get_last_line())
    
    def read_log(self):
        
        with open(self.file_path, 'r') as file:
            log = []
            for line in reversed(list(file)):
                log.append(line.rstrip())
            log = '\n'.join(log)
        self.gui.logText.setPlainText(log)
    
    def get_last_line(self):
        
        with open(self.file_path, 'r') as file:
            for line in reversed(list(file)):
                line = line.rstrip()
                break
        return line
