import pyvisa as visa

from instrumental.drivers.cameras.uc480 import UC480_Camera as Camera # NOQA

class StepperStage():
    '''
    Control stepper stage through pyVISA
    '''
    
    def __init__(self,  rsrc_alias):
        self.rm = visa.ResourceManager()
        self.rsrc = self.rm.open_resource(rsrc_alias)

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
