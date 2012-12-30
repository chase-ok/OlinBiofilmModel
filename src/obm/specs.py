'''
Created on Nov 12, 2012

@author: Chase Kernan
'''

import yaml
import uuid

SPEC_FILE_TYPE = "spec"
        
def generate_id():
    return str(uuid.uuid4())
        
DEFAULT_MODEL = "ProbabilisticAutomataModel"
DEFAULT_STOP_ON = dict(mass=5000, time=20000)
DEFAULT_PARAMETERS = dict(block_size=7,
                          rows=48, columns=256,
                          boundary_layer=6,
                          media_concentration=1.0,
                          media_penetration=4,
                          light_penetration=8,
                          distance_power=1.0,
                          tension_power=1.0,
                          initial_cell_spacing=2,
                          division_constant=1.0)

class Spec(object):
    
    def __init__(self, id=None,
                       model=DEFAULT_MODEL,
                       stop_on=DEFAULT_STOP_ON,
                       parameters=DEFAULT_PARAMETERS):
        self.id = generate_id() if id is None else id
        self.model = model
        self.stop_on = stop_on.copy()
        self.parameters = parameters.copy()
        
    def dump(self, stream):
        yaml_obj = dict(id=self.id,
                        model=self.model,
                        stop_on=self.stop_on,
                        parameters=self.parameters)
        yaml.safe_dump(yaml_obj, stream, default_flow_style=False)
            
    def __str__(self):
        return self.id

class SpecFileError(Exception):
    def __init__(self, path, cause):
        super(SpecFileError, self).__init__({'path':path, 'cause':cause})

def from_file(path):
    try:
        with open(path, 'id') as stream:
            yaml_obj = yaml.load(stream)
    except yaml.YAMLError as error:
        raise SpecFileError(path, error)
    except IOError as error:
        raise SpecFileError(path, error)
    
    try:
        return Spec(id=yaml_obj['id'],
                    model=yaml_obj['model'],
                    stop_on=yaml_obj['stop_on'],
                    parameters=yaml_obj['parameters'])
    except KeyError as e:
        raise SpecFileError(path, "Missing {0} section.".format(e.args[0]))

    
