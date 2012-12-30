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
        
    @property
    def quick_parameters(self):
        return QuickParameterObject(self.parameters)
        
    def dump(self, stream):
        yaml_obj = dict(id=self.id,
                        model=self.model,
                        stop_on=self.stop_on,
                        parameters=self.parameters)
        yaml.safe_dump(yaml_obj, stream, default_flow_style=False)
            
    def __str__(self):
        return self.id


class ParameterValueError(Exception):
    def __init__(self, name, value, reason=None):
        super(ParameterValueError, self).__init__({'name':name, 
                                                   'value':value, 
                                                   'reason':reason})

class MissingParameterError(Exception):
    def __init__(self, name):
        super(MissingParameterError, self).__init__(name)
    
class QuickParameterObject(object):
    
    def __init__(self, parameters):
        self.__dict__ = parameters
        
    def is_between(self, name, min_value, max_value, value_type=int):
        value = getattr(self, name)
        if not isinstance(value, value_type) or value < min_value \
                                            or value > max_value:
            raise ParameterValueError(name, value, 
                                      "Must be a {0} in the range {1} to {2}."\
                                      .format(value_type, min_value, max_value))
        
    def __getattr__(self, name):
        raise MissingParameterError(name)
        
    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__, self.__dict__)


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

    
