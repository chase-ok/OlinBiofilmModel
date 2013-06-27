'''
Created on Nov 12, 2012

@author: Chase Kernan
'''

import tables; tb = tables
import utils
import numpy as np
from itertools import product

DEFAULT_PARAMETERS = {
    'stop_on_mass': 4000,
    'stop_on_time': 30000,
    'stop_on_height': 64,
    'stop_on_no_growth': 500,
    'width': 128,
    'height': 64,
    'block_size': 11,
    'boundary_layer': 10,
    'media_concentration': 1.0,
    'light_penetration': 16,
    'distance_power': 2.0,
    'tension_power': 2.5,
    'initial_cell_spacing': 2,
    'division_constant': 1.0,
    'diffusion_constant': 0.5,
    'uptake_rate': 0.1,
    'monod_constant': 1.0,
    'max_diffusion_iterations': 5000,
    'diffusion_tol': 1e-4
}

class Spec(utils.TableObject):

    def __init__(self, uuid=None, **kwargs):
        utils.TableObject.__init__(self, uuid)
        params = _with_defaults(kwargs, DEFAULT_PARAMETERS)
        for name, value in params.iteritems():
            setattr(self, name, value)

    @property
    def shape(self): return (self.height, self.width)

    def is_between(self, name, min_value, max_value):
        value = getattr(self, name)
        if value < min_value or value > max_value:
            raise ParameterValueError(name, value, 
                                      "Must be in the range {1} to {2}."\
                                      .format(min_value, max_value))
        
    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__, self.__dict__)

class SpecTable(tb.IsDescription):
    uuid = utils.make_uuid_col()
    stop_on_mass = tb.UInt32Col()
    stop_on_time = tb.UInt32Col()
    stop_on_height = tb.UInt32Col()
    stop_on_no_growth = tb.UInt32Col()
    width = tb.UInt16Col()
    height = tb.UInt16Col()
    block_size = tb.UInt8Col()
    boundary_layer = tb.UInt8Col()
    media_concentration = tb.Float32Col()
    light_penetration = tb.UInt16Col()
    distance_power = tb.Float32Col()
    tension_power = tb.Float32Col()
    initial_cell_spacing = tb.UInt16Col()
    division_constant = tb.Float32Col()
    diffusion_constant = tb.Float32Col()
    diffusion_tol = tb.Float32Col()
    uptake_rate = tb.Float32Col()
    max_diffusion_iterations = tb.UInt32Col()
    monod_constant = tb.Float32Col()
Spec.setup_table("specs", SpecTable)

class ParameterValueError(Exception):
    def __init__(self, name, value, reason=None):
        super(ParameterValueError, self).__init__({'name':name, 
                                                   'value':value, 
                                                   'reason':reason})

def _with_defaults(d, defaults):
    new_d = defaults.copy()
    if d:
        for key, value in d.iteritems():
            if key not in defaults:
                raise ValueError("{0} not a valid key.".format(key))
            new_d[key] = value
    return new_d
    
class SpecBuilder(object):
    
    def __init__(self):
        self._all_values = {}
    
    def add(self, column, *values):
        self._all_values.setdefault(column, []).extend(values)
        
    @property
    def num_specs(self):
        return np.product([len(v) for v in self._all_values.itervalues()])

    @property
    def value_sets(self):
        return map(dict, product(*([(name, value) for value in values]
                                   for name, values 
                                   in self._all_values.iteritems())))
    
    def build(self):
        for value_set in self.value_sets:
            spec = Spec(**value_set)
            spec.save(flush=False)
        Spec.table.flush()
        self.clear()

    def clear(self):
        self._all_values = {}

def make_query(use_defaults=False, **conditions):
    specified = set()
    clauses = []
    for column, column_clauses in conditions.iteritems():
        specified.add(column)
        if isinstance(column_clauses, basestring):
            column_clauses = (column_clauses,)
        clauses.extend((column + clause) for clause in column_clauses)
    
    if use_defaults:
        for param, value in DEFAULT_PARAMETERS.iteritems():
            if param not in specified:
                clauses.append("{0}=={1}".format(param, value))
    
    return "&".join("({0})".format(clause) for clause in clauses)
