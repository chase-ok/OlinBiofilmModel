'''
Created on Nov 12, 2012

@author: Chase Kernan
'''

import tables; tb = tables
import utils
import numpy as np
from itertools import product

DEFAULT_MODEL = "ProbabilisticAutomataModel"
MODEL_ENUM = tb.Enum([DEFAULT_MODEL])

class Spec(tb.IsDescription):
    id = tb.Int32Col(dflt=0, pos=0)
    model = tb.EnumCol(MODEL_ENUM, DEFAULT_MODEL, base='uint8')
    stop_on_mass = tb.UInt32Col(dflt=2000)
    stop_on_time = tb.UInt32Col(dflt=10000)
    stop_on_height = tb.UInt32Col(dflt=0)
    stop_on_no_growth = tb.UInt32Col(dflt=200)
    block_size = tb.UInt8Col(dflt=15)
    boundary_layer = tb.UInt8Col(dflt=8)
    media_concentration = tb.Float32Col(dflt=1.0)
    light_penetration = tb.UInt16Col(dflt=8)
    distance_power = tb.Float32Col(dflt=2.0)
    tension_power = tb.Float32Col(dflt=1.0)
    initial_cell_spacing = tb.UInt16Col(dflt=2)
    division_constant = tb.Float32Col(dflt=1.0)
    diffusion_constant = tb.Float32Col(dflt=1.0)
    dt = tb.Float32Col(dflt=1.0)
    uptake_rate = tb.Float32Col(dflt=0.1)
    num_diffusion_iterations = tb.UInt32Col(dflt=20)
    monod_constant = tb.Float32Col(dflt=0.75)

specs = utils.QuickTable("specs", Spec,
                         filters=tb.Filters(complib='blosc', complevel=1),
                         sorted_indices=['id'])

def create_spec(**kwargs):
    spec = specs.table.row
    id = specs.table.nrows
    spec['id'] = id
    for key, value in kwargs.iteritems():
        spec[key] = value
    spec.append()
    
    specs.flush() # have to flush to update nrows
    return id

def get_spec(id, wrapped=True):
    spec = specs.read_single('id == {0}'.format(id))
    return _SpecWrapper(spec) if wrapped else spec
    
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
            create_spec(**value_set)
        specs.flush()
        self._all_values = {}


class ParameterValueError(Exception):
    def __init__(self, name, value, reason=None):
        super(ParameterValueError, self).__init__({'name':name, 
                                                   'value':value, 
                                                   'reason':reason})
    
class _QuickParameterObject(object):
    
    def __init__(self, parameters):
        self.__dict__ = parameters
        
    def is_between(self, name, min_value, max_value):
        value = getattr(self, name)
        if value < min_value or value > max_value:
            raise ParameterValueError(name, value, 
                                      "Must be in the range {1} to {2}."\
                                      .format(min_value, max_value))
        
    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__, self.__dict__)

class _SpecWrapper(object):
    
    def __init__(self, row):
        self._row = row
    
    @property
    def id(self):
        return self._row['id']
    
    @property
    def model(self):
        return MODEL_ENUM(self._row['model'])
    
    @property
    def stop_on(self):
        stop_on = {}
        for prop in ['mass', 'time', 'height', 'no_growth']:
            value = self._row['stop_on_' + prop]
            if value != 0:
                stop_on[prop] = value
        return stop_on
    
    @property
    def parameters(self):
        params = {}
        for param in ['block_size', 
                      'boundary_layer', 
                      'media_concentration', 
                      'light_penetration',
                      'distance_power', 
                      'tension_power', 
                      'initial_cell_spacing', 
                      'division_constant',
                      'diffusion_constant',
                      'dt',
                      'uptake_rate',
                      'num_diffusion_iterations',
                      'monod_constant']:
            params[param] = self._row[param]
        return params
    
    def make_quick_parameters(self):
        return _QuickParameterObject(self.parameters)

def make_query(use_defaults=False, **conditions):
    specified = set()
    clauses = []
    for column, column_clauses in conditions.iteritems():
        specified.add(column)
        if isinstance(column_clauses, basestring):
            column_clauses = (column_clauses,)
        clauses.extend((column + clause) for clause in column_clauses)
    
    if use_defaults:
        for column, data_type in Spec.columns.iteritems():
            if column != 'id' and column not in specified:
                clauses.append("{0}=={1}".format(column, data_type.dflt))
    
    return "&".join("({0})".format(clause) for clause in clauses)
        
boundary_vs_media_query = make_query(id='<100')

distance_vs_tension_query = make_query(id=('>=100', '<200'))
distance_vs_tension_gt0_query = make_query(id=('>=100', '<200'),
                                           distance_power='>0',
                                           tension_power='>0')

light_vs_division_query = make_query(id=('>=200', '<920'))

light_overhang_query = make_query(id=('>=200', '<920'),
                                  distance_power='==1.0',
                                  tension_power='==1.0',
                                  division_constant='==1.0')

