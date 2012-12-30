'''
Created on Dec 29, 2012

@author: Chase Kernan
'''

import numpy as np
import cv2
import random as rdm
import specs

def from_spec(spec):
    cls = _get_model_class(spec)
    return cls(spec)

def _get_model_class(spec):
    name = spec.model
    for match in [ProbabilisticAutomataModel]:
        if match.__name__ == name:
            return match
    raise specs.ParameterValueError("model", name,
                                    "No such model {0}.".format(name))

class Model(object):
    
    def __init__(self, spec):
        self.spec = spec
        self._p = spec.quick_parameters
        self._verify_parameters()
        
        self.num_cells = self._p.rows, self._p.columns
        self.min_dimension = min(self.num_cells)
        
    def _verify_parameters(self):
        self._p.is_between("rows", 1, 512, value_type=int)
        self._p.is_between("columns", 1, 512, value_type=int)
        
    @property
    def mass(self):
        raise NotImplementedError()
    
    @property
    def max_height(self):
        raise NotImplementedError()
    
    def step(self):
        raise NotImplementedError()
    
    def render(self):
        raise NotImplementedError()
    
    def reset(self):
        pass
    
    def run(self):
        self.reset()
        
        should_stop = self._get_stopping_function()
        t = 0
        while not should_stop(t):
            self.step()
            t += 1
            
    def _get_stopping_function(self):
        clauses = []
        for name, value in self.spec.stop_on.iteritems():
            if name == "mass":
                max_mass = int(value)
                clauses.append(lambda time: self.mass >= max_mass)
            elif name == "time":
                max_time = int(value)
                clauses.append(lambda time: time >= max_time)
            elif name == "max height":
                max_height = int(value)
                clauses.append(lambda time: self.max_height >= max_height)
            else:
                raise specs.ParameterValueError("stop_on", self.spec.stop_on,
                        "No such stopping function {0}.".format(name))
        
        return lambda time: any(clause(time) for clause in clauses)
    
ALIVE = 1
DEAD = 0
    
class CellularAutomataModel(Model):

    def reset(self):
        super(CellularAutomataModel, self).reset()
        
        self.cells = np.zeros(self.num_cells, np.uint8)
        self.boundary_layer = np.zeros(self.num_cells, np.uint8)
        self.media = np.zeros(self.num_cells, float)
        self.light = np.zeros(self.num_cells, float)
        self.division_probability = np.zeros(self.num_cells, float)
        self.dividing = np.zeros(self.num_cells, bool)
        self.surface_tension = np.zeros(self.num_cells, float)
        
        self.__max_row = self._p.rows-1
        self.__max_column = self._p.columns-1
        self.__mass = 0
        self.__max_height = 0
        
        self._place_cells_regularly()
    
    def _verify_parameters(self):
        super(CellularAutomataModel, self)._verify_parameters()
        
        self._p.is_between("boundary_layer", 0, 32)
        self._p.is_between("light_penetration", 0, 1024)
        self._p.is_between("media_concentration", 0.0, 10.0, value_type=float)
        self._p.is_between("media_penetration", 0, 32)
        self._p.is_between("division_constant", 0.00001, 10.0, value_type=float)
        self._p.is_between("initial_cell_spacing", 0, self._p.columns-1)
        
    def render(self):
        return self.cells*255
    
    @property
    def mass(self):
        return self.__mass
    
    @property
    def max_height(self):
        return self.__max_height
    
    def set_alive(self, row, column):
        row = max(0, min(self.__max_row, row))
        column = max(0, min(self.__max_column, column))
        
        if self.cells[row, column] == ALIVE:
            return
        
        self.cells[row, column] = ALIVE
        self.__mass += 1
        if row > self.__max_height:
            self.__max_height = row
    
    def _place_random_cells(self, probability=0.2):
        for column in range(self._p.columns):
            if rdm.random() < probability:
                self.set_alive(0, column)

    def _place_cells_regularly(self, spacing=None):
        if not spacing:
            spacing = self._p.initial_cell_spacing
        
        start = int(spacing/2)
        end = self._p.columns-int(spacing/2)
        for column in range(start, end+1, spacing):
            self.set_alive(0, column)
    
    def _calculate_media(self):
        kernel = _make_circular_kernel(self._p.boundary_layer)
        cv2.filter2D(self.cells, -1, kernel, self.boundary_layer)

        np.logical_not(self.boundary_layer, out=self.media)
        self.media *= self._p.media_concentration

        cv2.GaussianBlur(self.media, (0, 0), self._p.media_penetration,
                         dst=self.media)

    def _calculate_light(self):
        if self._p.light_penetration != 0:
            np.cumsum(self.cells, axis=0, out=self.light)
            self.light /= -self._p.light_penetration
            np.exp(self.light, out=self.light)
        else:
            self.light = np.ones(self.num_cells, dtype=float)

    def _calculate_surface_tension(self, center_factor=0):
        k = center_factor
        tension_kernel = np.array([[1, 2, 1],
                                   [2, k, 2],
                                   [1, 2, 1]], dtype=np.uint8)
        local_sum = cv2.filter2D(self.cells, -1, tension_kernel)
        self.surface_tension = local_sum/np.float(tension_kernel.sum())

    def _calculate_division_probability(self):
        self.division_probability = self._p.division_constant*\
                                   self.media*self.light
        self.division_probability[np.logical_not(self.cells)] = 0

    def _calculate_dividing_cells(self):
        self.dividing = np.random.ranf(self.num_cells) <= \
                        self.division_probability

    def _calculate_all_through_division_probability(self):
        self._calculate_media()
        self._calculate_light()
        self._calculate_division_probability()
        
    def make_presentation_image(self):
        image = np.zeros(self.num_cells + (3,), np.uint8)
        
        media = (self.media*127).astype(np.uint8)
        image[:, :, 0] = media
        image[:, :, 2] = media
        
        minValue = 64
        normalized = self.division_probability/self.division_probability.max()
        image[:, :, 1] = minValue + normalized*(255 - minValue)
        image[:, :, 1][np.logical_not(self.cells)] = 0
        image[:, :, 0][self.cells] = minValue
        
        very_likely = np.logical_and(self.cells, normalized > 0.5)
        image[:, :, 2][very_likely] = 127 + 127*normalized[very_likely]
        
        return image   
        
class ProbabilisticAutomataModel(CellularAutomataModel):

    def reset(self):
        super(ProbabilisticAutomataModel, self).reset()
        
        self.distance_kernel = _generate_distance_kernel(self._p.block_size)
        self.distance_kernel **= self._p.distance_power
        self.distance_kernel /= self.distance_kernel.sum()
        self.tension_kernel = np.array([[1, 2, 1],
                                        [2, 0, 2],
                                        [1, 2, 1]], float)
        self.tension_kernel /= self.tension_kernel.sum()
        self.tension_min = self.tension_kernel[0:2, 0].sum()
        
        shape = self._p.block_size, self._p.block_size
        self.probability = np.empty(shape, np.float32)
        self.cumulative = np.empty(self.probability.size, np.float32)
        self.indices = np.arange(self.probability.size)
        self.cell_block = np.empty(shape, np.uint8)
        

    def _verify_parameters(self):
        super(CellularAutomataModel, self)._verify_parameters()
        
        self._p.is_between("distance_power", 0.0, 4.0, value_type=float)
        self._p.is_between("tension_power", 0.0, 4.0, value_type=float)
        self._p.is_between("block_size", 3, 15)
        if self._p.block_size % 2 != 1:
            raise specs.ParameterValueError("block_size", self._p.block_size,
                                            "Must be an odd integer.")

    def step(self):
        self._calculate_all_through_division_probability()
        self._calculate_dividing_cells()
        self._divide()

    def _divide(self):        
        block_size = self._p.block_size # shortcut
        half_block = (block_size-1)/2
        
        rows, columns = map(list, self.dividing.nonzero())
        for row, column in zip(rows, columns):
            _write_block(self.cell_block, self.cells, row, column, block_size)
            cv2.filter2D(self.cell_block, cv2.cv.CV_32F, self.tension_kernel,
                         self.probability)
            cv2.threshold(self.probability, self.tension_min, 0, 
                          cv2.THRESH_TOZERO, self.probability)
            self.probability[self.cell_block] = 0
            self.probability **= self._p.tension_power
            self.probability *= self.distance_kernel
            
            # optimized version of np.random.choice
            np.cumsum(self.probability.flat, out=self.cumulative)
            total = self.cumulative[-1]
            if total < 1.0e-12:
                # no viable placements, we'll have precision problems anyways
                continue 
            self.cumulative /= total
            
            index = self.indices[np.searchsorted(self.cumulative, rdm.random())]
            local_row, local_column = np.unravel_index(index, 
                                                       self.probability.shape)
            self.set_alive(row+(local_row-half_block), 
                          column+(local_column-half_block))
    
def _make_circular_kernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))

def _generate_distance_kernel(size=7):
    kernel = np.empty((size, size), dtype=float)
    center = (size - 1)/2
    for row in range(size):
        for column in range(size):
            dx = row - center
            dy = column - center
            kernel[row, column] = dx**2 + dy**2
    kernel = np.sqrt(kernel)
    
    # avoid a 0 divide
    kernel[center, center] = 1.0
    kernel = 1.0/kernel
    kernel[center, center] = 0.0

    return kernel # we don't need to normalize here, we'll do it later

def _write_block(block, matrix, row, column, block_size, filler=0):
    x = (block_size-1)/2
    left = max(0, column-x)
    right = min(matrix.shape[1]-1, column+x+1)
    top = max(0, row-x)
    bottom = min(matrix.shape[0]-1, row+x+1)
    
    block.fill(filler)
    block[x-(row-top) : x+(bottom-row),
          x-(column-left) : x+(right-column)] = matrix[top:bottom, left:right]