'''
Created on Dec 29, 2012

@author: Chase Kernan
'''

import numpy as np
import cv2
import random as rdm
import specs
import tables; tb = tables
import utils
import re
import utils
from scipy.ndimage.filters import laplace

class Result(utils.TableObject):

    def __init__(self, uuid=None, spec_uuid=None, image=None, mass=None):
        utils.TableObject.__init__(self, uuid)
        self.spec_uuid = spec_uuid
        self.image = image; self._exclude.add('image')
        self.mass = mass; self._exclude.add('mass')

    @property
    @utils.memoized
    def spec(self): return specs.Spec.get(self.spec_uuid)

    @property
    @utils.memoized
    def int_image(self): return self.image.astype(np.uint8)

    def _on_get(self, row):
        self.image = _get_image(self.uuid, self.spec.shape)
        self.mass = _get_mass(self.uuid, self.spec.stop_on_time)

    def _fill_row(self, row):
        utils.TableObject._fill_row(self, row)
        _save_image(self.uuid, self.image)
        _save_mass(self.uuid, self.mass)

    def _on_delete(self):
        utils.TableObject._on_delete(self)
        _delete_image(self.uuid, self.spec.shape)
        _delete_image(self.uuid, self.spec.stop_on_time)

class ResultTable(tb.IsDescription):
    uuid = utils.make_uuid_col()
    spec_uuid = tb.StringCol(32)
Result.setup_table("results", ResultTable, 
                   sorted_indices=['uuid', 'spec_uuid'])

_get_image, _save_image, _delete_image \
        = utils.make_variable_data("_results_image", tb.BoolCol,
                                   filters=tb.Filters(complib='zlib', 
                                                      complevel=9))
_get_mass, _save_mass, _delete_mass \
        = utils.make_variable_data("_results_mass", tb.UInt16Col,
                                   filters=tb.Filters(complib='zlib', 
                                                      complevel=9))
@utils.memoized
def _get_image_table(size):
    class Image(utils.EasyTableObject): pass
    class ImageTable(tb.IsDescription):
        uuid = utils.make_uuid_col()
        image = tb.BoolCol(shape=size)

    name = "_result_images_{0}x{1}".format(*size)
    Image.setup_table(name, ImageTable,
                      filters=tb.Filters(complib='zlib', complevel=9))
    return Image

@utils.memoized
def _get_mass_table(length):
    class Mass(utils.EasyTableObject): pass
    class MassTable(tb.IsDescription):
        uuid = utils.make_uuid_col()
        mass = tb.UInt16Col(shape=(1, length))

    name = "_result_mass_{0}".format(length)
    Mass.setup_table(name, MassTable,
                     filters=tb.Filters(complib='zlib', complevel=9))
    return Mass

def compute_probabilistic(spec):
    model = ProbabilisticAutomataModel(spec)
    result = model.run()
    return result

class Model(object):
    
    def __init__(self, spec):
        self.spec = spec
        self._verify_parameters()
        
        self.num_cells = self.spec.shape
        self.min_dimension = min(self.num_cells)
        self.last_growth = 0
        self.time = 0
        
    def _verify_parameters(self):
        pass
        
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
        mass = np.zeros(self.spec.stop_on_time, np.uint16)
        self.time = 0
        while not should_stop():
            mass[self.time] = self.mass
            self.step()
            self.time += 1

        return Result(spec_uuid=self.spec.uuid, 
                      image=self.render(), 
                      mass=mass)
            
    def _get_stopping_function(self):
        clauses = []
        for full_name, value in self.spec.__dict__.iteritems():
            if value == 0: continue
            match = re.match(r"stop_on_(\w+)", full_name)
            if not match: continue
            name = match.group(1)

            if name == "mass":
                max_mass = int(value)
                clauses.append(lambda: self.mass >= max_mass)
            elif name == "time":
                max_time = int(value)
                clauses.append(lambda: self.time >= max_time)
            elif name == "height":
                max_height = int(value)
                clauses.append(lambda: self.max_height >= max_height)
            elif name == 'no_growth':
                no_growth = int(value)
                clauses.append(lambda: self.time >= self.last_growth+no_growth)
            else:
                raise specs.ParameterValueError("stop_on", self.spec.stop_on,
                        "No such stopping function {0}.".format(name))
        
        return lambda: any(clause() for clause in clauses)
    
ALIVE = 1
DEAD = 0
    
class CellularAutomataModel(Model):

    def reset(self):
        super(CellularAutomataModel, self).reset()
        
        self.cells = np.zeros(self.num_cells, np.uint8)
        self.boundary_layer = np.zeros(self.num_cells, np.uint8)
        self.light = np.zeros(self.num_cells, float)
        self.division_probability = np.zeros(self.num_cells, float)
        self.dividing = np.zeros(self.num_cells, bool)
        self.surface_tension = np.zeros(self.num_cells, float)
        
        self.__max_row = self.num_cells[0]-1
        self.__max_column = self.num_cells[1]-1
        self.__mass = 0
        self.__max_height = 0
        
        self._place_cells_regularly()
    
    def _verify_parameters(self):
        super(CellularAutomataModel, self)._verify_parameters()
        
        self.spec.is_between("boundary_layer", 0, 32)
        self.spec.is_between("light_penetration", 0, 1024)
        self.spec.is_between("media_concentration", 0.0, 10.0)
        self.spec.is_between("division_constant", 0.00001, 10.0)
        self.spec.is_between("initial_cell_spacing", 0, COLUMNS-1)
        self.spec.is_between("diffusion_constant", 0.01, 1000.0)
        self.spec.is_between("dt", 0.01, 5.0)
        self.spec.is_between("uptake_rate", 0.001, 5.0)
        self.spec.is_between("num_diffusion_iterations", 1, 10**6)
        self.spec.is_between("monod_constant", 0.0001, 2.0)
        
    def render(self):
        return self.cells.astype(bool)
    
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

        self.last_growth = self.time
    
    def _place_random_cells(self, probability=0.2):
        for column in range(COLUMNS):
            if rdm.random() < probability:
                self.set_alive(0, column)

    def _place_cells_regularly(self, spacing=None):
        if not spacing:
            spacing = self.spec.initial_cell_spacing
        
        start = int(spacing/2)
        end = self.num_cells[1]-int(spacing/2)
        for column in range(start, end+1, spacing):
            self.set_alive(0, column)

    def _make_boundary_layer(self, cells):
        kernel = _make_circular_kernel(self.spec.boundary_layer)
        boundary_layer = cv2.filter2D(cells.astype(np.uint8), -1, kernel)
        np.logical_not(boundary_layer, out=boundary_layer)

        #remove any non-connected segments
        fill_value = 2
        fill_source = boundary_layer.shape[0]-1, 0 # (x, y) not (r, c)
        cv2.floodFill(boundary_layer, None, fill_source, fill_value)

        return boundary_layer != fill_value
    
    def _calculate_media(self, check_start=40, check_interval_growth=5):
        check_interval = check_interval_growth
        dt = 1.0/(4*self.spec.diffusion_constant)

        cells = self._narrow_cells()
        boundary = self._make_boundary_layer(cells)

        media = cells*self.spec.media_concentration
        media_next = np.empty_like(media)

        for step in range(self.spec.max_diffusion_iterations):
            laplace(media, output=media_next)
            media_next *= self.spec.diffusion_constant*dt
            media_next += media
            media_next[cells] *= 1 - self.spec.uptake_rate*dt
            media_next[boundary] = self.spec.media_concentration
            media, media_next = media_next, media

            if step >= check_start and step%check_interval == 0:
                media_next -= media
                np.abs(media_next, out=media_next)
                error = media_next.sum()/(dt*media_next.size)
                if error <= self.spec.diffusion_tol: break
                check_interval += check_interval_growth

        self.media = media

    def _narrow_cells(self):
        return self.cells[0:self.max_height+self.spec.boundary_layer+1, :] > 0

    def _calculate_light(self):
        if self.spec.light_penetration != 0.0:
            np.cumsum(self.cells, axis=0, out=self.light)
            self.light /= -float(self.spec.light_penetration) # otherwise uint16
            np.exp(self.light, out=self.light)
        else:
            self.light.fill(1.0)

    def _calculate_surface_tension(self, center_factor=0):
        k = center_factor
        tension_kernel = np.array([[1, 2, 1],
                                   [2, k, 2],
                                   [1, 2, 1]], dtype=np.uint8)
        local_sum = cv2.filter2D(self.cells, -1, tension_kernel)
        self.surface_tension = local_sum/np.float(tension_kernel.sum())

    def _calculate_division_probability(self):
        media_prob = self.media/(self.media + self.spec.monod_constant)
        self.division_probability = self.spec.division_constant*self.light
        # since we narrowed the number of rows in the diffusion calc
        self.division_probability[0:media_prob.shape[0], :] *= media_prob
        self.division_probability[np.logical_not(self.cells)] = 0

    def _calculate_dividing_cells(self):
        self.dividing = np.random.ranf(self.num_cells) <= \
                        self.division_probability

    def _calculate_all_through_division_probability(self):
        if self.time - self.last_growth > 1: return
        self._calculate_media()
        self._calculate_light()
        self._calculate_division_probability()
        
    def make_presentation_image(self, prob_thresh=0.5):
        image = np.zeros(self.num_cells + (3,), np.uint8)
        
        media = (self.media*127).astype(np.uint8)
        image[:, :, 0] = media
        image[:, :, 2] = media
        
        minValue = 64
        normalized = self.division_probability/self.division_probability.max()
        image[:, :, 1] = minValue + normalized*(255 - minValue)
        image[:, :, 1][np.logical_not(self.cells)] = 0
        image[:, :, 0][self.cells] = minValue
        
        very_likely = np.logical_and(self.cells, normalized > prob_thresh)
        image[:, :, 2][very_likely] = 127 + 127*normalized[very_likely]
        
        return image   
        
class ProbabilisticAutomataModel(CellularAutomataModel):

    def reset(self):
        super(ProbabilisticAutomataModel, self).reset()
        
        self._distance_kernel = _generate_distance_kernel(self.spec.block_size)
        self._distance_kernel **= self.spec.distance_power
        self._distance_kernel /= self._distance_kernel.sum()
        self._tension_kernel = np.array([[1, 2, 1],
                                         [2, 0, 2],
                                         [1, 2, 1]], float)
        self._tension_kernel /= self._tension_kernel.sum()
        self.tension_min = self._tension_kernel[0:1, 0].sum()
        
        shape = self.spec.block_size, self.spec.block_size
        self._probability = np.empty(shape, np.float32)
        self._cumulative = np.empty(self._probability.size, np.float32)
        self._indices = np.arange(self._probability.size)
        self._cell_block = np.empty(shape, np.uint8)
        

    def _verify_parameters(self):
        super(CellularAutomataModel, self)._verify_parameters()
        
        self.spec.is_between("distance_power", 0.0, 4.0)
        self.spec.is_between("tension_power", 0.0, 4.0)
        self.spec.is_between("block_size", 3, 25)
        if self.spec.block_size % 2 != 1:
            raise specs.ParameterValueError("block_size", self.spec.block_size,
                                            "Must be an odd integer.")

    def step(self):
        self._calculate_all_through_division_probability()
        self._calculate_dividing_cells()
        self._divide()

    def _divide(self):        
        block_size = self.spec.block_size # shortcut
        half_block = (block_size-1)/2
        
        rows, columns = map(list, self.dividing.nonzero())
        for row, column in zip(rows, columns):
            _write_block(self._cell_block, self.cells, row, column, block_size)
            cv2.filter2D(self._cell_block, cv2.cv.CV_32F, self._tension_kernel,
                         self._probability, borderType=cv2.BORDER_CONSTANT)
            cv2.threshold(self._probability, self.tension_min, 0, 
                          cv2.THRESH_TOZERO, self._probability)
            self._probability[self._cell_block] = 0
            self._probability **= self.spec.tension_power
            self._probability *= self._distance_kernel
            
            # optimized version of np.random.choice
            np.cumsum(self._probability.flat, out=self._cumulative)
            total = self._cumulative[-1]
            if total < 1.0e-12:
                # no viable placements, we'll have precision problems anyways
                continue 
            self._cumulative /= total
            
            index = self._indices[np.searchsorted(self._cumulative, 
                                                  rdm.random())]
            local_row, local_column = np.unravel_index(index, 
                                                       self._probability.shape)
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
