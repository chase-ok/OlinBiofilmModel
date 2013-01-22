'''
Created on Dec 30, 2012

@author: chase_000
'''

import numpy as np
import cv2
import specs; sp = specs # fucking code analysis
import tables; tb = tables
import utils
import models
from scipy import interpolate
from matplotlib import pyplot as plt


BASIC_ANALYSIS_NODE = "basic_analysis"

class BasicAnalysis(tb.IsDescription):
    spec_id = tb.Int32Col(dflt=0, pos=0)
    model_id = tb.Int32Col(dflt=0, pos=1)
    mass = tb.Float32Col()
    perimeter = tb.Float32Col()
    heights = tb.UInt16Col(shape=(1, models.COLUMNS))
    max_height = tb.Float32Col()
    mean_height = tb.Float32Col()
    row_ffts = tb.Float32Col(shape=(models.ROWS, models.COLUMNS))
    coverages = tb.Float32Col(shape=(1, models.ROWS))
    convexity_max = tb.Float32Col()
    convexity_mean = tb.Float32Col()
    convexity_std = tb.Float32Col()
    x_correlations = tb.Float32Col(shape=(models.ROWS, models.COLUMNS/2-1))
    overhangs = tb.UInt8Col(shape=models.COLUMNS)
    overhang_heights = tb.UInt16Col(shape=models.COLUMNS)
    
basic_analysis = utils.QuickTable("basic_analysis", BasicAnalysis,
                                  filters=tb.Filters(complib='zlib', 
                                                     complevel=9),
                                  sorted_indices=['model_id', 'spec_id'])

def do_basic_analysis(model_id):
    result = models.model_results.table[model_id]
    analyzer = BasicAnalyzer(result['cells'])
    
    old = get_basic_analysis(model_id)
    row = basic_analysis.table.row if old is None else old
    row['spec_id'] = result['spec_id']
    row['model_id'] = model_id
    row['perimeter'] = analyzer.perimeter
    row['heights'] = analyzer.heights
    row['max_height'] = analyzer.max_height
    row['mean_height'] = analyzer.mean_height
    row['row_ffts'] = analyzer.row_ffts
    row['coverages'] = analyzer.coverages
    row['convexity_max'] = analyzer.convexity_max
    row['convexity_mean'] = analyzer.convexity_mean
    row['convexity_std'] = analyzer.convexity_std
    row['x_correlations'] = analyzer.x_correlations
    row['overhangs'] = analyzer.overhangs
    row['overhang_heights'] = analyzer.overhang_heights
    row['mass'] = analyzer.mass
    if old is None:
        row.append()
    else:
        row.update()
    basic_analysis.flush()
    
def do_basic_analysis_for_all_models(display_progress=True, recompute=False):
    for model in models.model_results.iter_rows(display_progress):
        model_id = model['id']
        if not recompute:
            if get_basic_analysis(model_id) is not None:
                continue
        do_basic_analysis(model_id)

def get_basic_analysis(model_id, compute_on_missing=False):
    basic = basic_analysis.read_single('model_id == {0}'.format(model_id))
    if basic:
        return basic
    elif compute_on_missing:
        do_basic_analysis(model_id)
        return basic_analysis.table[basic_analysis.table.nrows-1]
    else:
        return None

class BasicAnalyzer(object):
    
    def __init__(self, cells):
        self.cells = cells.astype(np.uint8)
        self.rows, self.columns = cells.shape
        
        self._calculate()
        self._finalize()
        
    def _calculate(self):
        self._calculate_mass()
        self._calculate_contours()
        self._calculate_perimeter()
        self._calculate_heights()
        self._calculate_max_height()
        self._calculate_mean_height()
        self._calculate_row_ffts()
        self._calculate_coverage()
        self._calculate_convexity_defects()
        self._calculate_x_correlations()
        self._calculate_overhang()
    
    def _calculate_mass(self):
        self.mass = np.sum(self.cells)/float(self.columns)

    def _calculate_contours(self):
        self.contours, _ = cv2.findContours(np.copy(self.cells), 
                                            cv2.RETR_LIST, 
                                            cv2.CHAIN_APPROX_SIMPLE)

    def _calculate_perimeter(self):
        self.perimeter = sum(cv2.arcLength(c, True) for c in self.contours)\
                         /float(self.columns)

    def _calculate_heights(self):
        heights = np.zeros(self.columns, dtype=int)
        for row in reversed(range(self.rows)):
            heights[np.logical_and((heights == 0), self.cells[row, :])] = row
        self.heights = heights

    def _calculate_max_height(self, top=0.05):
        heights = np.sort(self.heights)
        self.max_height = np.mean(heights[-np.ceil(top*len(heights)):])

    def _calculate_mean_height(self):
        self.mean_height = np.mean(self.heights)

    def _calculate_row_ffts(self):
        self.row_ffts = np.vstack(np.fft.fft(self.cells[row, :])
                                  for row in range(self.rows))

    def _calculate_coverage(self):
        self.coverages = self.cells.sum(axis=1)/float(self.columns)

    def _calculate_convexity_defects(self):
        self.convexity_defects = []
        for contour in self.contours:
            try:
                hull = cv2.convexHull(contour, returnPoints=False)
                defects = cv2.convexityDefects(contour, hull)
            except:
                continue
            if defects is None: 
                continue

            # defects is an Nx1x4 matrix
            for row in range(defects.shape[0]):
                depth = defects[row, 0, 3]/256.0
                self.convexity_defects.append(depth)
        
        if self.convexity_defects:
            self.convexity_max = max(self.convexity_defects)
            self.convexity_mean = np.mean(self.convexity_defects)
            self.convexity_std = np.std(self.convexity_defects)
        else:
            self.convexity_max = 0.0
            self.convexity_mean = 0.0
            self.convexity_std = 0.0

    def _calculate_x_correlations(self):
        distances = range(1, self.columns/2)
        self.x_correlations = np.zeros((self.rows, len(distances)), np.float32)
        found = np.empty(len(distances), int)
        count = np.empty_like(found)
        
        for row in range(self.rows):
            if not self.cells[row, :].any(): 
                break

            found.fill(0)
            count.fill(0)

            for col in range(self.rows):
                cell = self.cells[row, col]

                for i, distance in enumerate(distances):
                    for direction in [-1, 1]:
                        offset = col + distance*direction
                        if offset < 0 or offset >= self.columns:
                            continue

                        count[i] += 1
                        if self.cells[row, offset] == cell:
                            found[i] += 1

            probabilities = found.astype(float)/count
            self.x_correlations[row, :] = probabilities
            
    def _calculate_overhang(self):
        self.overhang_heights = np.zeros(self.columns, dtype=int)
        self.overhangs = np.zeros(self.columns, dtype=int)
        empty_count = np.zeros_like(self.overhangs)
        
        for row in range(self.heights.max()):
            alive = self.cells[row] > 0
            self.overhang_heights += empty_count*alive
            self.overhangs += alive
            empty_count += 1
            empty_count[alive] = 0
    
    def _finalize(self):
        del self.cells

SCALAR_PREFIX = "scalars_"

class ScalarField(object):
    
    def __init__(self, name, value_col=tb.Float32Col()):
        self.name = name
        self.value_col = value_col
        self._setup_tables()
    
    def _setup_tables(self):
        class ByModelDescriptor(tb.IsDescription):
            spec_id = tb.Int32Col(dflt=0, pos=0)
            model_id = tb.Int32Col(dflt=0, pos=1)
            value = self.value_col
        class BySpecDescriptor(tb.IsDescription):
            spec_id = tb.Int32Col(dflt=0, pos=0)
            mean = tb.Float32Col(dflt=0.0)
            std_dev = tb.Float32Col(dflt=0.0)
        self._by_model = utils.QuickTable(SCALAR_PREFIX + self.name, 
                                          ByModelDescriptor,
                                          filters=tb.Filters(complib='blosc', 
                                                             complevel=1),
                                          sorted_indices=['spec_id', 
                                                          'model_id'])
        self._by_spec = utils.QuickTable(SCALAR_PREFIX + self.name + "_summary", 
                                         BySpecDescriptor,
                                         filters=tb.Filters(complib='blosc', 
                                                            complevel=1),
                                         sorted_indices=['spec_id'])
    
    def compute(self):
        self._compute_by_model()
        self._summarize_by_spec()
    
    def _compute_by_model(self):
        raise NotImplemented()
    
    def _summarize_by_spec(self):
        self._by_spec.reset_table()
        
        # TODO: this is terribly inefficient
        row = self._by_spec.table.row
        for spec in specs.specs.table:
            spec_id = spec['id']
            row['spec_id'] = spec_id
            
            id_match = 'spec_id == {0}'.format(spec_id)
            values = np.array([match['value'] for match in 
                               self._by_model.table.where(id_match)])
            row['mean'] = np.mean(values)
            row['std_dev'] = np.std(values)
            row.append()
        self._by_spec.flush()     
    
    def histogram(self, numBins=20, show=False):
        values = np.array(self._by_spec.table.cols.mean)
        hist, binEdges = np.histogram(values, numBins)
        plt.bar(binEdges[:-1], hist, np.diff(binEdges))
        plt.xlim(np.min(binEdges), np.max(binEdges))
        plt.ylim(np.min(hist), np.max(hist))
        plt.xlabel(self.name)
        plt.ylabel("Frequency")
        if show:
            plt.show()
            
    def phase_diagram_2d(self, parameter1, parameter2, numCells=50, 
                      spec_query=None, show=False):
        if spec_query:
            spec_matches = sp.specs.table.readWhere(spec_query)
        else:
            spec_matches = sp.specs.table.read()
        
        shape = spec_matches.size, 1
        xs = np.empty(shape, float)
        ys = np.empty(shape, float)
        values = np.empty(shape, float)
        
        for i, spec in enumerate(spec_matches):
            xs[i] = float(spec[parameter1])
            ys[i] = float(spec[parameter2])
            values[i] = self._by_spec.read_single('spec_id == {0}'
                                                  .format(spec['id']))['mean']
        
        xMin, xMax = xs.min(), xs.max()
        yMin, yMax = ys.min(), ys.max()
        
        assert xMin != xMax
        assert yMin != yMax
        
        grid = np.mgrid[xMin:xMax:numCells*1j, 
                        yMin:yMax:numCells*1j]
        interp = interpolate.griddata(np.hstack((xs, ys)), 
                                      values, 
                                      np.vstack((grid[0].flat, grid[1].flat)).T, 
                                      'cubic')
        valueGrid = np.reshape(interp, grid[0].shape)
        
        plt.pcolormesh(grid[0], grid[1], valueGrid)
        plt.xlim(xMin, xMax)
        plt.ylim(yMin, yMax)
        plt.xlabel(parameter1)
        plt.ylabel(parameter2)
        plt.colorbar()
        plt.title(self.name)
        if show:
            plt.show()
            
    def phase_diagram_1d(self, parameter, spec_query=None, show=False, 
                         **line_args):
        if spec_query:
            spec_matches = sp.specs.table.readWhere(spec_query)
        else:
            spec_matches = sp.specs.table.read()
        
        xs = np.empty(spec_matches.size, float)
        ys = np.empty_like(xs)
        
        for i, spec in enumerate(spec_matches):
            xs[i] = float(spec[parameter])
            ys[i] = self._by_spec.read_single('spec_id == {0}'
                                              .format(spec['id']))['mean']
        
        plt.plot(xs, ys, **line_args)
        plt.xlabel(parameter)
        plt.ylabel(self.name)
        plt.title(self.name)
        if show:
            plt.show()
    

class ScalarBasicFuncField(ScalarField):
    
    def __init__(self, name, func):
        super(ScalarBasicFuncField, self).__init__(name)
        self.func = func
    
    def _compute_by_model(self):
        self._by_model.reset_table()
        
        row = self._by_model.table.row
        for model in models.model_results.iter_rows(display_progress=True):
            model_id = model['id']
            row['model_id'] = model_id
            row['spec_id'] = model['spec_id']
            
            basic = get_basic_analysis(model_id, compute_on_missing=True)
            row['value'] = self.func(basic)
            row.append()
        self._by_model.flush()                

convexity_mean_field = ScalarBasicFuncField('convexity_mean', 
                                            lambda row: row['convexity_mean'])

def _compute_overhang(basic):
    return basic['overhangs'].sum()/float(basic['overhangs'].size)
overhang_field = ScalarBasicFuncField('overhang', _compute_overhang)

def _compute_overhang_height(basic):
    return basic['overhang_heights'].sum()/float(basic['overhang_heights'].size)
overhang_heights_field = ScalarBasicFuncField('overhang_heights', 
                                              _compute_overhang_height)

def _compute_ptm(basic):
    return basic['perimeter']/float(basic['mass'])
perimeter_to_mass_field = ScalarBasicFuncField('perimeter_to_mass',
                                               _compute_ptm)

mean_height_field = ScalarBasicFuncField('mean_height', 
                                         lambda row: row['mean_height'])

max_height_field = ScalarBasicFuncField('max_height', 
                                        lambda row: row['max_height'])

def _compute_mtmh(basic):
    if basic['mean_height'] > 0.01 :
        return basic['max_height']/basic['mean_height']
    else:
        return 0.0
max_to_mean_height_field = ScalarBasicFuncField('max_to_mean_height',
                                                _compute_mtmh)

def _compute_negative_curvature(basic, smoothings=[15, 7, 3]):
    coverage = utils.smooth(basic['coverages'][0], smoothings[0])
    height = np.arange(1, len(coverage)+1)
    slope = utils.smooth(utils.derivative(coverage, height), smoothings[1])
    curvature = utils.smooth(utils.derivative(slope, height), smoothings[2])
    return -np.sum((curvature < 0)*curvature)
negative_coverage_curvature_field = ScalarBasicFuncField(
                                            'negative_coverage_curvature',
                                            _compute_negative_curvature)
