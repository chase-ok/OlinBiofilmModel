'''
Created on Dec 30, 2012

@author: chase_000
'''

import numpy as np
import cv2
import random as rdm
import specs
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
    overhangs = tb.UInt16Col(shape=(1, models.COLUMNS))
    
basic_analysis = utils.QuickTable("basic_analysis", BasicAnalysis,
                                  filters=tb.Filters(complib='zlib', 
                                                     complevel=9))

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
    if old is None:
        row.append()
    else:
        row.update()
    basic_analysis.flush()

def get_basic_analysis(model_id, compute_on_missing=False):
    id_match = 'model_id == {0}'.format(model_id)
    rows = [row for row in basic_analysis.table.where(id_match)]
    assert len(rows) <= 1
    
    if rows:
        return rows[0]
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
                
        self.convexity_max = max(self.convexity_defects)
        self.convexity_mean = np.mean(self.convexity_defects)
        self.convexity_std = np.std(self.convexity_defects)

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
        self.overhangs = np.zeros(self.columns, dtype=int)
        empty_count = np.zeros_like(self.overhangs)
        
        for row in range(self.heights.max()):
            alive = self.cells[row] > 0
            self.overhangs += empty_count*alive
            empty_count += 1
            empty_count[alive] = 0
    
    def _finalize(self):
        del self.cells

SCALAR_PREFIX = "scalars_"

class ScalarField(object):
    
    def __init__(self, name, value_col):
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
                                                             complevel=1))
        self._by_spec = utils.QuickTable(SCALAR_PREFIX + self.name + "_summary", 
                                         BySpecDescriptor,
                                         filters=tb.Filters(complib='blosc', 
                                                            complevel=1))
    
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

class ScalarBasicFuncField(ScalarField):
    
    def __init__(self, name, value_col, func):
        super(ScalarBasicFuncField, self).__init__(name, value_col)
        self.func = func
    
    def _compute_by_model(self):
        self._by_model.reset_table()
        
        row = self._by_model.table.row
        for model_result in models.model_results.table:
            model_id = model_result['id']
            row['model_id'] = model_id
            row['spec_id'] = model_result['spec_id']
            
            basic = get_basic_analysis(model_id, compute_on_missing=True)
            row['value'] = self.func(basic)
            print model_id, row['value']
            row.append()
        self._by_model.flush()                

convexity_mean_field = ScalarBasicFuncField('convexity_mean', tb.Float32Col(), 
                                            lambda row: row['convexity_mean'])

    
    
    
    
    
    