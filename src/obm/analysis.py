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
from itertools import chain

class Field(object):

    def __init__(self, func=None, path='', column=tb.Float32Col(), 
                 **table_args):
        self.func = func
        self.path = path
        self._make_tables(column, table_args)
    
    def _make_tables(self, column, table_args):
        self._make_by_result(column, table_args)
        self._make_by_spec()

    def _make_by_result(self, column, table_args):
        class ByResult(utils.EasyTableObject): pass
        class ByResultTable(tb.IsDescription):
            uuid = utils.make_uuid_col()
            data = column
        ByResult.setup_table(self.path, ByResultTable, **table_args)
        self._by_result = ByResult

    def _make_by_spec(self):
        class BySpec(utils.EasyTableObject): pass
        class BySpecTable(tb.IsDescription):
            uuid = utils.make_uuid_col()
            mean = tb.Float32Col()
            median = tb.Float32Col()
            std = tb.Float32Col()
            max = tb.Float32Col()
            min = tb.Float32Col()
        BySpec.setup_table(self.path + "_by_spec", BySpecTable, 
                           filters=tb.Filters(complib='blosc', complevel=1))
        self._by_spec = BySpec

    def reset(self):
        self._by_result.table.reset()
        self._by_spec.table.reset()

    def get_by_result(self, result):
        try:
            return self._by_result.get(result.uuid).data
        except KeyError:
            return self.compute_by_result(result)

    def compute_by_result(self, result):
        data = self.func(result)
        self._by_result(uuid=result.uuid, data=data).save()
        return data

    def delete_by_result(self, result):
        self._by_result(uuid=result.uuid).delete()

    def get_by_spec(self, spec):
        try:
            data = self._by_spec.get(spec.uuid)
            return dict(mean=data.mean, median=data.median, std=data.std,
                        max=data.max, min=data.min)
        except KeyError:
            return self.compute_by_spec(spec)

    def _wrap(self, data):
        summary = utils.row_to_dict(data)
        del summary['uuid']
        return summary

    def compute_by_spec(self, spec, recompute=False):
        matching = "spec_uuid=='{0}'".format(spec.uuid)
        func = self.compute_by_result if recompute else self.get_by_result
        data = np.array([func(res) for res in models.Result.where(matching)])
        summary = self._summarize(data)
        self._by_spec(uuid=spec.uuid, **summary).save()
        return summary

    def delete_by_spec(self, spec):
        self._by_spec(uuid=spec.uuid).delete()

    def _summarize(self, data):
        return dict(mean=data.mean(), median=np.median(data),
                    std=data.std(), max=data.max(), min=data.min())

    def phase_diagram_2d(self, parameter1, parameter2, num_cells=50, 
                         spec_query=None, statistic='mean', show=False, 
                         **plot_args):
        specs = self._query_specs(spec_query)
        
        shape = len(specs), 1
        xs = np.empty(shape, float)
        ys = np.empty(shape, float)
        values = np.empty(shape, float)
        
        for i, spec in enumerate(specs):
            xs[i] = float(getattr(spec, parameter1))
            ys[i] = float(getattr(spec, parameter2))
            values[i] = self._get_statistic(spec, statistic)
        
        xMin, xMax = xs.min(), xs.max()
        yMin, yMax = ys.min(), ys.max()
        
        assert xMin != xMax
        assert yMin != yMax
        
        grid = np.mgrid[xMin:xMax:num_cells*1j, 
                        yMin:yMax:num_cells*1j]
        interp = interpolate.griddata(np.hstack((xs, ys)), 
                                      values, 
                                      np.vstack((grid[0].flat, grid[1].flat)).T, 
                                      'cubic')
        valueGrid = np.reshape(interp, grid[0].shape)
        
        plt.pcolormesh(grid[0], grid[1], valueGrid, **plot_args)
        plt.xlim(xMin, xMax)
        plt.ylim(yMin, yMax)
        plt.xlabel(parameter1)
        plt.ylabel(parameter2)
        plt.colorbar()
        plt.title(self.path)
        if show: plt.show()

    def scatter_plot(self, y_field, spec_query=None, statistic='mean',
                     show=False, **plot_args):
        specs = self._query_specs(spec_query)
        xs = np.empty(len(specs), float)
        ys = np.empty_like(xs)

        for i, spec in enumerate(specs):
            xs[i] = self._get_statistic(spec, statistic)
            ys[i] = y_field._get_statistic(spec, statistic)

        plt.plot(xs, ys, '.', **plot_args)
        plt.xlabel(self.path)
        plt.ylabel(y_field.path)
        if show: plt.show()

        return xs, ys

    def _query_specs(self, spec_query):
        if spec_query:
            return list(sp.Spec.where(spec_query))
        else:
            return list(sp.Spec.all())

    def _get_statistic(self, spec, statistic):
        return self.get_by_spec(spec)[statistic]


class VariableField(Field):

    def __init__(self, shape=lambda r: 1, **field_args):
        Field.__init__(self, **field_args)
        self.shape = shape

    def _make_by_result(self, column, table_args):
        self._get_by_result, self._save_by_result, self._delete_by_result \
                = utils.make_variable_data(self.path, column, **table_args)

    def reset(self):
        raise NotImplemented()

    def get_by_result(self, result):
        try:
            return self._get_by_result(result.uuid, self.shape(result))
        except KeyError:
            return self.compute_by_result(result)

    def compute_by_result(self, result):
        data = self.func(result)
        self._save_by_result(result.uuid, data)
        return data

    def delete_by_result(self, result):
        self._delete_by_result(result.uuid, self.shape(result))

    def _summarize(self, data):
        combined = list(chain.from_iterable(data))
        return Field._summarize(self, np.array(combined))

class CurveAveragingField(VariableField):

    def __init__(self, spec_shape=lambda s: 1, **var_args):
        self.spec_shape = spec_shape
        VariableField.__init__(self, **var_args)

    def _make_by_spec(self):
        self._get_by_spec, self._save_by_spec, self._delete_by_spec \
                = utils.make_variable_data(self.path + "_by_spec", 
                                           tb.Float32Col)

    def get_by_spec(self, spec):
        try:
            return self._get_by_spec(spec.uuid, self.spec_shape(spec))
        except KeyError:
            return self.compute_by_spec(spec)

    def compute_by_spec(self, spec):
        matching = "spec_uuid=='{0}'".format(spec.uuid)
        data = [self.compute_by_result(res) 
                for res in models.Result.where(matching)]
        summary = self._summarize(data)
        self._save_by_spec(spec.uuid, summary)
        return summary

    def delete_by_result(self, spec):
        self._delete_by_result(result.uuid, self.spec_shape(spec))

    def _summarize(self, data):
        if not data: return None

        averaged = np.copy(data[0])
        for i in range(1, len(data)):
            averaged += data[i]
        return averaged/float(len(data))

    def curve_plot(self, spec_query=None, show=False):
        specs = self._query_specs(spec_query)

        plt.hold(True)
        for spec in specs:
            plt.plot(self.get_by_spec(spec))

        plt.ylabel(self.path)
        if show: plt.show()

    def average_curve_plot(self, spec_query=None, show=False, **plot_args):
        specs = self._query_specs(spec_query)
        data = [self.get_by_spec(spec) for spec in specs]
        plt.plot(self._summarize(data), **plot_args)
        plt.ylabel(self.path)
        if show: plt.show()

def _compute_heights(result):
    im = result.image
    heights = np.zeros(im.shape[1], dtype=int)
    for row in reversed(range(im.shape[0])):
        heights[np.logical_and((heights == 0), im[row, :])] = row
    return heights
heights = VariableField(func=_compute_heights,
                        shape=lambda r: r.image.shape[1],
                        path="heights",
                        column=tb.UInt16Col)

def _compute_contours(result):
    return cv2.findContours(np.copy(result.int_image), 
                            cv2.RETR_LIST, 
                            cv2.CHAIN_APPROX_SIMPLE)[0]

def _compute_perimeter(result):
    return sum(cv2.arcLength(c, True) for c in _compute_contours(result))\
           /float(result.spec.width)
perimeter = Field(func=_compute_perimeter, path="perimeter")

def _compute_coverages(result):
    return result.image.sum(axis=1)/float(result.spec.width)
coverages = CurveAveragingField(func=_compute_coverages,
                                shape=lambda r: r.spec.height,
                                spec_shape=lambda s: s.height,
                                path="coverages",
                                column=tb.Float32Col)


def _compute_overhangs(result):
    overhangs = np.zeros(result.spec.width, int)
    empty_count = np.zeros_like(overhangs)
        
    for row in range(get_height_dist(result)['max']):
        alive = result.image[row]
        overhangs += empty_count*alive
        empty_count += 1
        empty_count[alive] = 0

    return overhangs
overhangs = VariableField(func=_compute_overhangs,
                          shape=lambda r: r.spec.width,
                          path="overhangs",
                          column=tb.UInt16Col)

def _compute_x_correlations(result):
    distances = range(1, result.spec.width/2)
    x_correlations = np.zeros((result.spec.height, len(distances)), float)
    found = np.empty(len(distances), int)
    count = np.empty_like(found)

    for row in range(get_height_dist(result)['max']):
        found.fill(0)
        count.fill(0)

        for col in range(result.spec.width):
            cell = result.image[row, col]

            for i, distance in enumerate(distances):
                for direction in [-1, 1]:
                    offset = col + distance*direction
                    if offset < 0 or offset >= result.spec.width:
                        continue

                    count[i] += 1
                    if result.image[row, offset] == cell:
                        found[i] += 1

        x_correlations[row, :] = found.astype(float)/count

    return x_correlations
x_correlations = VariableField(func=_compute_x_correlations,
                               shape=lambda r: (r.spec.height, r.spec.width/2),
                               path="x_correlations",
                               column=tb.Float32Col)

def _compute_convex_hull_area(result):
    area = 0.0
    for contour in _compute_contours(result):
        try:
            hull = cv2.convexHull(contour, returnPoints=True)
            area += cv2.contourArea(hull)
        except:
            continue
    return area
convex_hull_area = Field(func=_compute_convex_hull_area,
                         path="convex_hull_area")

def _compute_convex_density(result):
    area = float(convex_hull_area.get_by_result(result))
    return 0.0 if area <= 1.0 else mass.get_by_result(result)/area
convex_density = Field(func=_compute_convex_density, path="convex_density")

def _compute_mass(result):
    return result.mass.max()
mass = Field(func=_compute_mass, path="mass", column=tb.UInt32Col())

def _compute_light_exposure(result):
    penetration_depth = 6.0 #result.spec.light_penetration
    cum_sum = result.image.cumsum(axis=0)
    light = np.exp(-cum_sum/penetration_depth)
    return (light*result.image).sum()
light_exposure = Field(func=_compute_light_exposure, path="light_exposure")


# class BasicAnalysisTable(tb.IsDescription):
#     uuid = tb.StringCol(32, pos=0)
#     spec_uuid = tb.StringCol(32, pos=1)
#     mass = tb.Float32Col()
#     perimeter = tb.Float32Col()
#     heights = tb.UInt16Col(shape=(1, models.COLUMNS))
#     max_height = tb.Float32Col()
#     mean_height = tb.Float32Col()
#     row_ffts = tb.Float32Col(shape=(models.ROWS, models.COLUMNS))
#     coverages = tb.Float32Col(shape=(1, models.ROWS))
#     convexity_max = tb.Float32Col()
#     convexity_mean = tb.Float32Col()
#     convexity_std = tb.Float32Col()
#     x_correlations = tb.Float32Col(shape=(models.ROWS, models.COLUMNS/2-1))
#     overhangs = tb.UInt8Col(shape=models.COLUMNS)
#     overhang_heights = tb.UInt16Col(shape=models.COLUMNS)
    
# basic_analysis = utils.QuickTable("basic_analysis", BasicAnalysis,
#                                   filters=tb.Filters(complib='zlib', 
#                                                      complevel=9),
#                                   sorted_indices=['model_id', 'spec_id'])

# def do_basic_analysis(model_id):
#     result = models.model_results.table[model_id]
#     analyzer = BasicAnalyzer(result['cells'])
    
#     old = get_basic_analysis(model_id)
#     row = basic_analysis.table.row if old is None else old
#     row['spec_id'] = result['spec_id']
#     row['model_id'] = model_id
#     row['perimeter'] = analyzer.perimeter
#     row['heights'] = analyzer.heights
#     row['max_height'] = analyzer.max_height
#     row['mean_height'] = analyzer.mean_height
#     row['row_ffts'] = analyzer.row_ffts
#     row['coverages'] = analyzer.coverages
#     row['convexity_max'] = analyzer.convexity_max
#     row['convexity_mean'] = analyzer.convexity_mean
#     row['convexity_std'] = analyzer.convexity_std
#     row['x_correlations'] = analyzer.x_correlations
#     row['overhangs'] = analyzer.overhangs
#     row['overhang_heights'] = analyzer.overhang_heights
#     row['mass'] = analyzer.mass
#     if old is None:
#         row.append()
#     else:
#         row.update()
#     basic_analysis.flush()
    
# def do_basic_analysis_for_all_models(display_progress=True, recompute=False):
#     for model in models.model_results.iter_rows(display_progress):
#         model_id = model['id']
#         if not recompute:
#             if get_basic_analysis(model_id) is not None:
#                 continue
#         do_basic_analysis(model_id)

# def get_basic_analysis(model_id, compute_on_missing=False):
#     basic = basic_analysis.read_single('model_id == {0}'.format(model_id))
#     if basic:
#         return basic
#     elif compute_on_missing:
#         do_basic_analysis(model_id)
#         return basic_analysis.table[basic_analysis.table.nrows-1]
#     else:
#         return None

# class BasicAnalyzer(object):
    
#     def __init__(self, cells):
#         self.cells = cells.astype(np.uint8)
#         self.rows, self.columns = cells.shape
        
#         self._compute()
#         self._finalize()
        
#     def _compute(self):
#         self._compute_mass()
#         self._compute_contours()
#         self._compute_perimeter()
#         self._compute_heights()
#         self._compute_max_height()
#         self._compute_mean_height()
#         self._compute_row_ffts()
#         self._compute_coverage()
#         self._compute_convexity_defects()
#         self._compute_x_correlations()
#         self._compute_overhang()
    
#     def _compute_mass(self):
#         self.mass = np.sum(self.cells)/float(self.columns)

#     def _compute_contours(self):
#         self.contours, _ = cv2.findContours(np.copy(self.cells), 
#                                             cv2.RETR_LIST, 
#                                             cv2.CHAIN_APPROX_SIMPLE)

#     def _compute_perimeter(self):
#         self.perimeter = sum(cv2.arcLength(c, True) for c in self.contours)\
#                          /float(self.columns)

#     def _compute_heights(self):
#         heights = np.zeros(self.columns, dtype=int)
#         for row in reversed(range(self.rows)):
#             heights[np.logical_and((heights == 0), self.cells[row, :])] = row
#         self.heights = heights

#     def _compute_max_height(self, top=0.05):
#         heights = np.sort(self.heights)
#         self.max_height = np.mean(heights[-np.ceil(top*len(heights)):])

#     def _compute_mean_height(self):
#         self.mean_height = np.mean(self.heights)

#     def _compute_row_ffts(self):
#         self.row_ffts = np.vstack(np.fft.fft(self.cells[row, :])
#                                   for row in range(self.rows))

#     def _compute_coverage(self):
#         self.coverages = self.cells.sum(axis=1)/float(self.columns)

#     def _compute_convexity_defects(self):
#         self.convexity_defects = []
#         for contour in self.contours:
#             try:
#                 hull = cv2.convexHull(contour, returnPoints=False)
#                 defects = cv2.convexityDefects(contour, hull)
#             except:
#                 continue
#             if defects is None: 
#                 continue

#             # defects is an Nx1x4 matrix
#             for row in range(defects.shape[0]):
#                 depth = defects[row, 0, 3]/256.0
#                 self.convexity_defects.append(depth)
        
#         if self.convexity_defects:
#             self.convexity_max = max(self.convexity_defects)
#             self.convexity_mean = np.mean(self.convexity_defects)
#             self.convexity_std = np.std(self.convexity_defects)
#         else:
#             self.convexity_max = 0.0
#             self.convexity_mean = 0.0
#             self.convexity_std = 0.0

#     def _compute_x_correlations(self):
#         distances = range(1, self.columns/2)
#         self.x_correlations = np.zeros((self.rows, len(distances)), np.float32)
#         found = np.empty(len(distances), int)
#         count = np.empty_like(found)
        
#         for row in range(self.rows):
#             if not self.cells[row, :].any(): 
#                 break

#             found.fill(0)
#             count.fill(0)

#             for col in range(self.rows):
#                 cell = self.cells[row, col]

#                 for i, distance in enumerate(distances):
#                     for direction in [-1, 1]:
#                         offset = col + distance*direction
#                         if offset < 0 or offset >= self.columns:
#                             continue

#                         count[i] += 1
#                         if self.cells[row, offset] == cell:
#                             found[i] += 1

#             probabilities = found.astype(float)/count
#             self.x_correlations[row, :] = probabilities
            
#     def _compute_overhang(self):
#         self.overhang_heights = np.zeros(self.columns, dtype=int)
#         self.overhangs = np.zeros(self.columns, dtype=int)
#         empty_count = np.zeros_like(self.overhangs)
        
#         for row in range(self.heights.max()):
#             alive = self.cells[row] > 0
#             self.overhang_heights += empty_count*alive
#             self.overhangs += alive
#             empty_count += 1
#             empty_count[alive] = 0
    
#     def _finalize(self):
#         del self.cells

# SCALAR_PREFIX = "scalars_"

# class ScalarField(object):
    
#     def __init__(self, name, value_col=tb.Float32Col()):
#         self.name = name
#         self.value_col = value_col
#         self._setup_tables()
    
#     def _setup_tables(self):
#         class ByModelDescriptor(tb.IsDescription):
#             spec_id = tb.Int32Col(dflt=0, pos=0)
#             model_id = tb.Int32Col(dflt=0, pos=1)
#             value = self.value_col
#         class BySpecDescriptor(tb.IsDescription):
#             spec_id = tb.Int32Col(dflt=0, pos=0)
#             mean = tb.Float32Col(dflt=0.0)
#             std_dev = tb.Float32Col(dflt=0.0)
#         self._by_model = utils.QuickTable(SCALAR_PREFIX + self.name, 
#                                           ByModelDescriptor,
#                                           filters=tb.Filters(complib='blosc', 
#                                                              complevel=1),
#                                           sorted_indices=['spec_id', 
#                                                           'model_id'])
#         self._by_spec = utils.QuickTable(SCALAR_PREFIX + self.name + "_summary", 
#                                          BySpecDescriptor,
#                                          filters=tb.Filters(complib='blosc', 
#                                                             complevel=1),
#                                          sorted_indices=['spec_id'])
    
#     def compute(self):
#         self._compute_by_model()
#         self._summarize_by_spec()
    
#     def _compute_by_model(self):
#         raise NotImplemented()
    
#     def _summarize_by_spec(self):
#         self._by_spec.reset_table()
        
#         # TODO: this is terribly inefficient
#         row = self._by_spec.table.row
#         for spec in specs.specs.table:
#             spec_id = spec['id']
#             row['spec_id'] = spec_id
            
#             id_match = 'spec_id == {0}'.format(spec_id)
#             values = np.array([match['value'] for match in 
#                                self._by_model.table.where(id_match)])
#             row['mean'] = np.mean(values)
#             row['std_dev'] = np.std(values)
#             row.append()
#         self._by_spec.flush()     
    
#     def histogram(self, numBins=20, show=False):
#         values = np.array(self._by_spec.table.cols.mean)
#         hist, binEdges = np.histogram(values, numBins)
#         plt.bar(binEdges[:-1], hist, np.diff(binEdges))
#         plt.xlim(np.min(binEdges), np.max(binEdges))
#         plt.ylim(np.min(hist), np.max(hist))
#         plt.xlabel(self.name)
#         plt.ylabel("Frequency")
#         if show:
#             plt.show()
            
#     def phase_diagram_2d(self, parameter1, parameter2, num_cells=50, 
#                       spec_query=None, show=False):
#         if spec_query:
#             spec_matches = sp.specs.table.readWhere(spec_query)
#         else:
#             spec_matches = sp.specs.table.reaod()
        
#         shape = spec_matches.size, 1
#         xs = np.empty(shape, float)
#         ys = np.empty(shape, float)
#         values = np.empty(shape, float)
        
#         for i, spec in enumerate(spec_matches):
#             xs[i] = float(spec[parameter1])
#             ys[i] = float(spec[parameter2])
#             values[i] = self._by_spec.read_single('spec_id == {0}'
#                                                   .format(spec['id']))['mean']
        
#         xMin, xMax = xs.min(), xs.max()
#         yMin, yMax = ys.min(), ys.max()
        
#         assert xMin != xMax
#         assert yMin != yMax
        
#         grid = np.mgrid[xMin:xMax:num_cells*1j, 
#                         yMin:yMax:num_cells*1j]
#         interp = interpolate.griddata(np.hstack((xs, ys)), 
#                                       values, 
#                                       np.vstack((grid[0].flat, grid[1].flat)).T, 
#                                       'cubic')
#         valueGrid = np.reshape(interp, grid[0].shape)
        
#         plt.pcolormesh(grid[0], grid[1], valueGrid)
#         plt.xlim(xMin, xMax)
#         plt.ylim(yMin, yMax)
#         plt.xlabel(parameter1)
#         plt.ylabel(parameter2)
#         plt.colorbar()
#         plt.title(self.name)
#         if show:
#             plt.show()
            
#     def phase_diagram_1d(self, parameter, spec_query=None, show=False, 
#                          **line_args):
#         if spec_query:
#             spec_matches = sp.specs.table.readWhere(spec_query)
#         else:
#             spec_matches = sp.specs.table.read()
        
#         xs = np.empty(spec_matches.size, float)
#         ys = np.empty_like(xs)
        
#         for i, spec in enumerate(spec_matches):
#             xs[i] = float(spec[parameter])
#             ys[i] = self._by_spec.read_single('spec_id == {0}'
#                                               .format(spec['id']))['mean']
        
#         plt.plot(xs, ys, **line_args)
#         plt.xlabel(parameter)
#         plt.ylabel(self.name)
#         plt.title(self.name)
#         if show:
#             plt.show()
    

# class ScalarBasicFuncField(ScalarField):
    
#     def __init__(self, name, func):
#         super(ScalarBasicFuncField, self).__init__(name)
#         self.func = func
    
#     def _compute_by_model(self):
#         self._by_model.reset_table()
        
#         row = self._by_model.table.row
#         for model in models.model_results.iter_rows(display_progress=True):
#             model_id = model['id']
#             row['model_id'] = model_id
#             row['spec_id'] = model['spec_id']
            
#             basic = get_basic_analysis(model_id, compute_on_missing=True)
#             row['value'] = self.func(basic)
#             row.append()
#         self._by_model.flush()                

# convexity_mean_field = ScalarBasicFuncField('convexity_mean', 
#                                             lambda row: row['convexity_mean'])

# def _compute_overhang(basic):
#     return basic['overhangs'].sum()/float(basic['overhangs'].size)
# overhang_field = ScalarBasicFuncField('overhang', _compute_overhang)

# def _compute_overhang_height(basic):
#     return basic['overhang_heights'].sum()/float(basic['overhang_heights'].size)
# overhang_heights_field = ScalarBasicFuncField('overhang_heights', 
#                                               _compute_overhang_height)

# def _compute_ptm(basic):
#     return basic['perimeter']/float(basic['mass'])
# perimeter_to_mass_field = ScalarBasicFuncField('perimeter_to_mass',
#                                                _compute_ptm)

# mean_height_field = ScalarBasicFuncField('mean_height', 
#                                          lambda row: row['mean_height'])

# max_height_field = ScalarBasicFuncField('max_height', 
#                                         lambda row: row['max_height'])

# def _compute_mtmh(basic):
#     if basic['mean_height'] > 0.01 :
#         return basic['max_height']/basic['mean_height']
#     else:
#         return 0.0
# max_to_mean_height_field = ScalarBasicFuncField('max_to_mean_height',
#                                                 _compute_mtmh)

# def _compute_negative_curvature(basic, smoothings=[15, 7, 3]):
#     coverage = utils.smooth(basic['coverages'][0], smoothings[0])
#     height = np.arange(1, len(coverage)+1)
#     slope = utils.smooth(utils.derivative(coverage, height), smoothings[1])
#     curvature = utils.smooth(utils.derivative(slope, height), smoothings[2])
#     return -np.sum((curvature < 0)*curvature)
# negative_coverage_curvature_field = ScalarBasicFuncField(
#                                             'negative_coverage_curvature',
#                                             _compute_negative_curvature)
