'''
Created on Dec 30, 2012

@author: chase_000
'''

import tables as tb
import numpy as np
from uuid import uuid4
import collections
import functools

def derivative(x, y):
    return np.gradient(x)/np.gradient(y)

def smooth(data, window):
    signal = np.r_[data[window-1:0:-1], data, data[-1:-window:-1]]
    kernel = np.ones(window, dtype=float)/window
    return np.convolve(signal, kernel, mode='valid')[:len(data)]

DEFAULT_H5_FILE = "data.h5"

_h5 = None
def get_h5():
    global _h5
    if _h5 is None:
        h5_file = tb.openFile(DEFAULT_H5_FILE, "a")
        _h5 = H5Context(h5_file, h5_file.root)
    return _h5

class H5Context(object):
    
    def __init__(self, hdf5, root):
        """Creates a new context from an hdf5 handle and parameter values.

        hdf5 is a handle to an exisiting hdf5 connection (i.e. from 
        tables.openFile).
        root is the node inside of the hdf5 file that this data set is going
        to run under.
        """
        self.hdf5 = hdf5
        self.root = root

    @property
    def attrs(self):
        """Returns the attrs of the root hdf5 node."""
        return self.root._v_attrs

    def has_node(self, name):
        """Returns true if root has a child node with the given name."""
        return hasattr(self.root, name)

    def node(self, name):
        return getattr(self.root, name)

    def clear_node(self, name):
        """Removes the node with the given name from root."""
        if self.has_node(name):
            getattr(self.root, name)._f_remove(recursive=True)

    def create_table(self, name, *args, **kwargs):
        """Creates a table with the given name under root.

        args and kwargs are passed on to the hdf5 create_table method.
        """
        self.clear_node(name)
        return self.hdf5.createTable(self.root, name, *args, **kwargs)

    def create_array(self, name, *args, **kwargs):
        """Creates an array with the given name under root.

        args and kwargs are passed on to the hdf5 create_table method.
        """
        self.clear_node(name)
        return self.hdf5.createArray(self.root, name, *args, **kwargs)

    def flush(self):
        """Flushes the entire hdf5 connection.
        """
        self.hdf5.flush()
        
class QuickTable(object):
    
    def __init__(self, node, description, indices=None, sorted_indices=None,
                 **kwargs):
        self.node = node
        self.description = description
        self.indices = [] if indices is None else indices
        self.sorted_indices = [] if sorted_indices is None else sorted_indices
        self.kwargs = kwargs
        self._raw = None
    
    @property
    def raw(self):
        if self._raw is None:
            h5 = get_h5()
            if not h5.has_node(self.node):
                self.reset()
            else:
                self._raw = h5.node(self.node)
        return self._raw

    @property
    def row(self):
        return self.raw.row
    
    def reset(self):
        self._raw = get_h5().create_table(self.node, self.description, 
                                           **self.kwargs)
        for index in self.indices:
            getattr(self._raw.cols, index).createIndex()
        for index in self.sorted_indices:
            getattr(self._raw.cols, index).createCSIndex()
    
    def flush(self):
        self.raw.flush()
        
    def iter_rows(self, display_progress=False):
        total = self.raw.nrows
        count = 0
        for row in self.raw:
            yield row
            count += 1
            if display_progress:
                print "Finished {0} of {1}".format(count, total)
        if display_progress:
            print "Done!"
    
    def read_single(self, query):
        rows = self.raw.where(query)
        for row in rows: return row
        return None

def make_uuid_col():
    return tb.StringCol(32, pos=0)

class TableObject(object):

    @classmethod
    def get(cls, uuid):
        row = cls._lookup_by_uuid(uuid)
        if row is None: raise KeyError(uuid)
        return cls._from_row(row)

    @classmethod
    def get_by_row(cls, row):
        return cls.get(cls.table.raw[row]['uuid'])

    @classmethod
    def _from_row(cls, row):
        obj = cls(uuid=row['uuid'])
        fields = row.fetch_all_fields()
        for name in fields.dtype.names:
            if name != 'uuid': setattr(obj, name, fields[name])
        obj._on_get(row)
        return obj

    @classmethod
    def all(cls):
        for row in cls.table.iter_rows():
            yield cls._from_row(row)

    @classmethod
    def reset(cls):
        for obj in cls.all(): obj._on_delete()
        cls.table.reset()

    @classmethod
    def where(cls, query):
        for row in cls.table.raw.where(query):
            yield cls._from_row(row)

    @classmethod
    def setup_table(cls, node, description, 
                    filters=tb.Filters(complib='blosc', complevel=1),
                    sorted_indices=['uuid']):
        cls.table = QuickTable(node, description, filters=filters, 
                               sorted_indices=sorted_indices)

    @classmethod
    def _lookup_by_uuid(cls, uuid):
        return cls.table.read_single('uuid=="{0}"'.format(uuid))

    def __init__(self, uuid=None):
        self.uuid = uuid if uuid else uuid4().hex
        self._exclude = set(['_exclude'])

    def save(self, flush=True):
        row = self._lookup_by_uuid(self.uuid)
        to_fill = self.table.row if row is None else row 
        self._fill_row(to_fill)
        if row is None:
            to_fill.append()
        else:
            to_fill.update()
        if flush: self.table.flush()

    def delete(self):
        try:
            self.table.raw.removeRows(self._lookup_by_uuid(self.uuid).nrow)
            self._on_delete()
            return True
        except NotImplementedError:
            # sometiems pytables can't remove all the rows of a table
            return False

    def _on_delete(self):
        pass

    def _on_get(self, row):
        pass

    def _fill_row(self, row):
        for name, value in self.__dict__.iteritems():
            if name not in self._exclude: row[name] = value

class EasyTableObject(TableObject):
    def __init__(self, uuid=None, **kwargs):
        TableObject.__init__(self, uuid=uuid)
        for name, value in kwargs.iteritems():
            setattr(self, name, value)

def row_to_dict(row):
    return dict((key, row[key]) for key in row.dtype.names)

# From the python decorator library
class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)

def make_variable_data(name, column_func, **table_args):
    @memoized
    def get_table(shape):
        if not isinstance(shape, tuple): 
            shape = (shape,)

        class Object(EasyTableObject): pass
        class ObjectTable(tb.IsDescription):
            uuid = make_uuid_col()
            data = column_func(shape)
        
        path = "{0}_{1}".format(name, '_'.join(map(str, shape)))
        Object.setup_table(path, ObjectTable, **table_args)
        return Object

    def get(uuid, shape): return get_table(shape).get(uuid).data
    def save(uuid, data): 
        return get_table(data.shape)(uuid=uuid, data=data).save()
    def delete(uuid, shape): return get_table(shape)(uuid=uuid).delete()
    return get, save, delete
