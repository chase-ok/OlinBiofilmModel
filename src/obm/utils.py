'''
Created on Dec 30, 2012

@author: chase_000
'''

import tables as tb

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

    def hasNode(self, name):
        """Returns true if root has a child node with the given name."""
        return hasattr(self.root, name)

    def node(self, name):
        return getattr(self.root, name)

    def clearNode(self, name):
        """Removes the node with the given name from root."""
        if self.hasNode(name):
            getattr(self.root, name)._f_remove(recursive=True)

    def createTable(self, name, *args, **kwargs):
        """Creates a table with the given name under root.

        args and kwargs are passed on to the hdf5 createTable method.
        """
        self.clearNode(name)
        return self.hdf5.createTable(self.root, name, *args, **kwargs)

    def createArray(self, name, *args, **kwargs):
        """Creates an array with the given name under root.

        args and kwargs are passed on to the hdf5 createTable method.
        """
        self.clearNode(name)
        return self.hdf5.createArray(self.root, name, *args, **kwargs)

    def flush(self):
        """Flushes the entire hdf5 connection.
        """
        self.hdf5.flush()
        
class QuickTable(object):
    
    def __init__(self, node, description, **kwargs):
        self.node = node
        self.description = description
        self.kwargs = kwargs
        self._table = None
    
    @property
    def table(self):
        if self._table is None:
            h5 = get_h5()
            if not h5.hasNode(self.node):
                self.reset_table()
            else:
                self._table = h5.node(self.node)
        return self._table
    
    def reset_table(self):
        self._table = get_h5().createTable(self.node, self.description, 
                                           **self.kwargs)
    
    def flush(self):
        self.table.flush()
    
