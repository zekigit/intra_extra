import scipy.io as spio
import numpy as np


def make_bnw_nodes(file_nodes, coords, colors, sizes):
    if isinstance(colors, float):
        colors = [colors] * len(coords)
    if isinstance(sizes, float):
        sizes = [sizes] * len(coords)

    nodes = np.column_stack((coords, colors, sizes))
    np.savetxt(file_nodes, nodes, delimiter='\t')



# Functions for loading mat files
def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict_in):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict_in:
        if isinstance(dict_in[key], spio.matlab.mio5_params.mat_struct):
            dict_in[key] = _todict(dict_in[key])
    return dict_in


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

