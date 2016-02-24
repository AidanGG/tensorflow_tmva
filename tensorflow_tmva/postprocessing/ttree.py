import numpy as np
import root_numpy as rnp

import ROOT


def join_struct_arrays(arrays):
    newdtype = sum((a.dtype.descr for a in arrays), [])
    newrecarray = np.empty(len(arrays[0]), dtype=newdtype)
    for a in arrays:
        for name in a.dtype.names:
            newrecarray[name] = a[name]
    return newrecarray


def name_classifier_branches(array, branch_names):
    array.dtype = {'names': branch_names,
                   'formats': [array.dtype] * len(array[0])}
    return np.reshape(array, [-1])
