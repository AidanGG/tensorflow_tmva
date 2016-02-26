import numpy as np
import root_numpy as rnp

import ROOT


def join_struct_arrays(arrays):
    new_dtype = sum((a.dtype.descr for a in arrays), [])
    new_recarray = np.empty(len(arrays[0]), dtype=new_dtype)
    for array in arrays:
        for name in array.dtype.names:
            new_recarray[name] = array[name]
    return new_recarray


def name_classifier_branches(array, branch_names):
    array.dtype = {'names': branch_names,
                   'formats': [array.dtype] * len(array[0])}
    return np.reshape(array, [-1])
