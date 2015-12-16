import ROOT
import numpy as np
from root_numpy import root2array, root2rec, tree2rec, tree2array, array2tree
from root_numpy.testdata import get_filepath


def concat_ttrees_to_array(ttrees, branches=None):
    for i in range(len(ttrees)):
        if i == 0:
            x = tree2array(ttrees[i], branches)
        else:
            x = np.hstack((x, tree2array(ttrees[i], branches)))

    return x


def ttrees_to_one_hot(ttrees):
    entries = 0
    for i in range(len(ttrees)):
        entries += ttrees[i].GetEntries()
    one_hot = np.zeros((entries, len(ttrees)))

    entry = 0
    for i in range(len(ttrees)):
        for j in range(ttrees[i].GetEntries()):
            one_hot[entry, i] = 1
            entry += 1

    return one_hot


def ttrees_to_arrays(ttrees, branches):
    one_hot = ttrees_to_one_hot(ttrees)

    x = concat_ttrees_to_array(ttrees, branches)

    return struct_array_to_array(x), one_hot


def struct_array_to_array(struct_array):
    return struct_array.view((np.float32, len(struct_array.dtype.names)))


def join_struct_arrays(arrays):
    sizes = np.array([a.itemsize for a in arrays])
    offsets = np.r_[0, sizes.cumsum()]
    n = len(arrays[0])
    joint = np.empty((n, offsets[-1]), dtype=np.uint8)
    for a, size, offset in zip(arrays, sizes, offsets):
        joint[:, offset:offset+size] = a.view(np.uint8).reshape(n, size)
    dtype = sum((a.dtype.descr for a in arrays), [])
    return joint.ravel().view(dtype)


def add_branch_to_array(struct_array, branch, tree_name):
    temp_array = join_struct_arrays([struct_array, branch])
    return array2tree(temp_array, name=tree_name)
