import ROOT
import numpy as np
from root_numpy import root2array, root2rec, tree2rec, tree2array
from root_numpy.testdata import get_filepath


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

    for i in range(len(ttrees)):
        if i == 0:
            x = tree2array(ttrees[i], branches=branches)
        else:
            x = np.hstack((x, tree2array(ttrees[i], branches=branches)))

    return np.array(x.tolist()), one_hot
