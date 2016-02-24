import numpy as np
import root_numpy as rnp

import data_set
import ROOT


def concat_ttrees_to_array(ttrees, branches=None):
    """Concatenates multiple TTrees of different classes into one ndarray."""
    rec = []

    for i in range(len(ttrees)):
        rec.append(rnp.tree2rec(ttrees[i], branches))

    return rnp.rec2array(rnp.stack(rec, fields=branches), fields=branches)


def ttrees_to_one_hot(ttrees):
    """Creates a one-hot array from TTrees representing different classes."""
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


def ttrees_to_binary(signal_ttree, background_ttree):
    """Creates a binary array (-1 to 1) from a signal and background TTree."""
    entries = signal_ttree.GetEntries() + background_ttree.GetEntries()

    binary = np.zeros((entries, 1))

    for i in range(signal_ttree.GetEntries()):
        binary[i, 0] = 1.0

    for j in range(background_ttree.GetEntries()):
        binary[i + j + 1, 0] = -1.0

    return binary


def ttrees_to_internal(ttrees, branches, binary=False):
    """Converts the TTrees to the main data structure."""
    data = concat_ttrees_to_array(ttrees, branches)
    if binary:
        labels = ttrees_to_binary(ttrees[0], ttrees[1])
    else:
        labels = ttrees_to_one_hot(ttrees)
    classes = len(ttrees)

    combined = data_set.DataSet(data, labels, classes)
    return combined
