import ROOT
import numpy as np
from root_numpy import root2array, root2rec, tree2rec
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


# def ttrees_to_arrays(ttrees, branches):
#     one_hot = ttrees_to_one_hot(ttrees)
#     x = []
#
#     for i in range(len(ttrees)):
#         ttrees[i].SetBranchStatus("*", 0)
#         for j in range(len(branches)):
#             ttrees[i].SetBranchStatus(branches[j], 1)
#
#         for row in range(ttrees[i].GetEntries()):
#             entry = []
#             for column in range(len(branches)):
#                 entry.append(0)
#                 ttrees[i].SetBranchAddress(branches[column], entry[column])
#             ttrees[i].GetEntry(row)
#             x.append(entry)
#
#     return x, one_hot
#
# x = ROOT.TNtuple("x", "", "a:b")
# y = ROOT.TNtuple("y", "", "a:b")
#
# x.Fill(1, 2)
# x.Fill(5, 6)
# y.Fill(3, 4)
#
# print ttrees_to_one_hot([x, y])
