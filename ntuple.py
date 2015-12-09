import ROOT
import tensorflow as tf


def tntuple_to_tensor(ntuple, classifier):
    for row in range(ntuple.GetEntries()):
        ntuple.GetEntry(row)

        for column in range(ntuple.GetNvar()):
            ntuple.GetArgs()[column]
