import ROOT
import tensorflow as tf


def tntuple_to_arrays(ntuple, classifier_index):
    data = []
    classifier_data = []

    for column in range(ntuple.GetNvar() - 1):
        data.append([])

    for row in range(ntuple.GetEntries()):
        classifier_marker = True
        ntuple.GetEntry(row)

        for column in range(ntuple.GetNvar()):
            if column == classifier_index:
                classifier_data.append(ntuple.GetArgs()[column])
                classifier_marker = False
            elif classifier_marker:
                data[column].append(ntuple.GetArgs()[column])
            else:
                data[column - 1].append(ntuple.GetArgs()[column])

    return data, classifier_data
