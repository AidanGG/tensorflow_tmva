import ROOT
import tensorflow as tf
import numpy as np


def vector_to_one_hot(vector, classes):
    one_hot = []

    for i in range(len(vector)):
        one_hot.append([])
        for j in range(classes):
            one_hot[i].append(0)
            if vector[i] == j:
                one_hot[i][j] = 1

    return one_hot


def tntuple_to_arrays(ntuple, classifier_index, classes):
    x = []
    classifier = []

    for row in range(ntuple.GetEntries()):
        x.append([])
        ntuple.GetEntry(row)

        for column in range(ntuple.GetNvar()):
            if column == classifier_index:
                classifier.append(ntuple.GetArgs()[column])
            else:
                x[row].append(ntuple.GetArgs()[column])

    y = vector_to_one_hot(classifier, classes)

    return x, y
