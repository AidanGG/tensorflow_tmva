import ROOT
import tensorflow as tf
import numpy as np


def vector_to_one_hot(vector, classes):
    one_hot = []

    for i in range(classes):
        one_hot.append([])

    for i in range(len(vector)):
        for j in range(classes):
            one_hot[j].append(0)
            if vector[i] == j:
                one_hot[j][i] = 1
    return one_hot


def tntuple_to_arrays(ntuple, classifier_index, classes):
    x = []
    classifier = []

    for column in range(ntuple.GetNvar() - 1):
        x.append([])

    for row in range(ntuple.GetEntries()):
        classifier_marker = True
        ntuple.GetEntry(row)

        for column in range(ntuple.GetNvar()):
            if column == classifier_index:
                classifier.append(ntuple.GetArgs()[column])
                classifier_marker = False
            elif classifier_marker:
                x[column].append(ntuple.GetArgs()[column])
            else:
                x[column - 1].append(ntuple.GetArgs()[column])

    y = vector_to_one_hot(classifier, classes)

    return x, y
