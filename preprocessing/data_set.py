import numpy as np


class DataSet(object):

    def __init__(self, data, one_hot):
        self._data = data
        self._one_hot = one_hot
        self._data_points = data.shape[0]
        self._dimensions = data.shape[1]
        self._classes = one_hot[1]

    def data(self):
        return self._data

    def one_hot(self):
        return self._one_hot

    def data_points(self):
        return self._data_points

    def dimensions(self):
        return self._dimensions

    def classes(self):
        return self._classes
