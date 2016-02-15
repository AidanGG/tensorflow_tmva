import numpy as np


class DataSet(object):
    """Based on the code provided in TensorFlow's MNIST example."""

    def __init__(self, data, labels, classes):
        self._data = data
        self._labels = labels
        self._data_points = data.shape[0]
        self._dimensions = data.shape[1]
        self._classes = classes
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def data(self):
        return self._data

    def labels(self):
        return self._labels

    def data_points(self):
        return self._data_points

    def dimensions(self):
        return self._dimensions

    def classes(self):
        return self._classes

    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._data_points:
            self._epochs_completed += 1
            perm = np.arange(self._data_points)
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._data_points
        end = self._index_in_epoch
        return self._data[start:end], self._labels[start:end]
