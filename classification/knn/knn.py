from ..preprocessing import ttree
import tensorflow as tf
import numpy as np


def knn(k, training, one_hot, testing, scale_frac=0.8, trim=False):
    # tr = tf.placeholder(tf.float32, shape=training.shape)
    # te = tf.placeholder(tf.float32, shape=testing.shape)
    # distance = metric(training, te, scale_frac)
    return None


def metric(training, test, scale_frac):
    if scale_frac == 0:
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(training, test)),
                                         reduction_indices=1, keep_dims=True))
    else:
        distance = None
    return distance


def scales(training, scale_frac):
    return None


def trim(training, one_hot):
    first_indices = []
    second_indices = []

    for row in range(len(one_hot)):
        if one_hot[row][0] == 0.:
            first_indices.append(row)
        else:
            second_indices.append(row)

    if len(first_indices) > len(second_indices):
        difference = len(first_indices) - len(second_indices)
        remove = np.random.choice(len(first_indices), size=difference,
                                  replace=False)
    elif len(first_indices) < len(second_indices):
        difference = len(second_indices) - len(first_indices)
        remove = np.random.choice(len(second_indices), size=difference,
                                  replace=False)
    else:
        return training, one_hot

    trimmed_training = np.delete(training, remove, axis=0)
    trimmed_one_hot = np.delete(one_hot, remove, axis=0)

    return trimmed_training, trimmed_one_hot
