from ..preprocessing import ttree
import tensorflow as tf
import numpy as np


def knn(signal, background, testing, branches, nkNN=20, scale_frac=0.8,
        trim=False, kernel="Gaus", use_kernel=False):
    return None


def metric(training, test, scale_frac):
    if scale_frac == 0:
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(training, test)),
                                         reduction_indices=1, keep_dims=True))
    else:
        distance = tf.sqrt(tf.reduce_sum(
            tf.square(tf.div(tf.sub(training, test),
                      scale(training, scale_frac))),
            reduction_indices=1, keep_dims=True))
    return distance


def scale(training, scale_frac):
    top = np.ravel(np.percentile(training, (1.0 + scale_frac) * 50.0, axis=0))
    bottom = np.ravel(np.percentile(training, (1.0 - scale_frac) * 50.0,
                                    axis=0))
    scales = top - bottom
    return scales


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


def kernel(x, kernel):
    if kernel == "Poln":
        if np.absolute(x) < 1:
            return np.power(1 - np.power(np.absolute(x), 3), 3)
        else:
            return 0
    elif kernel == "Gaus":
        return None


def kernel_weights():
    return signal_weight, background_weight
