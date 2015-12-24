import tensorflow as tf
import numpy as np


def knn(training, one_hot, testing, nkNN=20, scale_frac=0.8, trim=False,
        kernel="Gaus", use_kernel=False):
    return None


def model(input_dims, output_dims, scale_frac, scales, nkNN):
    training = tf.placeholder(tf.float32, shape=(None, input_dims))
    one_hot = tf.placeholder(tf.float32, shape=(None, output_dims))
    test = tf.placeholder(tf.float32, shape=(1, input_dims))
    distances = metric(training, test, scale_frac, scales)

    remaining_training = tf.identity(training)
    remaining_one_hot = tf.identity(one_hot)

    for i in range(nkNN):
        min_slice = tf.to_int64(tf.concat(0, [tf.argmin(remaining_training, 0),
                                              [-1]]))

        start = tf.slice(remaining_training, tf.to_int64([0, 0]), min_slice)
        end = tf.slice(remaining_training, min_slice + [1, 1], [-1, -1])
        remaining_training = tf.concat(0, [start, end])

        start = tf.slice(remaining_one_hot, tf.to_int64([0, 0]), min_slice)
        end = tf.slice(remaining_one_hot, min_slice + [1, 1], [-1, -1])
        class_slice = tf.slice(remaining_one_hot, min_slice + [0, 1], [1, -1])

        if i == 0:
            neighbour_one_hot = class_slice
        else:
            neighbour_one_hot = tf.concat(0, [neighbour_one_hot, class_slice])

        one_hot = tf.concat(0, [start, end])
    return tf.reduce_sum(neighbour_one_hot, reduction_indices=0)


def metric(training, test, scale_frac, scales):
    if scale_frac == 0:
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(training, test)),
                                         reduction_indices=1, keep_dims=True))
    else:
        distance = tf.sqrt(tf.reduce_sum(
            tf.square(tf.div(tf.sub(training, test), scales)),
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


def kernel(x, kernel, sigma_fact):
    if kernel == "Poln":
        if np.absolute(x) < 1:
            return np.power(1 - np.power(np.absolute(x), 3), 3)
        else:
            return 0
    elif kernel == "Gaus":
        return np.exp(-np.square(x / sigma_fact) / 2.0)


def kernel_weights():
    return signal_weight, background_weight
