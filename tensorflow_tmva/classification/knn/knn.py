import numpy as np
import tensorflow as tf


def knn(training, one_hot, testing, nkNN=20, scale_frac=0.8, trim=False,
        kernel="Gaus", use_kernel=False):
    """TO DO"""
    return None


def model_single(input_dims, output_dims, scale_frac, scales, nkNN):
    """
    Forms the knn model.

    Arguments:
    input_dim -- the dimension of the input data
    output_dim -- the number of classes
    scale_frac -- the fraction of events to use for finding widths
    scales -- list of distribution widths for each dimension
    nkNN -- the number of nearest neighbours to find

    Returns:
    A tensor with the number of neighbours in each class.
    """
    training = tf.placeholder(tf.float32, shape=(None, input_dims))
    one_hot = tf.placeholder(tf.float32, shape=(None, output_dims))
    test = tf.placeholder(tf.float32, shape=(1, input_dims))
    distances = metric_single(training, test, scale_frac, scales)

    remaining_training = tf.identity(training)
    remaining_one_hot = tf.identity(one_hot)
    remaining_distances = tf.identity(distances)

    for i in range(nkNN):
        # Gets the location of training entry currently closest to the test
        # entry.
        min_slice = tf.to_int64(
            tf.concat(0, [tf.argmin(remaining_distances, 0), [-1]]))

        # Cuts the nearest neighbour out of the training set.
        start = tf.slice(remaining_training, tf.to_int64([0, 0]), min_slice)
        end = tf.slice(remaining_training, min_slice + [1, 1], [-1, -1])
        remaining_training = tf.concat(0, [start, end])
        # Cuts the nearest neighbour out of the distances set.
        start = tf.slice(remaining_distances, tf.to_int64([0, 0]), min_slice)
        end = tf.slice(remaining_distances, min_slice + [1, 1], [-1, -1])
        remaining_training = tf.concat(0, [start, end])

        # Cuts the nearest neighbour's class and records it.
        start = tf.slice(remaining_one_hot, tf.to_int64([0, 0]), min_slice)
        end = tf.slice(remaining_one_hot, min_slice + [1, 1], [-1, -1])
        class_slice = tf.slice(remaining_one_hot, min_slice + [0, 1], [1, -1])
        remaining_one_hot = tf.concat(0, [start, end])
        if i == 0:
            neighbour_one_hot = class_slice
        else:
            neighbour_one_hot = tf.concat(0, [neighbour_one_hot, class_slice])

    return tf.reduce_sum(neighbour_one_hot, reduction_indices=0)


def metric_single(training, test, scale_frac, scales):
    """Calculates the distance between a training and test instance."""
    if scale_frac == 0:
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(training, test)),
                                         reduction_indices=1, keep_dims=True))
    else:
        distance = tf.sqrt(tf.reduce_sum(
            tf.square(tf.div(tf.sub(training, test), scales)),
            reduction_indices=1, keep_dims=True))
    return distance


def metric_multiple(training, testing, scale_frac, scales, input_dims):
    """Returns a tensor with distances between all training and testing
    instances."""
    dims = tf.slice(tf.shape(training), [1], [1])
    training_instances = tf.slice(tf.shape(training), [0], [1])
    testing_instances = tf.slice(tf.shape(testing), [0], [1])

    # Tiles the training, testing and scales tensors in the appropriate
    # directions.
    re_train = tf.tile(tf.reshape(training, tf.concat(0, [[1, -1], dims])),
                       tf.concat(0, [testing_instances, [1, 1]]))
    re_test = tf.tile(tf.reshape(testing, tf.concat(0, [[-1, 1], dims])),
                      tf.concat(0, [[1], training_instances, [1]]))
    re_scales = tf.tile(tf.reshape(scales, tf.concat(0, [[1, 1], dims])),
                        tf.concat(0, [testing_instances, training_instances,
                                      [1]]))

    # Does the distance calculations.
    if scale_frac == 0:
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(re_train, re_test)),
                                         reduction_indices=2, keep_dims=True))
    else:
        distance = tf.sqrt(tf.reduce_sum(
            tf.square(tf.div(tf.sub(training, test), re_scales)),
            reduction_indices=2, keep_dims=True))
    return distance


def scale(training, scale_frac):
    """Gets the scaling factors for each dimension from the training data."""
    top = np.ravel(np.percentile(training, (1.0 + scale_frac) * 50.0, axis=0))
    bottom = np.ravel(np.percentile(training, (1.0 - scale_frac) * 50.0,
                                    axis=0))
    scales = top - bottom
    return scales


def trim(training, one_hot):
    """Trims the training data so there are an equal number in each class."""
    first_indices = []
    second_indices = []

    # Counts the number of elements of each class.
    for row in range(len(one_hot)):
        if one_hot[row][0] == 0.:
            first_indices.append(row)
        else:
            second_indices.append(row)

    # Randomly chooses training elements from the class with more elements.
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

    # Deletes the chosen elements.
    trimmed_training = np.delete(training, remove, axis=0)
    trimmed_one_hot = np.delete(one_hot, remove, axis=0)

    return trimmed_training, trimmed_one_hot


def kernel(x, kernel, sigma_fact):
    """Chooses the kernel to use."""
    if kernel == "Poln":
        if np.absolute(x) < 1:
            return np.power(1 - np.power(np.absolute(x), 3), 3)
        else:
            return 0
    elif kernel == "Gaus":
        return np.exp(-np.square(x / sigma_fact) / 2.0)


def kernel_weights():
    """TO DO"""
    return signal_weight, background_weight
