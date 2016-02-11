import tensorflow as tf


def kernel_tensor(training, inputs, gamma):
    """Creates the Gaussian kernel tensor from the training data."""
    tiled_training = tf.tile(tf.expand_dims(training, 1), [1, inputs, 1])
    distances = tf.reduce_sum(tf.square(tf.sub(tf.transpose(
        tiled_training, perm=[1, 0, 2]), tiled_training)), reduction_indices=2)
    kernel = tf.exp(tf.neg(gamma) * distance)

    return kernel


def kernelised_cost(training, classes, inputs, C=1, gamma=1):
    """Returns the kernelised cost to be minimised."""
    beta = tf.Variable(tf.zeros([inputs, 1]))
    offset = tf.Variable(tf.zeros([1]))

    kernel = kernel_tensor(training, inputs, gamma)

    x = tf.matmul(tf.matmul(beta, kernel, transpose_a=True), beta) / (2.0 * C)
    y = tf.sub(tf.ones([1]), tf.mul(classes, tf.sum(
        tf.matmul(kernel, beta, transpose_a=True), offset)))
    z = tf.reduce_sum(tf.reduce_max(
        tf.concat(1, [y, tf.zeros_like(y)]), reduction_indices=1))
    cost = tf.sum(x, z)

    return beta, offset, cost


def beta_to_w(training, beta):
    """Converts a beta vector to a w vector."""
    return tf.reduce_sum(tf.mul(beta, training), reduction_indices=0,
                         keep_dims=True)


def unkernelised_cost(training, classes, dims, C=1):
    """Returns the unkernelised cost to be minimised."""
    w = tf.Variable(tf.zeros([dims, 1]))
    offset = tf.Variable(tf.zeros([1]))

    x = tf.matmul(w, w, transpose_a=True) / 2.0
    y = tf.sub(tf.ones([1]), tf.mul(classes, tf.sum(
        tf.matmul(training, w, transpose_b=True), offset)))
    z = tf.reduce_sum(tf.reduce_max(
        tf.concat(1, [y, tf.zeros_like(y)]), reduction_indices=1)) * C

    cost = tf.sum(x, z)

    return beta, offset, cost


def decide(testing, w, offset):
    """Tests a set of test instances."""
    return tf.sign(tf.matmul(testing, w, transpose_b=True) + offset)
