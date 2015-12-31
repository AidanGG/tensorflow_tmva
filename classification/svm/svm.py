import tensorflow as tf


def cost(data, classes, dims, softness=0):
    points = tf.slice(tf.shape(data), [0], [1])
    w = tf.variable(tf.zeros(tf.concat(0, [dims, [1]])))
    b = tf.variable(tf.zeros([1]))
    cost = tf.reduce_sum(tf.reduce_max(tf.concat(1, [1 - classes * (tf.matmul(data, w) + b), tf.zeros_like(classes)]), reduction_indices=1, keep_dims=True), reduction_indices=0) / tf.to_float(points) + softness * tf.reduce_sum(tf.square(w))
    return cost, w, b


def total_loss(transformed, classes):
    return tf.reduce_sum(tf.reduce_max(tf.concat(1, [tf.zeros_like(classes), tf.sub(tf.ones_like(classes), tf.mul(transformed, classes))]), reduction_indices=1, keep_dims=True))


def kernel_tensor(data, gamma=1):
    distance = tf.square(tf.abs(tf.matmul(data, tf.neg(data), transpose_b=True)))
    kernel = tf.exp(tf.neg(gamma) * distance)
    return kernel


def kernelised_cost(data, classes, dims, points, C=1, gamma=1):
    points = tf.slice(tf.shape(data), [0], [1])
    beta = tf.variable(tf.zeros([points, 1]))
    offset = tf.variable(tf.zeros([1]))

    kernel = kernel_tensor(data, gamma)
    transformed = tf.matmul(kernel, beta, transpose_a=True) + offset
    return tf.matmul(tf.matmul(beta, kernel, transpose_a=True), beta) + C * total_loss(transformed, classes)
