import tensorflow as tf


def cost(data, classes, dims, softness=0):
    points = tf.slice(tf.shape(data), [0], [1])
    w = tf.variable(tf.zeros(tf.concat(0, [dims, [1]])))
    b = tf.variable(tf.zeros([1]))
    cost = tf.reduce_sum(tf.reduce_max(tf.concat(1, [1 - classes * (tf.matmul(data, w) + b), tf.zeros_like(classes)]), reduction_indices=1, keep_dims=True), reduction_indices=0) / tf.to_float(points) + softness * tf.reduce_sum(tf.square(w))
    return cost, w, b
