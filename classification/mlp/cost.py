import tensorflow as tf


def mean_square_error(model, data):
    cost = tf.reduce_mean(tf.square(tf.sub(model, data)))
    return cost


def cross_entropy(model, data):
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    return cost


def linear(model, data):
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    return cost


def sigmoid(model, data):
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    return cost


def tanh(model, data):
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    return cost


def radial(model, data):
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    return cost
