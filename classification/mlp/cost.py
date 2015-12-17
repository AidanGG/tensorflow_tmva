import tensorflow as tf


def mean_square_error(model, data):
    cost = tf.reduce_mean(tf.square(tf.sub(model, data)))
    return cost


def cross_entropy(model, data):
    cost = -tf.reduce_sum(tf.mul(data, tf.log(model)))
    return cost
