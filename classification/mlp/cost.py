import tensorflow as tf


def cost(model, data, estimator_type="MSE"):
    if estimator_type == "CE":
        cost = -tf.reduce_sum(tf.mul(data, tf.log(model)))
    else:
        cost = tf.reduce_mean(tf.square(tf.sub(model, data)))
    return cost
