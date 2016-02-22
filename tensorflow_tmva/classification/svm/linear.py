import tensorflow as tf


def unkernelised_cost(training, classes, dims, C=1):
    """Returns the unkernelised cost to be minimised."""
    w = tf.Variable(tf.zeros([dims, 1]))
    offset = tf.Variable(tf.zeros([1]))

    x = tf.reshape(tf.div(
        tf.matmul(w, w, transpose_a=True), tf.constant([2.0])), [1])
    y = tf.sub(tf.ones([1]), tf.mul(classes, tf.add(
        tf.matmul(training, w, transpose_b=False), offset)))
    z = tf.mul(tf.reduce_sum(tf.reduce_max(
        tf.concat(1, [y, tf.zeros_like(y)]), reduction_indices=1)),
        tf.constant([C], dtype=tf.float32))

    cost = tf.add(x, z)

    return w, offset, cost


def decide(testing, w, offset):
    """Tests a set of test instances."""
    return tf.sign(tf.add(tf.matmul(testing, w), offset))
