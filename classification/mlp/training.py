import tensorflow as tf


def training(cost, learning_rate=0.02):
    """Applies a gradient descent training method."""
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
