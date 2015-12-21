import tensorflow as tf


def training(cost, learning_rate=0.02):
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
