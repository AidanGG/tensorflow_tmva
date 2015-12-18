import tensorflow as tf


def training(cost, method="BP"):
    if method == "BP":
        return tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    elif method == "GA"

    elif method == "BFGS"
