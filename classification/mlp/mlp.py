import tensorflow as tf
import numpy as np


def hidden_layers(layers):
    weights = []
    biases = []
    for layer in range(len(layers) - 1):
        weights.append(
            tf.Variable(tf.zeros([layers[layer], layers[layer + 1]])))
        biases.append(tf.Variable(tf.zeros([layers[layer + 1]])))
    return weights, biases


def model(x, weights, biases):
    y = tf.nn.bias_add(tf.matmul(x, weights[0]), biases[0])
    for layer in range(1, len(weights)):
        y = tf.nn.bias_add(tf.matmul(y, weights[layer]), biases[layer])
    return y


def run(input_dim, output_dim, layers):
    x = tf.placeholder(tf.float32, [None, input_dim])
    W, b = hidden_layers(layers)
    y = tf.nn.softmax(model(x, W, b))

    y_ = tf.placeholder(tf.float32, [None, output_dim])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(
        cross_entropy)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)
    return x, y, y_, sess, train_step
