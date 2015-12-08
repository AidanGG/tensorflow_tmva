import tensorflow as tf
import numpy as np


def hidden_layers(*layers):
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
