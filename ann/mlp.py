import tensorflow as tf
import numpy as np

def hidden_layers(*layers):
    weights = []
    biases = []
    for layer in range(len(layers) - 1):
        weights.append(tf.Variable(tf.zeros([layers[layer] + 1, layers[layer]])))
        biases.append(tf.Variable(tf.zeros([layers[layer] + 1, 1])))
    return weights, biases

def model(x, weights, biases):
    for layer in range(len(weights)):
        x = tf.matmul(weights[layer], x) + biases[layer]
    return x
