import tensorflow as tf


def model(input, output, hidden_layers, neuron_type="sigmoid",
          neuron_input_type="sum"):
    return y


def neuron_type(type, synapse):
    if type == "sigmoid":
        return tf.sigmoid(synapse)
    elif type == "linear":
        return synapse
    elif type == "tanh":
        return tf.tanh(synapse)
    elif type == "radial":
        return tf.sqrt(tf.exp(tf.neg(tf.square(synapse))))


def neuron_input_type(type, input_signal, weights, bias):
    if type == "sum":
        return tf.add(bias, tf.matmul(input_signal, weights))
    elif type == "sqsum":
        return tf.add(bias, tf.matmul(tf.square(input_signal),
                                      tf.square(weights)))
    elif type == "abssum":
        return tf.add(bias, tf.matmul(tf.abs(input_signal), tf.abs(weights)))
