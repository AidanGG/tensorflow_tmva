import tensorflow as tf


def model(input_dim, output_dim, hidden_layers, neuron_type="sigmoid",
          neuron_input_type="sum"):
    layer_weights = []
    layer_biases = []
    all_layers = [input_dim] + hidden_layers + [output_dim]
    for layer in range(len(all_layers) - 1):
        layer_weights.append(tf.Variable(tf.zeros([all_layers[layer],
                                                  all_layers[layer + 1]])))
        layer_biases.append(tf.Variable(tf.zeros([all_layers[layer + 1]])))
    x = tf.placeholder(tf.float32, [None, input_dim])
    layer_neurons = [x]
    for layer in range(len(all_layers) - 1):
        neuron = neuron_type(neuron_type, neuron_input_type(
            neuron_input_type, layer_weights[layer], layer_biases[layer]))
        # Combine neurons into layer here
        layer_neurons.append(working_layer)
    return x, layer_neurons[-1]


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
        return tf.nn.bias_add(tf.matmul(input_signal, weights), bias)
    elif type == "sqsum":
        return tf.nn.bias_add(tf.matmul(tf.square(input_signal),
                                        tf.square(weights)), bias)
    elif type == "abssum":
        return tf.nn.bias_add(tf.matmul(tf.abs(input_signal), tf.abs(weights)),
                              bias)
