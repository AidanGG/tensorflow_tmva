import tensorflow as tf


def model(input_dim, output_dim, hidden_layers, neuron_type="sigmoid",
          neuron_input_type="sum"):
    """
    Forms the mlp neural net architecture.

    Arguments:
    input_dim -- the dimension of the input data
    output_dim -- the number of classes
    hidden_layers -- a list of of neurons in each hidden layer
    neuron_type -- the activation function for each neuron (default "sigmoid")
    neuron_input_type -- the synapse function for each link (default "sum")

    Returns:
    x -- the input placeholder tensor
    y -- the model's classification
    layer_weights -- the weight variables for each layer
    layer_biases -- the bias variables for each layer
    """
    layer_weights = []
    layer_biases = []
    all_layers = [input_dim] + hidden_layers + [output_dim]

    # Creates new weights and biases for each layer.
    for layer in range(len(all_layers) - 1):
        layer_weights.append(fill_variable([all_layers[layer],
                                            all_layers[layer + 1]]))
        layer_biases.append(fill_variable([all_layers[layer + 1]]))

    x = tf.placeholder(tf.float32, [None, input_dim])
    layer_neurons = [x]

    # Creates the neuron layout layer by layer.
    for layer in range(len(hidden_layers)):
        neuron_layer = activation(neuron_type, synapse(
            neuron_input_type, layer_neurons[layer], layer_weights[layer],
            layer_biases[layer]))
        layer_neurons.append(neuron_layer)

    # Performs softmax normalisation after the final layer.
    y = tf.nn.softmax(activation("linear", synapse(
        neuron_input_type, layer_neurons[-1], layer_weights[-1],
        layer_biases[-1])))

    return x, y, layer_weights, layer_biases


def fill_variable(shape):
    """Initialises a weight or bias tensor with normally distributed values."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def activation(type, synapse):
    """Chooses the activation function to use."""
    if type == "sigmoid":
        return tf.sigmoid(synapse)
    elif type == "linear":
        return synapse
    elif type == "tanh":
        return tf.tanh(synapse)
    elif type == "radial":
        return tf.sqrt(tf.exp(tf.neg(tf.square(synapse))))


def synapse(type, input_signal, weights, bias):
    """Chooses the synapse function to use."""
    if type == "sum":
        return tf.nn.bias_add(tf.matmul(input_signal, weights), bias)
    elif type == "sqsum":
        return tf.nn.bias_add(tf.matmul(tf.square(input_signal),
                                        tf.square(weights)), bias)
    elif type == "abssum":
        return tf.nn.bias_add(tf.matmul(tf.abs(input_signal), tf.abs(weights)),
                              bias)
