import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import classification.mlp.cost
import classification.mlp.model
import classification.mlp.training
import preprocessing.data_set

# Reads the MNIST data into our internal data structure.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
my_train = preprocessing.data_set.DataSet(
    mnist.train.images, mnist.train.labels, 10)
my_test = preprocessing.data_set.DataSet(
    mnist.test.images, mnist.test.labels, 10)

# Gets the relevent MLP tensors.
x, y, W, b = classification.mlp.model.model(
    784, 10, [], neuron_type="linear", neuron_input_type="sum")

y_ = tf.placeholder(tf.float32, [None, 10])
cost = classification.mlp.cost.cost(y, y_, estimator_type="CE")

# Train step and Variable initialisation
train_step = classification.mlp.training.training(cost, learning_rate=0.01)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Runs the training step over batches of 100 each time.
# Would implement decay rate here by creating a new train_step with a
# lower learning_rate.
for i in range(1000):
    batch_xs, batch_ys = my_train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Calculates the prediction of each test case by taking the highest
# probability guess.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# From this prediction, calculate overall accuracy.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print sess.run(accuracy, feed_dict={x: my_test.data(), y_: my_test.labels()})
