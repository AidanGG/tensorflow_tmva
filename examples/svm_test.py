import numpy as np
import tensorflow as tf

import classification.svm.svm
import preprocessing.data_set
import preprocessing.ttree
import ROOT

# Generates two 2D normal distributions in two TTrees.
signal = ROOT.TNtuple("ntuple", "ntuple", "x:y:signal")
background = ROOT.TNtuple("ntuple", "ntuple", "x:y:signal")
for i in range(200):
    signal.Fill(ROOT.gRandom.Gaus(1, 1), ROOT.gRandom.Gaus(1, 1), 1)
    background.Fill(ROOT.gRandom.Gaus(-1, 1), ROOT.gRandom.Gaus(-1, 1), -1)

# Draws the distribution.
gcSaver = []
gcSaver.append(ROOT.TCanvas())
histo = ROOT.TH2F("histo", "", 1, -5, 5, 1, -5, 5)
histo.Draw()
signal.SetMarkerColor(ROOT.kRed)
signal.Draw("y:x", "signal > 0", "same")
background.SetMarkerColor(ROOT.kBlue)
background.Draw("y:x", "signal < 0", "same")

# Reads the TTrees into our data structure using {-1, 1} labels instead of
# a one-hot matrix.
binary = preprocessing.ttree.ttrees_to_internal(
    [signal, background], ["x", "y"], binary=True)

# Gets relevent tensors from SVM model.
input_tensor = tf.placeholder(tf.float32, [None, 2])
labels_tensor = tf.placeholder(tf.float32, [None, 1])
beta, offset, cost = classification.svm.svm.cost(
    input_tensor, labels_tensor, 400,
    kernel_type="gaussian", C=1, gamma=1)

# Sets up the optimiser and initialises Variables and session.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Runs the training step.
for i in range(100):
    sess.run(train_step, feed_dict={
             input_tensor: binary.data(), labels_tensor: binary.labels()})

# Generates a set of signal test data.
print("Generating non-deterministic test data set from signal distribution...")
test = np.random.normal(loc=1, size=[100, 2])
test_tensor = tf.placeholder(tf.float32, [None, 2])

# Classifies a test point from the trained SVM parameters.
model = classification.svm.svm.decide(
    input_tensor, 400, test_tensor, 100, beta, offset, kernel_type="gaussian",
    gamma=1)

print("Test data classified as signal: %f%%" % sess.run(
    tf.reduce_sum(model), feed_dict={input_tensor: binary.data(),
                                     test_tensor: test}))
