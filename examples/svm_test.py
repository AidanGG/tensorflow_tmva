import ROOT
import tensorflow as tf
import preprocessing.ttree
import preprocessing.data_set
import classification.svm.svm
import numpy as np

signal = ROOT.TNtuple("ntuple", "ntuple", "x:y:signal")
background = ROOT.TNtuple("ntuple", "ntuple", "x:y:signal")
for i in range(200):
    signal.Fill(ROOT.gRandom.Gaus(1, 1), ROOT.gRandom.Gaus(1, 1), 1)
    background.Fill(ROOT.gRandom.Gaus(-1, 1), ROOT.gRandom.Gaus(-1, 1), -1)

gcSaver = []

gcSaver.append(ROOT.TCanvas())

histo = ROOT.TH2F("histo", "", 1, -5, 5, 1, -5, 5)
histo.Draw()

signal.SetMarkerColor(ROOT.kRed)
signal.Draw("y:x", "signal > 0", "same")

background.SetMarkerColor(ROOT.kBlue)
background.Draw("y:x", "signal < 0", "same")

binary = preprocessing.ttree.ttrees_to_internal(
    [signal, background], ["x", "y"], binary=True)

input_tensor = tf.placeholder(tf.float32, [None, 2])
labels_tensor = tf.placeholder(tf.float32, [None, 1])
beta, offset, cost = classification.svm.svm.cost(
    input_tensor, labels_tensor, 400,
    kernel_type="gaussian", C=1, gamma=1)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(100):
    sess.run(train_step, feed_dict={
             input_tensor: binary.data(), labels_tensor: binary.labels()})

test = np.random.normal(loc=1, size=[100, 2])

test_tensor = tf.placeholder(tf.float32, [None, 2])

model = classification.svm.svm.decide(
    input_tensor, 400, test_tensor, 100, beta, offset,
    kernel_type="gaussian", gamma=1)

print(sess.run(beta))
print(sess.run(offset))
print(sess.run(model, feed_dict={
      input_tensor: binary.data(), test_tensor: test}))
print(sess.run(tf.reduce_sum(model) / 100, feed_dict={
      input_tensor: binary.data(), test_tensor: test}))
