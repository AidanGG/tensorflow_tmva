import ROOT
import tensorflow as tf
import tensorflow_tmva.preprocessing.ttree
import tensorflow_tmva.preprocessing.data_set
import tensorflow_tmva.classification.svm.cost

signal = ROOT.TNtuple("ntuple", "ntuple", "x:y")
background = ROOT.TNtuple("ntuple", "ntuple", "x:y")
for i in range(10000):
    signal.Fill(ROOT.gRandom.Gaus(1, 1), ROOT.gRandom.Gaus(1, 1))
    background.Fill(ROOT.gRandom.Gaus(-1, 1), ROOT.gRandom.Gaus(-1, 1))

binary = tensorflow_tmva.preprocessing.ttree.ttrees_to_internal(
    [signal, background], ["x", "y"], binary=True)

input_tensor = tf.placeholder(tf.float32, [None, 2])
labels_tensor = tf.placeholder(tf.float32, [None, 1])
w, offset, cost = tensorflow_tmva.classification.svm.cost.unkernelised_cost(
    input_tensor, labels_tensor, 2, C=10)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={
             input_tensor: binary.data(), labels_tensor: binary.labels()})

print sess.run(w)
print sess.run(offset)
