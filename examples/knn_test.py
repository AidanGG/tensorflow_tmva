import numpy as np
import tensorflow as tf

import classification.knn.knn as knn
import preprocessing.data_set
import preprocessing.ttree
import ROOT

# Generates two 3D normal distributions in two TTrees.
signal = ROOT.TNtuple("ntuple", "ntuple", "x:y:z")
background = ROOT.TNtuple("ntuple", "ntuple", "x:y:z")
for i in range(10000):
    signal.Fill(ROOT.gRandom.Gaus(1, 20), ROOT.gRandom.Gaus(
        1, 20), ROOT.gRandom.Gaus(1, 20))
    background.Fill(ROOT.gRandom.Gaus(-1, 20),
                    ROOT.gRandom.Gaus(-1, 20), ROOT.gRandom.Gaus(-1, 20))

# Reads the TTrees into our data structure.
data = preprocessing.ttree.ttrees_to_internal(
    [signal, background], ["x", "y", "z"], binary=False)

# Sets up a test instance.
test = np.array([[0., 0., 0.]])

# Gets the scales to use for the distribution widths
scale_frac = 0.8
scales = knn.scale(data.data(), scale_frac)

# Gets relevent tensors from k-NN model.
train_tensor, labels_tensor, test_tensor, neighbours = knn.model_single(
    3, 2, scale_frac, scales, 100)

sess = tf.Session()

print("Searching for 100 nearest neighbours")

# Feeds all data into model.
results = sess.run(neighbours,
                   feed_dict={train_tensor: data.data(),
                              labels_tensor: data.labels(), test_tensor: test})
print("Signal neighbours: %d" % results[0])
print("Background neighbours: %d" % results[1])
