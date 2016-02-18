import ROOT
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
