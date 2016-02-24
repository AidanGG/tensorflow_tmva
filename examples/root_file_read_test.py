import preprocessing.data_set
import preprocessing.ttree
import ROOT

branches = ["k0mm2", "k0et", "k0hso00", "k0hso01", "k0hso02", "k0hso03",
            "k0hso04", "k0hso10", "k0hso12", "k0hso14", "k0hso20", "k0hso22",
            "k0hso24", "k0hoo0", "k0hoo1", "k0hoo2", "k0hoo3", "k0hoo4",
            "cos_thr", "deltaz", "cosb"]

signal_file = ROOT.TFile("signal.root")
continuum_file = ROOT.TFile("continuum.root")
signal_tree = signal_file.Get("h101;1")
continuum_tree = continuum_file.Get("h101;1")

multiclass = preprocessing.ttree.ttrees_to_internal(
    [signal_tree, continuum_tree], branches, binary=False)

binary = preprocessing.ttree.ttrees_to_internal(
    [signal_tree, continuum_tree], branches, binary=True)

print(multiclass.data())
print(multiclass.labels())
print(multiclass.data_points())
print(multiclass.dimensions())
print(multiclass.classes())

print(binary.data())
print(binary.labels())
print(binary.data_points())
print(binary.dimensions())
print(binary.classes())
