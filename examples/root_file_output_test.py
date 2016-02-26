import numpy as np
import root_numpy as rnp

import postprocessing.ttree
import ROOT

# Gets a TTree from a ROOT file.
signal_file = ROOT.TFile("ROOT_data/signal1MTraining.root")
signal_tree = signal_file.Get("h101;1")

# Converts the TTree to a NumPy structured array.
array = rnp.tree2array(signal_tree)

# Generates three new branches to be added to the array.
classifier_branch = postprocessing.ttree.name_classifier_branches(
    np.random.rand(247015, 3), ['ClassA', 'ClassB', 'ClassC'])

# Attaches the new branches to the original array.
together = postprocessing.ttree.join_struct_arrays([array, classifier_branch])

# Outputs the total array as a ROOT file.
rnp.array2root(together, "ROOT_data/together.root")
