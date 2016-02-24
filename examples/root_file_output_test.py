import numpy as np
import root_numpy as rnp

import postprocessing.ttree
import ROOT

signal_file = ROOT.TFile("ROOT_data/signal1MTraining.root")
signal_tree = signal_file.Get("h101;1")

array = rnp.tree2array(signal_tree)

classifier_branch = postprocessing.ttree.name_classifier_branches(
    np.random.rand(247015, 3), ['ClassA', 'ClassB', 'ClassC'])

together = postprocessing.ttree.join_struct_arrays([array, classifier_branch])

rnp.array2root(together, "ROOT_data/together.root")
