import ROOT


def ttrees_to_one_hot(ttrees):
    one_hot = []

    for i in range(len(ttrees)):
        for j in range(ttrees[i].GetEntries()):
            entry = len(ttrees) * [0]
            entry[i] = 1
            one_hot.append(entry)

    return one_hot


def ttrees_to_arrays(ttrees, branches):
    one_hot = ttrees_to_one_hot(ttrees)
    x = []

    for i in range(len(ttrees)):
        ttrees[i].SetBranchStatus("*", 0)
        for j in range(len(branches)):
            ttrees[i].SetBranchStatus(branches[j], 1)

        for row in range(ttrees[i].GetEntries()):
            entry = []
            for column in range(len(branches)):
                entry.append(0)
                ttrees[i].SetBranchAddress(branches[column], entry[column])
            ttrees[i].GetEntry(row)
            x.append(entry)

    return x, one_hot
