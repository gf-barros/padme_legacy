import os
from itertools import islice

import h5py
import matplotlib.pyplot as plt
import meshio
import numpy as np


def readvtk(filename):
    with open(filename + ".vtk") as lines:
        array = np.genfromtxt(islice(lines, 125939, 210239))
    return array


def readh5(filename, dataset):
    f = h5py.File(filename + ".h5", "r")
    for key in f.keys():
        data = f[dataset]
    data_array = np.array(data, copy=True)
    f.close()
    return data_array


def snapshots_assembly(dir_input, file_type, compartments):
    if file_type == "libmesh_h5":
        path, dirs, files = next(os.walk(dir_input))
        m = len(dirs)
        n = (
            readh5(path + dirs[0] + "/out_1_000_00000", compartments[0])
        ).shape[0]
        X = np.zeros((len(compartments) * n, m))
        for j in range(m):
            i = 0
            for compartment in compartments:
                tmp = readh5(
                    path
                    + "step"
                    + str(j)
                    + "/out_1_000_"
                    + str(j).zfill(5),
                    compartment,
                )
                X[i * n : (i + 1) * n, j] = tmp[:]
                i += 1
    elif file_type == "freefem_vtk":
        path, dirs, files = next(os.walk(dir_input))
        compartments = ["susceptible", "infected", "recovered", "deceased"]
        single_comp = []
        for file in files:
            if (compartments[0] in file) and ("Init" not in file):
                single_comp.append(file)
        single_comp = sorted(single_comp, key=len)
        adjust = [
            x.replace(compartments[0] + ".vtk", "") for x in single_comp
        ]
        n = (readvtk(path + adjust[0] + compartments[0])).shape[0]
        m = len(adjust)
        X = np.zeros((len(compartments) * n, m))
        for j in range(m):
            i = 0
            print(j)
            for compartment in compartments:
                tmp = readvtk(path + adjust[j] + compartment)
                X[i * n : (i + 1) * n, j] = tmp[:]
                i += 1
    return X
