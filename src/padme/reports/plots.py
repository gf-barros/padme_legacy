import os
from itertools import islice

import h5py
import matplotlib.pyplot as plt
import meshio
import numpy as np


def export_figures(
    Mat, k, dir_name, compartment="susceptible", export_type="modes"
):
    [n, m] = Mat.shape
    if "delayed" in dir_name:
        delayed = True
    else:
        delayed = False
    if "uncoupled" in dir_name:
        uncoupled = True
    else:
        uncoupled = False
    if not (delayed):
        if not (uncoupled):
            # coupled requires splitting
            n = n // 5
            compartments = ["s", "e", "i", "r", "d"]
            aux_str = "coupled"
        else:
            comp_dict = {"s": 0, "e": 1, "i": 2, "r": 3, "d": 4}
            compartments = [compartment]
            aux_str = "uncoupled"
        print(f"Plotting {export_type} for standard {aux_str} DMD.")
    else:
        if not (uncoupled):
            # coupled requires splitting
            n = n // 4
            compartments = [
                "susceptible",
                "infected",
                "recovered",
                "deceased",
            ]
            aux_str = "coupled"
        else:
            comp_dict = {
                "susceptible": 0,
                "infected": 1,
                "recovered": 2,
                "deceased": 3,
            }
            compartments = [compartment]
            aux_str = "uncoupled"
        print(f"Plotting {export_type} for delayed {aux_str} DMD.")
    mesh = meshio.read(dir_name + "/mesh.vtk")
    if export_type == "modes":
        try:
            os.makedirs(dir_name + "/spatial")
        except OSError as error:
            print("Directory already created. Proceeding...")
        if uncoupled:
            for i in range(k):
                for compartment in compartments:
                    i_cont = comp_dict[compartment]
                    print(i_cont)
                    print(compartment)
                    mesh_out = meshio.Mesh(
                        mesh.points,
                        mesh.cells_dict,
                        point_data={compartment: (Mat[:, i])},
                    )
                    str_out = (
                        dir_name
                        + "/spatial"
                        + "/spatial_"
                        + compartment
                        + "_"
                        + str(i)
                        + ".vtk"
                    )
                    mesh_out.write(str_out)
        else:
            for i in range(k):
                i_cont = 0
                for compartment in compartments:
                    print(compartment)
                    print(i_cont)
                    print(n)
                    mesh_out = meshio.Mesh(
                        mesh.points,
                        mesh.cells_dict,
                        point_data={
                            compartment: (
                                Mat[i_cont * n : (i_cont + 1) * n, i]
                            )
                        },
                    )
                    str_out = (
                        dir_name
                        + "/spatial"
                        + "/spatial_"
                        + compartment
                        + "_"
                        + str(i)
                        + ".vtk"
                    )
                    mesh_out.write(str_out)
                    i_cont += 1


def export_plots(Mat, k, dir_name, type_sim):
    Mat = Mat[:k]
    x = Mat.real
    y = Mat.imag
    # plot the complex numbers
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(x, y, "*", label=type_sim)
    circ = plt.Circle((0, 0), radius=1, edgecolor="b", facecolor="None")
    plt.ylabel("Imaginary")
    plt.xlabel("Real")
    ax.add_patch(circ)
    plt.ylim([-1, 1])
    plt.xlim([-1, 1])
