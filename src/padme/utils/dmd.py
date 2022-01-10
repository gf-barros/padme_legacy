import os
from itertools import islice

import h5py
import matplotlib.pyplot as plt
import meshio
import numpy as np


def dmd(snapshots_matrix, dmd_list, dt_simulation):
    """
    Parameters
    ----------
    snapshots_matrix : NumPy 2D-array
        Snapshots matrix.
    dmd_list : list
        Contains DMD hyperparameters:
            - svd_flag (randomized, incremental, standard)
            - # of basis vectors
            - # of power iterations
            - # of oversampling
            - # of initial shift steps
    dt_simulation : float
        Time step used in the simulations (for input and output).

    Returns
    -------
    DMD_vals : Dict
        Dictionary containing modes, eigenvalues, singular values and approximate solution.

    """
    randomized_svd, r, q, p, initial_shift = dmd_list
    [n, m] = snapshots_matrix.shape
    x1 = snapshots_matrix[:, initial_shift:-1]
    x2 = snapshots_matrix[:, initial_shift + 1 :]
    if randomized_svd == True:
        u, s, vt = rsvd(x1, r, q, p)
        u = u[:, 0:r]
        s = s[0:r]
        vt = vt[0:r, :]
    else:
        u, s, vt = np.linalg.svd(
            x1,
            full_matrices=False,
            compute_uv=True,
            hermitian=False,
        )
        u = u[:, 0:r]
        s = s[0:r]
        vt = vt[0:r, :]

    s = np.divide(1.0, s)
    s = np.diag(s)
    u = np.transpose(u)
    vt = np.transpose(vt)
    a_tilde = np.linalg.multi_dot([u, x2, vt, s])
    eigenval, eigenvec = np.linalg.eig(a_tilde)
    eigenval = np.log(eigenval) / (dt_simulation)
    phi_dmd = np.linalg.multi_dot([x2, vt, s, eigenvec])
    phi_inv = np.linalg.pinv(phi_dmd)
    x0 = x1[:, 0]
    b = np.dot(phi_inv, x0)
    b = b[:, np.newaxis]
    t = np.arange(start=0, stop=(m - initial_shift + 1)) * dt_simulation
    t = t[np.newaxis, :]
    eigenval = eigenval[:, np.newaxis]
    temp = np.multiply(eigenval, t)
    temp = np.exp(temp)
    dynamics = np.multiply(b, temp)
    x_dmd = np.dot(phi_dmd, dynamics)
    x_dmd = np.real(x_dmd)
    dmd_vals = {
        "solution": x_dmd,
        "u": u,
        "s": s,
        "eigenvalues": eigenval,
        "eigenvectors": eigenvec,
        "phi": phi_dmd,
    }
    return dmd_vals


def rsvd(mat, r, q, p):
    ny = mat.shape[1]
    p = np.random.randn(ny, r + p)
    z = mat @ p
    for k in range(q):
        z = mat @ (mat.T @ z)
    q, r = np.linalg.qr(z, mode="reduced")
    y = q.T @ mat
    u_y, s, vt = np.linalg.svd(y, full_matrices=0)
    u = q @ u_y
    return u, s, vt
