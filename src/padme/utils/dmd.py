import os
from itertools import islice

import h5py
import matplotlib.pyplot as plt
import meshio
import numpy as np


def DMD(X, dmd_list, dt_simulation):
    """
    Parameters
    ----------
    X : NumPy 2D-array
        Snapshots matrix.
    dmd_list : TYPE
        DESCRIPTION.
    dt_simulation : TYPE
        DESCRIPTION.

    Returns
    -------
    DMD_vals : Dict
        DESCRIPTION.

    """
    randomized_svd, r, q, p, initial_shift = dmd_list
    [n, m] = X.shape
    i_cont = 0
    X1 = X[:, initial_shift:-1]
    X2 = X[:, initial_shift + 1 :]
    if randomized_svd == True:
        U, S, VT = rSVD(X1, r, q, p)
        U = U[:, 0:r]
        S = S[0:r]
        VT = VT[0:r, :]
    else:
        U, S, VT = np.linalg.svd(
            X1,
            full_matrices=False,
            compute_uv=True,
            hermitian=False,
        )
        U = U[:, 0:r]
        S = S[0:r]
        VT = VT[0:r, :]

    S = np.divide(1.0, S)
    S = np.diag(S)
    U = np.transpose(U)
    VT = np.transpose(VT)
    A_tilde = np.linalg.multi_dot([U, X2, VT, S])
    eigenVal, eigenVec = np.linalg.eig(A_tilde)
    eigenVal = np.log(eigenVal) / (dt_simulation)
    Phi_DMD = np.linalg.multi_dot([X2, VT, S, eigenVec])
    Phi_inv = np.linalg.pinv(Phi_DMD)
    x0 = X1[:, 0]
    b = np.dot(Phi_inv, x0)
    b = b[:, np.newaxis]
    t = np.arange(start=0, stop=(m - initial_shift + 1)) * dt_simulation
    t = t[np.newaxis, :]
    eigenVal = eigenVal[:, np.newaxis]
    temp = np.multiply(eigenVal, t)
    temp = np.exp(temp)
    Dynamics = np.multiply(b, temp)
    Xdmd = np.dot(Phi_DMD, Dynamics)
    Xdmd = np.real(Xdmd)
    DMD_vals = {
        "solution": Xdmd,
        "u": U,
        "s": S,
        "eigenvalues": eigenVal,
        "eigenvectors": eigenVec,
        "phi": Phi_DMD,
    }
    i_cont += 1
    return DMD_vals


def rSVD(X, r, q, p):
    ny = X.shape[1]
    P = np.random.randn(ny, r + p)
    Z = X @ P
    for k in range(q):
        Z = X @ (X.T @ Z)
    Q, R = np.linalg.qr(Z, mode="reduced")
    Y = Q.T @ X
    UY, S, VT = np.linalg.svd(Y, full_matrices=0)
    U = Q @ UY
    return U, S, VT
