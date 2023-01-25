"""Fluorescence intensity distribution analysis."""

import numpy as np
from scipy.integrate import quad

# i = 5
# n = np.array(range(30))
# N = S.CR[i]/S.G0_good[i]
# q = S.G0_good[i] *2*dt
# P3D = fidafunc3d_v2(n, N, q*np.sqrt(2), tol=1e-6)
# semilogy(n, Pexp, 'o', n, P, n, P3D)


def fidafunc3d_v2(n: np.ndarray, N, q, tol=1e-6, **kwargs) -> np.ndarray:
    """
    parameters -
    output: probability P(n) of having n counts during the sampling time interval
    inputs : n - counts per the sampling time interval
             N - number of molecules in the FCS area pi*wXY^2
             q - photon count from a single molecule in the field CENTER per sampling
             time interval (FCS count rate per molecule would be half of that)
    """

    def fida_f(xi, n, N, q, tol) -> np.ndarray:
        def fida_subint(x, xi, q) -> np.ndarray:
            return (np.exp((np.exp(np.sqrt(-1) * xi) - 1) * q * np.exp(-2 * x ** 2)) - 1) * x ** 2

        y = np.empty_like(xi)
        for j in range(len(xi)):
            y[j] = np.exp(
                4
                / np.sqrt(np.pi)
                * N
                * quad(lambda x: fida_subint(x, xi[j], q), 0, np.inf, epsrel=tol)
                - np.sqrt(-1) * n * xi[j]
            )
        return y

    P = np.empty_like(n)
    for k in range(len(n)):
        P[k] = quad(lambda xi: fida_f(xi, n[k], N, q, tol), -np.pi, np.pi, epsrel=tol) / (2 * np.pi)
    return P
