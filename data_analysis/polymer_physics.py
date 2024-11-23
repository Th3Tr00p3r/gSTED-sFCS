"""Polymer physics"""

import numpy as np
import scipy as sp
from matplotlib.axes import Axes

BP_TO_NM = 0.34
KUHN_LENGTH_NM = 100.0


def gyradius(L_bp: int, topology: str, model: str, kuhn_length_nm=KUHN_LENGTH_NM):
    """Doc."""

    L_nm = L_bp * BP_TO_NM

    # Linear Chains
    if topology == "linear":
        # Benoit-Doty
        if model == "wlc":
            lambda_ = L_nm / kuhn_length_nm
            return np.sqrt(
                (
                    lambda_ / 6
                    - 1 / 4
                    + 1 / (4 * lambda_)
                    - 1 / (8 * lambda_**2) * (1 - np.exp(-2 * lambda_))
                )
                * kuhn_length_nm**2
            )

        # Gaussian
        elif model == "gaussian":
            return np.sqrt(L_nm * kuhn_length_nm / 6)

    # Circular/Relaxed Ring Chains
    elif topology == "ring":
        if model == "wlc":
            return np.sqrt(L_nm * kuhn_length_nm / 12 - 7 / 72 * kuhn_length_nm**2)
        elif model == "gaussian":
            return np.sqrt(L_nm * kuhn_length_nm / 12)


def estimate_resolution_from_measured_field(w_meas_nm: float, Rg_nm: float):
    """Doc."""

    return np.sqrt(w_meas_nm**2 - 4 / 3 * Rg_nm**2)


def debye_structure_factor_fit(q, Rg: float) -> np.ndarray:
    """
    Static structure factor expression for Gaussian linear polymers.
    Sharp Bloomfield, Biopolymers (1968)
    See Yamakawa's book (Helical Worm-Like Chain in Polymer Solutions), Equation 5.30
    """
    x = (Rg * q) ** 2
    return 2 / x**2 * (x - 1 + np.exp(-x))


def dawson_structure_factor_fit(q, Rg: float) -> np.ndarray:
    """
    Static strucure factor expression for Gaussian ring polymers.
    See Yamakawa's book (Helical Worm-Like Chain in Polymer Solutions), Equation 5.70
    """
    x = Rg * q / np.sqrt(2)
    return 1 / x * sp.special.dawsn(x)


def wlc_rod_structure_factor_fit(q, L: float):
    """
    Static structure facor expression for short worm-like chain polymers.
    See Yamakawa's book (Helical Worm-Like Chain in Polymer Solutions), Equation 5.32.
    (Originally derived by Neugebauer, 1943)
    """
    x = L * q
    si_x, _ = sp.special.sici(x)
    return 2 / x**2 * (x * si_x + np.cos(x) - 1)


def plot_theoretical_structure_factor_in_ax(ax: Axes, q: np.ndarray, coeff: float, model: str):
    """
    Plot theoretical structure factor in an axis.
    """

    if model == "ideal":
        ax.plot(coeff * q, q ** (-2), "--k", label="$q^{-2}$" + f" ({model})")
    elif model == "fractal globule":
        ax.plot(coeff * q, q ** (-3), "-.k", label="$q^{-3}$" + f" ({model})")
    else:
        raise ValueError(f"Unknown model '{model}'. Choose 'ideal' or 'fractal globule'.")
