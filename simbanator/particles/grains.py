"""Unit conversion helpers for astrophysical quantities."""

import numpy as np


def Z_to_OH12(Z):
    """Convert metallicity *Z* (mass fraction) to 12 + log(O/H).

    Parameters
    ----------
    Z : float or array-like
        Metal mass fraction.

    Returns
    -------
    float or np.ndarray
        Oxygen abundance in solar-normalised logarithmic scale.
    """
    return np.log10(Z / 0.0127) + 8.69


def Dust_to_Metal(M_dust, M_h2, abundance):
    """Compute the dust-to-metal ratio following De Vis (2019).

    Parameters
    ----------
    M_dust : float or array-like
        Dust mass.
    M_h2 : float or array-like
        Molecular hydrogen mass.
    abundance : float or array-like
        12 + log(O/H) oxygen abundance.

    Returns
    -------
    float or np.ndarray
        Dust-to-metal ratio.
    """
    f_z = 27.36 * (10 ** (abundance - 12))
    M_z = f_z * M_h2 + M_dust
    return M_dust / M_z
