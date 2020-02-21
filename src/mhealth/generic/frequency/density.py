"""Module concerning spectrum density functions and features
"""
import numpy as np
import numba
from numba import types as ntypes
from typing import Optional


@numba.jit
def first_index(arr, x):
    for i in range(len(arr)):
        if x <= arr[i]:
            return i
    return len(arr)


@numba.jit
def peak_frequency(psd: np.ndarray, freqs: np.ndarray,
                   lower: Optional[float] = None,
                   upper: Optional[float] = None):
    """Find the peak frequency in a spectrum density.
    Params
        psd (np.ndarray[float]): Spectral density
        freqs (np.ndarray[float]): Frequency of the psd. Assumes ordered
        lower (float): Lower bound of the frequencies to consider
        upper (float): Upper board of the frequencies to consider
    Returns
        float: Frequency with maximum amplitude
    """
    lidx = 0 if lower is None else first_index(freqs, lower)
    uidx = len(psd) if upper is None else first_index(freqs, upper)
    return freqs[lidx + np.argmax(psd[lidx:uidx])]
