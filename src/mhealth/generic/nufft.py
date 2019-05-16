""" Non-Uniform FFT - Not implemented
https://cims.nyu.edu/cmcl/nufft
"""
from numba import njit
import numpy as np

def nufft_k_array(M):
    return np.arange(-M//2, (M//2))


def nudft(x, y, M):
    return (1 / len(x)) * np.dot(y, np.exp(1j * nufft_k_array(M) * x[:, np.newaxis]))


def nufft1d1freqs(ms, df=1.0):
    """Calculates 1D frequencies
    Params:
        ms (int): number of frequencies
        df (float): frequency spacing
    Returns:
        np.ndarray[float]: frequencies
    """
    return df * np.arange(-(ms//2), ms - (ms//2))
