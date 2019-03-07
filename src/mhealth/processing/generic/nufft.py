""" Non-Uniform FFT
https://cims.nyu.edu/cmcl/nufft
"""
from numba import njit
import numpy as np

def nufft_k_array(M):
    return np.arange(-M//2, (M//2))


def nudft(x, y, M):
    return (1 / len(x)) * np.dot(y, np.exp(1j * nufft_k_array(M) * x[:, np.newaxis]))
