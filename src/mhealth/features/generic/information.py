""" Generic information theory features and functions
"""
import numpy as np
from numba import njit


@njit
def entropy(x):
    """ Shannon entropy
    Args:
        x (np.ndarray[float]): Counts or probabilities of discrete distribution
    Returns:
        float: Shannon entropy of distribution
    """
    x = x / np.sum(x)
    return - np.sum(x * np.log(x))
