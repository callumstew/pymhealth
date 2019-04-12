""" Recurrence quantification analysis features
"""
import numpy as np
from numba import njit
from .information import entropy


def rq(x, radius=0):
    """ Recurrence matrix
    Params:
        x (np.ndarray): signal
        radius (int/float): Difference in values must be within the radius
            to count as a recurrence. Default = 0
    Returns:
        np.ndarray[bool, bool]: N*N boolean matrix where True corresponds to
            a recurrence, N = len(x)
    """
    return np.abs(np.subtract.outer(x, x)) <= radius


def recurrence_rate(r):
    """ The proportion of recurrence points in a recurrence matrix
    Params:
        r (np.ndarray[bool, bool]): Recurrence matrix
    Returns:
        float
    """
    return np.sum(r)/(r.shape[0]*r.shape[1])


@njit
def determinism(r):
    """ Determinism - proportion of recurrence points forming diagonal lines
    at least 2 points long.
    Params:
        r (np.ndarray[bool, bool]): Recurrence matrix
    Returns:
        float (0-1)
    """
    out = np.zeros(r.shape, dtype=np.bool_)
    for i in range(1, r.shape[0]-1):
        for j in range(1, r.shape[1]-1):
            out[i, j] = (r[i, j] & r[i-1, j-1]) | (r[i, j] & r[i+1, j+1])
        out[i, 0] = r[i, 0] & r[i+1, 1]
        out[i, -1] = r[i, -1] & r[i-1, -2]
    for j in range(1, r.shape[1]-1):
        out[0, j] = r[0, j] & r[1, j+1]
        out[-1, j] = r[-1, j] & r[-2, j-1]
    out[0, 0] = r[0, 0] & r[1, 1]
    out[-1, -1] = r[-1, -1] & r[-2, -2]
    return np.sum(out) / (r.shape[0] * r.shape[1])


@njit
def laminarity(r):
    """ Laminarity - proportion of recurrence points forming vertical lines
    at least 2 points long.
    Params:
        r (np.ndarray[bool, bool]): Recurrence matrix
    Returns:
        float (0-1)
    """
    out = np.zeros(r.shape, dtype=np.bool_)
    for i in range(r.shape[0]):
        out[i, 0] = r[i, 0] & r[i, 1]
        out[i, -1] = r[i, -1] & r[i, -2]
        for j in range(1, r.shape[1]-1):
            out[i, j] = (r[i, j] & r[i, j+1]) | (r[i, j] & r[i, j-1])
    return np.sum(out) / (r.shape[0] * r.shape[1])


def diagonal_lengths(r, minlen=2):
    """ The lengths of the contiguous diagonal lines
    Slower than simply counting points as in determinism.
    Params:
        r (np.ndarray[bool, bool]): Recurrence matrix
        minlen (int): Minimum length of a line
    Returns:
        np.ndarray[int]
    """
    @njit
    def diagonal_lengths_matrix(r):
        out = np.zeros(r.shape, dtype=np.int32)
        for i in range(1, r.shape[0]):
            for j in range(1, r.shape[1]):
                out[i, j] = (out[i-1, j-1] + 1) * (r[i, j] & r[i-1, j-1])
                out[i-1, j-1] = 0
        return out

    out = diagonal_lengths_matrix(r)
    out += 1
    return out[out >= minlen]


def vertical_lengths(r, minlen=2):
    """ The lengths of the contiguous vertical lines
    Slower than simply counting points as in laminarity.
    Params:
        r (np.ndarray[bool, bool]): Recurrence matrix
        minlen (int): Minimum length of a line
    Returns:
        np.ndarray[int]
    """
    @njit
    def vertical_lengths_matrix(r):
        out = np.zeros(r.shape, dtype=np.int32)
        for i in range(1, r.shape[0]):
            for j in range(r.shape[1]):
                out[i, j] = (out[i-1, j] + 1) * (r[i-1, j] & r[i, j])
                out[i-1, j] = 0
        return out

    out = vertical_lengths_matrix(r)
    out += 1
    return out[out >= minlen]


def length_entropy(segment_lengths):
    """ Entropy of the segment lengths (e.g. diagonals)
    Params:
        segment_lengths (np.ndarray[int]): Segment lengths
    Returns:
        float: Shannon entropy of distribution of segment lengths
    """
    counts = np.unique(segment_lengths, return_counts=True)[1]
    return entropy(counts)
