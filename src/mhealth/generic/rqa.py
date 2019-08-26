""" Recurrence quantification analysis features
"""
import numpy as np
from numba import jit
from scipy.spatial.distance import pdist, squareform
from .information import entropy


@jit(nopython=True)
def rq(x: np.ndarray, radius: float = 0.) -> np.ndarray:
    """Recurrence matrix.

    Args:
        x (np.ndarray): signal
        radius (int/float): Difference in values must be within the radius
            to count as a recurrence. Default = 0

    Returns:
        np.ndarray[bool, bool]: N*N boolean matrix where True corresponds to
            a recurrence, N = len(x)

    """
    n = len(x)
    out = np.zeros((n, n), dtype=np.bool_)
    for i in range(n):
        for j in range(n):
            out[i, j] = np.abs(x[i] - x[j]) <= radius
    return out


def rq2(x: np.ndarray, radius: float = 0.) -> np.ndarray:
    """Recurrence matrix multi-column.

    Can handle multi-column input (observations=vectors), but is not jit-able

    Args:
        x (np.ndarray): signal
        radius (int/float): Difference in values must be within the radius
            to count as a recurrence. Default = 0

    Returns:
        np.ndarray[bool, bool]: N*N boolean matrix where True corresponds to
            a recurrence, N = len(x)

    """
    DD = squareform(pdist(x))
    return DD <= radius


@jit
def recurrence_rate(r: np.ndarray) -> float:
    """Calculate proportion of recurrent points in RQ matrix.

    Args:
        r (np.ndarray[bool, bool]): Recurrence matrix

    Returns:
        float: (0-1)

    """
    return np.sum(r)/(r.shape[0]*r.shape[1])


@jit(nopython=True)
def determinism(r: np.ndarray) -> float:
    """Calculate determinism of RQ matrix.

    Proportion of recurrence points forming diagonal lines at
    least 2 points long.

    Args:
        r (np.ndarray[bool, bool]): Recurrence matrix

    Returns:
        float: (0-1)

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


@jit(nopython=True)
def laminarity(r: np.ndarray) -> float:
    """Laminarity of RQ matrix.

    Proportion of recurrence points forming vertical lines
    at least 2 points long.

    Args:
        r (np.ndarray[bool, bool]): Recurrence matrix

    Returns:
        float: (0-1)

    """
    out = np.zeros(r.shape, dtype=np.bool_)
    for i in range(r.shape[0]):
        out[i, 0] = r[i, 0] & r[i, 1]
        out[i, -1] = r[i, -1] & r[i, -2]
        for j in range(1, r.shape[1]-1):
            out[i, j] = (r[i, j] & r[i, j+1]) | (r[i, j] & r[i, j-1])
    return np.sum(out) / (r.shape[0] * r.shape[1])


@jit(nopython=True)
def diagonal_lengths(r: np.ndarray, minlen: int = 2) -> np.ndarray:
    """Calculate lengths of contiguous diagonal lines in RQ matrix.

    Slower than simply counting points as in determinism.

    Args:
        r (np.ndarray[bool, bool]): Recurrence matrix
        minlen (int): Minimum length of a line

    Returns:
        np.ndarray[int]: Lengths of diagonal lines greater than the minlen

    """
    out = np.zeros(r.shape, dtype=np.int32)
    for i in range(1, r.shape[0]):
        for j in range(1, r.shape[1]):
            out[i, j] = (out[i-1, j-1] + 1) * (r[i, j] & r[i-1, j-1])
            if out[i, j]:
                out[i-1, j-1] = 0
    out += 1
    out = out.reshape(out.shape[0] * out.shape[1])
    return out[out >= minlen]


@jit(nopython=True)
def vertical_lengths(r: np.ndarray, minlen: int = 2) -> np.ndarray:
    """Calculate lengths of contiguous vertical lines in RQ matrix.

    Slower than simply counting points as in laminarity.

    Args:
        r (np.ndarray[bool, bool]): Recurrence matrix
        minlen (int): Minimum length of a line

    Returns:
        np.ndarray[int]: Lengths of vertical lines greater than the minlen

    """
    out = np.zeros(r.shape, dtype=np.int32)
    n, m = r.shape
    for i in range(1, n):
        for j in range(m):
            out[i, j] = (out[i-1, j] + 1) * (r[i-1, j] & r[i, j])
        for j in range(m):
            if out[i, j] >= 1:
                out[i-1, j] = 0
    out += 1
    out = out.reshape(out.shape[0] * out.shape[1])
    return out[out >= minlen]


@jit
def length_entropy(r: np.ndarray, minlen: int = 2) -> float:
    """Calculate entropy of diagonal lengths in RQ matrix.

    Args:
        r (np.ndarray[bool, bool]): Recurrence matrix
        minlen (int): Minimum length of a line
    Returns:
        float: Shannon entropy of distribution of segment lengths
    """
    dlens = diagonal_lengths(r, minlen)
    counts = _dlen_counts(dlens, minlen, r.shape[0])
    return entropy(counts)


@jit(nopython=True)
def _dlen_counts(dlens: np.ndarray, minlen: int, N: int) -> np.ndarray:
    out = np.zeros(N, dtype=np.int64)
    for v in dlens:
        out[v] += 1
    return out[minlen:]
