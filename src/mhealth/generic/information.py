""" Generic information theory features and functions
"""
import numpy as np
from numba import jit


@jit
def entropy(x: np.ndarray) -> float:
    """ Shannon entropy
    Args:
        x (np.ndarray[float]): Counts or probabilities of discrete distribution
    Returns:
        float: Shannon entropy of distribution
    """
    x = x / np.sum(x)
    x += 1e-30
    return - np.sum(x * np.log(x))


# def sampen_bucket(x, m=2, r=0.2, rsplit=5):
#     """
#     doi:10.3390/e20010061
#     """
#     n = len(x)
#     x = np.convolve(x, np.ones(m) / m, mode='valid')
#     x = 1 + x - np.min(x)
#     Nb = np.max(x) / r / rsplit
#     bucket = [[]] * (n-m)
#     for i in range(n-m):
#         b = x[i] / r / rsplit
#         bucket[b] = bucket[b].append(x[i:i+m])
#     for i in range(n-m):
#         bucket[i].sort(key=lambda x: x[0])
#     for ib in range(Nb):
#         for jb in range(ib-m*rsplit, ib):
#             pass
#     pass


@jit(nopython=True)
def sampen(x: np.ndarray, mm: int = 2, r: float = 0.2) -> float:
    """
    Taken from https://github.com/raphaelvallat/entropy until
    lightweight or bucket-assisted algorithms implemented
    Fast evaluation of the sample entropy using Numba.
    """
    n = x.size
    n1 = n - 1
    mm += 1
    mm_dbld = 2 * mm

    # Define threshold
    r *= x.std()

    # initialize the lists
    run = np.zeros(n)
    run1 = np.zeros(n)
    r1 = np.zeros(n * mm_dbld)
    a = np.zeros(mm)
    b = np.zeros(mm)
    p = np.zeros(mm)

    for i in range(n1):
        nj = n1 - i

        for jj in range(nj):
            j = jj + i + 1
            if abs(x[j] - x[i]) < r:
                run[jj] = run1[jj] + 1
                m1 = mm if mm < run[jj] else run[jj]
                for m in range(m1):
                    a[m] += 1
                    if j < n1:
                        b[m] += 1
            else:
                run[jj] = 0
        for j in range(mm_dbld):
            run1[j] = run[j]
            r1[i + n * j] = run[j]
        if nj > mm_dbld - 1:
            for j in range(mm_dbld, nj):
                run1[j] = run[j]

    m = mm - 1

    while m > 0:
        b[m] = b[m - 1]
        m -= 1

    b[0] = n * n1 / 2
    p = np.true_divide(a, b)
    return -np.log(p[-1])
