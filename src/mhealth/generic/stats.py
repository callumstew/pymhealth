""" Generic statistical features
These functions are more rigid than their scipy counter-parts, but
the intention is that (using numba) they are faster.
However, scipy has a greater selection of statistical features and
may be preferable.
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def minmax(x):
    """ Minimum and maximum of an array looping once
    Params:
        x (np.ndarray[float/int]): Array
    Returns:
        (float/int, float/int): Tuple of minimum and maximum
    """
    x = x.ravel()
    minimum = x[0]
    maximum = x[0]
    for i in range(1, len(x)):
        if x[i] < minimum:
            minimum = x[i]
        if x[i] > maximum:
            maximum = x[i]
    return (minimum, maximum)


def drange(x):
    """ Range of data
    Params:
        x (np.ndarray[float/int])
    Returns
        float/int: max(x) - min(x)
    """
    minimum, maximum = minmax(x)
    return maximum - minimum


def interquartile_range(x):
    """ Interquartile range
    Params:
        x (np.ndarray[float/int])
    Returns
        float/int: 75th percentile - 25th percentile
    """
    return np.subtract(*np.percentile(x, [75, 25]))


@jit(nopython=True)
def skewness(x):
    """ Skewness (third-moment) of a distribution
    Params:
        x (np.ndarray): Distribution to find skew of
    Returns:
        float: skewness
    """
    sd = np.std(x)
    if sd == 0:
        return 0
    return np.sum(((x - np.mean(x))**3) / len(x)) / sd**3


@jit(nopython=True)
def kurtosis(x):
    """ Kurtosis B2 = mu_4 / mu_2^2
    Params:
        x (np.ndarray): Distribution to find kurtosis of
    Returns:
        float: kurtosis
    """
    v = np.var(x)
    if v == 0:
        return 0
    return np.sum(((x - np.mean(x))**4) / len(x)) / (v**2)


def kurtosis_excess(x):
    """ Kurtosis excess is the kurtosis - 3
    Params:
        x (np.ndarray): Distribution to find kurtosis excess of
    Returns:
        float: kurtosis excess
    """
    return kurtosis(x) - 3


mean = np.mean
median = np.median
std = np.std
var = np.var
dmin = np.min
dmax = np.max
percetile = np.percentile
