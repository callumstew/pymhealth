""" Generic time-domain features
"""
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.polynomial.polynomial import polyval, polyfit
from numba import njit, vectorize, guvectorize


@njit
def zero_crossings(x):
    """ Indices of zero-crossings in the input signal
    Params:
        x (np.ndarray): Signal
    Returns:
        np.ndarray[int]: Indices of zero-crossings
    """
    pos = x > 0
    return np.where(np.bitwise_xor(pos[:-1], pos[1:]))[0]


def zero_crossing_count(x):
    """ Number of zero-crossings in the input signal
    Params:
        x (np.ndarray): Signal
    Returns:
        int
    """
    return np.sum(zero_crossings(x))


def hjorth_mobility(x):
    """ Hjorth mobility
    sqrt of the variance of the first derivative of the signal divided by
    the variance of the signal
    Params:
        x (np.ndarray): Signal
    Returns:
        float: Hjorth mobility of signal
    """
    deriv = np.gradient(x)
    return np.sqrt(np.var(deriv) / np.var(x))


def hjorth_mobility_derivative(x, deriv):
    """ Hjorth mobility with precomputed first derivitive
    sqrt of the variance of the first derivative of the signal divided by
    the variance of the signal
    Params:
        x (np.ndarray): Signal
        deriv (np.ndarray): First derivative of the signal
    Returns:
        float: Hjorth mobility of signal
    """
    return np.sqrt(np.var(deriv) / np.var(x))


def hjorth_complexity(x):
    """ Hjorth complexity
    Hjorth mobility of the first derivitive divided by the hjorth mobility
    of the signal
    Params:
        x (np.ndarray): Signal
    Returns:
        float: Hjorth complexity of signal
    """
    deriv1 = np.gradient(x)
    return hjorth_mobility(deriv1) / hjorth_mobility_derivative(x, deriv1)


def hjorth_complexity_derivatives(x, deriv1, deriv2):
    """ Hjorth complexity with precomputed first and second derivitives
    Hjorth mobility of the first derivitive divided by the hjorth mobility
    of the signal
    Params:
        x (np.ndarray): Signal
        deriv1 (np.ndarray): First derivative of the signal
        deriv2 (np.ndarray): Second derivative of the signal
    Returns:
        float: Hjorth complexity of signal
    """
    return (hjorth_mobility_derivative(deriv1, deriv2) /
            hjorth_mobility_derivative(x, deriv1))


def dfa(x, windows, o=1):
    """ Detrended fluctuation analysis
    Params:
        x (np.ndarray): Signal
        windows (List[int[): List of window sizes
        o (int): Order of the polynomial to fit. Default 1
    Returns:
        float: scaling exponent
    """
    def profile(x):
        return np.cumsum(x - np.mean(x))

    def view(x, w, s):
        """ Strided window view of array - not necessary?
        Params:
            x (np.ndarray): Array to make window views of
            w (int): Window size
            s (int): Step size
        Returns:
            np.ndarray
        """
        stride = x.strides[0]
        N = x.shape[0]
        return as_strided(x, (((N - w) // s) + 1, w), (s * stride, stride))

    def fluctuations(xp, windows, o):
        s = windows[0]
        out = np.zeros((len(windows), len(xp)//s))
        for i, w in enumerate(windows):
            res = polyfit(np.arange(w), view(xp, w, s).T, o, full=True)[1][0]
            rms = np.sqrt(res / w)
            out[i, :rms.shape[0]] = rms
        return out.mean(1)

    F = fluctuations(profile(x), windows, o)
    scaling_exponent = o1fit(np.log(windows), np.log(F.mean(1)))
    return scaling_exponent


@njit
def o1fit(x, y):
    n = len(x)
    sumx = np.sum(x)
    b = (((n * np.sum(x * y)) - (sumx * np.sum(y))) /
         ((n * np.sum(x * x)) - (sumx * sumx)))
    A = np.mean(y) - (b * np.mean(x))
    return (A, b)


def o1fit_vec(x, ys):
    @guvectorize(["void(float64[:], float64[:], float64[:, :], float64[:, :])"],
                 "(t),(n),(n,m)->(t,m)")
    def o1fit_vector(two, x, ys, res):
        for i in range(ys.shape[1]):
            res[:, i] = o1fit(x, ys[:, i])

    return o1fit_vector(np.zeros(2), x, ys)
