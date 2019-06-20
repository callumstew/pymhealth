""" Generic time-domain features
"""
from functools import singledispatch
from typing import List
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy import polyfit
from numba import jit, guvectorize


@jit
def zero_crossings(x, th=0):
    """ Indices of zero-crossings in the input signal
    Params:
        x (np.ndarray): Signal
        th (float/int): Threshold for zero crossing
    Returns:
        np.ndarray[bool]: Whether there was a zero crossing at the index
    """
    x = x.copy()
    x[np.abs(x) <= th] = 0
    pos = x > 0
    return np.bitwise_xor(pos[:-1], pos[1:])


def zero_crossing_count(x, th=0):
    """ Number of zero-crossings in the input signal
    Params:
        x (np.ndarray): Signal
        th (float/int): Threshold for zero crossing
    Returns:
        int
    """
    return zero_crossings(x, th=th).sum()


def hjorth_activity(x):
    """ Hjorth activity
    The variance of the input signal
    Params:
        x (np.ndarray): Signal
    Returns
        float: Hjorth activity (variance)
    """
    return np.var(x)


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


def hjorth_parameters(x):
    """ Calculate all Hjorth parameters of a signal.
    doi:10.1016/0013-4694(70)90143-4
    Params:
        x (np.ndarray): Signal in time domain
    Returns
        (float, float, float): Tuple of activity, mobility, complexity.
    """
    deriv1 = np.gradient(x)
    deriv2 = np.gradient(deriv1)
    activity = hjorth_activity(x)
    mobility = hjorth_mobility_derivative(x, deriv1)
    complexity = hjorth_complexity_derivatives(x, deriv1, deriv2)
    return (activity, mobility, complexity)


def dfa(x: np.ndarray, windows: List[int], o: int = 1, overlap: float = 0):
    """ Detrended fluctuation analysis
    Params:
        x (np.ndarray): Signal
        windows (List[int]): List of window sizes
        o (int): Order of the polynomial to fit. Default 1
        overlap (float/int): Percentage overlap between windows. Default 0%
    Returns:
        float: scaling exponent
    """
    def profile(x):
        return np.cumsum(x - np.mean(x))

    def view(x, w, s):
        """ Strided window view of array
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
        min_step = max(int(np.min(windows) * (100 - overlap) / 100), 1)
        out = np.full((len(windows), len(xp) // min_step), np.nan)
        for i, w in enumerate(windows):
            s = max(int(w * (100 - overlap) / 100), 1)
            res = polyfit(np.arange(w), view(xp, w, s).T, o, full=True)
            res = res[1]
            rms = np.sqrt(res / w)
            out[i, :len(res)] = rms
        return np.nanmean(out, axis=1)

    F = fluctuations(profile(x), windows, o)
    scaling_exponent = np.polyfit(np.log(windows), np.log(F), 1)[0]
    return scaling_exponent


@singledispatch
def hurst(x, lags):
    """ Hurst exponent of a signal.
    A test for mean-reversion / trend
    H < 0.5 - mean reversion
    H = 0.5 - random walk
    H > 0.5 - trending
    Params:
        x (np.ndarray[int/float]): Signal to calculate hurst exponent on
        lags (np.ndarray[int]): Time-lags to calculate. Default = 2..64
    Returns:
        float: Hurst exponent
    """
    x = np.array(x)
    lags = np.array(lags)
    return np_hurst(x, lags)


@hurst.register(np.ndarray)
@jit
def np_hurst(x: np.ndarray, lags: np.ndarray = np.arange(2, 64)):
    """ Hurst exponent of a signal.
    A test for mean-reversion / trend
    H < 0.5 - mean reversion
    H = 0.5 - random walk
    H > 0.5 - trending
    Params:
        x (np.ndarray[int/float]): Signal to calculate hurst exponent on
        lags (np.ndarray[int]): Time-lags to calculate. Default = 2..64
    Returns:
        float: Hurst exponent
    """
    tau = np.zeros(lags.shape)
    for i in range(len(tau)):
        tau[i] = np.sqrt(np.std(np.subtract(x[lags[i]:], x[:-lags[i]])))
    cf = o1fit(np.log(lags), np.log(tau))
    return cf[1] * 2.


@jit
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
