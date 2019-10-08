""" Generic time-domain features
"""
from functools import singledispatch
from typing import List, Union, Tuple
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy import polyfit
from numba import jit, guvectorize


@jit
def gradient(x):
    """Calculate the derivative / gradient of the input [jit].

    The halved distance between x_{i-1} and x_{i+1}. The first
    and last gradient is the distance between x_{i} and the only
    neighbouring point.

    Params:
        x (np.ndarray)

    Returns:
        np.ndarray

    """
    out = np.zeros(len(x))
    out[0] = x[1] - x[0]
    out[-1] = x[-1] - x[-2]
    for i in range(1, len(x)-1):
        out[i] = (x[i+1] - x[i-1]) / 2
    return out


@jit
def zero_crossings(x: np.ndarray, th: Union[float, int] = 0) -> np.ndarray:
    """Calculate the indices of zero-crossings in the input signal [jit].

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


@jit
def zero_crossing_count(x: np.ndarray, th: Union[float, int] = 0) -> int:
    """Calculate number of zero-crossings in the input signal [jit].

    Params:
        x (np.ndarray): Signal
        th (float/int): Threshold for zero crossing

    Returns:
        int

    """
    return zero_crossings(x, th=th).sum()


@jit
def line_length(x: np.ndarray) -> float:
    """Sum the absolute differences between points in an array [jit].

    Params:
        x (np.ndarray): Signal

    Returns:
        float

    """
    return np.sum(np.abs(np.diff(x)))


@jit
def hjorth_activity(x: np.ndarray) -> float:
    """Calculate Hjorth activity of input signal [jit].

    The variance of the input signal

    Params:
        x (np.ndarray): Signal

    Returns
        float: Hjorth activity (variance)

    """
    return np.var(x)


@jit
def hjorth_mobility(x: np.ndarray) -> float:
    """Calculate Hjorth mobility of input signal [jit].

    sqrt of the variance of the first derivative of the signal divided by
    the variance of the signal

    Params:
        x (np.ndarray): Signal

    Returns:
        float: Hjorth mobility of signal

    """
    deriv = gradient(x)
    return np.sqrt(np.var(deriv) / np.var(x))


@jit
def hjorth_mobility_derivative(x: np.ndarray, deriv: np.ndarray) -> float:
    """Calculate Hjorth mobility with precomputed first derivitive [jit].

    sqrt of the variance of the first derivative of the signal divided by
    the variance of the signal

    Params:
        x (np.ndarray): Signal
        deriv (np.ndarray): First derivative of the signal

    Returns:
        float: Hjorth mobility of signal

    """
    return np.sqrt(np.var(deriv) / np.var(x))


@jit
def hjorth_complexity(x: np.ndarray) -> float:
    """Calculate Hjorth complexity of a signal. [jit]

    Hjorth mobility of the first derivitive divided by the hjorth mobility
    of the signal

    Params:
        x (np.ndarray): Signal

    Returns:
        float: Hjorth complexity of signal

    """
    deriv1 = gradient(x)
    return hjorth_mobility(deriv1) / hjorth_mobility_derivative(x, deriv1)


@jit
def hjorth_complexity_derivatives(x: np.ndarray, deriv1: np.ndarray,
                                  deriv2: np.ndarray) -> float:
    """Calculate Hjorth complexity with precomputed derivitives. [jit]

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


@jit
def hjorth_parameters(x: np.ndarray) -> Tuple[float, float, float]:
    """Calculate all Hjorth parameters of a signal. [jit]

    May be faster than calculating separately because it reuses computed
    gradients.

    Params:
        x (np.ndarray): Signal in time domain

    Returns:
        (float, float, float): Tuple of activity, mobility, complexity.

    See also:
        doi:10.1016/0013-4694(70)90143-4
    """
    deriv1 = gradient(x)
    deriv2 = gradient(deriv1)
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


@jit
def hurst(x: np.ndarray, lags: np.ndarray = np.arange(2, 64)):
    """Calculate Hurst exponent of a signal [jit].

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
def o1fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Calculate a first-order polynomial fit (line) [jit].

    Params:
        x (np.ndarray)
        y (np.ndarray)

    Returns:
        Tuple[float, float]: Fit parameters (intercept, gradient)

    """
    n = len(x)
    sumx = np.sum(x)
    b = (((n * np.sum(x * y)) - (sumx * np.sum(y))) /
         ((n * np.sum(x * x)) - (sumx * sumx)))
    A = np.mean(y) - (b * np.mean(x))
    return (A, b)


@jit
def o1fit_multiple(x, ys):
    """Fit multiple lines with the same x-axis [jit].

    Params:
        x (np.ndarray[n])
        ys (np.ndarray[n, m]): first dim equals length of x.

    Returns:
        np.ndarray[m, 2]

    """
    out = np.zeros((ys.shape[1], 2))
    for i in range(ys.shape[1]):
        A, b = o1fit(x, ys[:, i])
        out[i, 0] = A
        out[i, 1] = b
    return out
