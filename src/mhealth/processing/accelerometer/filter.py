""" Accelerometer-specific preprocessing filters
"""
import numpy as np
from ..generic.filter import butterworth


def acc_linear(acc, freq, cutoff=0.5, order=5):
    """ Filters input with a two-pass butterworth filter, returning
    the linear component of acceleration.
    To also de-noising using a bandpass filter, provide a both the
    low-pass and high-pass cutoff. e.g. (0.5, 10) to bandpass between
    0.5Hz and 10Hz

    Parameters:
        acc (array floats): A vector or array of acceleration values
            If multiple vectors are given in a 2d array, the 2nd dimension
            seperates vectors. i.e acc[m, n] where n is the dimension of
            acceleration in space
        freq (float): Sampling frequency
        cutoff (float or (float, float)): Cut-off frequency (Hz). Default: 0.5
        order (int): Order of the filter. Default: 5

    Returns:
        np.ndarray of floats: Linear acceleration
    """
    shape = acc.shape
    ftype = 'highpass' if np.shape(cutoff) == () else 'bandpass'
    acc = acc.reshape(shape[0], 1 if len(shape) == 1 else shape[1])
    res = np.zeros(acc.shape)
    for i in range(res.shape[1]):
        res[:, i] = butterworth(acc[:, i], cutoff=cutoff, freq=freq,
                                order=order, ftype=ftype)
    return res.reshape(shape)


def acc_gravity(acc, freq, cutoff=0.5, order=5):
    """ Filters acceleration with a two-pass Butterworth filter, returning
    the gravitational component
    Parameters:
        acc (array floats): A vector or array of acceleration values
            If multiple vectors are given in a 2d array, the 2nd dimension
            seperates vectors. i.e acc[m, n] where n is the dimension of
            acceleration in space
        freq (float): Sampling frequency
        cutoff (float): Cut-off frequency (Hz). Default: 0.5
        order (int): Order of the filter. Default: 5

    Returns:
        np.ndarray of floats: Gravitational component of acceleration
    """
    shape = acc.shape
    acc = acc.reshape(shape[0], 1 if len(shape) == 1 else shape[1])
    res = np.zeros(acc.shape)
    for i in range(res.shape[1]):
        res[:, i] = butterworth(acc[:, i], cutoff=cutoff, freq=freq,
                                order=order, ftype='lowpass')
    return res.reshape(shape)
