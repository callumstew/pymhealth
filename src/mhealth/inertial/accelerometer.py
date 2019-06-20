#!/usr/bin/env python3
""" Functions for use with acceleration data
"""
from typing import Optional, List
from functools import singledispatch
import numpy as np
from ..util.deps import pd
from ..generic.filters import butterworth


@singledispatch
def roll(y, z):
    """ Estimate angular roll from gravitational acceleration
    Params:
        y, z (float, int, array-like): y, and z acceleration
    Returns:
        (float, int, array-like): roll
    """
    return np.arctan2(y, z) * 180/np.pi


@roll.register(pd.DataFrame)
def _df_roll(df: pd.DataFrame, ycol: str = 'y', zcol: str = 'z'):
    """ Find angular roll for each row of a dataframe containing
    accelerometer data.
    Params:
        df (pd.DataFrame): accelerometer dataframe
        xcol, ycol, zcol (str): column names for x, y, and z acceleration
    Returns:
        pd.Series: roll
    """
    out = roll(df[ycol], df[zcol])
    out.name = 'roll'
    return out


@singledispatch
def pitch(x, y, z):
    """ Estimate angular pitch from gravitational acceleration
    Params:
        x, y, z (float, int, array-like): x, y, and z acceleration
    Returns:
        (float, int, array-like): pitch
    """
    return np.arctan2(-x, np.sqrt(y*y + z*z)) * 180/np.pi


@pitch.register(pd.DataFrame)
def _df_pitch(df: pd.DataFrame, xcol: str = 'x',
              ycol: str = 'y', zcol: str = 'z'):
    """ Find angular pitch for each row of a dataframe containing
    accelerometer data.
    Params:
        df (pd.DataFrame): accelerometer dataframe
        xcol, ycol, zcol (str): column names for x, y, and z acceleration
    Returns:
        pd.Series: pitch
    """
    out = pitch(df[xcol], df[ycol], df[zcol])
    out.name = 'pitch'
    return out


@singledispatch
def linear_filter(acc, freq, cutoff=0.5, order=5):
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


@linear_filter.register(pd.DataFrame)
def _df_linear_filter(df, freq, cutoff=0.5, order=5, columns=None):
    if columns:
        out = df[columns].copy()
    else:
        out = df._get_numeric_data().copy()
    for col in out.columns:
        out[col] = linear_filter(df[col].values, freq, cutoff, order)
    return out


@singledispatch
def gravity_filter(acc: np.ndarray, freq: float,
                   cutoff: float = 0.5, order: int = 5):
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


@gravity_filter.register(pd.DataFrame)
def _df_gravity_filter(df: pd.DataFrame, freq: float, cutoff: float = 0.5,
                       order:int = 5, columns: Optional[List[str]] = None):
    if columns:
        out = df[columns].copy()
    else:
        out = df._get_numeric_data().copy()
    for col in out.columns:
        out[col] = gravity_filter(df[col].values, freq, cutoff, order)
    return out


@singledispatch
def magnitude(x: float, y: float, z: float):
    return np.sqrt(x**2 + y**2 + z**2)


@magnitude.register(pd.DataFrame)
def _pd_magnitude(df, xcol: str = 'x', ycol: str = 'y', zcol: str = 'z'):
    out = magnitude(df['x'], df['y'], df['z'])
    out.name = 'magnitude'
    return out
