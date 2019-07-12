#!/usr/bin/env python3
""" Functions for use with acceleration data
"""
from typing import Optional, List, Union, Tuple
from functools import singledispatch
import numpy as np
from numba import jit
from ..util.deps import pd
from ..generic.filters import butterworth


@jit(nopython=True)
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


@jit(nopython=True)
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
def linear_filter(acc: np.ndarray, freq: float,
                  cutoff: Union[float, Tuple[float, float]] = 0.5,
                  order: int = 5) -> np.ndarray:
    """ Filters input with a two-pass butterworth filter, returning
    the linear component of acceleration.
    To also de-noising using a bandpass filter, provide a both the
    low-pass and high-pass cutoff. e.g. (0.5, 10) to bandpass between
    0.5Hz and 10Hz

    Params <np.ndarray[float]>:
        acc (np.ndarray[float]): A vector or array of acceleration values
            If multiple vectors are given in a 2d array, the 2nd dimension
            seperates vectors. i.e acc[m, n] where n is the dimension of
            acceleration.
        freq (float): Sampling frequency
        cutoff (float, Tuple[float, float]): Cut-off frequency. Default: 0.5
        order (int): Order of the filter. Default: 5
    Returns:
        np.ndarray[float]: Linear acceleration (above cut-off frequency)

    Params <pd.DataFrame>:
        df (pd.DataFrame): Dataframe containing acceleration
        freq (float): Sampling frequency
        cutoff (float, Tuple[float, float]): Cut-off frequency. Default: 0.5
        order (int): Order of filter. Default: 5
        columns (List[str]): List of column names. Optional
    Returns:
        pd.DataFrame: DataFrame containing filtered columns
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
def _df_linear_filter(df: pd.DataFrame,
                      freq: float,
                      cutoff: Union[float, Tuple[float, float]] = 0.5,
                      order: int = 5,
                      columns: List[str] = None) -> pd.DataFrame:
    if columns:
        out = df[columns].copy()
    else:
        out = df.select_dtypes(include=[float, int]).copy()
    out[:] = linear_filter(out.values, freq, cutoff, order)
    return out


@singledispatch
def gravity_filter(acc: np.ndarray, freq: float,
                   cutoff: float = 0.5, order: int = 5) -> np.ndarray:
    """ Filters acceleration with a two-pass Butterworth filter,
    returning the gravitational component

    Params <np.ndarray[float]>:
        acc (np.ndarray[float]): A vector or array of acceleration values
            If multiple vectors are given in a 2d array, the 2nd dimension
            seperates vectors. i.e acc[m, n] where n is the dimension of
            acceleration.
        freq (float): Sampling frequency
        cutoff (float): Cut-off frequency (Hz). Default: 0.5
        order (int): Order of the filter. Default: 5
    Returns:
        np.ndarray[float]: Gravitational component of acceleration

    Params <pd.DataFrame>:
        df (pd.DataFrame): Dataframe containing acceleration
        freq (float): Sampling frequency
        cutoff (float): Low-pass cut-off frequency. Default: 0.5
        order (int): Order of filter. Default: 5
        columns (List[str]): List of column names. Optional
    Returns:
        pd.DataFrame: DataFrame containing filtered columns
    """
    shape = acc.shape
    acc = acc.reshape(shape[0], 1 if len(shape) == 1 else shape[1])
    res = np.zeros(acc.shape)
    for i in range(res.shape[1]):
        res[:, i] = butterworth(acc[:, i], cutoff=cutoff, freq=freq,
                                order=order, ftype='lowpass')
    return res.reshape(shape)


@gravity_filter.register(pd.DataFrame)
def _df_gravity_filter(df: pd.DataFrame, freq: float,
                       cutoff: float = 0.5, order: int = 5,
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
    if columns:
        out = df[columns].copy()
    else:
        out = df.select_dtypes(include=[float, int]).copy()
    out[:] = gravity_filter(out.values, freq, cutoff, order)
    return out


@singledispatch
@jit(nopython=True)
def magnitude(x: float, y: float, z: float) -> float:
    """ Magnitude of x, y, z acceleration √(x²+y²+z²)

    Params <float>:
        x, y, z (float): Axes of acceleration
    Returns:
        float: Magnitude of acceleration

    Params <pd.Dataframe>:
        df (pd.DataFrame): Dataframe containing acceleration columns
        xcol, ycol, zcol (str): Column names. Default: 'x', 'y', 'z'
    Returns:
        float: Magnitude of acceleration
    """
    return np.sqrt(x**2 + y**2 + z**2)


@magnitude.register(pd.DataFrame)
def _pd_magnitude(df, xcol: str = 'x',
                  ycol: str = 'y', zcol: str = 'z') -> pd.DataFrame:
    out = magnitude(df[xcol], df[ycol], df[zcol])
    out.name = 'magnitude'
    return out


@singledispatch
@jit(nopython=True)
def magnitude_dot(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """ Magnitude of x, y, z acceleration arrays √(x²+y²+z²) using dot product for squares

    Params <float>:
        x, y, z (np.ndarray): Axes of acceleration
    Returns:
        float: Magnitude of acceleration

    Params <pd.Dataframe>:
        df (pd.DataFrame): Dataframe containing acceleration columns
        xcol, ycol, zcol (str): Column names. Default: 'x', 'y', 'z'
    Returns:
        float: Magnitude of acceleration
    """
    return np.sqrt(np.dot(x,x) + np.dot(y,y) + np.dot(z,z))

@magnitude_dot.register(pd.DataFrame)
def _pd_magnitude_dot(df, xcol: str = 'x',
                  ycol: str = 'y', zcol: str = 'z') -> float:
    return magnitude_dot(df[xcol], df[ycol], df[zcol])
