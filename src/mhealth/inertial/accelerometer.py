#!/usr/bin/env python3
""" Functions for use with acceleration data
"""
from functools import singledispatch
import numpy as np
import pandas as pd


@singledispatch
def accelerometer_roll(y, z):
    """ Estimate angular roll from gravitational acceleration
    Params:
        y, z (float, int, array-like): y, and z acceleration
    Returns:
        (float, int, array-like): roll
    """
    return np.arctan2(y, z) * 180/np.pi


@accelerometer_roll.register(pd.DataFrame)
def _df_accelerometer_roll(df: pd.DataFrame, ycol: str = 'y', zcol: str = 'z'):
    """ Find angular roll for each row of a dataframe containing
    accelerometer data.
    Params:
        df (pd.DataFrame): accelerometer dataframe
        xcol, ycol, zcol (str): column names for x, y, and z acceleration
    Returns:
        pd.Series: roll
    """
    out = accelerometer_roll(df[ycol], df[zcol])
    out.name = 'angular_roll'
    return out


@singledispatch
def accelerometer_pitch(x, y, z):
    """ Estimate angular pitch from gravitational acceleration
    Params:
        x, y, z (float, int, array-like): x, y, and z acceleration
    Returns:
        (float, int, array-like): pitch
    """
    return np.arctan2(-x, np.sqrt(y*y + z*z)) * 180/np.pi


@accelerometer_pitch.register(pd.DataFrame)
def _df_accelerometer_pitch(df: pd.DataFrame, xcol: str = 'x',
                            ycol: str = 'y', zcol: str = 'z'):
    """ Find angular pitch for each row of a dataframe containing
    accelerometer data.
    Params:
        df (pd.DataFrame): accelerometer dataframe
        xcol, ycol, zcol (str): column names for x, y, and z acceleration
    Returns:
        pd.Series: pitch
    """
    out = accelerometer_pitch(df[xcol], df[ycol], df[zcol])
    out.name = 'angular_pitch'
    return out
