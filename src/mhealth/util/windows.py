#!/usr/bin/env python3
""" Moving window operations
"""
import numpy as np
from numba import jit
from functools import lru_cache
from typing import Callable


def moving_window_func(func: Callable, min_window_len: int = 1):
    """ Create a rolling window aggregation function from a function
    This function is designed for rolling windows with a non-uniform index,
    particularly datetime indices. The returned function will aggregate windows
    of a specified size and stride.

    Params:
        func (Callable): A function which will be applied to each window
    Returns:
        Callable: A function with the signature (index, arr, wsize, wstep).
    """
    def moving_window(index: np.ndarray, arr: np.ndarray,
                      wsize: np.timedelta64, wstep: np.timedelta64,
                      min_window_len: int = min_window_len):
        """ Aggregate windows with the '{}' function
        Params:
            index (np.ndarray): Index of the array
            arr (np.ndarray): Array to perform windowed aggregation on
            wsize (np.timedelta64): Length of window
            wstep (np.timedelta64): Length of step
        Returns:
            np.ndarray: Window aggregations
        """
        indices = get_indices(index, wsize, wstep)
        out = windows_apply(indices, arr, min_window_len)
        return out
    windows_apply = windows_apply_func(func)
    moving_window.__doc__ = moving_window.__doc__.format(func.__name__)
    return moving_window


@lru_cache(64)
def windows_apply_func(func: Callable):
    """ Create a function to loop through known window indices and
    apply the supplied function.
    Params:
        func (Callable): The aggregation function to use
    Returns:
        Callable: function with signature (indices, arr)
    """
    @jit
    def windows_apply(indices: np.ndarray, arr: np.ndarray, min_window_len=1):
        """ Apply the '{}' function to windows with known indices
        Params:
            indices (np.ndarray[2, n]): Int array of start and end indices
            arr (np.ndarray): Array to apply windowed aggregation to
        Returns:
            np.ndarray: Windowed aggregations
        """
        n = indices.shape[1]
        out = np.zeros(n, arr.dtype)
        for i in range(n):
            si = indices[0, i]
            ei = indices[1, i]
            if ei - si >= min_window_len:
                out[i] = func(arr[indices[0, i]:indices[1, i]])
            else:
                out[i] = np.nan
        return out
    windows_apply.__doc__ = windows_apply.__doc__.format(func.__name__)
    return windows_apply


def get_indices(index: np.ndarray, wsize: np.timedelta64,
                wstep: np.timedelta64):
    """ Find the start and end indices of windows of a given step and size
    Params:
        index (np.ndarray[n]): Index of the array
        wsize (np.timedelta64): Length of window
        wstep (np.timedelta64): Length of step
    Returns:
        np.ndarray[2, n]: Start ([0, :]) and end ([1, :]) indices
    """
    starts = np.arange(index[0], index[-1], wstep)
    ends = starts + wsize
    starts_and_ends = np.concatenate((starts, ends))
    return np.searchsorted(index, starts_and_ends).reshape((2, len(starts)))
