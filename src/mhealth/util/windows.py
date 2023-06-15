#!/usr/bin/env python3
"""Rolling window operations."""
from typing import Callable, List, Dict, Optional
from functools import lru_cache, singledispatch
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numba import jit, prange, types
from numba.extending import register_jitable, overload


def singledispatchjit(func):
    """@singledispatch and @jit a function."""
    func = singledispatch(jit(func, nopython=True))
    @overload(func)
    def _(x):
        return func.py_func
    return func


def view(x: np.ndarray, w: int, s: int) -> np.ndarray:
    """Strided window view of array.

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


def array_shape(x):
    """Shape of the given array."""
    return x.shape


@overload(array_shape)
def _jit_array_shape(x):
    def _array_shape(x):
        return x.shape

    def _scalar_shape(x):
        return tuple()

    if isinstance(x, types.Array):
        return _array_shape
    return _scalar_shape


@singledispatch
@lru_cache(256)
def rolling_apply(func: Callable, wsize: Optional[int] = None,
                  wstep: int = 1) -> Callable:
    """Create a Callable to apply the given function to windows along an array.

    Params:
        func (Callable): Function to apply to each window
        wsize (int): Window size.
        wstep (int): Step size between start of windows.
    Returns:
        Callable: JITed function which will apply func to windows in an array

    """
    @jit(nopython=True, parallel=True)
    def windows_loop(arr, wsize, wstep, out):
        for i in prange(1, out.shape[0]):
            out[i] = func(arr[i*wstep:i*wstep + wsize])
        return out

    @jit(nopython=True)
    def loop_wrapper(arr, wsize=wsize, wstep=wstep):
        """Apply the function {} to windows in a given array.

        Params:
            arr (np.ndarray): Input array.
            wsize (int): Window size.
            wstep (int): Step size between starts of windows.
        Returns:
            np.ndarray: An array of length (len(arr) - wsize // wstep)

        """
        nw = max(0, 1 + (len(arr) - wsize) // wstep)
        init = func(arr[:wsize])
        shape = array_shape(init)
        out = np.zeros((nw, *shape))
        out[0] = init
        return windows_loop(arr, wsize, wstep, out)

    func = register_jitable(func)
    loop_wrapper.__doc__ = loop_wrapper.__doc__.format(func.__name__)
    return loop_wrapper


@rolling_apply.register(list)
@rolling_apply.register(tuple)
def _rolling_apply_coll(funcs: List[Callable], wsize: Optional[int] = None,
                        wstep: int = 1) -> Callable:
    def multi_funcs_rolling_apply(arr, wsize=wsize, wstep=wstep):
        out = []
        for f in funcs:
            out.append(rolling_apply(f)(arr, wsize, wstep))
        return out
    return multi_funcs_rolling_apply


@rolling_apply.register(dict)
def _rolling_apply_dict(funcs: Dict[str, Callable],
                        wsize: Optional[int] = None,
                        wstep: int = 1) -> Callable:
    def dict_funcs_rolling_apply(arr, wsize=wsize, wstep=wstep):
        vals = rolling_apply(list(funcs))(arr, wsize, wstep)
        return {zip(names, vals)}

    names, funcs = list(zip(*funcs.items()))
    return dict_funcs_rolling_apply


@lru_cache(256)
def indices_rolling_apply(func: Callable,
                          min_window_len: int = 1) -> Callable:
    """Create a Callable to apply func to windows with known indices.

    Params:
        func (Callable): The aggregation function to use
        min_window_len (int): Minimum length of window to apply func to.
    Returns:
        Callable: function with signature (indices, arr)

    """
    @jit
    def windows_loop(indices: np.ndarray,
                     arr: np.ndarray,
                     min_window_len: int = min_window_len) -> np.ndarray:
        """Apply the '{}' function to windows with known indices.

        Params:
            indices (np.ndarray[2, n]): Int array of start and end indices
            arr (np.ndarray): Array to apply windowed aggregation to
            min_window_len (int): Minimum length of window to apply func to.
        Returns:
            np.ndarray: Windowed aggregations

        """
        n = indices.shape[1]
        out = np.zeros(n, arr.dtype)
        for i in range(n):
            si = indices[0, i]
            ei = indices[1, i]
            if ei - si >= min_window_len:
                out[i] = func(arr[si:ei])
            else:
                out[i] = np.nan
        return out
    windows_loop.__doc__ = windows_loop.__doc__.format(str(func))
    return windows_loop


@register_jitable
def get_indices(index: np.ndarray, wsize: np.timedelta64,
                wstep: np.timedelta64) -> np.ndarray:
    """Find the start and end indices of windows of a given step and size.

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


@singledispatch
def nonuniform_rolling_apply(func: Callable,
                             min_window_len: int = 1) -> Callable:
    """Create a moving window aggregation function from a function.

    This function is designed for moving windows with a non-uniform index,
    particularly datetime indices. The returned function will aggregate windows
    of a specified size and stride.

    Params:
        func (Callable): A function which will be applied to each window
        min_window (int): The minimum length of a window to aggregate.
            Windows under this length will return nan.
    Returns:
        Callable: A function with the signature (index, arr, wsize, wstep).

    """
    def moving_window(index: np.ndarray,
                      arr: np.ndarray,
                      wsize: np.timedelta64,
                      wstep: np.timedelta64,
                      min_window_len: int = min_window_len) -> np.ndarray:
        """Aggregate windows with the '{}' function.

        Params:
            index (np.ndarray): Index of the array
            arr (np.ndarray): Array to perform windowed aggregation on
            wsize (np.timedelta64): Length of window
            wstep (np.timedelta64): Length of step
        Returns:
            np.ndarray: Window aggregations

        """
        indices = get_indices(index, wsize, wstep)
        out = f(indices, arr, min_window_len)
        return out

    f = indices_rolling_apply(func, min_window_len)
    moving_window.__doc__ = moving_window.__doc__.format(func.__name__)
    return moving_window


@nonuniform_rolling_apply.register(list)
@nonuniform_rolling_apply.register(tuple)
def _nu_rolling_apply_coll(funcs: List[Callable],
                           min_window_len: int = 1) -> Callable:
    def moving_window(index: np.ndarray, arr: np.ndarray,
                      wsize, wstep) -> List[np.ndarray]:
        indices = get_indices(index, wsize, wstep)
        out = []
        for f in funcs:
            out.append(f(indices, arr, min_window_len))
        return out
    funcs = [indices_rolling_apply(f, min_window_len) for f in funcs]
    return moving_window


@nonuniform_rolling_apply.register(dict)
def _nu_rolling_apply_dict(funcs: Dict[str, Callable],
                           min_window_len: int = 1) -> Callable:
    def moving_window(index: np.ndarray, arr: np.ndarray,
                      wsize, wstep) -> Dict[str, np.ndarray]:
        indices = get_indices(index, wsize, wstep)
        out = dict()
        for k, f in funcs.items():
            out[k] = f(indices, arr, min_window_len)
        return out
    funcs = {k: indices_rolling_apply(f) for k, f in funcs.items()}
    return moving_window
