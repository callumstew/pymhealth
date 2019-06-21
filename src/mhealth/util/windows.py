#!/usr/bin/env python3
""" Moving window operations
"""
from typing import Callable, List, Dict
from functools import lru_cache, singledispatch, wraps, partial
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numba import jit
from .deps import pd


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


@lru_cache(64)
def rolling_apply(func: Callable, **kwargs):
    """ Create a function to loop through an evenly-sampled array
    and apply the supplied function. Additional kwargs are optional
    and will be passed as the default value to the resulting function
    Params:
        func (Callable): Function to apply to each window
        wsize (int): Size of window
        wstep (int): Size of step between windows
        shape (Tuple[int]): Shape of output of func. Default: (1,)
        dtype (type): dtype of the output
    Returns:
        Callable: Function that will apply func to windows in a given array
    """

    @jit
    def windows_loop(arr, wsize, wstep, shape=tuple(), dtype=np.float64):
        nw = 1 + (len(arr) - wsize) // wstep
        out = np.zeros((nw, *shape), dtype=dtype)
        for i in range(nw):
            out[i] = func(arr[i*wstep:i*wstep + wsize])
        return out
    return partial(windows_loop, **kwargs)


@singledispatch
def rolling_window(func: Callable, *args, **kwargs) -> Callable:
    """ Create a function to loop through an evenly-sampled array
    and apply the supplied function. Additional args/kwargs are optional
    and will be passed as the default value to the resulting function
    Params:
        func (Callable): Function to apply to each window
        wsize (int): Size of window
        wstep (int): Size of step between windows
        shape (Tuple[int]): Shape of output of func. Default: (1,)
        dtype (type): dtype of the output
    Returns:
        Callable: Function that will apply func to windows in a given array
    """
    @singledispatch
    def rolling_dispatch(arr: np.ndarray, wsize: int, wstep: int,
                         shape: tuple = tuple(),
                         dtype: type = np.float64) -> np.ndarray:
        return rolling_apply(func)(arr, wsize, wstep, shape, dtype)

    @rolling_dispatch.register(pd.DataFrame)
    def _(df: pd.DataFrame, wsize: int, wstep: int,
          shape: tuple = tuple(), dtype: type = np.float64) -> pd.DataFrame:
        data = {}
        for c in df.columns:
            data[c] = rolling_apply(func)(df[c].values, wsize,
                                          wstep, shape, dtype)
        df = pd.DataFrame(data=data)
        tdstep = wstep * (df.index[1] - df.index[0])
        start = df.index[0]
        stop = start + tdstep * len(df)
        win_index = np.arange(start, stop, tdstep)
        win_index = win_index.astype(df.index.values.dtype)
        df = df.set_index(win_index)
        return df

    @wraps(rolling_dispatch)
    def wrapper(x, *a, **kw):
        return rolling_dispatch(x, *args, *a, **kwargs, **kw)

    return wrapper


@rolling_window.register(list)
@rolling_window.register(tuple)
def _(funcs: List[Callable], *args, **kwargs):

    @singledispatch
    def window_applyf(arr, wsize, wstep, shapes=None, dtypes=None):
        N = len(funcs)
        if shapes is None:
            shapes = [tuple() for i in range(N)]
        if dtypes is None:
            dtypes = [np.float64 for i in range(N)]
        out = []
        for f, s, d in zip(funcs, shapes, dtypes):
            out.append(rolling_window(f)(arr, wsize, wstep,
                                         shape=s, dtype=d))
        return out

    @window_applyf.register(pd.DataFrame)
    def _(df, wsize, wstep, shapes=None, dtypes=None):
        if shapes is None:
            shapes = [tuple() for i in range(len(funcs))]
        if dtypes is None:
            dtypes = [np.float64 for i in range(len(funcs))]
        cols = [c + '_' + str(i) for i in range(len(funcs))
                for c in df.columns]
        out = pd.concat([rolling_window(f)(df, wsize, wstep, s, dt)
                         for f, s, dt in zip(funcs, shapes, dtypes)], axis=1)
        out.columns = cols
        return out

    @wraps(window_applyf)
    def wrapper(x, *a, **kw):
        return window_applyf(x, *args, *a, **kwargs, **kw)

    return wrapper


@rolling_window.register(dict)
def _(funcs: Dict[str, Callable], *args, **kwargs):

    @singledispatch
    def window_applyf(arr, *args, **kwargs):
        vals = rolling_window(list(funcs))(arr, *args, **kwargs)
        return {n: y for n, y in zip(names, vals)}

    @window_applyf.register(pd.DataFrame)
    def _(df, *args, **kwargs):
        cols = [c + '_' + n for n in names for c in df.columns]
        out = rolling_window(funcs)(df, *args, **kwargs)
        out.columns = cols
        return out

    @wraps(window_applyf)
    def wrapper(x, *a, **kw):
        return window_applyf(x, *args, *a, **kwargs, **kw)

    names, funcs = list(zip(*funcs.items()))
    return wrapper


@lru_cache(64)
def nonuniform_window_aggregator(func: Callable):
    """ Create a function to loop through known window indices and
    apply the supplied function.
    Params:
        func (Callable): The aggregation function to use
    Returns:
        Callable: function with signature (indices, arr)
    """
    @jit
    def windows_loop(indices: np.ndarray, arr: np.ndarray, min_window_len=1):
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
    windows_loop.__doc__ = windows_loop.__doc__.format(str(func))
    return windows_loop


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


@singledispatch
def nonuniform_rolling_window(func: Callable, min_window: int = 1):
    """ Create a moving window aggregation function from a function
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
    @singledispatch
    def moving_window(index: np.ndarray,
                      arr: np.ndarray,
                      wsize: np.timedelta64,
                      wstep: np.timedelta64,
                      min_window: int = min_window):
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
        out = f(indices, arr, min_window)
        return out

    @moving_window.register(pd.DataFrame)
    def _(df, wsize, wstep, min_window=min_window):
        indices = get_indices(df.index.values, wsize, wstep)
        aggs = {c + '_' + func.__name__: f(indices, df[c].values, min_window)
                for c in df.columns}
        win_index = np.arange(df.index[0].value, df.index[-1].value, wstep)
        win_index = win_index.astype(df.index.values.dtype)
        return pd.DataFrame(data=aggs, index=win_index)

    f = nonuniform_window_aggregator(func)
    moving_window.__doc__ = moving_window.__doc__.format(func.__name__)
    return moving_window


@nonuniform_rolling_window.register(list)
def _(funcs: List[Callable], min_window_len: int = 1, method='uniform'):
    funcs = [nonuniform_window_aggregator(f) for f in funcs]

    @singledispatch
    def moving_window(index, arr, wsize, wstep):
        indices = get_indices(index, wsize, wstep)
        out = [] * len(funcs)
        for i, f in enumerate(funcs):
            out[i] = f(indices, arr, min_window_len)
        return out

    @moving_window.register(pd.DataFrame)
    def _(df, wsize, wstep):
        indices = get_indices(df.index.values, wsize, wstep)
        aggs = {c + '_' + f.__name__:
                    f(indices, df[c].values, min_window_len)
                for c in df.columns for f in funcs}
        win_index = np.arange(df.index[0].value, df.index[-1].value, wstep)
        win_index = win_index.astype(df.index.values.dtype)
        return pd.DataFrame(index=win_index, data=aggs)

    return moving_window


@nonuniform_rolling_window.register(dict)
def _(funcs: Dict[str, Callable], min_window_len: int = 1, method='uniform'):
    funcs = {k: nonuniform_window_aggregator(f) for k, f in funcs.items()}

    @singledispatch
    def moving_window(index, arr, wsize, wstep):
        indices = get_indices(index, wsize, wstep)
        out = dict()
        for k, f in funcs.items():
            out[k] = f(indices, arr, min_window_len)
        return out

    @moving_window.register(pd.DataFrame)
    def _(df, wsize, wstep):
        indices = get_indices(df.index.values, wsize, wstep)
        aggs = {c + '_' + name:
                    f(indices, df[c].values, min_window_len)
                for c in df.columns for name, f in funcs.items()}
        win_index = np.arange(df.index[0].value, df.index[-1].value, wstep)
        win_index = win_index.astype(df.index.values.dtype)
        return pd.DataFrame(index=win_index, data=aggs)

    return moving_window
