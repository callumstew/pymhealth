"""Functional tools for JIT functions."""
from numba import jit
from numba.extending import register_jitable


@jit
def identity(x):
    """Identity function"""
    return x


@register_jitable
def count(start=0, step=1):
    """Iterable counter.
    Params:
        start (int): Start of counter
        step (int): Step size between counts
    Returns:
        iterator of integers increasing by step
    """
    while True:
        yield start
        start += step


@jit
def pairwise(seq):
    """Iterate over pairs in the given sequence.
    Params:
        seq (List): A sequence
    Returns:
        iterator of 2-tuple pairs of the input sequence element
    """
    for i in range(len(seq) - 1):
        yield (seq[i], seq[i+1])


def compose(funcs, inner=identity):
    """Compose JITed functions together left to right.
    Params:
        funcs (List[Callable]): List of functions to compose
    Returns:
        Callable
    """
    @jit
    def wrap(x):
        return head(inner(x))

    head, tail = funcs[-1], funcs[:-1]
    if tail:
        return compose(tail, wrap)
    else:
        return wrap


def rcompose(funcs, inner=identity):
    """Compose JITed functions together right to left.
    Params:
        funcs (List[Callable]): List of functions to compose
    Returns:
        Callable
    """
    @jit
    def wrap(x):
        return head(inner(x))

    head, tail = funcs[0], funcs[1:]
    if tail:
        return rcompose(tail, wrap)
    else:
        return wrap
