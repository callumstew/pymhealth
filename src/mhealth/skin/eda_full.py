#!/usr/bin/env python3
""" Full electrodermal activity feature extraction
"""
import numpy as np
from itertools import count

def eda_tonic_feat(in_vec: np.ndarray, fs: float) -> np.ndarray:
    assert in_vec.shape[1] == 1, "input vector must have one column"

    out_vec = np.zeros(100) # may need to increase alloc if elements added
    ix = count()

    return out_vec[:next(ix)]


def eda_tonic_baseline_feat(in_vec: np.ndarray, fs: float) -> np.ndarray:
    assert in_vec.shape[1] == 1, "input vector must have one column"

    out_vec = np.zeros(100) # may need to increase alloc if elements added
    ix = count()

    return out_vec[:next(ix)]
