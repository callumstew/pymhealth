""" Generic information theory features and functions
TODO:
    * Sample entropy with bucket method
"""
from typing import Optional
import numpy as np
from numba import jit


@jit
def entropy(x: np.ndarray) -> float:
    """ Shannon entropy
    Args:
        x (np.ndarray[float]): Counts or probabilities of discrete distribution
    Returns:
        float: Shannon entropy of distribution
    """
    x = x / np.sum(x)
    x += 1e-30
    return - np.sum(x * np.log(x))


@jit(nopython=True)
def sampen(x: np.ndarray, mm: int = 2,
           r: float = 0.2, sd: Optional[float] = None) -> float:
    """Calculate the sample entropy of a 1D array.

    Sample entropy is a measure of a signal's complexity. It is the
    negative log of the ratio between the number of subsignals of length mm
    with a (Chebyshev) distance under r to the number of subsignals of
    length mm+1 with a (Chebyshev) distance under r.

    Params:
        x (np.ndarray): Input signal
        mm (int): Length of template
        r (float): Proportion of the standard deviation to indicate a match
        sd: (float/None): The standard deviation if not using the SD
            of the signal
    Returns:
        float: sample entropy

    Taken from https://github.com/raphaelvallat/entropy until a different
    version is implemented.

    Original License:

    BSD 3-Clause License

    Copyright (c) 2018, Raphael Vallat
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    n = len(x)
    n1 = n - 1
    mm += 1
    mm_dbld = 2 * mm
    r *= sd if sd is not None else x.std()
    run = np.zeros(n)
    run1 = np.zeros(n)
    r1 = np.zeros(n * mm_dbld)
    a = np.zeros(mm)
    b = np.zeros(mm)
    p = np.zeros(mm)
    for i in range(n1):
        nj = n1 - i
        for jj in range(nj):
            j = jj + i + 1
            if abs(x[j] - x[i]) < r:
                run[jj] = run1[jj] + 1
                m1 = mm if mm < run[jj] else run[jj]
                for m in range(m1):
                    a[m] += 1
                    if j < n1:
                        b[m] += 1
            else:
                run[jj] = 0
        for j in range(mm_dbld):
            run1[j] = run[j]
            r1[i + n * j] = run[j]
        if nj > mm_dbld - 1:
            for j in range(mm_dbld, nj):
                run1[j] = run[j]
    m = mm - 1
    while m > 0:
        b[m] = b[m - 1]
        m -= 1
    b[0] = n * n1 / 2
    p = np.true_divide(a, b)
    return -np.log(p[-1])
