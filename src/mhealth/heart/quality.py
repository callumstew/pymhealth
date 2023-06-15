"""A module for measuring the quality of a cardiac signal."""
import numpy as np
from numba import jit
from numba.extending import register_jitable


@register_jitable
def _corr_multi(x, y):
    corrs = np.zeros(x.shape[0])
    ysymean = y - y.mean()
    ystd = y.std() * x.shape[1]
    for i in range(x.shape[0]):
        denom = x[i, :].std() * ystd
        if denom:
            corrs[i] = (np.sum((x[i, :] - x[i, :].mean()) * ysymean) /
                        denom)
        else:
            corrs[i] = 0
    return corrs


@jit(nopython=True)
def beat_correlation(x, peaks, sampling_rate, buf_size=24):
    """Calculate correlation of the signal around nearby peaks."""
    hsr = sampling_rate // 2
    buf = np.zeros((buf_size, sampling_rate), dtype=np.float64)
    buf[:] = np.nan
    a = np.zeros(sampling_rate)
    quality = np.zeros(len(peaks))
    for i in range(1, 24):
        buf[i, :] = x[peaks[i]-hsr:peaks[i]+hsr]
    a[hsr - min(hsr, peaks[0]):] = \
        x[peaks[0] - min(hsr, peaks[0]):peaks[0] + hsr]
    quality[0] = np.nanmean(_corr_multi(buf, a))
    for i in range(1, len(peaks)-1):
        arr = x[peaks[i]-hsr:peaks[i]+hsr]
        quality[i] = np.nanmean(_corr_multi(buf, arr))
        buf[i % 24, :] = arr
    a[:] = 0
    a[:hsr + min(peaks[-1] + hsr, len(x)) - peaks[-1]] = \
        x[peaks[-1]-hsr:peaks[-1] + min(hsr, len(x)-peaks[1])]
    quality[-1] = np.nanmean(_corr_multi(buf, a))
    return quality


@jit(nopython=True)
def beat_correlation_bi(x, peaks, sampling_rate, buf_size=12):
    """Calculate correlation of the signal around nearby peaks.

    Takes the maximum correlation between the current peak and the previous
    n peaks or the following n peaks, where n = buf_size. This gives a clearer
    seperation between good quality and low quality peaks at the edge of
    noise.
    """
    def max_buf_corr(arr, prev_buf, succ_buf):
        prev = np.nanmean(_corr_multi(prev_buf, arr))
        succ = np.nanmean(_corr_multi(succ_buf, arr))
        return max(prev, succ)

    hsr = sampling_rate // 2
    prev_buf = np.zeros((buf_size, sampling_rate), dtype=np.float64)
    prev_buf[:] = 0
    succ_buf = prev_buf.copy()
    a = np.zeros(sampling_rate)
    quality = np.zeros(len(peaks))
    n = len(x)
    i = 0

    for i in range(1, buf_size):
        succ_buf[i, :] = x[peaks[i]-hsr:peaks[i]+hsr]

    while i < len(peaks):
        p = peaks[i]
        if p > hsr:
            break
        a[hsr - min(hsr, p):] = x[p - hsr:p + hsr]
        quality[p] = max_buf_corr(a, prev_buf, succ_buf)

    while i < len(peaks):
        p = peaks[i]
        if p + hsr > n:
            break
        arr = x[p-hsr:p+hsr]
        quality[i] = max_buf_corr(arr, prev_buf, succ_buf)
        prev_buf[i % buf_size, :] = arr
        if i + buf_size < len(peaks):
            if peaks[i + buf_size] < n - sampling_rate:
                succ_buf[i % buf_size, :] =\
                    x[peaks[i + buf_size] - hsr:peaks[i + buf_size] + hsr]
        else:
            succ_buf[i % buf_size, :] = 0
        i += 1

    while i < len(peaks):
        p = peaks[i]
        a[:] = 0
        a[:hsr + min(p + hsr, n) - p] = x[p-hsr:min(p + hsr, n)]
        quality[i] = np.nanmean(_corr_multi(prev_buf, arr))
        i += 1

    return quality
