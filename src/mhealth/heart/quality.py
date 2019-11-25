from numba import njit
import numpy as np


@njit
def beat_correlation(x, peaks, sampling_rate, buf_size=24):
    """Calculate correlation between the signal morphology around local peaks
    """
    def corr_multi(x, y):
        corrs = np.zeros(x.shape[0])
        ysymean = y - y.mean()
        ystd = y.std() * x.shape[1]
        for i in range(x.shape[0]):
            denom = x[i, :].std() * ystd
            if denom:
                corrs[i] = (np.sum((x[i, :] - x[i, :].mean()) * ysymean) /
                            denom)
            else:
                corrs[i] = np.nan
        return corrs
    hsr = sampling_rate // 2
    buf = np.zeros((24, sampling_rate), dtype=np.float64)
    buf[:] = np.nan
    a = np.zeros(sampling_rate)
    quality = np.zeros(len(peaks))
    for i in range(1, 24):
        buf[i, :] = x[peaks[i]-hsr:peaks[i]+hsr]
    a[hsr - min(hsr, peaks[0]):] = \
        x[peaks[0] - min(hsr, peaks[0]):peaks[0] + hsr]
    quality[0] = np.nanmean(corr_multi(buf, a))
    for i in range(1, len(peaks)-1):
        arr = x[peaks[i]-hsr:peaks[i]+hsr]
        quality[i] = np.nanmean(corr_multi(buf, arr))
        buf[i % 24, :] = arr
    a[:] = 0
    a[:peaks[-1] + min(hsr, peaks[-1])] = \
        x[peaks[-1]-hsr:peaks[-1] + min(hsr, len(x)-peaks[1])]
    quality[-1] = np.nanmean(corr_multi(buf, a))
    return quality
