from numba import njit
import numpy as np


@njit
def sqi(x, peaks):
    def corr_multi(x, y):
        corrs = np.zeros(x.shape[0])
        ysymean = y - y.mean()
        ystd = y.std() * x.shape[1]
        for i in range(x.shape[0]):
            denom = x[i, :].std() * ystd
            if denom:
                corrs[i] = (np.sum((x[i,:] - x[i,:].mean()) * ysymean) /
                            denom)
            else:
                corrs[i] = np.nan
        return corrs
    buf = np.zeros((24, 128), dtype=np.float64)
    buf[:] = np.nan
    a = np.zeros(128)
    quality = np.zeros(len(peaks))
    for i in range(1, 24):
        buf[i, :] = x[peaks[i]-64:peaks[i]+64]
    a[64 - min(64, peaks[0]):] = x[peaks[0] - min(64, peaks[0]):peaks[0] + 64]
    quality[0] = np.nanmean(corr_multi(buf, a))
    for i in range(1, len(peaks)-1):
        arr = x[peaks[i]-64:peaks[i]+64]
        quality[i] = np.nanmean(corr_multi(buf, arr))
        buf[i % 24, :] = arr
    a[:] = 0
    a[:peaks[-1] + min(64, peaks[-1])] = \
        x[peaks[-1]-64:peaks[-1] + min(64, len(x)-peaks[1])]
    quality[-1] = np.nanmean(corr_multi(buf, a))
    return quality
