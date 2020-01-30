""" Heart rate variability metrics
nni refers to normal R-peak intervals

Shaffer, F., & Ginsberg, J. P. (2017). An Overview of Heart Rate Variability
Metrics and Norms. Frontiers in public health, 5, 258.
https://dx.doi.org/10.3389/fpubh.2017.00258


Malik, M., Camm, A. J., Bigger, J. T., Breithardt, G., Cerutti, S.,
Cohen, R. J., ... Singer, D. H. (1996). Heart rate variability. Standards
of measurement, physiological interpretation, and clinical use.
European Heart Journal, 17(3), 354-381.
"""
from typing import Optional
import numpy as np
from numba import jit
from numba.extending import register_jitable
from ..util.windows import nonuniform_window_apply


_window_std = nonuniform_window_apply(np.std)
_window_mean = nonuniform_window_apply(np.mean)


@register_jitable
def td_factor(unit: str) -> float:
    if unit == 'ns':
        return 1.
    if unit == 'us':
        return 1e3
    if unit == 'ms':
        return 1e6
    if unit == 's':
        return 1e9
    raise ValueError('Unknown unit. Must be: "ns", "us", "ms", or "s"')


@register_jitable
def nni_to_ms(nni: np.ndarray, current_unit: str = 'ns') -> np.ndarray:
    return td_factor(current_unit) * nni.astype(float) / 1e6


@register_jitable
def nni_cumulative(nni: np.ndarray) -> np.ndarray:
    return np.cumsum(nni)


# Time domain
@register_jitable
def sdnn(nni: np.ndarray) -> float:
    """ The standard deviation of normalised R-peak intervals (nn).
    Typically taken over the period of 5 minutes or 24 hours.
    Params:
        nni (np.ndarray): Normal R-peak intervals

    Returns:
        float: σ(nni)

    See also:
        https://doi.org/10.3389/fpubh.2017.00258
    """
    return np.std(nni)


@jit(nopython=True)
def sdann(nni: np.ndarray, index: Optional[np.ndarray] = None,
          interval: float = 60*5, unit: Optional[str] = None):
    """
    Params:
        nni (np.ndarray): Normal R-peak intervals
        index (np.ndarray, optional): Time index
        interval (float): Segment interval length in seconds
        unit (str, optional): Units of nni.
            One of {'ns', 'us', 'ms', 's'}. Required if index not given.
    Returns:
        float: mean of the standard deviation of r-peak intervals
            in segments of length interval.
    """
    if index is None:
        if unit is not None:
            index = nni_cumulative(nni) * td_factor(unit)
        else:
            raise ValueError('index or unit must be specified')
    interval = interval * 1e9
    return _window_mean(index.astype(int), nni, interval, interval).std()


@jit(nopython=True)
def sdnni(nni: np.ndarray, index: Optional[np.ndarray] = None,
          interval: float = 60*5, unit: Optional[str] = None):
    """
    Params:
        nni (np.ndarray): Normal R-peak intervals
        index (np.ndarray, optional): Time index (integer in ns / datetime64)
        interval (float): Segment interval length in seconds
        unit (str, optional): Units of nni.
            One of {'ns', 'us', 'ms', 's'}. Required if index not given.
    Returns:
        float: mean of the standard deviation of r-peak intervals
            in segments of length interval.
    """
    if index is None:
        if unit is not None:
            index = nni_cumulative(nni) * td_factor(unit)
        else:
            raise ValueError('index or interval_unit must be specified')
    interval = interval * 1e9
    return _window_std(index, nni, interval, interval).mean()


@jit(nopython=True)
def pnn50(nni: np.ndarray, unit: str = 'ms') -> float:
    """ Proportion of successive differences over 50ms
    Params:
        nni (np.ndarray): R-peak intervals
        unit (str, optional): Unit the intervals are measured in. Default: 'ms'
    Returns:
        float
    """
    ms50 = 50 * 1e6 / td_factor(unit)
    return np.sum(np.abs(np.diff(nni)) > ms50) / (len(nni) - 1)


@jit(nopython=True)
def pnnx(nni: np.ndarray, unit: str = 'ms', x: float = 50.) -> float:
    """ Proportion of successive differences over X ms
    Params:
        nni (np.ndarray): R-peak intervals
        unit (str, optional): Unit the intervals are measured in. Default: 'ms'
        x (float): The cut-off point in milliseconds. Default: 50ms
    Returns:
        float
    """
    ms50 = x * 1e6 / td_factor(unit)
    return np.sum(np.abs(np.diff(nni)) > ms50) / (len(nni) - 1)


@jit(nopython=True)
def rmssd(nni: np.ndarray) -> float:
    """ Root mean square of differences.
    Params:
        nni (np.ndarray): R-peak intervals
    Returns:
        float
    """
    return np.sqrt(np.mean(np.square(np.diff(nni))))


@jit
def ssd(nni: np.ndarray) -> float:
    """ Sum of successive differences
    Params:
        nni (np.ndarray): R-peak intervals
    Returns:
        float
    """
    return np.sum(np.diff(nni))


@jit
def sdsd(nni: np.ndarray) -> float:
    """ Standard deviation of successive differences.
    Equivilent to rmssd because mean of differences should = 0
    Params:
        nni (np.ndarray): R-peak intervals
    Returns:
        float
    """
    return np.std(np.diff(nni))


# Freq domain
@register_jitable
def power_band(psd, freqs, lower=None, upper=None):
    if lower is None:
        lower = np.min(freqs)
    if upper is None:
        upper = np.max(freqs)
    return np.sum(np.abs(psd[np.logical_and(freqs >= lower, freqs <= upper)]))


@register_jitable
def peak_frequency(psd, freqs, lower=None, upper=None):
    if lower is None:
        lower = np.min(freqs)
    if upper is None:
        upper = np.max(freqs)
    mask = np.logical_and(freqs >= lower, freqs <= upper)
    return freqs[np.argmax(psd[mask])]


@register_jitable
def relative_power_band(psd, freqs, lower=None, upper=None):
    if lower is None:
        lower = np.min(freqs)
    if upper is None:
        upper = np.max(freqs)
    return power_band(psd, freqs, lower, upper) / np.sum(np.abs(psd))


@jit
def frequency_domain(rri: np.ndarray):
    pass


# Non-linear
@jit
def csi_sd1(rri: np.ndarray, factor: float = 1 / np.sqrt(2)) -> float:
    """ Poincare plot width of ellipsis. Equivilent to sdsd and rmss
    multiplied by some factor.
    Params:
        rri (np.ndarray): R-peak intervals
        factor (float): Proportion of points to include in ellipsis width
    Returns:
        float
    """
    return factor * np.std(np.diff(rri))


@jit
def csi_sd2(rri: np.ndarray, factor: float = 1 / np.sqrt(2)) -> float:
    """ Poincare plot length of ellipsis. Equivilent to sdsd and rmss
    multiplied by some factor.
    Equivilent to sqrt(2*(sdnn(rri)**2) - (sdsd(rri)**2)/2)
    Params:
        rri (np.ndarray): R-peak intervals
        factor (float): Proportion of points to include in ellipsis length
    Returns:
        float
    """
    return factor * np.std(rri[1:] + rri[:-1])


@jit
def lorenz_csi(rri: np.ndarray, factor: float = 1 / np.sqrt(2)) -> float:
    """ Cardiac Sympathetic Index
    Params:
        rri (np.ndarray): RR-intervals
    Returns:
        float: cardiac sympathetic index
    DOI: 10.1109/EMBC.2014.6944639
    """
    return csi_sd1(rri, factor) / csi_sd2(rri, factor)


@jit
def lorenz_cvi(rri: np.ndarray, factor: float = 1 / np.sqrt(2)) -> float:
    """
    """
    return np.log10(csi_sd1(rri, factor) * csi_sd2(rri, factor))


@jit
def lorenz_mcsi(rri: np.ndarray, factor: float = 1 / np.sqrt(2)) -> float:
    """
    A modified sympathetic index as used in [1]
    Uses the square of SD1 because it was found to have greater importance
    in seizure detection.
    Params:
        rri (np.ndarray): R-peak intervals
    Returns:
        float

    [1] DOI: 10.1109/EMBC.2014.6944639
    """
    return (csi_sd1(rri, factor) ** 2) / csi_sd2(rri, factor)
