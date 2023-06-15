""" ECG QRS-complex and R-peak detection algorithms
"""
from typing import Callable, Optional
from functools import singledispatch
import numpy as np
from numba import jit
from ..generic.filters import butterworth
from ..util.windows import view
from ..util.deps import pd


def pt_differentiate(x: np.ndarray) -> np.ndarray:
    """ Five-point derivative with transfer function
    H(z) = 1/8T ( -z^-2 - -2z^-1 + 2z + z^2 )
    """
    return np.convolve(x, [1, 2, 0, -2, -1][:-4]) / 8


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """ Moving average filter to integrate signal
    """
    return np.convolve(x, np.ones(window)/window, mode='same')


def bandpass(ecg: np.ndarray, fs: float, low: int = 5,
             high: int = 15, order: int = 5) -> np.ndarray:
    """ Bandpass filter for ECG
    Params:
        ecg (np.ndarray[float]): ECG signal array
        fs (float): Sampling frequency
        low (float): Low-frequency cutoff. Default 5
        high (float): High-frequency cutoff. Default 15
        order (int): Order of filter. Default 5
    Returns:
        np.ndarray[float]: Filtered ECG signal
    """
    return butterworth(ecg, freq=fs, cutoff=(low, high),
                       ftype='bandpass', order=order)


def filter_pan_tompkins(ecg: np.ndarray, fs: float) -> np.ndarray:
    """
    Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection
    algorithm. IEEE Trans. Biomed. Eng, 32(3), 230-236.
    DOI: 10.1109/TBME.1985.325532
    Params:
        ecg (np.array): Input ECG signal
        fs (float): Frequency of ECG signal.
    Returns:
        np.array: Filtered ECG
    """
    ecg = bandpass(ecg, fs)
    ecg = pt_differentiate(ecg)
    ecg = ecg ** 2
    window = int(0.2 * fs)
    return moving_average(ecg, window)


@singledispatch
def rpeaks_hamilton_tompkins(ecg: np.ndarray, fs: float) -> np.ndarray:
    """
    Uses Hamilton-Tompkins algorithm to detect R-peaks in an ECG signal.


    Params <np.ndarray>:
        ecg (np.ndarray[float]): ECG signal array
        fs (float): ECG sampling frequency
    Returns:
        np.ndarray: index of peaks


    Params <pd.DataFrame>:
        ecg (pd.DataFrame): Dataframe with ECG signal as one of the columns
        fs (float) (optional): Sampling frequency. If not given, uses index
        column (str) (optional): Column name of ECG signal. If not given, uses
            first column.
    Returns:
        pd.DataFrame: index of peaks


    See also:
        doi: 10.1109/TBME.1986.325695

        Hamilton, Patrick S. "Open source ECG analysis software documentation."
        Computers in cardiology 2002 (2002): 101-104.
    """
    ecg = np.array(ecg)
    return _np_hamilton_tompkins(ecg, fs)


@rpeaks_hamilton_tompkins.register(np.ndarray)
def _np_hamilton_tompkins(ecg: np.ndarray, fs: float) -> np.ndarray:
    fecg = filter_hamilton_tompkins(ecg, fs)
    peaks = find_peaks(fecg)
    return decision_rule_hamilton_tompkins(fecg, peaks, fs)


@rpeaks_hamilton_tompkins.register(pd.DataFrame)
def _df_hamilton_tompkins(ecg: pd.DataFrame, fs: Optional[float] = None,
                          column: Optional[str] = None) -> pd.DataFrame:
    column = column if column else ecg.columns[0]
    fs = fs if fs else (1e9 / (ecg.index[1]-ecg.index[0]).value)
    vals = _np_hamilton_tompkins(ecg[column].values, fs)
    return pd.DataFrame(vals, index=ecg.index[vals])


def filter_hamilton_tompkins(ecg: np.ndarray, fs: float) -> np.ndarray:
    """http://www.eplimited.com/osea13.pdf
    """
    ecg = bandpass(ecg, fs, 3, 25)
    ecg = np.abs(pt_differentiate(ecg))
    window = int(0.08 * fs)
    return moving_average(ecg, window)


@jit(nopython=True)
def decision_rule_hamilton_tompkins(fecg: np.ndarray, peaks: np.ndarray,
                                    fs: float, buf: int = 12,
                                    th: float = 0.3125) -> np.ndarray:
    """ Filters the given peaks according to Hamilton and Tompkins method


    Hamilton, Patrick S., and Willis J. Tompkins. "Quantitative investigation
    of QRS detection rules using the MIT/BIH arrhythmia database."
    IEEE transactions on biomedical engineering 12 (1986): 1157-1165.
    doi: 10.1109/TBME.1986.325695

    Hamilton, Patrick S. "Open source ECG analysis software documentation."
    Computers in cardiology 2002 (2002): 101-104.

    """
    peak_is_qrs = np.zeros(len(peaks), dtype=np.bool_)
    buf_qrs = np.zeros(buf, np.int64)
    buf_noise = np.zeros(buf, np.int64)
    i_buf_qrs = 0
    i_buf_noise = 0
    dth = 0
    prev_p = 0
    Nqrs = 0

    def local_maxima(p, lim):
        """ Peak is largest within 200ms
        """
        return fecg[p] >= np.max(fecg[max(0, p - lim):p + lim])

    def both_gradients(p):
        """ If both gradients are not present around peak, assumed to be
        baseline drift
        """
        x = fecg[max(0, p - int(fs * 0.05)):p + int(fs * 0.05)]
        pos = 0
        neg = 0
        for i in range(1, len(x)):
            if x[i] < x[i-1]:
                neg = 1
            elif x[i] > x[i-1]:
                pos = 1
            if pos and neg:
                return True
        return False

    def is_twave(p1, p2):
        """ Is the current peak (p2) part of a t-wave. compare against
        previous peak (p1)
        """
        lim50 = fs * 0.05

        def amplitude_over_half():
            return (np.max(np.diff(fecg[max(0, p2 - lim50):p2 + lim50])) <
                    0.5 * np.max(np.diff(fecg[max(0, p1 - lim50):p1 + lim50])))

        return p1 and p2 - p1 < (fs * 0.36) and amplitude_over_half()

    def sufficient_time_since_rr(p1, p2):
        """ If 1.5 * avg RR interval has elapsed, only threshold/2
        needs to be reached """
        avg_rr = np.mean(np.diff(np.sort(buf_qrs)))
        return Nqrs > 1 and fecg[p2] > 0.5 * dth and p2 - p1 >= 1.5 * avg_rr

    for i, p in enumerate(peaks):
        is_qrs = (local_maxima(p, int(fs * 0.2)) and
                  ((fecg[p] > dth and
                    both_gradients(p) and
                    not is_twave(prev_p, p))
                   or (sufficient_time_since_rr(prev_p, p))))
        if is_qrs:
            buf_qrs[i_buf_qrs % buf] = p
            i_buf_qrs += 1
            peak_is_qrs[i] = True
            prev_p = p
            Nqrs += 1
        else:
            buf_noise[i_buf_noise % buf] = p
            i_buf_noise += 1
        dth = (np.mean(fecg[buf_noise]) +
               (th * np.mean(fecg[buf_qrs] - fecg[buf_noise])))
    return peaks[peak_is_qrs]


def find_peaks(x: np.ndarray, comp: Callable = np.greater) -> np.ndarray:
    """ Find indices of peaks in an array.
    A peak is identified where the comparison function returns true
    comparing x_i to x_i-1 and x_i to x_i+1.
    Params:
        x (np.ndarray): Array to find peaks in
        comp (Callable): Comparison function to determine peak. Default: gt
    Returns:
        np.ndarray[int]: Indices of peaks
    """
    x = view(x, 3, 1)
    barr = np.logical_and(comp(x[:, 1], x[:, 0]), comp(x[:, 1], x[:, 2]))
    return np.where(barr)[0] + 1


@jit(nopython=True)
def nb_find_peaks(x: np.ndarray) -> np.ndarray:
    peaks = np.zeros(len(x), np.bool_)
    for i in range(1, len(x)-1):
        peaks[i] = x[i] > x[i-1] and x[i] > x[i+1]
    return np.where(peaks)[0]
