""" ECG QRS-complex and R-peak detection algorithms
"""
import numpy as np
from numba import njit

from ..generic.filters import butterworth

def bandpass(ecg, fs, low=5, high=15, order=5):
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


def pan_tompkins_detection(ecg, fs):
    """
    Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection
    algorithm. IEEE Trans. Biomed. Eng, 32(3), 230-236.
    DOI: 10.1109/TBME.1985.325532
    Params:
        ecg (np.array): Input ECG signal
        fs (float): Frequency of ECG signal.
    Returns:
        np.array: Indices of detected peaks
    """
    def differentiate(x):
        """ Five-point derivative with transfer function
        H(z) = 1/8T ( -z^-2 - -2z^-1 + 2z + z^2 )
        """
        return np.convolve(x, [1, 2, 0, -2, -1][:-4]) / 8


    def integrate(x, window):
        """ Moving average filter to integrate signal
        """
        return np.convolve(x, np.ones(window)/window)[:-window + 1]

    def detect_peaks(x):

        return x

    ecg = bandpass(ecg, fs)
    ecg = differentiate(ecg)
    ecg = ecg ** 2
    window = int(0.2 * fs)
    ecg = integrate(ecg, window)
    peaks = detect_peaks(ecg)
    return peaks
