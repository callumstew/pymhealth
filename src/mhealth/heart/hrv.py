""" Heart rate variability metrics

Shaffer, F., & Ginsberg, J. P. (2017). An Overview of Heart Rate Variability
Metrics and Norms. Frontiers in public health, 5, 258.
https://dx.doi.org/10.3389%2Ffpubh.2017.00258


Malik, M., Camm, A. J., Bigger, J. T., Breithardt, G., Cerutti, S.,
Cohen, R. J., ... Singer, D. H. (1996). Heart rate variability. Standards
of measurement, physiological interpretation, and clinical use.
European Heart Journal, 17(3), 354-381.
"""
import numpy as np
import pandas as pd
from numba import njit


def nni_timestamps(nni, interval_unit='ms'):
    return pd.to_timedelta(np.cumsum(nni), unit=interval_unit)


# Time domain

def sdnn(nni):
    return np.std(nni)


def sdrr(rri):
    return np.std(rri)


def sdann(nni, interval='5T'):
    """
    nni (pd.Series)

    """
    return nni.resample(interval).mean().std()


def sdnni(nni, interval='5T'):
    return nni.resample(interval).std().mean()


def pnn50(nni):
    return (np.diff(nni) > 50) / len(nni)


def rmssd(nni):
    return np.sqrt(np.mean(np.square(np.diff(nni))))


def ssd(nni):
    """ Sum of successive differences
    """
    return np.sum(np.diff(nni))


# Freq domain

# Non-linear
def csi_sd1(rri):
    return np.std((np.sqrt(2) / 2) * (rri[:-1] - rri[1:]))

def csi_sd2(rri):
    return np.std((np.sqrt(2) / 2) * (rri[:-1] + rri[1:]))


def lorenz_csi(rri):
    """ Cardiac Sympathetic Index
    Params:
        rri (np.ndarray): RR-intervals
    Returns:
        Cardiac sympathetic index
    DOI: 10.1109/EMBC.2014.6944639
    """
    return csi_sd1(rri) / csi_sd2(rri)


def lorenz_cvi(rri):
    return np.log10(csi_sd1(rri) * csi_sd2(rri))


def lorenz_mcsi(rri):
    return (csi_sd1(rri) ** 2) / csi_sd2(rri)
