import numpy as np
from numba import jit
from ..generic.filters import butterworth


def incremental_merge_segmentation(x, m=64):
    """Detect PPG pulse peaks
    10.1109/EMBC.2012.6346628
    """
    raise NotImplementedError
    seg = 0
    z = 0
    seg_in_line = 1
    # line = construct_line(x[seg*m:(seg*m)+m])
    # lines = 
    for seg in range(1, len(x) // m):
        pass


def pulse_onset_zong(x, sampling_rate):
    """ Pulse onset detection based on (Zong et al 2003) WIP
    10.1109/CIC.2003.1291140
    """
    x = butterworth(x, [0.5, 20], sampling_rate, ftype='bandpass')
    w = int(sampling_rate / (1000 / 150))
    x = slope_sum(x, w)
    onsets = physionet_decision_rule(x, sampling_rate)
    return onsets


@jit
def slope_sum(x, w):
    out = np.zeros(len(x))
    dx = np.diff(x)
    for i in range(w, len(x)-1):
        out[i] = np.sum(dx[i-w:i])
    return out


@jit
def physionet_decision_rule(x, sampling_rate):
    th = 2 * np.std(x[:sampling_rate * 10]) + np.mean(x[:sampling_rate * 10])
    th_sub_std = th - np.std(x[:sampling_rate * 10])
    w150 = int(sampling_rate / (1000 / 150))
    onsets = []
    i = w150
    j = 0
    amps_idx = 0
    prev_amps = np.zeros(10)
    prev_amps[:] = th
    while i < len(x) - sampling_rate * 10:
        if x[i] > (th_sub_std):
            largest_nearby = i - w150 + np.argmax(x[i-w150:i+w150])
            j = largest_nearby
            """
            while x[j] > 0.01 * x[largest_nearby]:
                j -= 1
            """
            onsets.append(j)
            i += (3 * w150) - 1
            amps_idx = (amps_idx + 1) % 10
            prev_amps[amps_idx] = x[largest_nearby]
            th = np.median(prev_amps)
            th_sub_std = th - 2 * np.std(prev_amps)
        i += 1

        if j < i - (sampling_rate * 5):
            th = 3 * np.mean(x[j + (sampling_rate):j + (sampling_rate * 11)])
            prev_amps[:] = th
            i = j + w150

    return onsets
