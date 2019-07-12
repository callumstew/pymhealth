#!/usr/bin/env python3
""" Full accelerometer feature extraction
"""
import numpy as np
from itertools import count
from .accelerometer import *
from ..generic.stats import *
from ..generic.timedom import *
from ..generic.rqa import *

def acc_timedom_feat(in_vec: np.ndarray, fs: float) -> np.ndarray:
    assert in_vec.shape[1] == 3, "input vector must have three columns"

    X = in_vec[:,0]
    Y = in_vec[:,1]
    Z = in_vec[:,2]
    T = X + Y + Z

    out_vec = np.zeros(100) # may need to increase alloc if elements added
    ix = count()

    # mean
    out_vec[next(ix)] = mean(X)
    out_vec[next(ix)] = mean(Y)
    out_vec[next(ix)] = mean(Z)
    out_vec[next(ix)] = mean(T)
    out_vec[next(ix)] = mean(absolute(X))
    out_vec[next(ix)] = mean(absolute(Y))
    out_vec[next(ix)] = mean(absolute(Z))
    out_vec[next(ix)] = mean(absolute(X)+absolute(Y)+absolute(Z))

    # magnitude
    out_vec[next(ix)] = magnitude_dot(X,Y,Z)

    # distance
    out_vec[next(ix)] = mean(X-Y)
    out_vec[next(ix)] = mean(X-Z)
    out_vec[next(ix)] = mean(Y-Z)

    # skewness
    out_vec[next(ix)] = skewness(X)
    out_vec[next(ix)] = skewness(Y)
    out_vec[next(ix)] = skewness(Z)
    out_vec[next(ix)] = skewness(T)

    # kurtosis
    out_vec[next(ix)] = kurtosis(X)
    out_vec[next(ix)] = kurtosis(Y)
    out_vec[next(ix)] = kurtosis(Z)
    out_vec[next(ix)] = kurtosis(T)

    # variance
    out_vec[next(ix)] = var(X)
    out_vec[next(ix)] = var(Y)
    out_vec[next(ix)] = var(Z)
    out_vec[next(ix)] = var(T)

    # standard deviation
    out_vec[next(ix)] = std(X)
    out_vec[next(ix)] = std(Y)
    out_vec[next(ix)] = std(Z)
    out_vec[next(ix)] = std(T)

    # coefficient of variation (%)
    out_vec[next(ix)] = variation(X)*100
    out_vec[next(ix)] = variation(Y)*100
    out_vec[next(ix)] = variation(Z)*100
    out_vec[next(ix)] = variation(T)*100

    # amplitude range, max, min
    out_vec[next(ix)] = drange(X)
    out_vec[next(ix)] = drange(Y)
    out_vec[next(ix)] = drange(Z)
    mmX = minmax(X)
    mmY = minmax(Y)
    mmZ = minmax(Z)
    out_vec[next(ix)] = mmX[1]
    out_vec[next(ix)] = mmX[0]
    out_vec[next(ix)] = mmY[1]
    out_vec[next(ix)] = mmY[0]
    out_vec[next(ix)] = mmZ[1]
    out_vec[next(ix)] = mmZ[0]

    # correlation
    out_vec[next(ix)] = np.corrcoef(X,Y)[0,1]
    out_vec[next(ix)] = np.corrcoef(Y,Z)[0,1]
    out_vec[next(ix)] = np.corrcoef(X,Z)[0,1]

    # interquartile range
    out_vec[next(ix)] = interquartile_range(X)
    out_vec[next(ix)] = interquartile_range(Y)
    out_vec[next(ix)] = interquartile_range(Z)
    out_vec[next(ix)] = interquartile_range(T)

    # median
    out_vec[next(ix)] = median(X)
    out_vec[next(ix)] = median(Y)
    out_vec[next(ix)] = median(Z)
    out_vec[next(ix)] = median(T)

    # zero crossing rate (counts per minute)
    time = in_vec.shape[0]/fs
    out_vec[next(ix)] = zero_crossing_count(X)*60/time
    out_vec[next(ix)] = zero_crossing_count(Y)*60/time
    out_vec[next(ix)] = zero_crossing_count(Z)*60/time

    # area
    out_vec[next(ix)] = np.trapz(X, dx=1/fs)
    out_vec[next(ix)] = np.trapz(Y, dx=1/fs)
    out_vec[next(ix)] = np.trapz(Z, dx=1/fs)
    out_vec[next(ix)] = np.trapz(T, dx=1/fs)

    # area abs
    out_vec[next(ix)] = np.trapz(absolute(X), dx=1/fs)
    out_vec[next(ix)] = np.trapz(absolute(Y), dx=1/fs)
    out_vec[next(ix)] = np.trapz(absolute(Z), dx=1/fs)
    out_vec[next(ix)] = np.trapz(absolute(T), dx=1/fs)

    return out_vec[:next(ix)]


def acc_freqdom_feat(in_vec: np.ndarray, fs: float) -> np.ndarray:
    assert in_vec.shape[1] == 3, "input vector must have three columns"

    X = in_vec[:,0]
    Y = in_vec[:,1]
    Z = in_vec[:,2]

    out_vec = np.zeros(100) # may need to increase alloc if elements added
    ix = count()

    N = in_vec.shape[0]
    freq = (np.linspace(0,N-1,N) * fs/N)[1:]
    w = np.hamming(N)

    ampSpectX = np.abs(np.fft.fft(X*w))
    ampSpectY = np.abs(np.fft.fft(Y*w))
    ampSpectZ = np.abs(np.fft.fft(Z*w))
    ampSpectX[0] = 0
    ampSpectY[0] = 0
    ampSpectZ[0] = 0

    powSpectX = ((2*ampSpectX)**2/N)[:round(len(powSpectX)/2)]
    powSpectY = ((2*ampSpectY)**2/N)[:round(len(powSpectX)/2)]
    powSpectZ = ((2*ampSpectZ)**2/N)[:round(len(powSpectX)/2)]

    # signal power
    out_vec[next(ix)] = sum(powSpectX)
    out_vec[next(ix)] = sum(powSpectY)
    out_vec[next(ix)] = sum(powSpectZ)

    # signal power (low)

    # signal power (high)

    # entropy

    # fft main peak

    return out_vec[:next(ix)]


def acc_rp_feat(in_vec: np.ndarray, epsilon: float) -> np.ndarray:
    assert in_vec.shape[1] == 3, "input vector must have three columns"

    out_vec = np.zeros(4)

    rq_mat = rq2(in_vec, epsilon)
    out_vec[0] = recurrence_rate(rq_mat)
    out_vec[1] = determinism(rq_mat)
    out_vec[2] = length_entropy(rq_mat)
    out_vec[3] = mean(diagonal_lengths(rq_mat))

    return out_vec
