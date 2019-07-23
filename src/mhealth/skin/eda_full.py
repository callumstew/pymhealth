#!/usr/bin/env python3
""" Full electrodermal activity feature extraction
"""
import numpy as np
from itertools import count

def smoothSignal(input: np.ndarray, smoothingSpan: int, edgeSpan: int = 0, repeat: int = 0) -> np.ndarray:
    output = np.convolve(input, np.ones((smoothingSpan,))/smoothingSpan, mode='same')
    if edgeSpan > 0:
        output[:edgeSpan] = output[edgeSpan]
        output[-edgeSpan:] = output[-edgeSpan]
    if repeat > 0:
        for i in range(repeat-1): output = smoothSignal(output, smoothingSpan, edgeSpan)
    return output

def getSCRR(input: np.ndarray, smoothingFactor: int, threshold: int, fs: float) -> float:
    diffSC = smoothSignal(np.diff(input), smoothingFactor) * fs
    thIx = np.where(diffSC >= threshold)[0]
    thSet = np.diff(thIx)

    SCR = np.empty(0)

    for outerIx in range(len(thSet)):
        if thSet[outerIx] > 1 or outerIx == len(thSet):
            thSelect = thIx[outerIx]
            innerIx = diffSC[thSelect]
            while innerIx > 0 and thSelect < len(diffSC):
                thSelect += 1
                innerIx = diffSC[thSelect]
            SCR = np.append(SCR, thSelect)
    
    if SCR.size == 0: return 0
    else: return len(SCR) / (len(input) / fs)
            
            



def eda_tonic_feat(in_vec: np.ndarray, fs: float, params: dict) -> np.ndarray:
    assert len(in_vec.shape) == 1, "input vector must have one column"

    N = in_vec.shape[0]
    smoothSCRR1 = params['smoothSCRR1'] * fs
    smoothSCRR2 = params['smoothSCRR2'] * fs
    smoothDiff = params['smoothDiff'] * fs
    smoothOPD = params['smoothOPD'] * fs
    edgeDuration = params['edgeDuration'] * fs
    thresholdSCRR = params['thresholdSCRR']
    thresholdOPD = params['thresholdOPD']

    inputSCRR = smoothSignal(in_vec, smoothSCRR1, edgeDuration)
    
    inputDiff2 = smoothSignal(in_vec, smoothDiff, edgeDuration)
    inputDiff2 = np.diff(inputDiff2, 2)
    inputDiff2 = np.abs(smoothSignal(inputDiff2, smoothDiff, repeat=3))

    inputOPD = smoothSignal(in_vec, smoothOPD, edgeDuration)
    inputOPD = np.diff(inputOPD, 2)
    inputOPD = np.abs(smoothSignal(inputOPD, smoothOPD, repeat=3))

    out_vec = np.zeros(100) # may need to increase alloc if elements added
    ix = count()

    out_vec[next(ix)] = np.mean(in_vec)
    out_vec[next(ix)] = getSCRR(inputSCRR, smoothSCRR2, thresholdSCRR, fs)
    out_vec[next(ix)] = np.sum(inputDiff2**2) / len(inputDiff2)
    out_vec[next(ix)] = np.mean(inputOPD[inputOPD > thresholdOPD])

    return out_vec[:next(ix)]


def eda_tonic_baseline_feat(in_vec: np.ndarray, fs: float) -> np.ndarray:
    assert in_vec.shape[1] == 1, "input vector must have one column"

    out_vec = np.zeros(100) # may need to increase alloc if elements added
    ix = count()

    return out_vec[:next(ix)]
