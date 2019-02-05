=======
Generic
=======
Generic preprocessing functions that may apply to any or multiple signals

-------
Filters
-------

Butterworth
-----------
.. py:function:: butterworth(arr, cutoff, freq, order=5, ftype='highpass')
    Generic Butterworth function

    :param acc: Signal vector 
    :type acc: float[n]
    :param cutoff: Cutoff value(s) 
    :type cutoff: float or (float, float)
    :param freq: Sampling frequency of signal
    :type freq: float
    :param order: Order of Butterworth filter. Default - 5
    :type order: int
    :param ftype: Filter type {'highpass', 'lowpass', 'bandpass'} Default - 'highpass'
    :type ftype: str
    :rtype: float[n]
