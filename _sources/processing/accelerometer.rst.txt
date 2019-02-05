.. highlight:: python3
=============
Accelerometer
=============

-------
Filters
-------
Convenience functions around scipy.signal for filtering accelerometer data, using a 5th order Butterworth filter.

Linear acceleration
-------------------
Filter out the linear component of acceleration from a raw accelerometer signal.

.. py:function:: acc_linear(acc, freq, cutoff, order) 
    Estimate linear acceleration using a high-pass Butterworth filter.
    Optionally, you can specify a low-pass cutoff to perform a band-pass filter by passing two values to cutoff.

    :param acc: acceleration vector or matrix. If a matrix, columns correspond to dimension in space
    :type acc: float[n,m]
    :param freq: Sampling frequency of acceleration
    :type freq: float
    :param cutoff: Cutoff value(s) for highpass or bandpass filter
    :type cutoff: float or (float, float)
    :param order: Order of Butterworth filter
    :type order: int
    :rtype: float[n,m]



Gravitational component
-----------------------
Filter out the gravitational component of acceleration from a raw accelerometer signal.

.. py:function:: acc_gravity(acc, freq, cutoff, order) 
    Estimate gravitational component of acceleration using a low-pass Butterworth filter.

    :param acc: acceleration vector or matrix.
        If a matrix, columns correspond to dimension in space
    :type acc: float[n,m]
    :param freq: Sampling frequency of acceleration
    :type freq: float
    :param cutoff: Low-pass cutoff value(s)
    :type cutoff: float
    :param order: Order of Butterworth filter
    :type order: int
    :rtype: float[n,m]
