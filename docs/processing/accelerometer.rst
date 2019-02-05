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



.. py:function haversine_vector(lat1, lon1, latcol, loncol)
    The haversine distance between a fixed point and a set of
    latitude / longitude vectors

    :param lat1: fixed latitude
    :type lat1: float64
    :param lon1: fixed longitude
    :type lon1: float64
    :param latcol: latitude vector
    :type latcol: float64[n]
    :param loncol: longitude vector
    :type loncol: float64[n]
    :rtype: float64[n]
