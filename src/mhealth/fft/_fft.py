import numpy as np
from numba import jit
from numba.extending import overload, register_jitable
import numba.cffi_support
from . import _fftw


FFTW_FORWARD = -1
FFTW_BACKWARD = 1


numba.cffi_support.register_module(_fftw)
numba.cffi_support.register_type(_fftw.ffi.typeof('fftw_complex'),
                                 numba.complex128)
fftw_fft = _fftw.lib.fftw_fft


@jit
def fft(a):
    out = np.zeros(a.shape, dtype=np.complex128)
    a = a.astype(np.complex128)
    n = a.shape[0]
    fftw_fft(
        n,
        _fftw.ffi.from_buffer(a),
        _fftw.ffi.from_buffer(out),
        FFTW_FORWARD
    )
    return out


@register_jitable
def _unscaled_ifft(a):
    out = np.zeros(a.shape, dtype=np.complex128)
    a = a.astype(np.complex128)
    n = a.shape[0]
    fftw_fft(
        n,
        _fftw.ffi.from_buffer(a),
        _fftw.ffi.from_buffer(out),
        FFTW_BACKWARD
    )
    return out


@jit
def ifft(a):
    return _unscaled_ifft(a) / a.shape[0]


@overload(np.fft.fft)
def _jit_np_fft(a):
    return fft.py_func


@overload(np.fft.ifft)
def _jit_np_ifft(a):
    return ifft.py_func
