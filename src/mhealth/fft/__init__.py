import importlib as _imp

if _imp.util.find_spec('mhealth.fft._fftw'):
    from ._fft import fft, ifft
else:
    print('Error loading JIT-able FFT functions - falling back to numpy')
    from numpy.fft import fft, ifft
