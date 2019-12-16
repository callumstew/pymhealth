"""Generate CFFI bindings to FFTW."""
from cffi import FFI


ffi = FFI()
ffi.set_source(
    '_fftw',
    """
    #include <fftw3.h>

    static void fftw_fft(int N, fftw_complex* in,
                         fftw_complex* out, int direction) {
        fftw_plan p = fftw_plan_dft_1d(N, in, out, direction,
                                       FFTW_ESTIMATE);
        fftw_execute(p);
        fftw_destroy_plan(p);
    };
    """,
    libraries=['fftw3']
)

ffi.cdef(
    """
    #define FFTW_FORWARD ...
    #define FFTW_BACKWARD ...
    typedef ... fftw_complex;
    void fftw_fft(int, fftw_complex *, fftw_complex *, int);
    """
)


if __name__ == '__main__':
    ffi.compile()
