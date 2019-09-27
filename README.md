# Pymhealth

Pymhealth is a python package for processing and extracting features from
mHealth sensors and data streams, particularly those from smartphones
and common wearable devices. It uses numba to compile functions where
it will provide a significant improvement over popular python data analysis
and signal processing packages, but will otherwise use and integrate itself
with the standard python data science stack.


## Package structure
---
Documentation is provided at # OUTDATED
[callumstew.github.io/pymhealth](https://callumstew.github.io/pymhealth/)


## Quick contribution notes

The idea of this project is to have functions either JIT compiled with numba,
or such that they can be jitted within other numba functions (i.e. extracting
features looping through windows in a large array). An introduction to numba is available [here](https://numba.pydata.org/numba-do/latest/user/5minguide.html)

The three decorators we will typically use are: @jit, @register_jitable, and @overload.


@jit will cause the decorated function to be just-in-time compiled when first run.
It is therefore important that @jit-ed functions use only numba-compatible
python and numpy features.


@register_jitable allows numba to compile a non-jited function which is referenced
in a jited function. It can be useful in small untility functions that you want to
seperate for readability reasons. It can also be useful where the jit compiled version of
a function does not provide any performance increase (or even a decrease), and
should only be used when necessary as part of another function.


@overload allows you to write a numba-compatible function which will be used in
place of the overloaded function. For example, if there is a
function in scipy which you want to use, it will not be possible to compile it
with numba directly. If it is easy to reimplement, you can overload it with your
own numba-compatible function. For more information, see [this page](https://numba.pydata.org/numba-doc/dev/extending/high-level.html)
