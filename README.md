# Pymhealth

Pymhealth is a python package for processing and extracting features from
mHealth sensors and data streams, particularly those from smartphones
and common wearable devices. It uses numba to compile functions where
it will provide a significant improvement over popular python data analysis
and signal processing packages, but will otherwise use and integrate itself
with the standard python data science stack.


## Package structure
There are two main subpackages: processing and features, for processing and
extracting features from mHealth data respectively. Each contains submodules
corresponding to common mHealth data streams (accelerometer, eda, telephony,
etc).

Documentation is provided at
[callumstew.github.io/pymhealth](https://callumstew.github.io/pymhealth/)
