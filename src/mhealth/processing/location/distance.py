import numpy as np
from numba import jit, njit, vectorize, guvectorize

@jit
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine distance in kilometers between two points given in degrees.
    r = 6371.009, 2r = 12742.018
    Because the haversine distance assumes a spherical Earth, it will only
    be accurate within ~0.5%. The mean earth radius is used (6371km).
    """
    λ1 = np.radians(lat1)
    λ2 = np.radians(lat2)
    φ1 = np.radians(lon1)
    φ2 = np.radians(lon2)
    Δλ = λ2 - λ1
    Δφ = φ2 - φ1
    return 12742.018 * np.arcsin(np.sqrt(
        np.sin(Δλ/2.0)**2 + (np.cos(λ1) * np.cos(λ2) * np.sin(Δφ/2.0)**2)))


@guvectorize(["void(float64[:], float64[:], float64[:], float64[:], float64[:])"],
             "(n),(n),(n),(n)->(n)")
def haversine_elementwise(lat1, lon1, lat2, lon2, res):
    """ Elementwise haversine distance between vectors
        of latitudes and longitudes

        lat1, lon1: vectors for latitude and longitude of locations
        lat2, lon2: vectors for latitude and longitude of locations
    """
    for i in range(lat1.shape[0]):
        res[i] = haversine(lat1[i], lon1[i], lat2[i], lon2[i])


@guvectorize(["void(float64, float64, float64[:], float64[:], float64[:])"],
             "(),(),(n),(n)->(n)")
def haversine_vector(lat1, lon1, latcol, loncol, res):
    """ Haversine distance between a fixed lat/lon and vectors of lat/lon
    lat1, lon1: Fixed latitude and longitude
    latvec, lonvec: vector of latitudes and longitudes
    """
    for i in range(latcol.shape[0]):
        res[i] = haversine(lat1, lon1, latcol[i], loncol[i])


@guvectorize(["void(float64[:], float64[:], float64[:], float64[:], float64[:,:])"],
             "(n),(n),(m),(m)->(n,m)")
def haversine_outer_product(lat1, lon1, lat2, lon2, res):
    """ Haversine distance between vectors of latitudes and longitudes
        (outer product)

        lat1, lon1: vectors for latitude and longitude of locations
        lat2, lon2: vectors for latitude and longitude of locations
    """
    for i in range(lat1.shape[0]):
        for j in range(lat2.shape[0]):
            res[i,j] = haversine(lat1[i], lon1[i], lat2[j], lon2[j])
