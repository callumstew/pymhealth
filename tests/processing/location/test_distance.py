import pytest
import numpy as np
import numpy.testing as npt

from mhealth.processing.location import distance

POINTS = np.array([
    (0.1532, 86.675),
    (33.123, 21.541),
    (41.507483, -99.436554),
    (38.504048, -98.315949),
    (51.5074, 0.1278),
    (41.3851, 2.1734)])


def test_haversine():
    """
    Test haversine distance, assumes r=6371
    """
    lat1, lon1 = POINTS[0]
    lat2, lon2 = POINTS[1]
    d = 7704.777296228049
    assert distance.haversine(lat1, lon1, lat2, lon2) == pytest.approx(d)


def test_haversine_elementwise():
    lats = POINTS[:, 0]
    lons = POINTS[:, 1]
    desired = np.array([7704.77729623, 9756.94118642,
                    347.32834804, 7275.82114826,
                    1136.28562666])
    out = distance.haversine_elementwise(lats[:-1], lons[:-1],
                                         lats[1:], lons[1:])
    npt.assert_almost_equal(out, desired)


def test_haversine_vector():
    lat_fixed, lon_fixed = POINTS[0]
    lats = POINTS[1:, 0]
    lons = POINTS[1:, 1]
    desired = np.array([7704.77729623, 15341.98217643, 15686.42408015,
                        9755.32422594, 9537.84258146])
    out = distance.haversine_vector(lat_fixed, lon_fixed, lats, lons)
    npt.assert_almost_equal(out, desired)


def test_haversine_outer_product():
    lats = POINTS[:, 0]
    lons = POINTS[:, 1]
    desired = np.array(
            [[0.0 , 7704.77729623, 15341.98217643, 15686.42408015, 9755.32422594, 9537.84258146],
            [7704.77729623, 0.0, 9756.94118642, 9918.88428512, 2677.52968247, 1938.58116302],
            [15341.98217643, 9756.94118642, 0.0,  347.32834804, 7096.01276647, 7898.26438152],
            [15686.42408015, 9918.88428512,  347.32834804,   0.0, 7275.82114826, 8034.9315799],
            [9755.32422594, 2677.52968247, 7096.01276647, 7275.82114826, 0.0, 1136.28562666],
            [9537.84258146, 1938.58116302, 7898.26438152, 8034.9315799, 1136.28562666, 0.0]])
    out = distance.haversine_outer_product(lats, lons, lats, lons)
    npt.assert_almost_equal(out, desired)
