""" Calculate location distance-based features.
Typically each feature has two functions - one that takes a
dataframe and the other that takes numpy ndarrays.
The dataframe is assumed to have a datetime index and a latitude and
longitude column, each measured in degrees.
"""
import numpy as np
from . import distance


def determine_home_coords(df, start_time='23:00', end_time='06:00'):
    """ Returns median latitude and longitude during nighttime
    Args:
        df: (pandas.DataFrame) RADAR android_phone_location dataframe
        start_time (str): Optional time to consider locations from
        end_time (str): Optional time to consider locations until
    Returns:
        (float, float): latitude, longitude
    """
    lat, lon = df[['latitude', 'longitude']]\
        .between_time(start_time, end_time)\
        .median()\
        .values
    return (lat, lon)


def distance_from_home(df, home_coords=None):
    """ Calculate distance to a home coordinate
    Params:
        df (pandas.Dataframe): Location dataframe
        home_coords ((float, float)): Latitude, longitude of the home point.
            (Optional) If not given, it is calculated.
    Returns:
        pd.Series[float]: Distance from home_coords in km
    """
    if home_coords is None:
        home_coords = determine_home_coords(df)
    out = arr_distance_from_home(df['latitude'], df['longitude'], home_coords)
    out.name = 'home_distance'
    return out


def arr_distance_from_home(latitude, longitude, home_coords):
    """ Distance between an array of latitude/longitude and a home point
    Params:
        latitude (np.ndarray[float]): Array of latitudes in degrees
        longitude (np.ndarray[float]): Array of longitudes in degrees
        home_coords ((float, float)): Latitude, longitude of the home point
    Returns:
        np.ndarray[float]: Distance from home_coords in km
    """
    lat, lon = home_coords
    return distance.haversine_vector(lat, lon, latitude, longitude)


def proportion_home_stay(df, limit=0.1, home_coords=None):
    """ The proportion of points within range of the home coordinates
    Params:
        df (pandas.DataFrame): Location dataframe
        limit (float): Distance to home_coord within which the point is
            assumed to be at home (km) Default: 0.1km
        home_coords ((float, float)): Latitude, longitude of the home point.
            (Optional) If not given, it is calculated.
    Returns:
        float: (0 - 1.0) proportion of coordinates within range of
            the home coordinates
    """
    return (distance_from_home(df, home_coords) < limit).sum() / len(df)


def arr_proportion_home_stay(latitude, longitude, limit, home_coords):
    """ The proportion of points within range of the home coordinates
    Params:
        latitude (np.ndarray[float]): Latitude array
        longitude (np.ndarray[float]): Longitude array
        limit (float): Distance to home_coord within which the point is
            assumed to be at home (km)
        home_coords ((float, float)): Latitude, longitude of the home point.
    Returns:
        float: (0 - 1.0) proportion of coordinates within range of
            the home coordinates
    """
    return (arr_distance_from_home(latitude, longitude, home_coords) <
            limit).sum() / len(latitude)


def successive_distance(df):
    """ The distance between successive points.
    Params:
        df (pandas.DataFrame): Location dataframe
    Returns:
        pd.Series[float]: Distance (km) between a point and the
            previous point. Initial distance is 0.
    """
    return arr_successive_distance(df['latitude'], df['longitude'])


def arr_successive_distance(latitude, longitude):
    """ The distance between successive points.
    Params:
        latitude (np.ndarray[float]): Latitude coordinates
        longitude (np.ndarray[float]): Longitude coordinates
    Returns:
        np.ndarray[float]: Distance (km) between a point and the
            previous point. Initial distance is 0.
    """
    dist = latitude.copy()
    dist[0] = 0
    dist[1:] = distance.haversine_elementwise(
        latitude[:-1], longitude[:-1],
        latitude[1:], longitude[1:]
    )
    return dist
