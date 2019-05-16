""" Calculate location distribution-based features
Typically each feature has two functions - one that takes a
dataframe and the other that takes numpy ndarrays.
The dataframe is assumed to have a datetime index and a latitude and
longitude column, each measured in degrees.
Cluster labels should be integers from 0 upwards. Non-clustered points
should be labelled -1.
"""
import numpy as np
import hdbscan
from numba import njit

from ..generic.information import entropy


def location_variance(df):
    """ Variance in locations
    Defined as the log of the sum of the variances of latitude and longitude.
    var(latitude) + var(longitude)
    Args:
        df (pandas.DataFrame): RADAR android_phone_location dataframe
    Returns:
        float: location variance
    """
    return arr_location_variance(df['latitude'].values, df['longitude'].values)


@njit
def arr_location_variance(latitude, longitude):
    """ Variance in locations
    Defined as the log of the sum of the variances of latitude and longitude.
    log(var(latitude) + var(longitude))
    Args:
        latitude (np.ndarray[float]): Array of latitudes in degrees
        longitude (np.ndarray[float]): Array of longitudes in degrees
    Returns:
        float: location variance
    """
    return np.var(latitude) + np.var(longitude)


def cluster_locations(df, **kwargs):
    """ Cluster locations with HDBSCAN
    Args:
        df (pandas.DataFrame): RADAR android_phone_location dataframe
        min_samples (int): Minimum number of samples to form a cluster
            Default = N/20
        kwargs: Key-word arguments to provide HDBSCAN
    Returns:
        np.ndarray[int]: Cluster labels
    """
    min_samples = kwargs.pop('min_samples', 1 + len(df)//20)
    clusterer = hdbscan.HDBSCAN(metric='haversine', min_samples=min_samples)
    clusterer.fit(df[['latitude', 'longitude']])
    return clusterer.labels_


def num_clusters(cluster_labels):
    """ Number of unique labels
    Args:
        cluster_labels (list/np.ndarray): Cluster labels
    Returns:
        int: number of unique labels
    """
    return len(np.unique(cluster_labels))


def cluster_totals(cluster_labels):
    """ Occurences in each cluster
    Args:
        cluster_labels (list/np.ndarray): Cluster labels
    Returns:
        dict: cluster label key with number of occurences as value
    """
    return dict(((c, n) for c, n in
                 zip(*np.unique(cluster_labels, return_counts=True))))


def cluster_entropy(cluster_labels):
    """ Entropy of location clusters
    Args:
        cluster_labels (list/np.ndarray): Cluster labels
    Returns:
        float: Shannon entropy
    """
    # c_total = cluster_totals(cluster_labels)
    # counts = list(c_total.values())
    counts = np.unique(cluster_labels, return_counts=True)[1]
    return entropy(counts)


def normalized_cluster_entropy(cluster_labels, n_clusters=None):
    """ Cluster entropy normalized by the log of the number of clusters.
    Args:
        cluster_labels (list/np.ndarray): Cluster labels
    Returns:
        float: Shannon entropy / log(n_clusters)
    """
    if n_clusters is None:
        n_clusters = len(np.unique(cluster_labels))
    counts = np.unique(cluster_labels, return_counts=True)[1]
    return entropy(counts) / np.log(n_clusters)
