import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors


class Clustering:
    """Class with clustering techniques implemented for unsupervised learning"""

    @staticmethod
    def perform_kmeans(X: pd.DataFrame, **kwargs):
        """Static method wrapped in Clustering class which performs Kmeans algorithm

        :param X: Dataset with samples
        :type X: pd.DataFrame
        :return: Inertias from data samples to clusters
        :rtype: dict
        """
        k_list = np.arange(1, len(X.columns) + 1)
        inertia_dic = {}
        for k in k_list:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
            inertia_dic[k] = kmeans.inertia_
        return inertia_dic

    @staticmethod
    def perform_DBSCAN(X: pd.DataFrame, **kwargs):
        """Static method wrapped in Clustering class which performs

        :param X: Dataset with samples
        :type X: pd.DataFrame
        :return: Distances for all data to its k-nearest-neighbor
        :rtype: list
        """
        neighbors = NearestNeighbors(n_neighbors=2 * len(X.columns))
        neighbors_fit = neighbors.fit(X)
        distances, _ = neighbors_fit.kneighbors(X)
        distances = np.sort(distances, axis=0)
        distances = distances[:, -1]
        return distances
