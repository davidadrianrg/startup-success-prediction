import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors


class Clustering:
    @staticmethod
    def perform_kmeans(X: pd.DataFrame, **kwargs):
        k_list = np.arange(1, len(X.columns) + 1)
        inertia_dic = {}
        for k in k_list:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
            inertia_dic[k] = kmeans.inertia_
        return inertia_dic

    @staticmethod
    def perform_DBSCAN(X: pd.DataFrame, **kwargs):
        neighbors = NearestNeighbors(n_neighbors=2 * len(X.columns))
        neighbors_fit = neighbors.fit(X)
        distances, _ = neighbors_fit.kneighbors(X)
        distances = np.sort(distances, axis=0)
        distances = distances[:, -1]
        return distances
