import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors


class Clustering:
    @staticmethod
    def perform_kmeans(X, **kwargs):
        k_list = np.arange(1, len(X.columns) + 1)
        inertia_dic = {}
        for k in k_list:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
            inertia_dic[k] = kmeans.inertia_
        plt.plot(inertia_dic.keys(), inertia_dic.values(), marker="x")
        plt.xlabel("k")
        plt.xticks(range(1, len(X.columns) + 1))
        plt.ylabel("Inercia")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def perform_DBSCAN(X, **kwargs):
        neighbors = NearestNeighbors(n_neighbors=2 * len(X.columns))
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)
        distances = np.sort(distances, axis=0)
        distances = distances[:, -1]
        plt.plot(distances)
        plt.xlabel("Puntos ordenados por distancia al k-vecino más cercano")
        plt.ylabel("Distancia al k-vecino más cercano")
        plt.show()


"""
X = pd.read_csv("test\X.csv")
X = X.drop("Unnamed: 0", axis=1)
Clustering.perform_DBSCAN(X)
"""
