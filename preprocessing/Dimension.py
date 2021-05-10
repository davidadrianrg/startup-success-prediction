import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA, FastICA


class Dimension:
    @staticmethod
    def get_correlation_matrix(X):
        samples, nvar = X.shape
        corr_mat = np.corrcoef(np.c_[X].T)
        etiquetas = X.columns.values.tolist()
        # DataFrame with correlations between variables
        matrix = pd.DataFrame(corr_mat, columns=etiquetas, index=etiquetas)

    @staticmethod
    def get_PCA(X, **kwargs):
        samples, nvar = X.shape
        pca = PCA(**kwargs)
        pca.fit(X)
        plt.figure(figsize=(10, 5))
        plt.bar(
            X.columns.values.tolist(),
            pca.explained_variance_ratio_ * 100,
            color="b",
            align="center",
            tick_label=X.columns.values.tolist(),
        )
        plt.xticks(rotation="vertical")
        indices = np.argsort(pca.explained_variance_ratio_)
        plt.xlabel("Componentes principales")
        plt.ylabel("% de varianza explicada")
        plt.tight_layout()
        plt.show()


"""
X = pd.read_csv("test\X.csv")
X = X.drop("Unnamed: 0", axis=1)
Dimension.get_correlation_matrix(X)
"""
