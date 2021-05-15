import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA, FastICA


class Dimension:
    """Class with implemented techniques for reducing dimensions of features"""

    @staticmethod
    def get_correlation_matrix(X: pd.DataFrame):
        """Calculates the correlation matriz of data features

        :param X: Dataset of features and samples
        :type X: pd.DataFrame
        :return: Dataframe with correlation between variables
        :rtype: pd.DataFrame
        """
        samples, nvar = X.shape
        corr_mat = np.corrcoef(np.c_[X].T)
        etiquetas = X.columns.values.tolist()
        # DataFrame with correlations between variables
        return pd.DataFrame(corr_mat, columns=etiquetas, index=etiquetas)

    @staticmethod
    def get_PCA(X: pd.DataFrame, **kwargs):
        """Applies PCA technique to reduce dimensionality

        :param X: Dataset of features and samples
        :type X: pd.DataFrame
        :return: PCA results
        :rtype: obj
        """
        samples, nvar = X.shape
        pca = PCA(**kwargs)
        pca.fit(X)
        return pca
