"""Module to implement the anomalies detection methodes."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class Anomalies:
    """Class with clustering techniques implemented for unsupervised learning."""

    def __init__(self, X, t, train_size, anomalies_size):
        """Contructor method which customized dataset to apply anomalies detection techniques.

        :param X: Dataset samples
        :type X: pd.DataFrame
        :param t: Dataset labels
        :type t: pd.Series
        :param train_size: percentage of samples used for training
        :type train_size: float
        :param anomalies_size: percentage of samples included as anomalies
        :type anomalies_size: float
        """
        M = X
        M["label"] = t.values
        M = M.sort_values(by=["label"], ascending=False)
        clean = M["label"].tolist().count(1)
        X_train = M[: round(clean * train_size)].iloc[:, :-1]
        X_test = M[
            round(clean * train_size) : clean + round(clean * anomalies_size)
        ].iloc[:, :-1]
        self.t_test = M[
            round(clean * train_size) : clean + round(clean * anomalies_size)
        ].iloc[:, -1]

        scaler = StandardScaler()
        scaler.fit(X_train)
        self.X_train = scaler.transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.autoencoder = None

    def perform_IsolationForest(
        self, n_estimators=100, target_names=None, contamination=0, **kwargs
    ):
        """Isolation Forest algorithm.

        :param n_estimators: number of estimators used, defaults to 100
        :type n_estimators: int, optional
        :param contamination: percentage os samples included as anomalies in trining set, defaults to 0
        :type contamination: int, optional
        :return: classification report with results
        :rtype: str
        """
        model = IsolationForest(
            n_estimators=n_estimators, contamination=contamination, **kwargs
        )
        model.fit(self.X_train)
        y_test = model.predict(self.X_test)
        return classification_report(
            self.t_test, y_test, target_names=target_names
        )

    def perform_LOF(self, n_neighbors=10, target_names=None, novelty=True):
        """LOF algorithm.

        :param n_neighbors: number of data neighbours used, defaults to 10
        :type n_neighbors: int, optional
        :param novelty: param necessary to detect anomalies, defaults to True
        :type novelty: bool, optional
        :return: classification report with results
        :rtype: str
        """
        model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=novelty)
        model.fit(self.X_train)
        y_test = model.predict(self.X_test)
        return classification_report(
            self.t_test, y_test, target_names=target_names
        )

    def perform_autoencoding(self):
        """Create an autoencoder neural network."""
        _, variables = self.X_train.shape
        autoencoder = models.Sequential()
        autoencoder.add(layers.Dense(1, input_dim=variables, activation="relu"))
        autoencoder.add(layers.Dense(variables, activation="relu"))
        opt = keras.optimizers.Adam(learning_rate=0.001)
        autoencoder.compile(loss="mean_squared_error", optimizer=opt)
        self.autoencoder = autoencoder

    def train_autoencoding(self, epochs=200, batch_size=100, **kwargs):
        """Fits autoencoder neural network.

        :param epochs: number of times all data is passed to network, defaults to 200
        :type epochs: int, optional
        :param batch_size: number of splits of data in batchs, defaults to 100
        :type batch_size: int, optional
        """
        self.history = self.autoencoder.fit(
            self.X_train,
            self.X_train,
            validation_data=(self.X_test, self.X_test),
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )

    def plot_autoencoder_validation(
        self,
        xlabel: str = "Mean Square Error (MSE)",
        ylabel: str = "Iteration (epoch)",
        legend: tuple = ("Entrenamiento", "Test"),
        figsize: tuple = (12, 4),
    ):
        """Plot autoencoder validation curve.

        :param xlabel: plot xlabel, defaults to "Mean Square Error (MSE)"
        :type xlabel: str, optional
        :param ylabel: plot ylabel, defaults to "Iteration (epoch)"
        :type ylabel: str, optional
        :param legend: plot legend, defaults to ("Entrenamiento", "Test")
        :type legend: tuple, optional
        :param figsize: size of plot, defaults to (12, 4)
        :type figsize: tuple, optional
        :return: plot figure
        :rtype: obj
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.history.history["loss"])
        ax.plot(self.history.history["val_loss"])
        ax.set_ylabel(xlabel)
        ax.set_xlabel(ylabel)
        ax.legend(legend, loc="upper right")
        return fig

    def plot_autoencoder_threshold(
        self,
        xlabel: str = "Reconstruction error (training)",
        ylabel: str = "Number of data",
        legend: tuple = ("Threshold"),
        figsize: tuple = (12, 4),
    ):
        """Predicts anomalies by recontructing samples passed to encoder.

        :param xlabel: xlabel plot, defaults to "Reconstruction error (training)"
        :type xlabel: str, optional
        :param ylabel: ylabel plot, defaults to "Number of data"
        :type ylabel: str, optional
        :param legend: legend plot, defaults to ("Threshold")
        :type legend: tuple, optional
        :param figsize: size of plot, defaults to (12, 4)
        :type figsize: tuple, optional
        :return: figure
        :rtype: obj
        """
        y_train = self.autoencoder.predict(self.X_train)
        self.mse_train = np.mean(np.power(self.X_train - y_train, 2), axis=1)
        threshold = np.max(self.mse_train)

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(self.mse_train, bins=50)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axvline(threshold, color="r", linestyle="--")
        ax.legend(legend, loc="upper center")
        return fig

    def plot_autoencoder_error(
        self,
        xlabel: str = "Data Index",
        ylabel: str = "Reconstruction Error",
        legend: tuple = ("Training", "Test", "Threshold"),
        figsize: tuple = (12, 4),
    ):
        """Plot autoencoder error.

        :param xlabel: xlabel plot, defaults to "Data Index"
        :type xlabel: str, optional
        :param ylabel: ylabel plot, defaults to "Reconstruction Error"
        :type ylabel: str, optional
        :param legend: legend plot, defaults to ("Training", "Test", "Threshold")
        :type legend: tuple, optional
        :param figsize: size of figure, defaults to (12, 4)
        :type figsize: tuple, optional
        :return: figure
        :rtype: obj
        """
        e_test = self.autoencoder.predict(self.X_test)
        self.mse_test = np.mean(np.power(self.X_test - e_test, 2), axis=1)
        threshold = np.max(self.mse_train)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(1, self.X_train.shape[0] + 1), self.mse_train, "b.")
        ax.plot(
            range(
                self.X_train.shape[0] + 1,
                self.X_train.shape[0] + self.X_test.shape[0] + 1,
            ),
            self.mse_test,
            "r.",
        )
        ax.axhline(threshold, color="r", linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(legend, loc="upper left")
        return fig

    def get_autoencoder_clreport(self, target_names=None):
        """Generate classification report with autoencoders results.

        :return: classification report with results
        :rtype: str
        """
        y_test = np.ones((self.t_test.shape))
        threshold = np.max(self.mse_train)
        y_test[self.mse_test > threshold] = -1

        return classification_report(
            self.t_test, y_test, target_names=target_names
        )
