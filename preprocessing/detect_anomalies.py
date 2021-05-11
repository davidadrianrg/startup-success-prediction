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
    def __init__(self, X, t, train_size, anomalies_size):
        M = X
        M["label"] = t.values
        M = M.sort_values(by=["label"], ascending=False)
        clean = M["label"].tolist().count(1)
        X_train = M[: round(clean * train_size)].iloc[:, :-1]
        X_test = M[round(clean * train_size) : clean + round(clean * anomalies_size)].iloc[:, :-1]
        self.t_test = M[round(clean * train_size) : clean + round(clean * anomalies_size)].iloc[:, -1]

        scaler = StandardScaler()
        scaler.fit(X_train)
        self.X_train = scaler.transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.autoencoder = None

    def perform_IsolationForest(self):
        model = IsolationForest(n_estimators=100, contamination=0)
        model.fit(self.X_train)
        y_test = model.predict(self.X_test)
        target_names = ["Anomalía", "Normal"]
        return classification_report(self.t_test, y_test, target_names=target_names)

    def perform_LOF(self):
        model = LocalOutlierFactor(n_neighbors=10, novelty=True)
        model.fit(self.X_train)
        y_test = model.predict(self.X_test)
        target_names = ["Anomalía", "Normal"]
        return classification_report(self.t_test, y_test, target_names=target_names)

    def perform_autoencoding(self):
        _, variables = self.X_train.shape
        autoencoder = models.Sequential()
        autoencoder.add(layers.Dense(1, input_dim=variables, activation="relu"))
        autoencoder.add(layers.Dense(variables, activation="relu"))
        opt = keras.optimizers.Adam(learning_rate=0.001)
        autoencoder.compile(loss="mean_squared_error", optimizer=opt)
        self.autoencoder = autoencoder

    def train_autoencoding(self, epochs=200, batch_size=100, **kwargs):
        self.history = self.autoencoder.fit(
            self.X_train, self.X_train, validation_data=(self.X_test, self.X_test), **kwargs
        )

    def plot_autoencoder_validation(
        self,
        xlabel: str = "Mean Square Error (MSE)",
        ylabel: str = "Iteration (epoch)",
        legend: tuple = ("Entrenamiento", "Test"),
        figsize: tuple = (12, 4),
    ):
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
        e_test = self.autoencoder.predict(self.X_test)
        self.mse_test = np.mean(np.power(self.X_test - e_test, 2), axis=1)
        threshold = np.max(self.mse_train)
        fig, ax = plt.subplots(figsize=(12, 4))
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

    def get_autoencoder_clreport(self):
        y_test = np.ones((self.t_test.shape))
        threshold = np.max(self.mse_train)
        y_test[self.mse_test > threshold] = -1

        target_names = ["Anomalía", "Normal"]
        return classification_report(self.t_test, y_test, target_names=target_names)
