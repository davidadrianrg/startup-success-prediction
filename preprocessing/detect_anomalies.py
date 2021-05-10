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
    def __init__(self, X, t, train_size, anomlies_size):
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

    def perform_IsolationForest(self):
        model = IsolationForest(n_estimators=100, contamination=0)
        model.fit(self.X_train)
        y_test = model.predict(X_test)
        target_names = ["Anomalía", "Normal"]
        return classification_report(
            self.t_test, y_test, target_names=target_names
        )

    def perform_LOF(self):
        model = LocalOutlierFactor(n_neighbors=10, novelty=True)
        model.fit(self.X_train)
        y_test = model.predict(self.X_test)
        target_names = ["Anomalía", "Normal"]
        return classification_report(
            self.t_test, y_test, target_names=target_names
        )

    def perform_autoencoding(self):
        samples, variables = self.X_train.shape
        autoencoder = models.Sequential()
        autoencoder.add(layers.Dense(1, input_dim=variables, activation="relu"))
        autoencoder.add(layers.Dense(variables, activation="relu"))
        opt = keras.optimizers.Adam(learning_rate=0.001)
        autoencoder.compile(loss="mean_squared_error", optimizer=opt)

    def train_autoencoding(self, epochs=200, batch_size=100, **kwargs):
        self.history = autoencoder.fit(
            self.X_train,
            self.X_train,
            validation_data=(self.X_test, self.X_test),
            **kwargs
        )

    def plot_autoencoder_validation(self):
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.ylabel("Error cuadrático medio (MSE)")
        plt.xlabel("Iteración (epoch)")
        plt.legend(["Entrenamiento", "Test"], loc="upper right")

    def plot_autoencoder_threshold(self):
        y_train = autoencoder.predict(self.X_train)
        self.mse_train = np.mean(np.power(self.X_train - y_train, 2), axis=1)
        umbral = np.max(self.mse_train)

        plt.figure(figsize=(12, 4))
        plt.hist(self.mse_train, bins=50)
        plt.xlabel("Error de reconstrucción (entrenamiento)")
        plt.ylabel("Número de datos")
        plt.axvline(umbral, color="r", linestyle="--")
        plt.legend(["Umbral"], loc="upper center")

    def plot_autoencoder_error(self):
        e_test = autoencoder.predict(self.X_test)
        self.mse_test = np.mean(np.power(self.X_test - e_test, 2), axis=1)

        plt.figure(figsize=(12, 4))
        plt.plot(range(1, self.X_train.shape[0] + 1), self.mse_train, "b.")
        plt.plot(
            range(
                self.X_train.shape[0] + 1,
                self.X_train.shape[0] + self.X_test.shape[0] + 1,
            ),
            self.mse_test,
            "r.",
        )
        plt.axhline(umbral, color="r", linestyle="--")
        plt.xlabel("Índice del dato")
        plt.ylabel("Error de reconstrucción")
        plt.legend(["Entrenamiento", "Test", "Umbral"], loc="upper left")

    def get_autoencoder_clreport(self):
        y_test = np.ones((t_test.shape))
        y_test[self.mse_test > umbral] = -1

        target_names = ["Anomalía", "Normal"]
        print(
            classification_report(
                self.t_test, y_test, target_names=target_names
            )
        )


X = pd.read_csv("test\X.csv")
# t = pd.read_csv("test\t.csv")
X = X.drop("Unnamed: 0", axis=1)
print(X)
anomalies = Anomalies(X, t)
