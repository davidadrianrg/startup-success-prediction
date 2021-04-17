"""Module with utils functions to optimize hyperparameters and get the best DNN model."""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from models import customized_metrics as cm
from sklearn.model_selection import train_test_split, KFold
from tensorflow import keras
from keras.utils import to_categorical
from scipy.stats import loguniform


def create_random_network(
    m,
    n_classes,
    metrics="accuracy",
    layers=(1, 5),
    n=(10, 20),
    activation=("relu", "sigmoid"),
    lr=(1e-5, 1e-3),
    optimizer=keras.optimizers.Adam,
    loss="categorical_crossentropy",
):
    """Return a deep neural network model with pseudo-random hyperparameters according to its args, each time it is called."""
    # Defining the model class Sequential in order to add layers one by one
    model = keras.models.Sequential()
    # First layer of de network
    model.add(
        keras.layers.Dense(
            random.randint(n[0], n[1]),
            input_dim=m,
            activation=random.choice(activation),
        )
    )
    # Input Normalization, working as a StandardScaler() tranformer for input data
    keras.layers.BatchNormalization()
    # Loop which adds layers following a uniform random distribution
    for i in range(random.randint(layers[0], layers[1])):
        model.add(
            keras.layers.Dense(
                random.randint(n[0], n[1]), activation=random.choice(activation)
            )
        )
    model.add(keras.layers.Dense(n_classes, activation="softmax"))
    # Choosing a learning rate from a logarithmic uniform distrbution
    optimizer = optimizer(
        learning_rate=round(
            loguniform.rvs(lr[0], lr[1]), int(abs(math.log(lr[0], 10)))
        )
    )
    # Define some characteristics for the training process
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model, optimizer, loss


def get_hyperparams(model):
    """Return the hyperparams of the model passed as an argument"""
    neurons = []
    activation = []
    layers = []
    nlayers = len(model.layers)

    for i in range(nlayers):
        neurons.append(model.layers[i].units)
        activation.append(model.layers[i].get_config()["activation"])
        layers.append("layer " + str(i))
    lr = model.optimizer.learning_rate.value().numpy()
    params = {
        "neurons": neurons,
        "activation": activation,
    }
    comp = {"optimizer": model.optimizer.get_config()["name"], "lr": lr}
    dfparams = pd.DataFrame(params, index=layers)
    dfcomp = pd.DataFrame(comp, index=["compiler"])

    return dfparams, dfcomp


def split_history_metrics(historial):
    d = {}
    for k in historial[0].keys():
        d[k] = [d[k] for d in historial]
    return d


def optimize_DNN(
    X,
    t,
    kfolds=10,
    train_size=0.8,
    trials=5,
    epochs=50,
    batch_size=40,
    metrics=(
        "accuracy",
        "Recall",
        cm.specificity,
        "Precision",
        cm.f1_score,
        "AUC",
    ),
):
    """Train the current model using cross validation and register its score comparing with new ones in each trial."""
    # Its needed the accuracy metric, if it is not passed it will be auto-included
    if "accuracy" not in metrics:
        metrics.append("accuracy")

    n, m = X.shape
    n_classes = len(np.unique(t))
    cv = KFold(kfolds)
    last_mean = 0

    # Split the data into train and test sets. Note that test set will be reserved for evaluate the best model later with data unseen previously for it
    X_train_set, X_test, t_train_set, t_test = train_test_split(
        X, t, train_size=train_size
    )

    # Loop for different trials or models to train in order to find the best
    for row in range(trials):

        model_aux, optimizer, loss = create_random_network(
            m, n_classes, metrics=metrics
        )
        params, comp = get_hyperparams(model_aux)

        print(f"\n***Trial {row+1} hyperparameters***", end="\n\n")
        print(params, "\n", comp)
        fold = 1
        # Lists to store historical data and means per fold for models
        historial = []
        mean_folds = []

        # Loop that manage cross validation using training set
        for train_index, test_index in cv.split(X_train_set):
            # This sentence carefully clones the untrained model in each fold in order to avoid unwanted learning weights between them
            model = keras.models.clone_model(model_aux)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            X_train, X_val = X_train_set[train_index], X_train_set[test_index]
            t_train, t_val = t_train_set[train_index], t_train_set[test_index]

            t_train = to_categorical(t_train, num_classes=n_classes)
            t_val = to_categorical(t_val, num_classes=n_classes)

            # Training of the model
            history = model.fit(
                X_train,
                t_train,
                validation_data=(X_val, t_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
            )
            # Save the train and test results,of the current model
            historial.append(history.history)
            mean = np.mean(history.history["val_accuracy"])
            mean_folds.append(mean)

            print(f"\nKFold{fold} --- Current score: {mean}")
            fold += 1
        # Criterion to choose the best model with the highest accuracy score in validation
        means = np.mean(mean_folds)
        history_metrics = split_history_metrics(historial)
        print(
            f"\n\tMin. Train score: {min(np.concatenate(history_metrics['accuracy']))} | Max. Train score: {max(np.concatenate(history_metrics['accuracy']))}"
        )
        if means > last_mean:
            # Save the model and its historial results
            best_model = (model, history_metrics)
            test_set = (X_test, t_test)
            last_mean = means
    return best_model, test_set


def plot_best_DNN(best_model, metric="accuracy"):
    """Plots the validation curve of the model using its historial results"""
    plt.plot(np.mean(best_model[0][1][metric], axis=0))
    plt.plot(np.mean(best_model[0][1]["val_" + metric], axis=0))
    plt.title("Model " + metric)
    plt.ylabel(metric)
    plt.xlabel("Iteration (epoch)")
    plt.legend(["Trainning", "Test"], loc="lower right")
    plt.show()


"""
# Example of code
X = pd.read_csv("./test/X.csv")
X = X.drop(["Unnamed: 0"], axis=1).values
t = pd.read_csv("./test/t.csv")
t = t["labels"].values
n, m = X.shape
n_classes = len(np.unique(t))

best = optimize_DNN(X, t, epochs=200, kfolds=10, trials=2)
plot_best_DNN(best)
"""
