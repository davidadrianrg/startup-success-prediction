"""Module with utils functions to optimize hyperparameters and get the best DNN model."""

import math
import random

import numpy as np
import pandas as pd
from keras.utils import to_categorical
from scipy.stats import loguniform
from sklearn.model_selection import KFold, train_test_split
from tensorflow import keras

from models import customized_metrics as cm
from models import multithreading as mth


def create_random_network(
    m: int,
    n_classes: int,
    metrics: str = "accuracy",
    layers: tuple = (1, 5),
    n: tuple = (10, 20),
    activation: tuple = ("relu", "sigmoid"),
    lr: tuple = (1e-5, 1e-3),
    optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam,
    loss: str = "categorical_crossentropy",
) -> (keras.models.Sequential, keras.optimizers.Optimizer, str):
    """Return a deep neural network model with pseudo-random hyperparameters according to its arguments, each time it is called.

    :param m: Number of input variables which will enter in the neural network
    :type m: int
    :param n_classes: Number of classes to be classificated
    :type n_classes: int
    :param metrics: String with the metric to evaluate and compare the different random networks, defaults to "accuracy"
    :type metrics: str, optional
    :param layers: Tuple with the number of layers of the neural network as hyperparameter, defaults to (1, 5)
    :type layers: tuple, optional
    :param n: Tuple with the min and max number of neurons per layer, defaults to (10,20)
    :type n: tuple, optional
    :param activation: Tuple with the activation functions as hyperparameters, defaults to ("relu","sigmoid")
    :type activation: tuple, optional
    :param lr: Tuple with the min and max values for the learning rate as hyperparameter, defaults to (1e-5, 1e-3)
    :type lr: tuple, optional
    :param optimizer: A keras Optimizer to be used to train the neural network, defaults to keras.optimizers.Adam
    :type optimizer: keras.optimizers.Optimizer
    :param loss: String with the lossing function to be used to measure the error, defaults to "categorical_crossentropy"
    :type loss: str
    :return: A tuple formed by the keras.models.Sequential, keras.optimizers.Optimizer, and the lossing function
    :rtype: tuple
    """
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
    for _ in range(random.randint(layers[0], layers[1])):
        model.add(keras.layers.Dense(random.randint(n[0], n[1]), activation=random.choice(activation)))
    model.add(keras.layers.Dense(n_classes, activation="softmax"))
    # Choosing a learning rate from a logarithmic uniform distrbution
    optimizer = optimizer(learning_rate=round(loguniform.rvs(lr[0], lr[1]), int(abs(math.log(lr[0], 10)))))
    # Define some characteristics for the training process
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model, optimizer, loss


def get_hyperparams(model: keras.models.Sequential) -> (pd.DataFrame, pd.DataFrame):
    """Return the hyperparams of the model passed as an argument.

    :param model: A keras.models.Sequential neural network model
    :type model: keras.models.Sequential
    :return: A tuple with a pandas Dataframe with the hyperparameters of the neural network and a pandas Dataframe with the parameters of the optimezer used in the model
    :rtype: tuple
    """
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


def split_history_metrics(historial: list) -> dict:
    """Transform the given list which contains the history results to a dictionary with the metrics organized.

    :param historial: A list with the historical results of the trained model
    :type historial: list
    :return: A dictionary with the metrics obtained from the trained model
    :rtype: dict
    """
    d = {}
    for k in historial[0].keys():
        d[k] = [d[k] for d in historial]
    return d


def optimize_DNN(
    X: np.ndarray,
    t: np.ndarray,
    kfolds: int = 10,
    train_size: float = 0.8,
    trials: int = 5,
    epochs: int = 50,
    batch_size: int = 40,
    metrics: tuple = ("accuracy", "Recall", cm.specificity, "Precision", cm.f1_score, "AUC"),
) -> tuple:
    """Train the current model using cross validation and register its score comparing with new ones in each trial.

    :param X: Characteristic matrix numpy array of the dataset which will be evaluated
    :type X: np.ndarray
    :param t: Vector labels numpy array of the dataset which will be evaluated
    :type t: np.ndarray
    :param kfolds: Number of folds for the cross validation algorithm, defaults to 10
    :type kfolds: int, optional
    :param train_size: % of the data to be splitted into train and test values, defaults to 0.8
    :type train_size: float, optional
    :param trials: Number of trials used to generate random neural networks with different hyperparameters, defaults to 5
    :type trials: int, optional
    :param epochs: Number of maximum iterations allow to converge the algorithm, defaults to 50
    :type epochs: int, optional
    :param batch_size: Size of the batch used to calculate lossing function, defaults to 40
    :type batch_size: int, optional
    :param metrics: Tuple with the metrics to compare and evaluate the differene neural networks, defaults to ("accuracy","Recall",cm.specificity,"Precision",cm.f1_score,"AUC")
    :type metrics: tuple, optional
    :return: A tuple containing a tuple with the model and the history of the training, and another tuple with the X_test and t_test arrays to evaluate the model
    :rtype: tuple
    """
    # Its needed the accuracy metric, if it is not passed it will be auto-included
    if "accuracy" not in metrics:
        metrics.append("accuracy")

    _, m = X.shape
    n_classes = len(np.unique(t))
    cv = KFold(kfolds)
    last_mean = 0

    # Split the data into train and test sets. Note that test set will be reserved for evaluate the best model later with data unseen previously for it
    X_train_set, X_test, t_train_set, t_test = train_test_split(X, t, train_size=train_size)

    # Loop for different trials or models to train in order to find the best
    for row in range(trials):

        model_aux, optimizer, loss = create_random_network(m, n_classes, metrics=metrics)
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

def optimize_DNN_multithread(
    X: np.ndarray,
    t: np.ndarray,
    kfolds: int = 10,
    train_size: float = 0.8,
    trials: int = 5,
    epochs: int = 50,
    batch_size: int = 40,
    metrics: tuple = ("accuracy", "Recall", cm.specificity, "Precision", cm.f1_score, "AUC"),
) -> tuple:
    """Train the current model using cross validation and multithreading and register its score comparing with new ones in each trial.

    :param X: Characteristic matrix numpy array of the dataset which will be evaluated
    :type X: np.ndarray
    :param t: Vector labels numpy array of the dataset which will be evaluated
    :type t: np.ndarray
    :param kfolds: Number of folds for the cross validation algorithm, defaults to 10
    :type kfolds: int, optional
    :param train_size: % of the data to be splitted into train and test values, defaults to 0.8
    :type train_size: float, optional
    :param trials: Number of trials used to generate random neural networks with different hyperparameters, defaults to 5
    :type trials: int, optional
    :param epochs: Number of maximum iterations allow to converge the algorithm, defaults to 50
    :type epochs: int, optional
    :param batch_size: Size of the batch used to calculate lossing function, defaults to 40
    :type batch_size: int, optional
    :param metrics: Tuple with the metrics to compare and evaluate the differene neural networks, defaults to ("accuracy","Recall",cm.specificity,"Precision",cm.f1_score,"AUC")
    :type metrics: tuple, optional
    :return: A tuple containing a tuple with the model and the history of the training, and another tuple with the X_test and t_test arrays to evaluate the model
    :rtype: tuple
    """
    # Its needed the accuracy metric, if it is not passed it will be auto-included
    if "accuracy" not in metrics:
        metrics.append("accuracy")

    _, m = X.shape
    n_classes = len(np.unique(t))
    cv = KFold(kfolds)
    last_mean = 0

    # Split the data into train and test sets. Note that test set will be reserved for evaluate the best model later with data unseen previously for it
    X_train_set, X_test, t_train_set, t_test = train_test_split(X, t, train_size=train_size)

    # Loop for different trials or models to train in order to find the best
    threads_dict = {}
    for row in range(trials):

        model_aux, optimizer, loss = create_random_network(m, n_classes, metrics=metrics)
        params, comp = get_hyperparams(model_aux)

        print(f"\n***Trial {row+1} hyperparameters***", end="\n\n")
        print(params, "\n", comp)
        # Lists to store threads to manage them
        threads_dict[row]=[]
        # Loop that manage cross validation using training set
        for train_index, test_index in cv.split(X_train_set):
            # This sentence carefully clones the untrained model in each fold in order to avoid unwanted learning weights between them
            model = keras.models.clone_model(model_aux)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            X_train, X_val = X_train_set[train_index], X_train_set[test_index]
            t_train, t_val = t_train_set[train_index], t_train_set[test_index]

            t_train = to_categorical(t_train, num_classes=n_classes)
            t_val = to_categorical(t_val, num_classes=n_classes)

            # Training of the model using multithreading
            thread = mth.DnnThread(X_train,t_train, X_val, t_val, model, epochs, batch_size)
            thread.start()
            threads_dict[row].append(thread)

    # Save the train and test results,of the current model
    for row in range(trials):
        # Lists to store historical data and means per fold
        historial = []
        mean_folds = []
        for thread in threads_dict[row]:
            thread.join()
            historial.append(thread.history.history)
            mean = np.mean(thread.history.history["val_accuracy"])
            mean_folds.append(mean)
            print(f"\nKFold{threads_dict[row].index(thread)+1} --- Current score: {mean}")

        # Criterion to choose the best model with the highest accuracy score in validation
        means = np.mean(mean_folds)
        history_metrics = split_history_metrics(historial)
        print(
            f"\n\tMin. Train score: {min(np.concatenate(history_metrics['accuracy']))} | Max. Train score: {max(np.concatenate(history_metrics['accuracy']))}"
        )
        if means > last_mean:
            # Save the model and its historial results
            best_model = (thread.model, history_metrics)
            test_set = (X_test, t_test)
            last_mean = means
    return best_model, test_set
