"""Wrapper module to make more usable th hyperparametersTunning and hyperparametersDNN modules."""

import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from tensorflow.keras.models import Sequential

from models import customized_metrics as cm
from models.hyperparametersDNN import HpDNN
from models.hyperparametersTunning import HpModels


def get_best_models(
    X: np.ndarray,
    t: np.ndarray,
    models: list = HpModels.select_models(),
    cv: int = 10,
    train_size: float = 0.8,
    scoring: dict = {
        "accuracy": "accuracy",
        "recall": "recall",
        "specificity": make_scorer(recall_score, pos_label=0),
        "precision": "precision",
        "f1": "f1",
        "AUC": "roc_auc",
    },
    trials: int = 2,
    epochs: int = 50,
    batch_size: int = 40,
    metrics: tuple = (
        "accuracy",
        "Recall",
        cm.specificity,
        "Precision",
        cm.f1_score,
        "AUC",
    ),
    is_mthreading: bool = False,
) -> tuple:
    """Return the best models generated with random hyperparameters using the arguments as training hyperparameters.

    :param X: Characteristic matrix numpy array of the dataset which will be evaluated
    :type X: np.ndarray
    :param t: Vector labels numpy array of the dataset which will be evaluated
    :type t: np.ndarray
    :param models: List with the selected model tags, defaults to HpModels.select_models()
    :type models: list, optional
    :param cv: Number of folds for the cross validation algorithm, defaults to 10
    :type cv: int, optional
    :param train_size: % of the data to be splitted into train and test values, defaults to 0.8
    :type train_size: float, optional
    :param scoring: A dictionary with the wanted metrics to compare and evaluate the different models, defaults to { "accuracy": "accuracy", "recall": "recall", "specificity": make_scorer(recall_score, pos_label=0), "precision": "precision", "f1": "f1", "AUC": "roc_auc", }
    :type scoring: dict, optional
    :param trials: Number of trials used to generate random models with different hyperparameters, defaults to 2
    :type trials: int, optional
    :param epochs: Number of maximum iterations allow to converge the algorithm, defaults to 50
    :type epochs: int, optional
    :param batch_size: Size of the batch used to calculate lossing function, defaults to 40
    :type batch_size: int, optional
    :param metrics: Tuple with the metrics to compare and evaluate the differene neural networks, defaults to ( "accuracy", "Recall", cm.specificity, "Precision", cm.f1_score, "AUC", )
    :type metrics: tuple, optional
    :param is_mthreading: Boolean to enable/disable multithreading during the training of the models, defaults to False.
    :type is_mthreading: bool, optional
    :return: A tuple containing a tuple with the best_models and train_size and a tuple with the bestDNN model
    :rtype: tuple
    """
    if is_mthreading:
        best_models, _ = HpModels().optimizing_models_multithread(
            models,
            X,
            t,
            cv=cv,
            train_size=train_size,
            scoring=scoring,
            trials=trials,
        )
        best_DNN, _ = HpDNN().optimize_DNN_multithread(
            X,
            t,
            kfolds=cv,
            train_size=train_size,
            trials=trials,
            epochs=epochs,
            batch_size=batch_size,
            metrics=metrics,
        )
    else:
        best_models, _ = HpModels().optimizing_models(
            models,
            X,
            t,
            cv=cv,
            train_size=train_size,
            scoring=scoring,
            trials=trials,
        )
        best_DNN, _ = HpDNN().optimize_DNN(
            X,
            t,
            kfolds=cv,
            train_size=train_size,
            trials=trials,
            epochs=epochs,
            batch_size=batch_size,
            metrics=metrics,
        )
    return (best_models, train_size), best_DNN


def get_results(best_models: tuple, best_DNN: tuple) -> pd.DataFrame:
    """Get the best models and best neural network and returns a pandas Dataframe with the metric results.

    :param best_models: Tuple with the best models
    :type best_models: tuple
    :param best_DNN: Tuple with the best neural network
    :type best_DNN: tuple
    :return: A pandas Dataframe with the metric results
    :rtype: pd.DataFrame
    """
    best_models = best_models[0]
    best_DNN = best_DNN[0]
    metrics = best_DNN[0].metrics_names
    DNN_means = dict()
    for tag in best_DNN[1]:
        DNN_means[tag] = []
        for i in best_DNN[1][tag]:
            DNN_means[tag].append(np.mean(i))

    tags = list(best_models.keys())
    data = dict()

    for metric in metrics:
        if "test_" + metric in best_models[tags[0]][0]:

            for tag in tags:
                data[tag + "_train_" + metric] = best_models[tag][0][
                    "train_" + metric
                ]
                data[tag + "_val_" + metric] = best_models[tag][0][
                    "test_" + metric
                ]

            data["DNN_train_" + metric] = DNN_means[metric]
            data["DNN_val_" + metric] = DNN_means["val_" + metric]
    results = pd.DataFrame(data, index=np.arange(1, len(DNN_means[metric]) + 1))
    results.index.name = "Folds"
    return results


def analize_performance_DNN(
    best_DNN: tuple,
) -> tuple:
    """Get the best DNN model and returns the results and the numpy arrays with the test values and predicted values.

    :param best_DNN: Tuple with the best neural network
    :type best_DNN: tuple
    :return: A tuple with the numpy arrays with the test values and predicted values
    :rtype: tuple
    """
    X_test, t_test = best_DNN[1]
    n_classes = len(np.unique(t_test))
    t_test_bin = to_categorical(t_test, num_classes=n_classes)
    results = pd.DataFrame(columns=best_DNN[0][0].metrics_names)
    results.loc["DNN_test"] = best_DNN[0][0].evaluate(
        X_test, t_test_bin, batch_size=None, verbose=0
    )
    y_pred_proba = best_DNN[0][0].predict(X_test)
    y_pred = np.ndarray.tolist(y_pred_proba)
    y_pred_len = len(y_pred)
    y_out = [round(y_pred[i].index(max(y_pred[i]))) for i in range(y_pred_len)]
    return results, X_test, t_test, y_out, y_pred_proba


def analize_performance_models(
    best_models: tuple, X: np.ndarray, t: np.ndarray
) -> tuple:
    """Get the best models and dataset values and returns the models trained and the numpy arrays with the test values and predicted values.

    :param best_models: Tuple with the best models
    :type best_models: tuple
    :param X: Characteristic matrix numpy array of the dataset which will be evaluated
    :type X: np.ndarray
    :param t: Vector labels numpy array of the dataset which will be evaluated
    :type t: np.ndarray
    :return: A tuple with the models trained and the numpy arrays with the test values and predicted values
    :rtype: tuple
    """
    train_size = best_models[1]
    best_models = best_models[0]
    y_pred = dict()
    y_score = dict()

    X_train, X_test, t_train, t_test = train_test_split(
        X, t, train_size=train_size
    )
    for model in best_models:
        best_models[model][1].fit(X_train, t_train)
        y_pred[model] = best_models[model][1].predict(X_test)
        y_score[model] = best_models[model][1].predict_proba(X_test)

    return best_models, X_test, t_test, y_pred, y_score


def get_hyperparams(model: BaseEstimator, tag: str) -> pd.DataFrame:
    """Wrapp the get_hyperparameters function in hyperparametersTunning module.

    Return a pandas DataFrame with the hyperparameters of the given model.

    :param model: Model to get hyperparameters
    :type model: BaseEstimator
    :param tag: Name of the model to get hyperparameters
    :type tag: str
    :return: A pandas DataFrame with the hyperparameters used in the given model.
    :rtype: pd.DataFrame
    """
    return HpModels.get_hyperparameters(model, tag)


def get_hyperparams_DNN(
    model: Sequential,
) -> tuple((pd.DataFrame, pd.DataFrame)):
    """Wrapp the get_hyperparams function in hyperparametersDNN module.

    Return the hyperparams of the model passed as an argument.

    :param model: A keras.models.Sequential neural network model
    :type model: keras.models.Sequential
    :return: A tuple with a pandas Dataframe with the hyperparameters of the neural network and a pandas Dataframe with the parameters of the optimezer used in the model
    :rtype: tuple
    """
    return HpDNN.get_hyperparams(model)
