"""Module to implement customized metrics for the neural networks."""

import numpy as np
from keras import backend as K


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Define the specifity metric to implement in keras models.

    :param y_true: Numpy array containing the true values
    :type y_true: np.ndarray
    :param y_pred: Numpy array containing the predicted values
    :type y_pred: np.ndarray
    :return: A float number with the value of the metric calculated
    :rtype: float
    """
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Define the f1_score metric to implement in keras models.

    :param y_true: Numpy array containing the true values
    :type y_true: np.ndarray
    :param y_pred: Numpy array containing the predicted values
    :type y_pred: np.ndarray
    :return: A float number with the value of the metric calculated
    :rtype: float
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    precision = true_positives / predicted_positives
    recall = true_positives / possible_positives
    return 2 * ((precision * recall) / (precision + recall))
