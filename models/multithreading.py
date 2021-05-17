"""Multithreading module with different Thread classes to implement training models and DNN with cross validation."""

from threading import Thread

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from keras import models


class ModelThread(Thread):
    """Thread class to implement the training model using cross validation."""

    def __init__(self, X_train:np.ndarray, t_train:np.ndarray, model:BaseEstimator, cv:int, scoring:dict, **kwargs):
        """Create an instance of the ModelThread class and set their attributes.

        :param X_train: Characteristic matrix numpy array of the dataset which will be evaluated
        :type X_train: np.ndarray
        :param t_train: Vector labels numpy array of the dataset which will be evaluated
        :type t_train: np.ndarray
        :param model: Estimator to be trained
        :type model: BaseEstimator
        :param cv: Number of folds for the cross validation algorithm
        :type cv: int
        :param scoring: A dictionary with the wanted metrics to compare and evaluate the different models
        :type scoring: dict
        """
        super().__init__(**kwargs)
        self.X_train = X_train
        self.t_train = t_train
        self.model = model
        self.cv = cv
        self.scoring = scoring
        self.scores = None

    def run(self):
        """Run the cv_multithread_model function in an isolated thread."""
        self.cv_multithread_model()

    def cv_multithread_model(self):
        """Perform the cross validation using the class attributes."""
        self.scores = cross_validate(
            self.model,
            self.X_train,
            self.t_train,
            cv=self.cv,
            scoring=self.scoring,
            return_train_score=True,
        )


class DnnThread(Thread):
    """Thread class to implement the training dnn fitting."""

    def __init__(self, X_train:np.ndarray, t_train:np.ndarray, X_val:np.ndarray, t_val:np.ndarray, model:models.Sequential, epochs:int, batch_size:int, **kwargs):
        """Create an instance of the DnnThread class and set their attributes.

        :param X_train: Characteristic matrix numpy array of the dataset which will be used for the training
        :type X_train: np.ndarray
        :param t_train: Vector labels numpy array of the dataset which will be used for the training
        :type t_train: np.ndarray
        :param X_val: Characteristic matrix numpy array of the dataset which will be evaluated
        :type X_val: np.ndarray
        :param t_val: Vector labels numpy array of the dataset which will be evaluated
        :type t_val: np.ndarray
        :param model: Deep Neural Network to be trained
        :type model: models.Sequential
        :param epochs: Number of maximum iterations allow to converge the algorithm
        :type epochs: int
        :param batch_size: Size of the batch used to calculate lossing function
        :type batch_size: int
        """
        super().__init__(**kwargs)
        self.X_train = X_train
        self.t_train = t_train
        self.X_val = X_val
        self.t_val = t_val
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.history = None

    def run(self):
        """Run the fit_dnn function in an isolated thread."""
        self.fit_dnn()

    def fit_dnn(self):
        """Train the model using the class attributes."""
        self.history = self.model.fit(
            self.X_train,
            self.t_train,
            validation_data=(self.X_val, self.t_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )
