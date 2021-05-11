"""Module to optimize hyperparameters and get the best Models."""

import math
import random

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from models import multithreading as mth
import timeit


class HpModels:
    def __init__(self):
        self.best_models = dict()
        self.threads_dict = dict()

    @staticmethod
    def select_models(LR: bool = True, LDA: bool = True, KNN: bool = True, SVC: bool = True) -> list:
        """Create a list which contains the tags of the models that are going to be tested for the best.

        :param LR: If True, appends the Logistic Regression model tag, defaults to True
        :type LR: bool, optional
        :param LDA: If True, appends the Linear Discriminant Analysis model tag, defaults to True
        :type LDA: bool, optional
        :param KNN: If True, appends the K-nearest neighbors model tag, defaults to True
        :type KNN: bool, optional
        :param SVC: If True, appends the Support Vector Machine model tag, defaults to True
        :type SVC: bool, optional
        :return: A list with the selected model tags.
        :rtype: list
        """
        models = []
        if LR:
            models.append("LR")
        if LDA:
            models.append("LDA")
        if KNN:
            models.append("KNN")
        if SVC:
            models.append("SVC")
        elif models is []:
            models.append("LR")
        return models

    @staticmethod
    def create_random_LR(
        penalty: tuple = ("none", "l1", "l2", "elasticnet"),
        solver: tuple = ("liblinear", "sag", "saga", "newton-cg", "lbfgs"),
        max_iter: tuple = (100, 1000),
        C: tuple = (1e-5, 10),
    ) -> LogisticRegression:
        """Create a Logistic Regression model using random hyperparameters.

        :param penalty: Tuple with the penalty hyperparameters, defaults to ("none", "l1", "l2", "elasticnet")
        :type penalty: tuple, optional
        :param solver: Tuple with the solver hyperparameters, defaults to ("liblinear", "sag", "saga", "newton-cg", "lbfgs")
        :type solver: tuple, optional
        :param max_iter: Tuple with the max_iter hyperparameters, defaults to (100, 1000)
        :type max_iter: tuple, optional
        :param C: Tuple with the C hyperparameters, defaults to (1e-5, 10)
        :type C: tuple, optional
        :return: A LogisticRegresion model with the random hyperparameters.
        :rtype: LogisticRegression
        """
        model = LogisticRegression(
            penalty=random.choice(penalty),
            solver=random.choice(solver),
            max_iter=round(loguniform.rvs(max_iter[0], max_iter[1])),
            C=round(loguniform.rvs(C[0], C[1]), int(abs(math.log(C[0], 10)))),
        )
        return model

    @staticmethod
    def create_random_LDA(
        solver: tuple = ("svd", "lsqr", "eigen"),
        shrinkage: tuple = ("auto", round(random.uniform(1e-5, 1), 5), "none"),
        tol: tuple = (1e-5, 1e-3),
    ) -> LinearDiscriminantAnalysis:
        """Create a Linear Discriminant model using random hyperparameters.

        :param solver: Tuple with the solver hyperparameters, defaults to ("svd", "lsqr", "eigen")
        :type solver: tuple, optional
        :param shrinkage: Tuple with the shrinkage hyperparameters, defaults to ("auto", round(random.uniform(1e-5, 1), 5), "none")
        :type shrinkage: tuple, optional
        :param tol: Tuple with the tolerance hyperparameters, defaults to (1e-5, 1e-3)
        :type tol: tuple, optional
        :return: A LinearDiscriminantAnalysis model with the random hyperparameters.
        :rtype: LinearDiscriminantAnalysis
        """
        model = LinearDiscriminantAnalysis(
            solver=random.choice(solver),
            shrinkage=random.choice(shrinkage),
            tol=round(loguniform.rvs(tol[0], tol[1]), int(abs(math.log(tol[0], 10)))),
        )
        return model

    @staticmethod
    def create_random_KNN(
        n_neighbors: tuple = (5, 200),
        weights: tuple = ("uniform", "distance"),
        algorithm: tuple = ("auto", "ball_tree", "kd_tree", "brute"),
        leaf_size: tuple = (15, 150),
    ) -> KNeighborsClassifier:
        """Create a K-nearest neighbors model using random hyperparameters.

        :param n_neighbors: Tuple with the number of neighbors hyperparameters, defaults to (5, 200)
        :type n_neighbors: tuple, optional
        :param weights: Tuple with the weights hyperparameters, defaults to ("uniform", "distance")
        :type weights: tuple, optional
        :param algorithm: Tuple with the algorithms to use as hyperparameters, defaults to ("auto", "ball_tree", "kd_tree", "brute")
        :type algorithm: tuple, optional
        :param leaf_size: Tuple with the leaf_size hyperparameters, defaults to (15, 150)
        :type leaf_size: tuple, optional
        :return: A KNeighborsClassifier model with the random hyperparameters.
        :rtype: KNeighborsClassifier
        """
        model = KNeighborsClassifier(
            n_neighbors=random.randint(n_neighbors[0], n_neighbors[1]),
            weights=random.choice(weights),
            algorithm=random.choice(algorithm),
            leaf_size=random.randint(leaf_size[0], leaf_size[1]),
        )
        return model

    @staticmethod
    def create_random_SVC(
        kernel: tuple = ("linear", "poly", "rbf", "sigmoid", "precomputed"),
        gamma: tuple = ("scale", "auto"),
        C: tuple = (1e-5, 10),
        decision_function_shape: tuple = ("ovo", "ovr"),
        probability: bool = True,
    ) -> SVC:
        """Create a Support Vector Machine model using random hyperparameters.

        :param kernel: Tuple with the kernel hyperparameters, defaults to ("linear", "poly", "rbf", "sigmoid", "precomputed")
        :type kernel: tuple, optional
        :param gamma: Tuple with the gamma hyperparameters, defaults to ("scale", "auto")
        :type gamma: tuple, optional
        :param C: Tuple with C hyperparameters, defaults to (1e-5, 10)
        :type C: tuple, optional
        :param decision_function_shape: Tuple with the decision_function_shape hyperparameters, defaults to ("ovo", "ovr")
        :type decision_function_shape: tuple, optional
        :param probability: Boolean to enable probability prediction, defaults to True
        :type probability: tuple, optional
        :return: A SVC model with the random hyperparameters.
        :rtype: SVC
        """
        model = SVC(
            kernel=random.choice(kernel),
            gamma=random.choice(gamma),
            C=round(loguniform.rvs(C[0], C[1]), int(abs(math.log(C[0], 10)))),
            decision_function_shape=random.choice(decision_function_shape),
            probability=probability,
        )
        return model

    @staticmethod
    def get_hyperparameters(model: BaseEstimator, tag: str) -> pd.DataFrame:
        """Return a pandas DataFrame with the hyperparameters of the given model.

        :param model: Model to get hyperparameters
        :type model: BaseEstimator
        :param tag: Name of the model to get hyperparameters
        :type tag: str
        :return: A pandas DataFrame with the hyperparameters used in the given model.
        :rtype: pd.Dataframe
        """
        hp = dict()
        all_hp = model.get_params()
        if tag == "LR":
            for i in all_hp:
                if i in ["penalty", "solver", "max_iter", "C"]:
                    hp[i] = all_hp[i]
        elif tag == "LDA":
            for i in all_hp:
                if i in ["solver", "shrinkage", "tol"]:
                    hp[i] = all_hp[i]
        elif tag == "KNN":
            for i in all_hp:
                if i in ["n_neighbors", "weights", "algorithm, leaf_size"]:
                    hp[i] = all_hp[i]
        elif tag == "SVC":
            for i in all_hp:
                if i in [
                    "kernel",
                    "gamma",
                    "C",
                    "decision_function_shape",
                    "probability",
                ]:
                    hp[i] = all_hp[i]

        return pd.DataFrame.from_records([hp], index=["hyperparams"])

    @staticmethod
    def build_macro_model(model: BaseEstimator, scaler: TransformerMixin = StandardScaler()) -> Pipeline:
        """Create a macro model pipeline using the given scaler.

        :param model: A model to use in the macro model pipeline
        :type model: BaseEstimator
        :param scaler: An scaler to preprocess de model, defaults to StandardScaler()
        :type scaler: TransformerMixin, optional
        :return: A macro model pipeline with the scaler preprocessing feature.
        :rtype: Pipeline
        """
        macro_model = make_pipeline(scaler, model)
        return macro_model

    @staticmethod
    def random_model(tag: str) -> BaseEstimator:
        """Wrapp the functions of create_random model.

        :param tag: String tag with the choosen model to be created
        :type tag: str
        :return: The selected model with random hyperparameters.
        :rtype: BaseEstimator
        """
        if tag == "LR":
            model = HpModels.create_random_LR()
        if tag == "LDA":
            model = HpModels.create_random_LDA()
        if tag == "KNN":
            model = HpModels.create_random_KNN()
        if tag == "SVC":
            model = HpModels.create_random_SVC()
        return model

    def perform_optimizing_model(
        self,
        tag,
        X: np.ndarray,
        t: np.ndarray,
        train_size: float = 0.85,
        scoring: dict = {
            "accuracy": "accuracy",
            "recall": "recall",
            "specificity": make_scorer(recall_score, pos_label=0),
            "precision": "precision",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        cv: int = 10,
        trials: int = 25,
    ):
        """Optimize hyperparameters of the given model and returns the best of them using cross validation.

        :param tag:  model to be evaluated.
        :type tag: str
        :param X: Characteristic matrix numpy array of the dataset which will be evaluated
        :type X: np.ndarray
        :param t: Vector labels numpy array of the dataset which will be evaluated
        :type t: np.ndarray
        :param train_size: % of the data to be splitted into train and test values, defaults to 0.85
        :type train_size: float, optional
        :param scoring: A dictionary with the wanted metrics to compare and evaluate the different models, defaults to { "accuracy": "accuracy", "recall": "recall", "specificity": make_scorer(recall_score, pos_label=0), "precision": "precision", "f1": "f1", "roc_auc": "roc_auc", }
        :type scoring: dict, optional
        :param cv: Number of folds for the cross validation algorithm, defaults to 10
        :type cv: int, optional
        :param trials: Number of trials used to generate random models with different hyperparameters, defaults to 25
        :type trials: int, optional
        :return: A dictionary with the scores and models trained using cross validation
        :rtype: dict
        """
        if "accuracy" not in scoring:
            scoring["accuracy"] = "accuracy"

        X_train, _, t_train, _ = train_test_split(X, t, train_size=train_size)

        last_accuracy = 0
        print(f"\n***Optimizing {tag} hyperparameters***")
        for _ in range(trials):
            current_model = HpModels.random_model(tag)
            macro_model = HpModels.build_macro_model(current_model)
            scores = cross_validate(
                macro_model,
                X_train,
                t_train,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
            )
            mean_accuracy = np.mean(scores["test_accuracy"])
            if mean_accuracy > last_accuracy:
                self.best_models[tag] = (scores, macro_model)
                last_accuracy = mean_accuracy
                print("\nBest accuracy so far: ", last_accuracy)
            print(".", end="")
        print(
            "\nScore:",
            round(np.mean(last_accuracy), 4),
            "-",
            macro_model.steps[1][1],
        )

    def optimizing_models(self, models: list, *args, **kwargs) -> dict:

        time_models = dict()
        for tag in models:
            time_models[tag] = round(timeit("perform_optimizing_model(tag, *args, **kwargs)"), 4)

        return self.best_models, pd.DataFrame(time_models, index=["Time Models"])

    def perform_optimizing_models_multithread(
        self,
        tag,
        X: np.ndarray,
        t: np.ndarray,
        train_size: float = 0.85,
        scoring: dict = {
            "accuracy": "accuracy",
            "recall": "recall",
            "specificity": make_scorer(recall_score, pos_label=0),
            "precision": "precision",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        cv: int = 10,
        trials: int = 25,
    ) -> dict:
        """Optimize hyperparameters of the given model and returns the best of them using cross validation and multithreading.

        :param tag: model to be evaluated.
        :type tag: str
        :param X: Characteristic matrix numpy array of the dataset which will be evaluated
        :type X: np.ndarray
        :param t: Vector labels numpy array of the dataset which will be evaluated
        :type t: np.ndarray
        :param train_size: % of the data to be splitted into train and test values, defaults to 0.85
        :type train_size: float, optional
        :param scoring: A dictionary with the wanted metrics to compare and evaluate the different models, defaults to { "accuracy": "accuracy", "recall": "recall", "specificity": make_scorer(recall_score, pos_label=0), "precision": "precision", "f1": "f1", "roc_auc": "roc_auc", }
        :type scoring: dict, optional
        :param cv: Number of folds for the cross validation algorithm, defaults to 10
        :type cv: int, optional
        :param trials: Number of trials used to generate random models with different hyperparameters, defaults to 25
        :type trials: int, optional
        :return: A dictionary with the scores and models trained using cross validation
        :rtype: dict
        """
        if "accuracy" not in scoring:
            scoring["accuracy"] = "accuracy"

        X_train, _, t_train, _ = train_test_split(X, t, train_size=train_size)

        print(f"\n***Optimizing {tag} hyperparameters***")
        self.threads_dict[tag] = []
        for _ in range(trials):
            current_model = HpModels.random_model(tag)
            macro_model = HpModels.build_macro_model(current_model)
            model_thread = mth.ModelThread(X_train, np.ravel(t_train), macro_model, cv, scoring)
            model_thread.start()
            self.threads_dict[tag].append(model_thread)

        last_accuracy = 0
        print(f"\n***Cross validation results for {tag}***")
        for thread in self.threads_dict[tag]:
            thread.join()
            mean_accuracy = np.mean(thread.scores["test_accuracy"])
            if mean_accuracy > last_accuracy:
                self.best_models[tag] = (thread.scores, thread.model)
                last_accuracy = mean_accuracy
                print("\nBest accuracy so far: ", last_accuracy)
        print(
            "\nScore:",
            round(np.mean(last_accuracy), 4),
            "-",
            self.best_models[tag][1].steps[1][1],
        )

    def optimizing_models_multithread(self, models: list, *args, **kwargs):

        time_models = dict()
        for tag in models:
            time_models[tag] = round(timeit("self.perform_optimizing_models_multithread(tag, *args, **kwargs)"))

        return self.best_models, pd.DataFrame(time_models, index=["Time Models"])
