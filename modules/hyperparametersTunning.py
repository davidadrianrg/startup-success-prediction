"""Module to optimize hyperparameters and get the best Models"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import warnings
from scipy.stats import loguniform
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate


def warn(*args, **kwargs):
    pass


def select_models(LR=True, LDA=True, KNN=True, SVC=True):
    # This is a list which contains the tags of the models that are going to be tested for the best
    models = []
    if LR:
        models.append("LR")
    if LDA:
        models.append("LDA")
    if KNN:
        models.append("KNN")
    if SVC:
        models.append("SVC")
    return models


def create_random_LR(
    penalty=["none", "l1", "l2", "elasticnet"],
    solver=["liblinear", "sag", "saga", "newton-cg", "lbfgs"],
    max_iter=[100, 1000],
    C=[1e-5, 10],
):
    model = LogisticRegression(
        penalty=random.choice(penalty),
        solver=random.choice(solver),
        max_iter=round(loguniform.rvs(max_iter[0], max_iter[1])),
        C=round(loguniform.rvs(C[0], C[1]), int(abs(math.log(C[0], 10)))),
    )
    return model


def create_random_LDA(
    solver=["svd", "lsqr", "eigen"],
    shrinkage=["auto", round(random.uniform(1e-5, 1), 5), "none"],
    tol=[1e-5, 1e-3],
):
    model = LinearDiscriminantAnalysis(
        solver=random.choice(solver),
        shrinkage=random.choice(shrinkage),
        tol=round(
            loguniform.rvs(tol[0], tol[1]), int(abs(math.log(tol[0], 10)))
        ),
    )
    return model


def create_random_KNN(
    n_neighbors=[5, 200],
    weights=["uniform", "distance"],
    algorithm=["auto", "ball_tree", "kd_tree", "brute"],
    leaf_size=[15, 150],
):
    model = KNeighborsClassifier(
        n_neighbors=random.randint(n_neighbors[0], n_neighbors[1]),
        weights=random.choice(weights),
        algorithm=random.choice(algorithm),
        leaf_size=random.randint(leaf_size[0], leaf_size[1]),
    )
    return model


def create_random_SVC(
    kernel=["linear", "poly", "rbf", "sigmoid", "precomputed"],
    gamma=["scale", "auto"],
    C=[1e-5, 10],
    decision_function_shape=["ovo", "ovr"],
):
    model = SVC(
        kernel=random.choice(kernel),
        gamma=random.choice(gamma),
        C=round(loguniform.rvs(C[0], C[1]), int(abs(math.log(C[0], 10)))),
        decision_function_shape=random.choice(decision_function_shape),
    )
    return model


def build_macro_model(model, scaler=StandardScaler()):
    macro_model = make_pipeline(scaler, model)
    return macro_model


def random_model(tag):
    if tag == "LR":
        model = create_random_LR()
    if tag == "LDA":
        model = create_random_LDA()
    if tag == "KNN":
        model = create_random_KNN()
    if tag == "SVC":
        model = create_random_SVC()
    return model


def optimizing_models(
    models,
    X,
    t,
    test_size=0.15,
    scoring={"accuracy": "accuracy"},
    cv=20,
    trials=25,
):

    if "accuracy" not in scoring:
        scoring["accuracy"] = "accuracy"

    warnings.warn = warn
    best_models = dict()
    X_train, X_test, t_train, t_test = train_test_split(
        X, t, test_size=test_size
    )
    for tag in models:
        last_accuracy = 0
        print(f"\n***Optimizing {tag} hyperparameters***")
        for i in range(trials):
            current_model = random_model(tag)
            macro_model = build_macro_model(current_model)
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
                best_models[tag] = (
                    scores["train_accuracy"],
                    scores["test_accuracy"],
                    macro_model,
                )
                last_accuracy = mean_accuracy
                print("\nBest accuracy so far: ", last_accuracy)
            print(".", end="")
        print(
            "\nScore:",
            round(np.mean(best_models[tag][1]), 4),
            "-",
            best_models[tag][2].steps[1][1],
        )
    return best_models


def plot_best_model(best_models, tag="LR"):
    """Plots the validation curve of the model using its historial results"""
    plt.plot(best_models[tag][0])
    plt.plot(best_models[tag][1])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration (cv)")
    plt.legend(["Trainning", "Test"], loc="lower right")
    plt.show()


"""Example of code
X = pd.read_csv("X.csv")
X = X.drop(["Unnamed: 0"], axis=1).values
t = pd.read_csv("t.csv")
t = t["labels"].values
n, m = X.shape
n_classes = len(np.unique(t))

models = select_models()
best = optimizing_models(models, X, t)
plot_best_model(best)"""
