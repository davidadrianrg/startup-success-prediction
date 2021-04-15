from models import hyperparametersTunning as hpTune
from models import hyperparametersDNN as hpDNN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from models import customized_metrics as cm
from tensorflow.math import confusion_matrix
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from sklearn.metrics import (
    make_scorer,
    recall_score,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    plot_confusion_matrix,
    classification_report,
)
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison


def get_best_models(
    # This function is carefull to pass same cv an train_size to models an DNN
    X,
    t,
    models=hpTune.select_models(),
    cv=10,
    train_size=0.8,
    scoring={
        "accuracy": "accuracy",
        "recall": "recall",
        "specificity": make_scorer(recall_score, pos_label=0),
        "precision": "precision",
        "f1": "f1",
        "AUC": "roc_auc",
    },
    trials=2,
    epochs=5,
    batch_size=40,
    metrics=[
        "accuracy",
        "Recall",
        cm.specificity,
        "Precision",
        cm.f1_score,
        "AUC",
    ],
):
    best_models = hpTune.optimizing_models(
        models,
        X,
        t,
        cv=cv,
        train_size=train_size,
        scoring=scoring,
        trials=trials,
    )
    best_DNN = hpDNN.optimize_DNN(
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


def get_results(best_models, best_DNN):
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
                data[tag + "_train_" + metric] = best_models[tag][0]["train_" + metric]
                data[tag + "_val_" + metric] = best_models[tag][0]["test_" + metric]

            data["DNN_train_" + metric] = DNN_means[metric]
            data["DNN_val_" + metric] = DNN_means["val_" + metric]
    results = pd.DataFrame(data, index=np.arange(1, len(DNN_means[metric]) + 1))
    results.index.name = "Folds"
    return results


def compare_models(results):

    tags = []
    for i in results.columns:
        if len(i.split("_val_accuracy")) > 1:
            tags.append(i.split("_val_accuracy")[0])

    data = []
    labels = []
    for tag in tags:
        data.append(results[tag + "_val_accuracy"])
        labels.append(tag)

    fig, ax = plt.subplots()
    ax.set_title("Models")
    ax.boxplot(data, labels=labels)
    plt.show()

    alpha = 0.05
    F_statistic, pVal = stats.kruskal(*data)
    print("p-value KrusW:", pVal)
    if pVal <= alpha:
        print("Reject hipothesis: models are diferent\n")
        stacked_data = np.vstack(data).ravel()
        cv = len(data[0])
        model_rep = []
        for i in labels:
            model_rep.append(np.repeat("model" + i, cv))
        stacked_model = np.vstack(model_rep).ravel()
        MultiComp = MultiComparison(stacked_data, stacked_model)
        comp = MultiComp.tukeyhsd(alpha=0.05)
        print(comp)
    else:
        print("Aceptamos la hipótesis: los modelos son iguales")


def analize_performance_DNN(
    best_DNN,
):
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
    y_out_bin = to_categorical(t_test, num_classes=n_classes)

    m = calculate_cmatrix_DNN(t_test, y_out)
    calculate_roc_curve(t_test, y_pred_proba, "DNN")
    return results, m


def calculate_cmatrix_DNN(t_test, y_pred):
    # t_test = to_categorical(t_test, num_classes=n_classes)
    m = confusion_matrix(t_test, y_pred).numpy()
    classes = len(m)
    columns = []
    index = []
    for i in range(classes):
        columns.append("Clase real " + str(i))
        index.append("Clase predicha " + str(i))
    dfm = pd.DataFrame(m, index=index, columns=columns)

    return dfm


def calculate_cmatrix_models(model, tag, X_test, t_test):
    disp = plot_confusion_matrix(model, X_test, t_test)
    disp.figure_.suptitle("Matriz de confusión - " + tag)
    disp.figure_.set_dpi(100)
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    return disp


def calculate_roc_curve(t, y, tag):
    n_classes = len(np.unique(t))
    t_bin = to_categorical(t, num_classes=n_classes)

    plt.figure(figsize=(10, 8))
    colors = [
        "aqua",
        "blue",
        "violet",
        "gold",
        "orange",
        "pink",
        "tan",
        "purple",
        "lime",
        "red",
    ]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(t_bin[:, i], y[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(
            fpr[i],
            tpr[i],
            color=colors[i],
            lw=1,
            label="ROC clase %i (area = %0.3f)" % (i, roc_auc[i]),
        )

    fpr_micro, tpr_micro, _ = roc_curve(t_bin.ravel(), y.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(
        fpr_micro,
        tpr_micro,
        color="red",
        lw=2,
        linestyle=":",
        label="Curva ROC micro-average (AUC = %0.3f)" % roc_auc_micro,
    )
    plt.plot([0, 1], [0, 1], color="k", lw=2, linestyle="--")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curva ROC por clase - " + tag)
    plt.legend(loc="lower right")


def analize_performance_models(best_models, X, t):
    train_size = best_models[1]
    best_models = best_models[0]
    y_pred = dict()
    y_score = dict()
    cmx_list = []

    X_train, X_test, t_train, t_test = train_test_split(X, t, train_size=train_size)
    for model in best_models:
        best_models[model][1].fit(X_train, t_train)
        y_pred[model] = best_models[model][1].predict(X_test)
        y_score[model] = best_models[model][1].predict_proba(X_test)

    return best_models, X_test,t_test, y_pred, y_score



"""
# Example of code
X = pd.read_csv("test/X.csv")
X = X.drop(["Unnamed: 0"], axis=1).values
t = pd.read_csv("test/t.csv")
t = t["labels"].values
n, m = X.shape
n_classes = len(np.unique(t))

a, b = get_best_models(X, t)
# c = get_results(a, b)
# compare_models(c)
r, m = analize_performance_DNN(b)
plt.show()
print(r)
print(m)
# analize_performance_models(a, X, t)
"""
