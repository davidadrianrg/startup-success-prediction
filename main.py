"""Main file of the program to train a model for startup success prediction."""

# Importing required modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from test import old_preprocessing as oldpr
from modules.postprocessing import Report


if __name__ == "__main__":
    data, labels, X, t = oldpr.preprocess()
    metadata = {
        "Title": "Startup Success Prediction Model Comparison",
        "Author": "David Adrián Rodríguez García",
    }

    with Report(generate_pdf=True) as report:
        # Testing diferent modules to print the report
        report.print_title("Startup Success Prediction Model")
        report.print_title("David Adrián Rodríguez García & Víctor Caínzos López", 2)
        report.print_line()
        report.print_title("BoxPlot Graph", 3)
        report.print_boxplot(data, labels)
        scores_list = np.load("./test/hptest.npz")
        labels_list = ["modelLR", "modelLDA", "modelKNN"]
        report.print_title("Hypotheses Contrast for models", 3)
        report.print_hpcontrast(scores_list['arr_0'], scores_list['arr_1'], scores_list['arr_2'], labels=labels_list)
        cl_list = np.load("./test/clreport.npz")
        report.print_title("Classification Report for the model", 3)
        report.print_clreport(cl_list["arr_0"], cl_list["arr_1"])
        roc_list = np.load("./test/roc_test.npz")
        report.print_title("ROC Curve representation for the labels", 3)
        report.print_roc_curve(roc_list["arr_0"],roc_list["arr_1"])