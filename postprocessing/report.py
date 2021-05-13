"""Postprocessing module to implement graph plot and file output."""

# Importing required modules
import copy
from os import mkdir, path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from keras.utils import to_categorical
from md2pdf.core import md2pdf
from preprocessing.detect_anomalies import Anomalies
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    plot_confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import validation_curve
from statsmodels.stats.multicomp import MultiComparison
from sklearn.decomposition import PCA


class Report:
    """Markdown Wrapper Class to implement matplolib graphs and file output with markdown syntax."""

    def __init__(self, metadata: dict = None, generate_pdf: bool = False):
        """Init Function to set attributes to the class.

        :param metadata: Dictionary with metadata to be included in the file, defaults to None
        :type metadata: dict, optional
        :param generate_pdf: Boolean to enable/disable pdf generation, defaults to False
        :type generate_pdf: bool, optional
        """
        self.metadata = metadata
        self.generate_pdf = generate_pdf
        self.filepath = None
        self.report_file = None
        self.img_directory = None

    def __enter__(self, filepath: str = "./report.md"):
        """Enter Function to implement with open method to manage report file.

        :param filepath: String with the filepath of the output report file, defaults to "./report.md"
        :type filepath: str, optional
        :return: An instance of the Report class
        :rtype: Report
        """
        self.filepath = filepath
        self.report_file = open(filepath, "w", encoding="utf-8")
        self.img_directory = path.dirname(filepath) + "/img/"

        # If directory not exists, will be created
        try:
            mkdir(self.img_directory)
        except FileExistsError:
            print("Image directory already exists, overwritting images")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit Function to implement with open method to manage report file.

        :param exc_type: Indicates class of exception
        :param exc_value: Indicates type of exception
        :param traceback: Report with all of the information needed to solve the exception
        """
        self.report_file.close()
        # Opening the file in reading format
        with open(self.filepath, "r", encoding="utf-8") as filemd:
            file_str = filemd.read()

        # Parse incomptibility syntax
        mdtopdf = self.parse_pdf(file_str)
        # Generate pdf report if generate_pdf is selected
        if self.generate_pdf:
            directory = path.dirname(path.abspath(self.filepath))
            filename, _ = path.splitext(path.basename(self.filepath))
            md2pdf(
                f"{directory}/{filename}.pdf",
                md_content=mdtopdf,
                md_file_path=None,
                css_file_path="./templates/markdown-pdf.css",
                base_url=directory,
            )

    @staticmethod
    def parse_image(img_path: str, title: str = "") -> str:
        """Parse image to Markdown syntax.

        :param img_path: String with the filepath of the image
        :type img_path: str
        :param title: String with the title of the image, defaults to ""
        :type title: str, optional
        :return: String with the image in markdown syntax
        :rtype: str
        """
        return f"![{title}]({img_path})\n"

    @staticmethod
    def parse_title(title: str, h: int = 1) -> str:
        """Parse title to Markdown syntax.

        :param title: String with the title
        :type title: str
        :param h: Number with the header level, defaults to 1
        :type h: int, optional
        :return: String with the title in markdown syntax
        :rtype: str
        """
        h_title = "#" * h
        return h_title + " " + title + "\n"

    @staticmethod
    def parse_list(listmd: list, unordered: str = "") -> str:
        """Parse list to Markdown syntax.

        :param listmd: List with values to be printed
        :type listmd: list
        :param unordered: String with the character of the ul element, defaults to ""
        :type unordered: str, optional
        :return: String with the list in markdown syntax
        :rtype: str
        """
        output = ""
        if unordered:
            # Argument must include one of the following characters: -,*,+
            for element in listmd:
                output += unordered + " " + str(element) + "\n"
        else:
            for element in listmd:
                output += str(listmd.index(element) + 1) + ". " + str(element) + "\n"
        return output

    @staticmethod
    def parse_code(codeblock: str, language: str = "python") -> str:
        """Parse code to Markdown syntax.

        :param codeblock: String input with the codeblock
        :type codeblock: str
        :param language: Coding language to lint the code, defaults to "python"
        :type language: str, optional
        :return: String with the codeblock in markdown syntax
        :rtype: str
        """
        return f"```{language}\n{codeblock}\n```"

    @staticmethod
    def parse_noformat(paragraph: str) -> str:
        """Parse paragraph to avoid Markdown syntax.

        :param paragraph: String paragraph input
        :type paragraph: str
        :return: String avoiding markdown syntax, returned as a plain textblock
        :rtype: str
        """
        return f"\n```no-format\n{paragraph}\n```\n"

    @staticmethod
    def parse_dataframe(data: pd.DataFrame, rows: int = 5) -> str:
        """Parse dataframe to a markdown table showing the rows given by rows argument.

        :param data: Pandas dataframe to be parsed to markdown table
        :type data: pd.DataFrame
        :param rows: Number of rows to be parsed, defaults to 5
        :type rows: int, optional
        :return: Markdown table in string format
        :rtype: str
        """
        return f"{data.head(rows).to_markdown()}\n\n"

    @staticmethod
    def parse_pdf(mdstring: str):
        """Parse Markdown string to avoid incompatibility syntax with md2pdf.

        :param mdstring: String input in markdown syntax
        :type mdstring: str
        :return: String output with html <pre> tags
        :rtype: [type]
        """
        pdfstring = mdstring.replace("```no-format", "<pre>").replace("```", "</pre>")
        return pdfstring

    # Print methods to write markdown report
    def print(self, paragraph: str):
        """Print plain text to report file in Markdown syntax.

        :param paragraph: String paragraph to be printed in the report file
        :type paragraph: str
        """
        self.report_file.write(f"{paragraph}\n")

    def print_noformat(self, paragraph: str):
        """Print plain text to report file escaping Markdown syntax.

        :param paragraph: String paragraph to be printed in the report file with no format
        :type paragraph: str
        """
        self.report_file.write(f"{self.parse_noformat(paragraph)}\n")

    def print_line(self):
        """Print horizontal line to report file in Markdown syntax."""
        self.report_file.write("\n---\n")

    def print_title(self, title: str, h: int = 1):
        """Print title to report file in Markdown syntax.

        :param title: String title to be printed
        :type title: str
        :param h: Number with the header level, defaults to 1
        :type h: int, optional
        """
        self.report_file.write(f"{self.parse_title(title,h)}\n")

    def print_code(self, codeblock: str, language: str = "python"):
        """Print code to report file in Markdown syntax.

        :param codeblock: String input with the codeblock
        :type codeblock: str
        :param language: Coding language to lint the code, defaults to "python"
        :type language: str, optional
        """
        self.report_file.write(f"{self.parse_code(codeblock, language)}\n")

    def print_boxplot(
        self,
        data: pd.DataFrame,
        labels: list,
        filename: str = "boxplot.png",
        img_title: str = "Box Plot",
        figsize: tuple = (15, 7),
        color: str = "green",
        orient: str = "v",
        same_scale: bool = False,
        **kwargs,
    ):
        """Print to file the boxplot of the given data.

        :param data: Pandas Dataframe with data to be plotted
        :type data: pd.DataFrame
        :param labels: List containing the column tags to be plotted
        :type labels: list
        :param filename: String with the filename of the output image, defaults to "boxplot.png"
        :type filename: str, optional
        :param img_title: String with the title of the output image, defaults to "Box Plot"
        :type img_title: str, optional
        :param figsize: Tuple with the image dimensions, defaults to (15, 7)
        :type figsize: tuple, optional
        :param color: String with the color of the boxplot, defaults to "green"
        :type color: str, optional
        :param orient: String for the orientation of the boxplot, defaults to "v"
        :type orient: str, optional
        :param same_scale: Boolean to use the same scale for different boxes, defaults to False
        :type same_scale: bool, optional
        """
        fig = plt.figure(figsize=figsize)
        if same_scale:
            ax = plt.axes()
            ax.set_title(img_title)
            sns.boxplot(data=data)
        else:
            for label in labels:
                fig.add_subplot(1, len(labels), labels.index(label) + 1)
                sns.boxplot(y=data[label], color=color, orient=orient)
                fig.tight_layout()

        # Saving image to file and report
        self.save_image(fig, filename, img_title, **kwargs)

    def print_confusion_matrix(
        self,
        model: BaseEstimator,
        X_test: np.ndarray,
        t_test: np.ndarray,
        xlabel: str,
        ylabel: str,
        filename: str = "confusion_matrix.png",
        img_title: str = "Confusion Matrix",
        size_dpi: int = 100,
        **kwargs,
    ):
        """Print confusion matrix plot of the given model.

        :param model: Model which be used to print the confusion matrix
        :type model: BaseEstimator
        :param X_test: Numpy array with the X_test values
        :type X_test: np.ndarray
        :param t_test: Numpy array with the t_test values
        :type t_test: np.ndarray
        :param xlabel: String with the label of the x axe
        :type xlabel: str
        :param ylabel: String with the label of the y axe
        :type ylabel: str
        :param filename: String with the filename of the output image, defaults to "confusion_matrix.png"
        :type filename: str, optional
        :param img_title: String with the title of the output image, defaults to "Confusion Matrix"
        :type img_title: str, optional
        :param size_dpi: Number for the resolution dpi of the image, defaults to 100
        :type size_dpi: int, optional
        """
        disp = plot_confusion_matrix(model, X_test, t_test)  # Show confusion matrix plot
        disp.figure_.suptitle(img_title)  # Add title to the confusion matrix
        disp.figure_.set_dpi(size_dpi)  # Set figure dpi
        disp.ax_.set_xlabel(xlabel)
        disp.ax_.set_ylabel(ylabel)

        # Saving image to file and report
        self.save_image(disp.figure_, filename, img_title, **kwargs)

    def print_confusion_matrix_DNN(
        self,
        t_test: np.ndarray,
        y_pred: np.ndarray,
        xlabel: str,
        ylabel: str,
        title: str,
    ):
        """Print confusion matrix plot of the given DNN model.

        :param t_test: Numpy array with the t_test values
        :type t_test: np.ndarray
        :param y_pred: Numpy array with the y predicted values
        :type y_pred: np.ndarray
        :param xlabel: String with the label of the x axe
        :type xlabel: str
        :param ylabel: String with the label of the y axe
        :type ylabel: str
        :param title: String with the title for the dataframe
        :type title: str
        """
        m = confusion_matrix(t_test, y_pred)
        classes = len(m)
        columns = []
        index = []
        for iclass in range(classes):
            columns.append(ylabel + " " + str(iclass))
            index.append(xlabel + " " + str(iclass))
        df_matrix = pd.DataFrame(m, index=index, columns=columns)
        self.print(f"**{title}**\n")
        self.print_dataframe(df_matrix)

    def print_roc_curve(
        self,
        t_test: np.ndarray,
        y_score: np.ndarray,
        filename: str = "roc_curve.png",
        img_title: str = "Roc Curve per class",
        figsize: tuple = (10, 8),
        xlabel: str = "False Positive Rate",
        ylabel: str = "True Positive Rate",
        legend_loc: str = "lower right",
        **kwargs,
    ):
        """Print roc curves for any labels of the given model.

        :param t_test: Numpy array with the t_test values
        :type t_test: np.ndarray
        :param y_score: Numpy array with the y score values
        :type y_pred: np.ndarray
        :param filename: String with the filename of the output image, defaults to "roc_curve.png"
        :type filename: str, optional
        :param img_title: String with the title of the output image, defaults to "Roc Curve per class"
        :type img_title: str, optional
        :param figsize: Tuple with the image dimensions, defaults to (10, 8)
        :type figsize: tuple, optional
        :param xlabel: String with the label of the x axe
        :type xlabel: str
        :param ylabel: String with the label of the y axe
        :type ylabel: str
        :param legend_loc: String with the position of the legend in the figure
        :type legend_loc: str
        """
        # Binarizing classes
        n_classes = len(np.unique(t_test))  # Calculate the number of classes in the problem
        t_test_bin = to_categorical(t_test, num_classes=n_classes)  # Recoding the labels to binary values

        # Ploting the figure with each roc curve per class
        fig = plt.figure(figsize=figsize)
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
        fig, ax = plt.subplots()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(t_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            ax.plot(
                fpr[i],
                tpr[i],
                color=colors[i],
                lw=1,
                label="ROC class %i (area = %0.3f)" % (i, roc_auc[i]),
            )

        # Using the micro-average to calculate the main ROC Curve
        # False_positives and True_positives
        fpr_micro, tpr_micro, _ = roc_curve(t_test_bin.ravel(), y_score.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)

        ax.plot(
            fpr_micro,
            tpr_micro,
            color="red",
            lw=2,
            linestyle=":",
            label="ROC micro-average (AUC = %0.3f)" % roc_auc_micro,
        )
        ax.plot([0, 1], [0, 1], color="k", lw=2, linestyle="--")
        ax.set(xlabel=xlabel, ylabel=ylabel, title=img_title)
        ax.legend(loc=legend_loc, fontsize="x-small")

        # Saving image to file and report
        self.save_image(fig, filename, img_title, **kwargs)

    def print_clreport(
        self,
        t_test: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Classification report",
    ):
        """Print the classification report to the report file.

        :param t_test: Numpy array with the t_test values
        :type t_test: np.ndarray
        :param y_pred: Numpy array with the y predicted values
        :type y_pred: np.ndarray
        :param title: String title for the classification report, defaults to "Classification report"
        :type title: str, optional
        """
        self.print_noformat(title + ":\n\n" + str(classification_report(t_test, y_pred)))

    def print_hpcontrast(self, data: list, labels: list, alpha: float = 0.05):
        """Contrast the hypoteses of the scores given in the list and print the results in the report.

        Using Kurskal-Wallis and Tuckeyhsd tests.

        :param data: List containing the metric results of the models
        :type data: list
        :param labels: List containg the models tags
        :type labels: list
        :param alpha: Number for the pValue of the test, defaults to 0.05
        :type alpha: float, optional
        """
        _, pVal = stats.kruskal(*data)
        str_toprint = f"p-valor KrusW:{pVal}\n"
        if pVal <= alpha:
            str_toprint += "Hypotheses are being rejected: the models are different\n"
            stacked_data = np.vstack(data).ravel()
            cv = len(data[0])
            model_rep = []
            for i in labels:
                model_rep.append(np.repeat("model" + i, cv))
            stacked_model = np.vstack(model_rep).ravel()
            multi_comp = MultiComparison(stacked_data, stacked_model)
            comp = multi_comp.tukeyhsd(alpha=alpha)
            str_toprint += str(comp)
        else:
            str_toprint = str_toprint + "Hypotheses are being accepted: the models are equal"
        self.print_noformat(str_toprint)

    def print_validation_curve(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        param_name: str,
        param_range=np.logspace(-6, -1, 5),
        ylabel: str = "Score",
        filename: str = "validation_curve.png",
        img_title: str = "Validation Curve",
        figsize: tuple = (10, 8),
        **kwargs,
    ):
        """Print validation curve plot of the given model.

        :param model: Model which be used to plot the validation curve
        :type model: BaseEstimator
        :param X: Numpy array with the characteristic matrix
        :type X: np.ndarray
        :param y: Numpy array with the predicted values
        :type y: np.ndarray
        :param param_name: String with the name of the hyperparameter
        :type param_name: str
        :param param_range: Logarithmic range for the hyperparameter values, defaults to np.logspace(-6, -1, 5)
        :type param_range: range, optional
        :param ylabel: String with the label for the y axe, defaults to "Score"
        :type ylabel: str, optional
        :param filename: String with the filename of the output image, defaults to "validation_curve.png"
        :type filename: str, optional
        :param img_title: String with the title of the output image, defaults to "Validation Curve"
        :type img_title: str, optional
        :param figsize: Tuple with the image dimensions, defaults to (10, 8)
        :type figsize: tuple, optional
        """
        train_scores, test_scores = validation_curve(
            model,
            X,
            y,
            param_name=param_name,
            param_range=param_range,
            scoring="accuracy",
            n_jobs=-1,
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        fig, ax = plt.subplots()
        fig.figsize = figsize
        ax.set(title=img_title, xlabel=param_name, ylabel=ylabel, ylim=(0.0, 1.1))
        lw = 2
        ax.semilogx(
            param_range,
            train_scores_mean,
            label="Training score",
            color="darkorange",
            lw=lw,
        )
        ax.fill_between(
            param_range,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.2,
            color="darkorange",
            lw=lw,
        )
        ax.semilogx(
            param_range,
            test_scores_mean,
            label="Cross-validation score",
            color="navy",
            lw=lw,
        )
        ax.fill_between(
            param_range,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.2,
            color="navy",
            lw=lw,
        )
        ax.legend(loc="best", fontsize="x-small")

        # Saving image to file and report
        self.save_image(fig, filename, img_title, **kwargs)

    def print_mean_acc_model(
        self,
        best_models: tuple,
        tag: str,
        filename: str = "mean_accuracy.png",
        img_title: str = "Mean Accuracy",
        **kwargs,
    ):
        """Plot the mean accuracy of the model using its historial results to the report.

        :param best_models: Tuple with the best models to print their mean accuracy curves
        :type best_models: tuple
        :param tag: String with the tag of the model to be printed
        :type tag: str
        :param filename: String with the filename of the output image, defaults to "mean_accuracy.png"
        :type filename: str, optional
        :param img_title: String with the title of the output image, defaults to "Mean Accuracy"
        :type img_title: str, optional
        """
        fig, ax = plt.subplots()
        ax.plot(best_models[tag][0]["train_accuracy"])
        ax.plot(best_models[tag][0]["test_accuracy"])
        ax.set_title(img_title)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Iteration (cv)")
        ax.legend(["Trainning", "Test"], loc="lower right")

        self.save_image(fig, filename, img_title, **kwargs)

    def print_val_curve_dnn(
        self,
        best_DNN: tuple,
        metric: str = "accuracy",
        filename: str = "validation_curve_dnn.png",
        img_title: str = "Validation Curve",
        **kwargs,
    ):
        """Plot the validation curve of the model using its historial results to the report.

        :param best_DNN: Tuple with the best models to print their validation curves
        :type best_DNN: tuple
        :param metric: String with the metric used to perform the validation curve, defaults to "accuracy"
        :type metric: str, optional
        :param filename: String with the filename of the output image, defaults to "validation_curve_dnn.png"
        :type filename: str, optional
        :param img_title: String with the title of the output image, defaults to "Validation Curve"
        :type img_title: str, optional
        """
        fig, ax = plt.subplots()
        ax.plot(np.mean(best_DNN[0][1][metric], axis=0))
        ax.plot(np.mean(best_DNN[0][1]["val_" + metric], axis=0))
        ax.set_title("DNN Model " + metric)
        ax.set_ylabel(metric)
        ax.set_xlabel("Iteration (epoch)")
        ax.legend(["Trainning", "Test"], loc="lower right")

        self.save_image(fig, filename, img_title, **kwargs)

    def print_dataframe(self, data: pd.DataFrame, rows: int = 5):
        """Print the dataframe with the given rows to the report file.

        If the dataframe have more than 5 columns, will be divided to print in the report.

        :param data: Pandas dataframe to be printed
        :type data: pd.DataFrame
        :param rows: Number of rows of the dataframe to be printed, defaults to 5
        :type rows: int, optional
        """
        if (data.shape[1]) > 5:
            colum_start = 0
            colum_end = 5
            ntables = data.shape[1] // 5
            for _ in range(0, ntables):
                data_split = data.iloc[:, colum_start:colum_end]
                self.report_file.write(self.parse_dataframe(data_split, rows))
                colum_start = copy.copy(colum_end)
                colum_end += 5
            colums_left = data.shape[1] - ntables * 5
            colum_end = colum_end + colums_left - 5
            data_split = data.iloc[:, colum_start:colum_end]
            self.report_file.write(self.parse_dataframe(data_split, rows))
        else:
            self.report_file.write(self.parse_dataframe(data, rows))

    def print_kmeans(
        self,
        inertia_dic: dict,
        X: np.ndarray,
        ylabel: str = "Inertia",
        filename: str = "kmeans.png",
        img_title: str = "KMeans Clustering",
        figsize: tuple = (10, 8),
    ):
        """Print the kmeans clustering for the given X dataframe and inertias dictionary.

        :param inertia_dic: Dictionary with the inertias obtained from kmeans model
        :type inertia_dic: dict
        :param X: Dataframe to be clustered
        :type X: np.ndarray
        :param ylabel: String with the label for the y axe, defaults to "Inertia"
        :type ylabel: str, optional
        :param filename: String with the filename of the output image, defaults to "kmeans.png"
        :type filename: str, optional
        :param img_title: String with the title of the output image, defaults to "KMeans Clustering"
        :type img_title: str, optional
        :param figsize: Tuple with the image dimensions, defaults to (10, 8)
        :type figsize: tuple, optional
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(inertia_dic.keys(), inertia_dic.values(), marker="x")
        ax.set_xlabel("k")
        ax.set_xticks(range(1, len(X.columns) + 1))
        ax.set_ylabel(ylabel)
        fig.set_tight_layout(True)
        self.save_image(fig, filename, img_title)

    def print_dbscan(
        self,
        distances: dict,
        xlabel: str = "Ordered points per distance to the nearest k-neighbor",
        ylabel: str = "Distance to the nearest k-neighbor",
        filename: str = "dbscan.png",
        img_title: str = "DBSCAN Clustering",
        figsize: tuple = (10, 8),
    ):
        """Print the dbscan clustering for the given distances list.

        :param distances: List with the distances between data points
        :type distances: list
        :param X: Dataframe to be clustered
        :type X: np.ndarray
        :param ylabel: String with the label for the y axe, defaults to "Inertia"
        :type ylabel: str, optional
        :param filename: String with the filename of the output image, defaults to "kmeans.png"
        :type filename: str, optional
        :param img_title: String with the title of the output image, defaults to "KMeans Clustering"
        :type img_title: str, optional
        :param figsize: Tuple with the image dimensions, defaults to (10, 8)
        :type figsize: tuple, optional
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(distances)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self.save_image(fig, filename, img_title)

    def print_autoencoder_validation(
        self,
        anomalies: Anomalies,
        filename: str = "autoencoder_validation.png",
        img_title: str = "Autoencoder Validation",
        **kwargs,
    ):
        """Print the autoencoder validation graph to the report.

        :param Anomalies: An instance of the class Anomalies to be printed
        :type Anomalies: preprocessing.Anomalies
        :param filename: String with the filename of the output image, defaults to "autoencoder_validation.png"
        :type filename: str, optional
        :param img_title: String with the title of the output image, defaults to "Autoencoder Validation"
        :type img_title: str, optional
        """
        self.save_image(anomalies.plot_autoencoder_validation(**kwargs), filename, img_title)

    def print_autoencoder_threshold(
        self,
        anomalies: Anomalies,
        filename: str = "autoencoder_threshold.png",
        img_title: str = "Autoencoder Threshold",
        **kwargs,
    ):
        """Print the autoencoder threshold graph to the report.

        :param Anomalies: An instance of the class Anomalies to be printed
        :type Anomalies: preprocessing.Anomalies
        :param filename: String with the filename of the output image, defaults to "autoencoder_threshold.png"
        :type filename: str, optional
        :param img_title: String with the title of the output image, defaults to "Autoencoder Threshold"
        :type img_title: str, optional
        """
        self.save_image(anomalies.plot_autoencoder_threshold(**kwargs), filename, img_title)

    def print_autoencoder_error(
        self,
        anomalies: Anomalies,
        filename: str = "autoencoder_error.png",
        img_title: str = "Autoencoder Error",
        **kwargs,
    ):
        """Print the autoencoder error graph to the report.

        :param Anomalies: An instance of the class Anomalies to be printed
        :type Anomalies: preprocessing.Anomalies
        :param filename: String with the filename of the output image, defaults to "autoencoder_threshold.png"
        :type filename: str, optional
        :param img_title: String with the title of the output image, defaults to "Autoencoder Threshold"
        :type img_title: str, optional
        """
        self.save_image(anomalies.plot_autoencoder_error(**kwargs), filename, img_title)

    def print_PCA(
        self,
        X: pd.DataFrame,
        pca: PCA,
        xlabel: str = "Main Components",
        ylabel: str = "Variance %",
        filename: str = "pca.png",
        img_title: str = "PCA",
        figsize: tuple = (10, 5),
    ):
        """Print the PCA dimension graph to the report.

        :param X: Pandas Dataframe with the characteristic matrix
        :type X: pd.DataFrame
        :param pca: [description]
        :type pca: PCA
        :param xlabel: String with the label for the x axe, defaults to "Main Components"
        :type xlabel: str, optional
        :param ylabel: String with the label for the y axe, defaults to "Variance %"
        :type ylabel: str, optional
        :param filename: String with the filename of the output image, defaults to "pca.png"
        :type filename: str, optional
        :param img_title: String with the title of the output image, defaults to "PCA"
        :type img_title: str, optional
        :param figsize: Tuple with the image dimensions, defaults to (10, 5)
        :type figsize: tuple, optional
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(
            X.columns.values.tolist(),
            pca.explained_variance_ratio_ * 100,
            color="b",
            align="center",
            tick_label=X.columns.values.tolist(),
        )
        ax.set_xticks(rotation="vertical")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.set_tight_layout(True)
        self.save_image(fig, filename, img_title)

    def save_image(self, figure: plt.Figure, filename: str, img_title: str, **kwargs):
        """Image saving method to file and report.

        :param figure: Matplotlib figure to be printed in the report file
        :type figure: plt.Figure
        :param filename: String with the filename of the output image
        :type filename: str
        :param img_title: String with the title of the output image
        :type img_title: str
        """
        # Saving image to file
        img_path = self.img_directory + filename
        figure.savefig(img_path, **kwargs)
        # Include image in the report
        self.report_file.write(self.parse_image(img_path, img_title))
