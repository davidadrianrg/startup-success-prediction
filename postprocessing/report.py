"""Postprocessing module to implement graph plot and file output."""

# Importing required modules
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from os import path, mkdir
from md2pdf.core import md2pdf
import scipy.stats as stats
from statsmodels.stats.multicomp import MultiComparison
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import validation_curve
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    roc_curve,
    auc,
    plot_confusion_matrix,
    classification_report,
)


class Report:
    """Markdown Wrapper Class to implement matplolib graphs and markdown syntax easily."""

    def __init__(self, metadata: dict = None, generate_pdf: bool = False):
        """Init Function to set attributes to the class."""
        self.metadata = metadata
        self.generate_pdf = generate_pdf
        self.filepath = None
        self.report_file = None
        self.img_directory = None

    def __enter__(self, filepath: str = "./report.md"):
        """Enter Function to implement with open method to manage report file."""
        self.filepath = filepath
        self.report_file = open(filepath, "w", encoding="utf-8")
        self.img_directory = path.dirname(filepath) + "/img/"

        # If directory not exists, will be created
        try:
            mkdir(self.img_directory)
        except FileExistsError:
            print("Directory already exists")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit Function to implement with open method to manage report file."""
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
        """Parse image to Markdown syntax."""
        return f"![{title}]({img_path})\n"

    @staticmethod
    def parse_title(title: str, h: int = 1) -> str:
        """Parse title to Markdown syntax."""
        h_title = "#" * h
        return h_title + " " + title + "\n"

    @staticmethod
    def parse_list(listmd: list, unordered: str = "") -> str:
        """Parse list to Markdown syntax."""
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
        """Parse code to Markdown syntax."""
        return f"```{language}\n{codeblock}\n```"

    @staticmethod
    def parse_noformat(paragraph: str) -> str:
        """Parse paragraph to avoid Markdown syntax."""
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
        """Parse Markdown string to avoid incompatibility syntax with md2pdf."""
        pdfstring = mdstring.replace("```no-format", "<pre>").replace("```", "</pre>")
        return pdfstring

    # Print methods to write markdown report
    def print(self, paragraph: str):
        """Print plain text to report file in Markdown syntax."""
        self.report_file.write(f"{paragraph}\n")

    def print_noformat(self, paragraph: str):
        """Print plain text to report file escaping Markdown syntax."""
        self.report_file.write(f"{self.parse_noformat(paragraph)}\n")

    def print_line(self):
        """Print horizontal line to report file in Markdown syntax."""
        self.report_file.write("\n---\n")

    def print_title(self, title: str, h: int = 1):
        """Print title to report file in Markdown syntax."""
        self.report_file.write(f"{self.parse_title(title,h)}\n")

    def print_code(self, codeblock: str, language: str = "python"):
        """Print code to report file in Markdown syntax."""
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
        **kwargs,
    ):
        """Print to file the boxplot of the given data."""
        fig = plt.figure(figsize=figsize)
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
        size: int = 100,
        **kwargs,
    ):
        """Print confusion matrix plot of the given model."""
        disp = plot_confusion_matrix(
            model, X_test, t_test, labels=[xlabel, ylabel]
        )  # Show confusion matrix plot
        disp.figure_.suptitle(img_title)  # Add title to the confusion matrix
        disp.figure_.set_dpi(size)  # Set figure size

        # Saving image to file and report
        self.save_image(disp.figure_, filename, img_title, **kwargs)

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
        """Print roc curves for any labels of the given model."""
        # Binarizing classes
        n_classes = len(
            np.unique(t_test)
        )  # Calculate the number of classes in the problem
        t_test_bin = label_binarize(
            t_test, classes=np.arange(0, n_classes, 1)
        )  # Recoding the labels to binary values

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
        """Print the classification report to the report file."""
        self.print_noformat(
            title + ":\n\n" + str(classification_report(t_test, y_pred))
        )

    def print_hpcontrast(self, *args: list, labels: list, alpha: float = 0.05):
        """
        Contrast the hypoteses of the scores given in the list and print the results in the report.

        Using Kurskal-Wallis and Tuckeyhsd tests.
        """
        _, pVal = stats.kruskal(*args)
        str_toprint = f"p-valor KrusW:{pVal}\n"
        if pVal <= alpha:
            str_toprint += "Hypotheses are being rejected: the models are different\n"
            stacked_data = np.vstack(args).ravel()
            cv_len = args[0].size
            labels_model = []
            for i in range(0, len(args)):
                labels_model.append(np.repeat(labels[i], cv_len))
            stacked_model = np.vstack(labels_model).ravel()
            multi_comp = MultiComparison(stacked_data, stacked_model)
            comp = multi_comp.allpairtest(stats.ttest_rel, method="Holm")
            str_toprint += str(comp[0]) + "\n"
            str_toprint += str(multi_comp.tukeyhsd(alpha=alpha))
            self.print_noformat(str_toprint)
        else:
            self.print_noformat(
                str_toprint + "Hypotheses are being accepted: the models are equal"
            )

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
        """Print validation curve plot of the given model."""
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
            for i in range(0, ntables):
                data_split = data.iloc[:,  colum_start : colum_end]
                self.report_file.write(self.parse_dataframe(data_split, rows))
                colum_start = copy.copy(colum_end)
                colum_end += 5
            colums_left = data.shape[1] - ntables*5
            colum_end = colum_end + colums_left - 5
            data_split = data.iloc[:, colum_start : colum_end]
            self.report_file.write(self.parse_dataframe(data_split, rows))
        else:
            self.report_file.write(self.parse_dataframe(data, rows))

    def save_image(self, figure: plt.Figure, filename: str, img_title: str, **kwargs):
        """Image saving method to file and report."""
        # Saving image to file
        img_path = self.img_directory + filename
        figure.savefig(img_path, **kwargs)
        # Include image in the report
        self.report_file.write(self.parse_image(img_path, img_title))
