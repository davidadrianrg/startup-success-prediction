'''Main file of the program to train a model for startup success prediction'''

# Importing required modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from test import old_preprocessing as oldpr
from modules.postprocessing import Report


if __name__ == '__main__':
    data, labels, X, t = oldpr.preprocess()
    metadata = {
        "Title" : "Startup Success Prediction Model Comparison", 
        "Author" : "David Adrián Rodríguez García"
    }

    with Report(generate_pdf=True) as report:
        report.print_title("Startup Success Prediction Model Comparison")
        report.print_title("David Adrián Rodríguez García & Víctor Caínzos López",2)
        report.print_line()
        report.print_title("BoxPlot Graph",3)
        report.print_boxplot(data, labels)