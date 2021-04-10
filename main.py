"""Main file of the program to train a model for startup success prediction."""

# Importing required modules

import numpy as np
import pandas as pd
from test import old_preprocessing as oldpr
from modules.postprocessing import Report
from modules import preprocessing as prp
from modules import hyperparametersTunning as hpt
from modules import hyperparametersDNN as hpdnn


def make_report_test():
    """Generate a report taking into account the given data (TEST ONLY)."""
    with Report(generate_pdf=True) as report:
        # Testing diferent modules to print the report
        data, labels, _, _ = oldpr.preprocess()
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
        report.print_roc_curve(roc_list["arr_0"], roc_list["arr_1"])

def make_preprocessing(filepath: str) -> pd.DataFrame:
    """Clean and prepare the given dataframe to train ML models."""
    # Load a dataframe to preprocess from the given filepath
    data = prp.read_dataset(filepath)

    # Clean the dataset of lost values
    drop_duplicates = ["name"]
    to_drop = [
                "Unnamed: 0",
                "Unnamed: 6",
                "latitude",
                "longitude",
                "zip_code",
                "object_id",
                "status",
            ]
    data, data_missing = prp.drop_values(data, drop_duplicates, to_drop)

    # Fill empty values with zeros
    empty_labels = ["age_first_milestone_year", "age_last_milestone_year"]
    data = prp.fill_empty_values(data, empty_labels)

    # Transform data values to datatime
    date_labels = ["closed_at", "founded_at"]
    data = prp.to_datetime(data, date_labels)

    # Use the last date of the last year registered considered as the end of the study period
    data = prp.to_last_date(data, "closed_at")

    # Define the ages of the startups being active as a new feature
    data["age"] = data["last_date"] - data["founded_at"]
    data["age"] = round(data.age / np.timedelta64(1, "Y"))

    # Clean the dataset from spurious data, negative years and non-sense data
    data, data_spurious = prp.eliminate_spurious_data(data, "age")

    # Recode non numerical columns to numerical ones
    non_numerical = ["state_code"]
    data = prp.non_numerical_recoding(data, non_numerical)

    # Normalize quantitative data
    data, data_skewness, features, norm_features = prp.data_normalization(data)

    # Get X and t from the dataset using the column tags
    x_tags = [
                "state_code",
                "age_last_funding_year",
                "age_first_milestone_year",
                "age_last_milestone_year",
                "funding_rounds",
                "milestones",
                "is_CA",
                "is_NY",
                "is_MA",
                "is_TX",
                "is_otherstate",
                "is_software",
                "is_web",
                "is_mobile",
                "is_enterprise",
                "is_advertising",
                "is_gamesvideo",
                "is_ecommerce",
                "is_biotech",
                "is_consulting",
                "is_othercategory",
                "has_VC",
                "has_angel",
                "has_roundA",
                "has_roundB",
                "has_roundC",
                "has_roundD",
                "avg_participants",
                "is_top500",
                "age",
                "norm_funding_total_usd",
                "norm_age_first_funding_year",
                "norm_relationships",
            ]
    label_tag = "labels"
    X, t = prp.split_X_t(data, x_tags, label_tag )

    return X, t

def train_models(X: pd.DataFrame, t: pd.DataFrame):
    """Train different models using the dataset and its labels."""
    models = hpt.select_models()
    # Get best models with optimized hyperparameters
    best_models = hpt.optimizing_models(models, X, t)
    # Plot the results
    hpt.plot_best_model(best_models)
    # Get the best DNN model optimizing hyperparameters
    best_dnn = hpdnn.get_best_DNN(X, t)
    # Plot the results
    hpdnn.plot_best_DNN(best_dnn)
    

def make_report():
    """Generate a report taking into account the given data."""
    


if __name__ == "__main__":
    
    # Loading and preprocessing the startup dataframe
    X, t = make_preprocessing("data/startup_data.csv")

    # Training different models using the previous dataset
    train_models(X, t)

    # Generating the report with the results of the training
    make_report()