"""Preprocessing script to clean the Dataset."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from os import path


def read_dataset(filepath: str) -> pd.DataFrame:
    """Read a pandas DataFrame from csv and return data."""
    _, file_extension = path.splitext(filepath)
    if file_extension == ".csv":
        data = pd.read_csv(filepath)
        return data
    elif file_extension == ".pkl":
        data = pd.read_pickle(filepath)
        return data


def drop_values(
    data: pd.DataFrame, drop_duplicates: list, to_drop: list
) -> pd.DataFrame:
    """Drop data with no sense, duplicates or NaN values."""
    # Dropping out data registered with the same name, and some meaningless features
    data = data.drop_duplicates(subset=drop_duplicates)
    data = data.drop(to_drop, axis=1)

    data_missing = data.isnull().sum().reset_index()
    data_missing.columns = ["feature", "missing"]
    data_missing = (
        data_missing[data_missing["missing"] > 0]
        .sort_values("missing", ascending=False)
        .reset_index(drop=True)
    )
    data_missing["(%) of total"] = round((data_missing["missing"] / len(data)) * 100, 2)

    return data, data_missing


def fill_empty_values(data: pd.DataFrame, labels: list) -> pd.DataFrame:
    """Fill empty values with zeros."""
    for label in labels:
        data[label] = data[label].fillna(0)

    return data


def to_datetime(data: pd.DataFrame, labels: list) -> pd.DataFrame:
    """Change format to datetime."""
    for label in labels:
        data[label] = pd.to_datetime(data[label])

    return data


def to_last_date(data: pd.DataFrame, ref_label: str) -> pd.DataFrame:
    """Use the last date of the last year registered considered as the end of the study period."""
    # Define an auxiliar feature 'last date' to help on calculating life age
    data["last_date"] = data[ref_label]
    last_date = data[ref_label].max()
    data["last_date"] = data["last_date"].fillna(last_date)
    data["last_date"] = pd.to_datetime(data["last_date"])

    return data


def eliminate_spurious_data(data: pd.DataFrame, sort_value: str) -> pd.DataFrame:
    """Clean the dataset from spurious data, negative years and non-sense data."""
    # From all numerical data
    numeric = data.select_dtypes(include=np.number)
    spurious_list = []
    for i in numeric.columns:
        if numeric[i].min() < 0:
            spurious_list.append(i)
    data_spurious = data[spurious_list].sort_values(sort_value).head()
    # Drop samples of those ones with negative values that make no sense (all data ages)
    for i in spurious_list:
        data = data.drop(data[data[i] < 0].index)

    return data, data_spurious


def non_numerical_recoding(data: pd.DataFrame, non_numerical: list) -> pd.DataFrame:
    """Map all the non numerical data to a categorical number."""
    for label in non_numerical:
        data[label] = data[label].map(pd.Series(data[label].unique()))

    return data


def data_normalization(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize quantitative data of the given dataframe."""
    features = []
    numeric = data.select_dtypes(include=np.number)
    # Include only quantitative numerical data
    for i in numeric.columns:
        if data[i].min() != 0 or data[i].max() != 1:
            features.append(i)
    # Feature sample skewness
    data_skewness = (
        data[features]
        .skew(axis=0, skipna=True)
        .sort_values(ascending=False)
        .reset_index()
    )
    data_skewness.columns = ["feature", "skewness"]
    # Exclude those variables with smaller skewness
    features = data_skewness[data_skewness["skewness"] > 2]["feature"].values

    # Create new columns in the DataFrame to normalized features
    norm_features = []
    # Log transformation
    for var in features:
        data["norm_" + var] = np.log1p(data[var])
        norm_features.append("norm_" + var)
    # Normalization
    for var in norm_features:
        data[var] = MinMaxScaler().fit_transform(data[var].values.reshape(len(data), 1))

    return data, data_skewness, features, norm_features


def split_X_t(data: pd.DataFrame, x_tags: list, label_tag: str) -> pd.DataFrame:
    """Split X and t from a dataset using their colum tags."""
    X = data[x_tags]
    t = data[label_tag]
    return X, t
