"""Preprocessing script to clean the Dataset."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from os import path


def read_dataset(filepath: str) -> pd.DataFrame:
    """Read a pandas DataFrame from csv and return data.

    :param filepath: String with the filepath of the file
    :type filepath: str
    :return: A pandas Dataframe with the data inside the file
    :rtype: pd.DataFrame
    """
    _, file_extension = path.splitext(filepath)
    if file_extension == ".csv":
        data = pd.read_csv(filepath)
        return data
    elif file_extension == ".pkl":
        data = pd.read_pickle(filepath)
        return data


def drop_values(data: pd.DataFrame, drop_duplicates: list, to_drop: list) -> pd.DataFrame:
    """Drop data with no sense, duplicates or NaN values.

    :param data: Pandas Dataframe with the data
    :type data: pd.DataFrame
    :param drop_duplicates: List with the column tags to drop the duplicates
    :type drop_duplicates: list
    :param to_drop: List with the column tags to be dropped
    :type to_drop: list
    :return: A pandas Dataframe with the data dropped
    :rtype: pd.DataFrame
    """
    # Dropping out data registered with the same name, and some meaningless features
    data = data.drop_duplicates(subset=drop_duplicates)
    data = data.drop(to_drop, axis=1)

    data_missing = data.isnull().sum().reset_index()
    data_missing.columns = ["feature", "missing"]
    data_missing = (
        data_missing[data_missing["missing"] > 0].sort_values("missing", ascending=False).reset_index(drop=True)
    )
    data_missing["(%) of total"] = round((data_missing["missing"] / len(data)) * 100, 2)

    return data, data_missing


def fill_empty_values(data: pd.DataFrame, labels: list) -> pd.DataFrame:
    """Fill empty values with zeros.

    :param data: Pandas Dataframe with the data
    :type data: pd.DataFrame
    :param labels: List with column tags where fill empty values
    :type labels: list
    :return: A pandas Dataframe with no empty values
    :rtype: pd.DataFrame
    """
    for label in labels:
        data[label] = data[label].fillna(0)

    return data


def to_datetime(data: pd.DataFrame, labels: list) -> pd.DataFrame:
    """Change format to datetime.

    :param data: Pandas Dataframe with the data
    :type data: pd.DataFrame
    :param labels: List with column tags where dates will be transform to datetime format
    :type labels: list
    :return: A pandas Dataframe with dates in datetime format
    :rtype: pd.DataFrame
    """
    for label in labels:
        data[label] = pd.to_datetime(data[label])

    return data


def to_last_date(data: pd.DataFrame, ref_label: str) -> pd.DataFrame:
    """Use the last date of the last year registered considered as the end of the study period.

    :param data: Pandas Dataframe with the data
    :type data: pd.DataFrame
    :param ref_label: List with column tags where dates will be filled with the last date registered
    :type ref_label: str
    :return: A pandas Dataframe with all dates filled
    :rtype: pd.DataFrame
    """
    # Define an auxiliar feature 'last date' to help on calculating life age
    data["last_date"] = data[ref_label]
    last_date = data[ref_label].max()
    data["last_date"] = data["last_date"].fillna(last_date)
    data["last_date"] = pd.to_datetime(data["last_date"])

    return data


def eliminate_spurious_data(data: pd.DataFrame, sort_value: str) -> pd.DataFrame:
    """Clean the dataset from spurious data, negative years and non-sense data.

    :param data: Pandas Dataframe with the data
    :type data: pd.DataFrame
    :param sort_value: String tag value that will be used to sort the Dataframe
    :type sort_value: str
    :return: A pandas Dataframe with spurious data cleaned
    :rtype: pd.DataFrame
    """
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


def non_numerical_recoding(data: pd.DataFrame, non_numerical: str) -> pd.DataFrame:
    """Map all the non numerical data to a categorical number.

    :param data: Pandas Dataframe with the data
    :type data: pd.DataFrame
    :param non_numerical: String tag with the column with non numerical values
    :type non_numerical: str
    :return: A pandas Dataframe with the non_numerical column transformed to numerical data
    :rtype: pd.DataFrame
    """
    # Looking for different values
    diff_values = data[non_numerical].unique()
    # Store them in a dictionary
    dict_states = dict()
    for i in range(len(diff_values)):
        dict_states[diff_values[i]] = i
    # Mapping all the state codes to a categorical number
    data[non_numerical] = data[non_numerical].map(dict_states)

    return data


def data_normalization(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize quantitative data of the given dataframe if their skewness is greater than 2.

    :param data: Pandas Dataframe with the data
    :type data: pd.DataFrame
    :return: A pandas dataframe with the data with skewness > 2 been normalized
    :rtype: pd.DataFrame
    """
    features = []
    numeric = data.select_dtypes(include=np.number)
    # Include only quantitative numerical data
    for i in numeric.columns:
        if data[i].min() != 0 or data[i].max() != 1:
            features.append(i)
    # Feature sample skewness
    data_skewness = data[features].skew(axis=0, skipna=True).sort_values(ascending=False).reset_index()
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


def split_X_t(data: pd.DataFrame, x_tags: list, label_tag: str) -> tuple:
    """Split X and t from a dataset using their colum tags.

    :param data: Pandas Dataframe with the data
    :type data: pd.DataFrame
    :param x_tags: List with the column tags to be used for the characteristics matrix
    :type x_tags: list
    :param label_tag: List with the column tag of the label column
    :type label_tag: str
    :return: A tuple containing the X pandas Dataframe and the t pandas Dataframe
    :rtype: tuple
    """
    X = data[x_tags]
    t = data[label_tag]
    return X, t
