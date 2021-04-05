"""Preprocessing script to clean the Dataset"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


class Preprocessing:
    def read_dataset(self):
        data = pd.read_csv("data\startup_data.csv")
        return data

    def impute_lost_values(self):

        data = self.read_dataset()
        # Dropping out startups registered with the same name, and some meaningless features
        data = data.drop_duplicates(subset=["name"])
        data = data.drop(
            [
                "Unnamed: 0",
                "Unnamed: 6",
                "latitude",
                "longitude",
                "zip_code",
                "object_id",
                "status",
            ],
            axis=1,
        )

        data_missing = data.isnull().sum().reset_index()
        data_missing.columns = ["feature", "missing"]
        data_missing = (
            data_missing[data_missing["missing"] > 0]
            .sort_values("missing", ascending=False)
            .reset_index(drop=True)
        )
        data_missing["(%) of total"] = round(
            (data_missing["missing"] / len(data)) * 100, 2
        )

        # Fill missing values with 0 ages
        data["age_first_milestone_year"] = data[
            "age_first_milestone_year"
        ].fillna(0)
        data["age_last_milestone_year"] = data[
            "age_last_milestone_year"
        ].fillna(0)

        # Change format to datetime
        data["closed_at"] = pd.to_datetime(data["closed_at"])
        data["founded_at"] = pd.to_datetime(data["founded_at"])

        # Define an auxiliar feature 'last date' to help on calculating startup's life age
        data["last_date"] = data["closed_at"]

        # Use the last date of the last year registered considered as the end of the study period
        data["last_date"] = data["last_date"].fillna("2013-12-31")
        data["last_date"] = pd.to_datetime(data["last_date"])

        # Define the ages of the startups being active as a new feature
        data["age"] = data["last_date"] - data["founded_at"]
        data["age"] = round(data.age / np.timedelta64(1, "Y"))

        return data, data_missing

    def eliminate_spurious_data(self):

        data, _ = self.impute_lost_values()
        # From all numerical data
        self.numeric = data.select_dtypes(include=np.number)
        spurious_list = []
        for i in self.numeric.columns:
            if self.numeric[i].min() < 0:
                spurious_list.append(i)
        data_spurious = data[spurious_list].sort_values("age").head()
        # Drop samples of those ones with negative values that make no sense (all data ages)
        for i in spurious_list:
            data = data.drop(data[data[i] < 0].index)

        return data, data_spurious

    def non_numerical_recoding(self):

        data, _ = self.eliminate_spurious_data()
        # Looking for different state codes
        states = data.state_code.unique()
        # Store them in a dictionary
        dict_states = dict()
        for i in range(len(states)):
            dict_states[states[i]] = i
        # Mapping all the state codes to a categorical number
        data["state_code"] = data["state_code"].map(dict_states)

        return data

    def quantitative_data_normalization(self):

        data = self.non_numerical_recoding()
        features = []
        # Include only quantitative numerical data
        for i in self.numeric.columns:
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
        features = data_skewness[data_skewness["skewness"] > 2][
            "feature"
        ].values

        # Create new columns in the DataFrame to normalized features
        norm_features = []
        # Log transformation
        for var in features:
            data["norm_" + var] = np.log1p(data[var])
            norm_features.append("norm_" + var)
        # Normalization
        for var in norm_features:
            data[var] = MinMaxScaler().fit_transform(
                data[var].values.reshape(len(data), 1)
            )

        return data, data_skewness, features, norm_features

    def get_X_t(self):

        data, _, _, _ = self.quantitative_data_normalization()
        X = data[
            [
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
        ]

        t = data["labels"]

        return X, t
