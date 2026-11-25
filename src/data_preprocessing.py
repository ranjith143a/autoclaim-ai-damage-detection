"""
Data Preprocessing and Feature Pipeline Module.
Implements leak-free train/validation/test splitting (70/15/15),
missing value imputation, categorical encoding, and numerical scaling.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib


CATEGORICAL_FEATURES = ["vehicle_segment", "repair_region"]

NUMERICAL_METADATA_FEATURES = [
    "vehicle_age_years",
    "vehicle_insured_value_inr",
    "labor_rate_per_hour_inr",
    "parts_price_index",
    "policy_deductible_inr",
]

CNN_PROBABILITY_FEATURES = [
    "front_bumper_cnn_probability",
    "hood_cnn_probability",
    "headlight_cnn_probability",
    "front_door_cnn_probability",
    "back_door_cnn_probability",
    "back_bumper_cnn_probability",
    "fender_cnn_probability",
    "windshield_cnn_probability",
]

CNN_FLAG_FEATURES = [
    "front_bumper_cnn_flag",
    "hood_cnn_flag",
    "headlight_cnn_flag",
    "front_door_cnn_flag",
    "back_door_cnn_flag",
    "back_bumper_cnn_flag",
    "fender_cnn_flag",
    "windshield_cnn_flag",
]

NUMERICAL_FEATURES = (
    NUMERICAL_METADATA_FEATURES + CNN_PROBABILITY_FEATURES + CNN_FLAG_FEATURES
)

ALL_FEATURE_COLUMNS = (
    NUMERICAL_METADATA_FEATURES[:1]  # vehicle_age_years
    + CATEGORICAL_FEATURES[:1]      # vehicle_segment
    + NUMERICAL_METADATA_FEATURES[1:2]  # vehicle_insured_value_inr
    + CATEGORICAL_FEATURES[1:2]      # repair_region
    + NUMERICAL_METADATA_FEATURES[2:]   # labor_rate, parts_price_index, deductible
    + CNN_PROBABILITY_FEATURES
    + CNN_FLAG_FEATURES
)

TARGET_COLUMN = "claim_amount_inr"


def load_dataset(csv_path=None):
    """Loads the 40,000 synthetic insurance claim dataset."""
    if csv_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Check data/ first, then root
        csv_path = os.path.join(base_dir, "data", "synthetic_vehicle_claims_40000.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(base_dir, "synthetic_vehicle_claims_40000.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Claims dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def split_data(df, random_state=42):
    """
    Splits dataset into 70% Train, 15% Validation, and 15% Test.
    Strictly partitions data before any fit transformation to avoid data leakage.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].values

    # Step 1: 70% train, 30% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state
    )

    # Step 2: Split 30% temp equally into 15% val and 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessors():
    """
    Constructs Scikit-learn ColumnTransformers:
    1. Base preprocessor: OneHotEncoder for categoricals + SimpleImputer for numericals (ideal for trees).
    2. Scaled preprocessor: Same as base, but scales all features with StandardScaler (ideal for linear/SVR/KNN).
    """
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    base_transformer = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
            ("num", num_pipeline, NUMERICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    scaled_transformer = Pipeline([
        ("base", base_transformer),
        ("scaler", StandardScaler()),
    ])

    return base_transformer, scaled_transformer


def prepare_preprocessed_datasets(csv_path=None, random_state=42):
    """
    Executes complete data loading, splitting, and fitting of preprocessors.
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test,
        base_preprocessor, scaled_preprocessor,
        feature_names
    """
    df = load_dataset(csv_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, random_state=random_state)

    base_prep, scaled_prep = build_preprocessors()

    # Fit base preprocessor ONLY on training data
    base_prep.fit(X_train)
    scaled_prep.fit(X_train)

    feature_names = list(base_prep.get_feature_names_out())

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "base_preprocessor": base_prep,
        "scaled_preprocessor": scaled_prep,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    data_dict = prepare_preprocessed_datasets()
    print("Train shape:", data_dict["X_train"].shape, "y:", data_dict["y_train"].shape)
    print("Val shape:  ", data_dict["X_val"].shape, "y:", data_dict["y_val"].shape)
    print("Test shape: ", data_dict["X_test"].shape, "y:", data_dict["y_test"].shape)
    print(f"Total features ({len(data_dict['feature_names'])}):", data_dict["feature_names"])
