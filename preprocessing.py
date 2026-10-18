"""Prepare the Seaborn Titanic dataset for machine-learning models."""

from __future__ import annotations

import pandas as pd
import seaborn as sns
from scipy.sparse import spmatrix
from numpy import ndarray
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "survived"


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create a transformer for numeric, categorical, and boolean columns."""
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = features.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def preprocess_data(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[ndarray | spmatrix, ndarray | spmatrix, pd.Series, pd.Series, ColumnTransformer]:
    """Remove duplicate rows, split data, and fit preprocessing on training data."""
    cleaned_data = data.drop_duplicates().copy()
    if TARGET_COLUMN not in cleaned_data:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column.")

    features = cleaned_data.drop(columns=[TARGET_COLUMN])
    target = cleaned_data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )
    preprocessor = build_preprocessor(features)
    x_train_preprocessed = preprocessor.fit_transform(x_train)
    x_test_preprocessed = preprocessor.transform(x_test)
    return x_train_preprocessed, x_test_preprocessed, y_train, y_test, preprocessor


def main() -> None:
    """Run preprocessing for the Titanic dataset and print a compact summary."""
    titanic_data = sns.load_dataset("titanic")
    duplicate_count = titanic_data.duplicated().sum()
    x_train, x_test, y_train, y_test, _ = preprocess_data(titanic_data)

    print(f"Dataset shape: {titanic_data.shape}")
    print(f"Duplicates removed: {duplicate_count}")
    print(f"Training rows: {len(y_train)}, test rows: {len(y_test)}")
    print(f"Preprocessed shapes: train={x_train.shape}, test={x_test.shape}")


if __name__ == "__main__":
    main()
