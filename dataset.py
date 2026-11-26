from __future__ import annotations

from typing import Any
from pathlib import Path
from sklearn.model_selection import train_test_split

import pandas as pd

from schemas import DatasetRowChurn

DATA_PATH = Path(__file__).resolve().parent / "data" / "churn_dataset.csv"

NUM_COLS = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
]

CAT_COLS = [
    "region",
    "device_type",
    "payment_method",
    "autopay_enabled",
]


def prepare_data(df: pd.DataFrame):
    y = df["churn"]
    X = df.drop(columns=["churn"]).copy()

    cat_cols = [c for c in CAT_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    if X.isna().sum().sum() > 0:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
        X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])

    return X, y, num_cols, cat_cols

def split_train_test(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

def class_distribution(y: pd.Series) -> dict[str, Any]:
    counts = {int(k): int(v) for k, v in y.value_counts().items()}
    shares = {
        int(k): float(v) for k, v in y.value_counts(normalize=True).items()
    }
    return {"counts": counts, "shares": shares}


def check_churn_split(y_train: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    return {
        "train": class_distribution(y_train),
        "test": class_distribution(y_test),
    }

def load_dataframe(path: Path | str | None = None) -> pd.DataFrame:
    csv_path = path if path is not None else DATA_PATH
    return pd.read_csv(csv_path)

def to_records(df: pd.DataFrame) -> list[dict]:
    return df.to_dict(orient="records")

def to_rows(df: pd.DataFrame) -> list[DatasetRowChurn]:
    return[DatasetRowChurn.model_validate(record) for record in to_records(df)]

class ChurnDataset:
    """Holds the CSV as a DataFrame, dicts, and Pydantic rows."""
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else DATA_PATH
        self.df = load_dataframe(self.path)
        self.records = to_records(self.df)
        self.rows = to_rows(self.df)

    def preview(self, n: int = 5) -> pd.DataFrame:
        return self.df.head(n)

    def prepare(self):
        return prepare_data(self.df)

    def split(self):
        X, y, num_cols, cat_cols = prepare_data(self.df)
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        return X_train, X_test, y_train, y_test, num_cols, cat_cols

    def info(self) -> dict[str, Any]:
        counts = {int(k): int(v) for k, v in self.df["churn"].value_counts().items()}
        return {
            "num_rows": len(self.df),
            "num_columns": len(self.df.columns),
            "feature_names": [col for col in self.df.columns if col != "churn"],
            "churn_distribution": counts
        }

    def split_info(self) -> dict[str, Any]:
        X_train, X_test, y_train, y_test, _, _  = self.split()
        return {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "churn_split": check_churn_split(y_train, y_test)
        }