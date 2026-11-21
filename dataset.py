from __future__ import annotations

from typing import Any
from pathlib import Path

import pandas as pd

from schemas import DatasetRowChurn

DATA_PATH = Path(__file__).resolve().parent / "data" / "churn_dataset.csv"

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

    def info(self) -> dict[str, Any]:
        counts = {int(k): int(v) for k, v in self.df["churn"].value_counts().items()}
        return {
            "num_rows": len(self.df),
            "num_columns": len(self.df.columns),
            "feature_names": [col for col in self.df.columns if col != "churn"],
            "churn_distribution": counts
        }