from __future__ import annotations

from typing import Any
from fastapi import FastAPI

from schemas import FeatureVectorChurn, DatasetRowChurn
from dataset import ChurnDataset

app = FastAPI() 
dataset = ChurnDataset()

@app.get("/")
def ml_churn_service_health_check():
    return {"message": "ml churn service is running"}

@app.post("/predict")
def predict(features: FeatureVectorChurn) -> DatasetRowChurn:
    return features

@app.get("/dataset/preview")
def dataset_preview(top: int = 5) -> list[DatasetRowChurn]:
    return dataset.rows[:top]

@app.get("/dataset/info")
def dataset_info() -> dict[str, Any]:
    return dataset.info()