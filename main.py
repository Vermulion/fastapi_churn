from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class FeatureVectorChurn(BaseModel):

    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int

class DatasetRowChurn(FeatureVectorChurn):
    churn: int

@app.get("/")
def ml_churn_service_health_check():
    return {"message": "ml churn service is running"}

@app.post("/predict")
def predict(features: FeatureVectorChurn) -> DatasetRowChurn:
    return features