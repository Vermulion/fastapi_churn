from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def ml_churn_service_health_check():
    return {"message": "ml churn service is running"}