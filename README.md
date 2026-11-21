# fastapi_churn

Training **HTTP service** (Python + FastAPI) for **customer churn**: predict whether a customer leaves from usage and account fields.

This is homework. The model is not trained yet. `POST /predict` currently **echoes** the request body so the Pydantic schema can be checked.

## Setup

Python 3.9+. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload
```

- API: <http://127.0.0.1:8000>
- Interactive docs: <http://127.0.0.1:8000/docs>

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Body: feature vector (no `churn`). Echoes the same JSON. |
| `GET` | `/dataset/preview` | First `N` CSV rows as JSON. Query: `?top=5` |
| `GET` | `/dataset/info` | Row/column counts, feature names, `churn` class counts |

Example health check:

```bash
curl -s http://127.0.0.1:8000/
```

Example predict (after Uvicorn is up):

```bash
curl -s http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"monthly_fee":9.99,"usage_hours":27.92,"support_requests":1,"account_age_months":14,"failed_payments":1,"region":"america","device_type":"desktop","payment_method":"card","autopay_enabled":1}'
```

If this returns **500**, the handler return type still expects `churn` in the response. It should return `FeatureVectorChurn`, not `DatasetRowChurn`.

## Dataset

File: `data/churn_dataset.csv` (~2000 rows).

| Column | Role |
|--------|------|
| `monthly_fee`, `usage_hours` | float features |
| `support_requests`, `account_age_months`, `failed_payments`, `autopay_enabled` | int features |
| `region`, `device_type`, `payment_method` | string features |
| `churn` | label: `1` = left, `0` = stayed |

## Project layout

```
main.py       FastAPI routes
schemas.py    Pydantic models
dataset.py    Load and summarize the CSV
data/         churn_dataset.csv
```

- **`FeatureVectorChurn`**: inputs for prediction (no label).
- **`DatasetRowChurn`**: one training row (features + `churn`).

## Libraries

- **FastAPI** — HTTP API, validation, OpenAPI `/docs`.
- **Uvicorn** — server that runs the FastAPI app.
- **pandas** — read the CSV and compute dataset info (shape, `value_counts`).

## Git

`.venv/` and `.cursor/` are gitignored. Do not commit the virtualenv.
