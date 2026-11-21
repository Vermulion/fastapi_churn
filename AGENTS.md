# AGENTS.md

Short context for coding agents. Do not duplicate `.cursor/rules/student-workflow.mdc`.

## What this is

Python **FastAPI** service for **customer churn** (homework). Dataset: `data/churn_dataset.csv`. GitHub repo name: `fastapi_churn`.

## Stack

- Python 3.9 (venv `.venv/`)
- FastAPI 0.128.x, Uvicorn 0.39.x, pandas 2.3.x, Pydantic v2 (via FastAPI)
- Run: `uvicorn main:app --reload` → `http://127.0.0.1:8000` (`/docs`)

## Layout

```
main.py              # FastAPI app; loads ChurnDataset once at import
schemas.py           # FeatureVectorChurn, DatasetRowChurn
dataset.py           # CSV → DataFrame, records, list[DatasetRowChurn]
data/churn_dataset.csv
requirements.txt
```

## Data

CSV columns: `monthly_fee`, `usage_hours`, `support_requests`, `account_age_months`, `failed_payments`, `region`, `device_type`, `payment_method`, `autopay_enabled`, `churn`.

- **Features** (`FeatureVectorChurn`): all except `churn`
- **Target** `churn`: `int` 0/1 (training rows only)
- ~2000 rows. Column name is lowercase `churn`, not `Churn`.

`ChurnDataset`: `df` (pandas), `records` (list of dicts), `rows` (Pydantic). Path: `dataset.py` next to `data/`.

## Public HTTP

| Method | Path | In | Out |
|--------|------|-----|-----|
| GET | `/` | — | `{"message": "ml churn service is running"}` |
| POST | `/predict` | JSON `FeatureVectorChurn` (no `churn`) | **Temporary echo** of the same fields |
| GET | `/dataset/preview?top=5` | query `top` (default 5) | JSON array of `DatasetRowChurn` |
| GET | `/dataset/info` | — | `num_rows`, `num_columns`, `feature_names` (no `churn`), `churn_distribution` (class counts) |

JSON keys are snake_case. Do not return a pandas `DataFrame` from a route (not JSON). Use `list[DatasetRowChurn]` or `dict[str, Any]`, not `dict[str, str]` for mixed values.

**Known bug:** `predict` is annotated `-> DatasetRowChurn` but returns features without `churn` → response 500. Intended echo is `-> FeatureVectorChurn`. Do not change unless asked.

## Conventions

- Plan first. Do not add dependencies unless asked.
- Optional extras must not change existing public routes/defaults unless asked.
- After a code feature, add an offline pytest. Tests need `httpx` (not in `requirements.txt` yet).
- Do not commit `.venv/`, `__pycache__/`, `.env`, `.cursor/`.

## Next likely steps

Train a model; make `POST /predict` return a predicted `churn` (not an echo).
