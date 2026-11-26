from dataset import (
    CAT_COLS,
    NUM_COLS,
    ChurnDataset,
    check_churn_split,
    prepare_data,
)
import json
import sys
import time

# region agent log
_log_path = "/Users/user/Documents/ML/HW/HW_3/fast_api/.cursor/debug-507622.log"
try:
    import httpx  # noqa: F401

    _httpx_ok, _httpx_err = True, None
except Exception as _e:
    _httpx_ok, _httpx_err = False, type(_e).__name__ + ": " + str(_e)
_payload = {
    "sessionId": "507622",
    "runId": "pre-fix",
    "hypothesisId": "C",
    "location": "tests/test_dataset.py:import",
    "message": "pytest import environment",
    "data": {
        "executable": sys.executable,
        "httpx_ok": _httpx_ok,
        "httpx_err": _httpx_err,
        "cwd_in_path": any(
            p.endswith("fast_api") or p == "" or p == "."
            for p in sys.path
        ),
        "sys_path_head": sys.path[:5],
    },
    "timestamp": int(time.time() * 1000),
}
with open(_log_path, "a") as _f:
    _f.write(json.dumps(_payload) + "\n")
# endregion

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_prepare_drops_churn_and_splits_column_types():
    X, y, num_cols, cat_cols = prepare_data(ChurnDataset().df)

    assert "churn" not in X.columns
    assert len(y) == len(X)
    assert set(num_cols) == set(NUM_COLS)
    assert set(cat_cols) == set(CAT_COLS)
    assert "autopay_enabled" in cat_cols
    assert "autopay_enabled" not in num_cols


def test_split_sizes_and_alignment():
    X_train, X_test, y_train, y_test, num_cols, cat_cols = ChurnDataset().split()

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == 2000
    assert len(X_test) == 400  # test_size=0.2


def test_churn_shares_are_close_in_train_and_test():
    X_train, X_test, y_train, y_test, _, _ = ChurnDataset().split()
    info = check_churn_split(y_train, y_test)

    train_churn = info["train"]["shares"][1]
    test_churn = info["test"]["shares"][1]
    assert abs(train_churn - test_churn) < 0.05


def test_split_info_endpoint():
    response = client.get("/dataset/split-info")
    assert response.status_code == 200
    body = response.json()
    assert body["train_size"] == 1600
    assert body["test_size"] == 400
    assert "churn_split" in body
    assert body["churn_split"]["train"]["counts"]["0"] + body[
        "churn_split"
    ]["train"]["counts"]["1"] == 1600