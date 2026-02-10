from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from fil_rouge.split import time_split
from fil_rouge.evaluate import rmse, mape, smape
from fil_rouge.registry import save_joblib, save_json

@dataclass
class MLRunResult:
    leaderboard: pd.DataFrame
    run_dir: Path
    best_model_name: str
    best_model_path: Path

def _build_pipeline(model, cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), num_cols),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("model", model)])

def benchmark_ml(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    test_size: float,
    run_root: Path,
    model_out_dir: Path,
    extra_meta: dict | None = None,
) -> MLRunResult:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, target_col]).sort_values(date_col).reset_index(drop=True)

    train_df, test_df = time_split(df, date_col, test_size=test_size)

    y_train = train_df[target_col].astype(float)
    y_test = test_df[target_col].astype(float)

    X_train = train_df.drop(columns=[target_col], errors="ignore")
    X_test = test_df.drop(columns=[target_col], errors="ignore")

    # Drop datetime columns (Date) - we keep engineered calendar features instead
    for col in list(X_train.columns):
        if np.issubdtype(X_train[col].dtype, np.datetime64):
            X_train = X_train.drop(columns=[col])
            X_test = X_test.drop(columns=[col])

    # cat/num split
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    models = [
        ("Ridge", Ridge(alpha=1.0, random_state=42)),
        ("RandomForest", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)),
        ("XGBoost", XGBRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=8,
            subsample=0.9, colsample_bytree=0.9,
            random_state=42, n_jobs=-1
        )),
    ]

    results = []
    best_rmse = float("inf")
    best = None

    for name, model in models:
        pipe = _build_pipeline(model, cat_cols, num_cols)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        row = {
            "model": name,
            "MAE": float(mean_absolute_error(y_test, pred)),
            "RMSE": float(rmse(y_test, pred)),
            "MAPE(%)": float(mape(y_test, pred)),
            "sMAPE(%)": float(smape(y_test, pred)),
        }
        results.append(row)

        if row["RMSE"] < best_rmse:
            best_rmse = row["RMSE"]
            best = (name, pipe)

    leaderboard = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    leaderboard.to_csv(run_dir / "metrics.csv", index=False)

    save_json(run_dir / "run_config.json", {
        "run_id": run_id,
        "date_col": date_col,
        "target_col": target_col,
        "test_size": test_size,
        "n_rows": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "extra": extra_meta or {},
    })

    assert best is not None
    best_name, best_pipe = best

    model_out_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_out_dir / "pipeline.joblib"
    save_joblib(best_pipe, best_path)

    save_json(model_out_dir / "metrics.json", {
        "best_model": best_name,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "run_id": run_id,
    })

    return MLRunResult(
        leaderboard=leaderboard,
        run_dir=run_dir,
        best_model_name=best_name,
        best_model_path=best_path,
    )