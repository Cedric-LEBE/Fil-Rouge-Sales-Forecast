from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

from fil_rouge.config import INTERIM_DIR, ARTEFACTS_DIR, MODEL_STORE_DIR, DATE_COL, GROUP_COL, TARGET_COL, TEST_SIZE
from fil_rouge.io import read_parquet
from fil_rouge.split import time_split
from fil_rouge.evaluate import mae, rmse, mape

from fil_rouge.pipelines.ts.models_hw import fit_hw, forecast_hw
from fil_rouge.pipelines.ts.models_sarimax import fit_sarimax, forecast_sarimax
from fil_rouge.pipelines.ts.models_prophet import prophet_available, fit_prophet, forecast_prophet

def _time_exog(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    return pd.DataFrame({
        "dow": df[date_col].dt.dayofweek.astype(int),
        "is_weekend": df[date_col].dt.dayofweek.isin([5,6]).astype(int),
        "month": df[date_col].dt.month.astype(int),
    })

def run_train_ts_region() -> None:
    base = read_parquet(INTERIM_DIR / "sales_region_day_base.parquet").copy()
    base[DATE_COL] = pd.to_datetime(base[DATE_COL], errors="coerce")
    base = base.dropna(subset=[DATE_COL, GROUP_COL, TARGET_COL])

    regions = sorted(base[GROUP_COL].unique().tolist())
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for region in regions:
        df_r = base[base[GROUP_COL] == region].sort_values(DATE_COL).reset_index(drop=True)
        if len(df_r) < 60:
            continue

        train_df, test_df = time_split(df_r, DATE_COL, test_size=TEST_SIZE)
        y_train = train_df[TARGET_COL].astype(float)
        y_test = test_df[TARGET_COL].astype(float)

        arte_dir = ARTEFACTS_DIR / "runs_ts" / f"region_{region}" / f"run_{run_id}"
        arte_dir.mkdir(parents=True, exist_ok=True)

        results = []

        # 1) Holt-Winters
        try:
            hw = fit_hw(y_train)
            pred_hw = forecast_hw(hw, steps=len(y_test))
            results.append({
                "model": "Holt-Winters",
                "MAE": mae(y_test, pred_hw),
                "RMSE": rmse(y_test, pred_hw),
                "MAPE(%)": mape(y_test, pred_hw),
            })
        except Exception as e:
            results.append({"model": "Holt-Winters", "error": str(e)})

        # 2) SARIMAX (avec exog calendrier)
        try:
            exog_train = _time_exog(train_df, DATE_COL)
            exog_test = _time_exog(test_df, DATE_COL)
            sarx = fit_sarimax(y_train, exog=exog_train, order=(1,1,1), seasonal_order=(0,0,0,0))
            pred_sarx = forecast_sarimax(sarx, steps=len(y_test), exog_future=exog_test)
            results.append({
                "model": "SARIMAX",
                "MAE": mae(y_test, pred_sarx),
                "RMSE": rmse(y_test, pred_sarx),
                "MAPE(%)": mape(y_test, pred_sarx),
            })
        except Exception as e:
            results.append({"model": "SARIMAX", "error": str(e)})

        if prophet_available():
            try:
                p_train = train_df[[DATE_COL, TARGET_COL]].rename(columns={DATE_COL: "ds", TARGET_COL: "y"})
                p_test = test_df[[DATE_COL]].rename(columns={DATE_COL: "ds"})
                pm = fit_prophet(p_train)
                fc = forecast_prophet(pm, p_test)
                pred_p = fc["yhat"].astype(float).values
                results.append({
                    "model": "Prophet",
                    "MAE": mae(y_test, pred_p),
                    "RMSE": rmse(y_test, pred_p),
                    "MAPE(%)": mape(y_test, pred_p),
                })
            except Exception as e:
                results.append({"model": "Prophet", "error": str(e)})

        bench = pd.DataFrame(results)
        bench.to_csv(arte_dir / "benchmark.csv", index=False)

        # sélection best (RMSE min)
        bench_ok = bench.dropna(subset=["RMSE"]).copy()
        if bench_ok.empty:
            continue
        best_row = bench_ok.sort_values("RMSE").iloc[0]
        best_name = str(best_row["model"])

        # refit sur tout l'historique (train+test) pour le modèle best
        full = df_r.copy()
        y_full = full[TARGET_COL].astype(float)
        model_obj = None
        model_type = best_name

        if best_name == "Holt-Winters":
            model_obj = fit_hw(y_full)
        elif best_name == "SARIMAX":
            exog_full = _time_exog(full, DATE_COL)
            model_obj = fit_sarimax(y_full, exog=exog_full, order=(1,1,1), seasonal_order=(0,0,0,0))
        elif best_name == "Prophet" and prophet_available():
            p_full = full[[DATE_COL, TARGET_COL]].rename(columns={DATE_COL: "ds", TARGET_COL: "y"})
            model_obj = fit_prophet(p_full)

        # save champion per region
        out_dir = MODEL_STORE_DIR / "latest" / "ts_region" / str(region)
        out_dir.mkdir(parents=True, exist_ok=True)

        # sauvegarde robuste
        if best_name == "SARIMAX":
            model_obj.save(out_dir / "model.pkl")
        else:
            joblib.dump(model_obj, out_dir / "model.joblib")

        (out_dir / "model_type.json").write_text(f'{{"model":"{model_type}","run_id":"{run_id}"}}', encoding="utf-8")
        (out_dir / "metrics.json").write_text(bench_ok.to_json(orient="records"), encoding="utf-8")

    print("✅ TS REGION training finished")