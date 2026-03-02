from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib


@dataclass
class ForecastResult:
    scope: str  # "global" or "region"
    region: Optional[str]
    horizon_days: int
    last_date: pd.Timestamp
    forecast_date: pd.Timestamp
    y_at_horizon: float
    future_sum: float
    df: pd.DataFrame  # Date, y_true, y_pred


# -------------------------
# Core ML iterative forecast
# -------------------------
def iterative_forecast_ml(
    pipeline,
    base_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    group_col: str | None = None,
    group_value: str | None = None,
    horizon: int = 30,
) -> pd.DataFrame:
    base_df = base_df.copy()
    base_df[date_col] = pd.to_datetime(base_df[date_col], errors="coerce")
    base_df = base_df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    if group_col and group_value is not None:
        base_df = base_df[base_df[group_col] == group_value].copy().reset_index(drop=True)

    if base_df.empty:
        raise ValueError("Historique vide après filtrage (région/période).")

    last_date = base_df[date_col].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    ext = pd.DataFrame({date_col: pd.concat([base_df[date_col], pd.Series(future_dates)], ignore_index=True)})
    if group_col and group_value is not None:
        ext[group_col] = group_value

    ext[target_col] = pd.concat(
        [base_df[target_col].astype(float), pd.Series([np.nan] * horizon)],
        ignore_index=True,
    )

    # time features
    ext["Year"] = ext[date_col].dt.year
    ext["Month"] = ext[date_col].dt.month
    ext["Quarter"] = ext[date_col].dt.quarter
    ext["Day"] = ext[date_col].dt.day
    ext["DayOfWeek"] = ext[date_col].dt.dayofweek
    ext["DayOfYear"] = ext[date_col].dt.dayofyear
    ext["WeekOfYear"] = ext[date_col].dt.isocalendar().week.astype(int)
    ext["IsWeekend"] = ext["DayOfWeek"].isin([5, 6]).astype(int)

    lags = (1, 7, 14, 30)
    windows = (7, 14, 30)

    for i in range(len(base_df), len(ext)):
        for lag in lags:
            ext.loc[i, f"sales_lag_{lag}"] = ext.loc[i - lag, target_col] if i - lag >= 0 else np.nan

        for w in windows:
            vals = ext.loc[max(0, i - w): i - 1, target_col].dropna().astype(float)
            ext.loc[i, f"rolling_mean_{w}"] = vals.mean() if len(vals) >= 2 else np.nan
            ext.loc[i, f"rolling_std_{w}"] = vals.std(ddof=1) if len(vals) >= 2 else np.nan

        X_row = ext.loc[[i]].drop(columns=[target_col], errors="ignore")
        ext.loc[i, target_col] = float(pipeline.predict(X_row)[0])

    out = pd.DataFrame({date_col: ext[date_col]})
    out["y_true"] = np.nan
    out.loc[: len(base_df) - 1, "y_true"] = base_df[target_col].astype(float).values
    out["y_pred"] = np.nan
    out.loc[len(base_df):, "y_pred"] = ext.loc[len(base_df):, target_col].astype(float).values
    return out


# -------------------------
# Parsing helpers
# -------------------------
def parse_horizon_days(question: str, default: int = 30) -> int:
    m = re.search(r"(\d+)\s*(jour|jours|day|days)\b", (question or "").lower())
    if not m:
        return default
    n = int(m.group(1))
    return max(1, min(n, 365))


def _strip_accents(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )


def normalize_region_token(s: str) -> str:
    if s is None:
        return ""

    s = str(s).lower().strip()

    # remove accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # unify separators
    s = s.replace("_", " ").replace("-", " ")

    # keep only letters/numbers/spaces
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # common typos
    s = s.replace("southest", "southeast")

    # FR -> EN aliases (important: multi-words first)
    s = s.replace("sud est", "southeast").replace("sudest", "southeast").replace("sud-est", "southeast")
    s = s.replace("nord est", "northeast").replace("nord-est", "northeast").replace("nordest", "northeast")
    s = s.replace("centre ouest", "central west").replace("centre-ouest", "central west").replace("centreouest", "central west")

    # normalize "central west" back to dataset style if needed later
    s = s.replace("central west", "central-west")

    return s


def detect_region(question: str, available_regions: list[str]) -> Optional[str]:
    q = normalize_region_token(question)

    mapping = {normalize_region_token(r): r for r in available_regions}

    # longest first avoids "south" matching inside "southeast"
    keys_sorted = sorted(mapping.keys(), key=len, reverse=True)

    for k in keys_sorted:
        pattern = rf"\b{re.escape(k)}\b"
        if re.search(pattern, q):
            return mapping[k]

    return None

def _wants_global(question: str) -> bool:
    q = normalize_region_token(question)
    # mots-clés explicitement "global"
    return any(w in q for w in ["global", "toutes", "total", "overall", "all regions"])


# -------------------------
# Public API
# -------------------------
def forecast_sales(project_root: Path, question: str) -> ForecastResult:
    horizon = parse_horizon_days(question, default=30)

    data_interim = project_root / "data" / "interim"
    model_latest = project_root / "model_store" / "latest"

    # Load region base once to get available regions
    region_base_path = data_interim / "sales_region_day_base.parquet"
    region_base = pd.read_parquet(region_base_path)
    regions = sorted(region_base["Region"].dropna().unique().tolist()) if "Region" in region_base.columns else []

    detected_region = detect_region(question, regions)

    # Routing:
    # - Si région détectée -> REGION (même si user n'a pas écrit "région")
    # - Sinon -> GLOBAL (ou si user demande explicitement global)
    use_region = detected_region is not None and not _wants_global(question)
    region = detected_region if use_region else None

    if region is None:
        base = pd.read_parquet(data_interim / "sales_global_day_base.parquet")
        pipe = joblib.load(model_latest / "ml_global" / "pipeline.joblib")

        base["Date"] = pd.to_datetime(base["Date"], errors="coerce")
        base = base.dropna(subset=["Date"]).sort_values("Date")
        base["Region"] = "GLOBAL"

        out = iterative_forecast_ml(
            pipe, base, "Date", "daily_sales_global", "Region", "GLOBAL", horizon
        )
        last_date = base["Date"].max()
        scope = "global"
    else:
        base = region_base.copy()
        pipe = joblib.load(model_latest / "ml_region" / "pipeline.joblib")

        base["Date"] = pd.to_datetime(base["Date"], errors="coerce")
        base = base.dropna(subset=["Date"]).sort_values("Date")

        out = iterative_forecast_ml(
            pipe, base, "Date", "daily_sales", "Region", region, horizon
        )
        last_date = base.loc[base["Region"] == region, "Date"].max()
        scope = "region"

    forecast_date = pd.to_datetime(last_date) + pd.Timedelta(days=horizon)
    future = out.dropna(subset=["y_pred"]).copy()

    if future.empty:
        raise ValueError("Impossible de générer des prédictions (future vide).")

    idx = min(horizon - 1, len(future) - 1)
    y_at = float(future.iloc[idx]["y_pred"])
    future_sum = float(future["y_pred"].sum())

    return ForecastResult(
        scope=scope,
        region=region,
        horizon_days=horizon,
        last_date=pd.to_datetime(last_date),
        forecast_date=forecast_date,
        y_at_horizon=y_at,
        future_sum=future_sum,
        df=out,
    )