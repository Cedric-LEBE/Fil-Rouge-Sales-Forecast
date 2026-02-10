from __future__ import annotations
import numpy as np
import pandas as pd

def iterative_forecast_ml(
    pipeline,
    base_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    group_col: str | None,
    group_value: str | None,
    horizon: int,
    lags=(1, 7, 14, 30),
    windows=(7, 14, 30),
) -> pd.DataFrame:
    """
    Forecast itératif: on prédit jour par jour en recalculant lags/rolling à partir
    de l'historique + prédictions précédentes.
    base_df doit contenir : Date, target_col, + time features (Year/Month/...)
    et si régional: la colonne group_col.
    """
    hist = base_df.copy()
    hist[date_col] = pd.to_datetime(hist[date_col], errors="coerce")
    hist = hist.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    if group_col and group_value is not None:
        hist = hist[hist[group_col] == group_value].copy().reset_index(drop=True)

    if hist.empty:
        raise ValueError("Historique vide pour la sélection donnée.")

    last_date = hist[date_col].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    ext = pd.DataFrame({date_col: pd.concat([hist[date_col], pd.Series(future_dates)], ignore_index=True)})
    if group_col and group_value is not None:
        ext[group_col] = group_value

    ext[target_col] = pd.concat([hist[target_col].astype(float), pd.Series([np.nan] * horizon)], ignore_index=True)

    # si time features déjà dans base_df, on les recalcule en amont (côté dataset)
    # ici on suppose qu'elles sont déjà présentes dans base_df et alignées;
    # donc on les reconstruit minimalement:
    ext["Year"] = ext[date_col].dt.year
    ext["Month"] = ext[date_col].dt.month
    ext["Quarter"] = ext[date_col].dt.quarter
    ext["Day"] = ext[date_col].dt.day
    ext["DayOfWeek"] = ext[date_col].dt.dayofweek
    ext["DayOfYear"] = ext[date_col].dt.dayofyear
    ext["WeekOfYear"] = ext[date_col].dt.isocalendar().week.astype(int)
    ext["IsWeekend"] = ext["DayOfWeek"].isin([5, 6]).astype(int)

    # forecast
    for i in range(len(hist), len(ext)):
        for lag in lags:
            ext.loc[i, f"sales_lag_{lag}"] = ext.loc[i - lag, target_col] if (i - lag) >= 0 else np.nan

        for w in windows:
            window_vals = ext.loc[max(0, i - w): i - 1, target_col].dropna().astype(float)
            ext.loc[i, f"rolling_mean_{w}"] = window_vals.mean() if len(window_vals) >= 2 else np.nan
            ext.loc[i, f"rolling_std_{w}"] = window_vals.std(ddof=1) if len(window_vals) >= 2 else np.nan

        X_row = ext.loc[[i]].drop(columns=[target_col], errors="ignore")
        yhat = float(pipeline.predict(X_row)[0])
        ext.loc[i, target_col] = yhat

    out_hist = hist[[date_col, target_col]].rename(columns={target_col: "y_true"}).copy()
    out_hist["y_pred"] = np.nan
    out_pred = ext.iloc[len(hist):][[date_col, target_col]].rename(columns={target_col: "y_pred"}).copy()
    out_pred["y_true"] = np.nan
    out = pd.concat([out_hist, out_pred], ignore_index=True)
    return out