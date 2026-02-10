from __future__ import annotations
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarimax(y: pd.Series, exog: pd.DataFrame | None = None,
                order=(1,1,1), seasonal_order=(0,0,0,0)):
    m = SARIMAX(
        endog=y.astype(float).values,
        exog=exog.values if exog is not None else None,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    return m

def forecast_sarimax(model, steps: int, exog_future: pd.DataFrame | None = None):
    return model.forecast(steps=steps, exog=exog_future.values if exog_future is not None else None)