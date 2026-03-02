from __future__ import annotations
import pandas as pd

def prophet_available() -> bool:
    try:
        import prophet  
        return True
    except Exception:
        return False

def fit_prophet(df: pd.DataFrame):
    from prophet import Prophet
    m = Prophet()
    m.fit(df)
    return m

def forecast_prophet(model, future_df: pd.DataFrame) -> pd.DataFrame:
    return model.predict(future_df)