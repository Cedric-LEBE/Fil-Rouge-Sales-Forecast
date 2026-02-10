from __future__ import annotations
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def fit_hw(y: pd.Series):
    model = ExponentialSmoothing(y.astype(float), trend="add", seasonal=None).fit()
    return model

def forecast_hw(model, steps: int):
    return model.forecast(steps)