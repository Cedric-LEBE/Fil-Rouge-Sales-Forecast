from __future__ import annotations
import pandas as pd

def time_split(df: pd.DataFrame, date_col: str, test_size: float = 0.2):
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()