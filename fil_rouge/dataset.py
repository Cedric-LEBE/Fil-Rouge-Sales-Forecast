from __future__ import annotations
import pandas as pd

def build_sales_region_day(
    df: pd.DataFrame,
    date_col: str,
    group_col: str,
    sales_col: str,
    target_col: str,
) -> pd.DataFrame:
    out = (
        df.groupby([date_col, group_col], as_index=False)
          .agg(**{target_col: (sales_col, "sum")})
          .sort_values([group_col, date_col])
          .reset_index(drop=True)
    )
    return out

def build_sales_global_day(
    sales_region_day: pd.DataFrame,
    date_col: str,
    target_col: str,
    global_target_col: str = "daily_sales_global",
) -> pd.DataFrame:
    out = (
        sales_region_day.groupby(date_col, as_index=False)
        .agg(**{global_target_col: (target_col, "sum")})
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    return out