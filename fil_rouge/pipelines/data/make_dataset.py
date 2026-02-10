from __future__ import annotations

import pandas as pd

from fil_rouge.config import (
    RAW_DIR, INTERIM_DIR, PROCESSED_DIR, RAW_FILES,
    ORDER_TS_COL, DATE_COL, GROUP_COL,
    PRICE_COL, SALES_COL, TARGET_COL,
    ensure_dirs,
)
from fil_rouge.io import load_raw_olist, write_parquet
from fil_rouge.preprocess import ensure_datetime, merge_and_clean_olist
from fil_rouge.features import add_business_features, add_time_features, add_lags_and_rollings
from fil_rouge.dataset import build_sales_region_day, build_sales_global_day

def run_make_dataset() -> None:
    ensure_dirs()

    dfs = load_raw_olist(RAW_DIR, RAW_FILES)

    data = merge_and_clean_olist(
        orders=dfs["orders"],
        customers=dfs["customers"],
        items=dfs["items"],
        payments=dfs["payments"],
        products=dfs["products"],
        delivered_only=True,
    )

    data = ensure_datetime(data, ORDER_TS_COL).dropna(subset=[ORDER_TS_COL])
    data[DATE_COL] = data[ORDER_TS_COL].dt.floor("D")

    if PRICE_COL not in data.columns:
        raise KeyError(f"Missing '{PRICE_COL}' column (expected from OrderItems).")
    data[SALES_COL] = pd.to_numeric(data[PRICE_COL], errors="coerce").astype(float)

    data = add_business_features(data)

    # INTERIM 1: transaction-level merged clean
    write_parquet(data, INTERIM_DIR / "data_merged_clean.parquet")

    # Base region time series
    sales_region_day = build_sales_region_day(
        data, date_col=DATE_COL, group_col=GROUP_COL, sales_col=SALES_COL, target_col=TARGET_COL
    )
    sales_region_day = add_time_features(sales_region_day, DATE_COL)

    # INTERIM 2: region base
    write_parquet(sales_region_day, INTERIM_DIR / "sales_region_day_base.parquet")

    # Base global time series
    sales_global_day = build_sales_global_day(
        sales_region_day, date_col=DATE_COL, target_col=TARGET_COL, global_target_col="daily_sales_global"
    )
    sales_global_day = add_time_features(sales_global_day, DATE_COL)

    # INTERIM 3: global base
    write_parquet(sales_global_day, INTERIM_DIR / "sales_global_day_base.parquet")

    # Processed ML datasets (lags/rolling)
    train_region = add_lags_and_rollings(
        sales_region_day, group_col=GROUP_COL, date_col=DATE_COL, target_col=TARGET_COL
    ).dropna().reset_index(drop=True)
    write_parquet(train_region, PROCESSED_DIR / "train_region.parquet")

    train_global = add_lags_and_rollings(
        sales_global_day.assign(**{GROUP_COL: "GLOBAL"}),  # trick pour réutiliser la même fonction
        group_col=GROUP_COL, date_col=DATE_COL, target_col="daily_sales_global"
    ).dropna().reset_index(drop=True)
    # Retire la colonne group qui est artificielle si tu veux
    write_parquet(train_global, PROCESSED_DIR / "train_global.parquet")

    print("✅ Dataset pipeline finished")
    print("   - data/interim/data_merged_clean.parquet")
    print("   - data/interim/sales_region_day_base.parquet")
    print("   - data/interim/sales_global_day_base.parquet")
    print("   - data/processed/train_region.parquet")
    print("   - data/processed/train_global.parquet")