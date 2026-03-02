from __future__ import annotations
import pandas as pd
from fil_rouge.config import (
    ORDER_ID_COL, CUSTOMER_ID_COL, PRODUCT_ID_COL,
    ORDER_STATUS_COL, DELIVERED_VALUE,
)

def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out

def merge_and_clean_olist(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    items: pd.DataFrame,
    payments: pd.DataFrame,
    products: pd.DataFrame,
    delivered_only: bool = True,
) -> pd.DataFrame:
    # dédup produits
    products = products.drop_duplicates(subset=[PRODUCT_ID_COL], keep="first").copy()

    df = orders.merge(customers, on=CUSTOMER_ID_COL, how="left")
    df = df.merge(items, on=ORDER_ID_COL, how="left")
    df = df.merge(payments, on=ORDER_ID_COL, how="left")
    df = df.merge(products, on=PRODUCT_ID_COL, how="left")

    if delivered_only and ORDER_STATUS_COL in df.columns:
        df = df[df[ORDER_STATUS_COL] == DELIVERED_VALUE].copy()

    # fill NA 
    if "product_category_name" in df.columns:
        df["product_category_name"] = df["product_category_name"].fillna("Unknown")

    return df.reset_index(drop=True)