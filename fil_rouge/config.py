from __future__ import annotations
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODEL_STORE_DIR = PROJECT_ROOT / "model_store"
ARTEFACTS_DIR = PROJECT_ROOT / "artefacts"

# Raw input files (noms attendus dans data/raw/)
RAW_FILES = {
    "orders": "df_Orders.csv",
    "customers": "df_Customers.csv",
    "items": "df_OrderItems.csv",
    "payments": "df_Payments.csv",
    "products": "df_Products.csv",
}

ORDER_ID_COL = "order_id"
CUSTOMER_ID_COL = "customer_id"
PRODUCT_ID_COL = "product_id"

ORDER_STATUS_COL = "order_status"
DELIVERED_VALUE = "delivered"

ORDER_TS_COL = "order_purchase_timestamp"  # datetime
PRICE_COL = "price"                        # float

# Cibles / colonnes pipeline
DATE_COL = "Date"
GROUP_COL = "Region"           # niveau régional
SALES_COL = "Sales"
TARGET_COL = "daily_sales"     # cible après agrégation (Date x Region)

# Paramètres split
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# Horizon par défaut (app)
DEFAULT_HORIZON_DAYS = int(os.getenv("HORIZON_DAYS", "30"))

def ensure_dirs() -> None:
    for p in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODEL_STORE_DIR, ARTEFACTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)