from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = ROOT / "data" / "interim" / "data_merged_clean.parquet"

DB_URL = os.getenv(
    "ANALYTICS_DATABASE_URL",
    "postgresql+psycopg2://analytics:analytics@localhost:5432/analytics",
)


def main() -> None:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Missing parquet: {PARQUET_PATH}")

    df = pd.read_parquet(PARQUET_PATH)

    # Normalize column names to snake_case expected by the chatbot SQL generator
    rename = {}
    for c in df.columns:
        cc = c.strip().lower().replace(" ", "_")
        cc = cc.replace("-", "_")
        rename[c] = cc
    df = df.rename(columns=rename)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Keep a compact set (but don't drop unknown columns)
    engine = create_engine(DB_URL, pool_pre_ping=True)

    with engine.begin() as conn:
        # pgvector extension is optional; doesn't hurt if missing
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception:
            pass

    df.to_sql("sales", engine, if_exists="replace", index=False, chunksize=50_000, method="multi")

    with engine.begin() as conn:
        # Useful indexes
        for stmt in [
            "CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(date)",
            "CREATE INDEX IF NOT EXISTS idx_sales_region ON sales(region)",
            "CREATE INDEX IF NOT EXISTS idx_sales_macro_category ON sales(macro_category)",
        ]:
            try:
                conn.execute(text(stmt))
            except Exception:
                pass

    print("✅ Loaded analytics DB table: sales")


if __name__ == "__main__":
    main()
