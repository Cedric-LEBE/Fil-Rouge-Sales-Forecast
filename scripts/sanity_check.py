from pathlib import Path

REQUIRED = [
    "data/interim/data_merged_clean.parquet",
    "data/interim/sales_region_day_base.parquet",
    "data/interim/sales_global_day_base.parquet",
    "data/processed/train_region.parquet",
    "data/processed/train_global.parquet",
    "model_store/latest/ml_global/pipeline.joblib",
    "model_store/latest/ml_region/pipeline.joblib",
]

if __name__ == "__main__":
    missing = [p for p in REQUIRED if not Path(p).exists()]
    if missing:
        print("❌ Missing files:")
        for m in missing:
            print(" -", m)
        raise SystemExit(1)
    print("✅ Sanity check OK")