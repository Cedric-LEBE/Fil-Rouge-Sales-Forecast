from __future__ import annotations
from fil_rouge.config import PROCESSED_DIR, ARTEFACTS_DIR, MODEL_STORE_DIR, TEST_SIZE
from fil_rouge.io import read_parquet
from fil_rouge.pipelines.ml.benchmark_ml import benchmark_ml

def run_train_ml_global() -> None:
    df = read_parquet(PROCESSED_DIR / "train_global.parquet")

    run_root = ARTEFACTS_DIR / "runs_ml" / "global"
    model_out = MODEL_STORE_DIR / "latest" / "ml_global"

    res = benchmark_ml(
        df=df,
        date_col="Date",
        target_col="daily_sales_global",
        test_size=TEST_SIZE,
        run_root=run_root,
        model_out_dir=model_out,
        extra_meta={"level": "global"},
    )
    print("✅ ML GLOBAL done")
    print(res.leaderboard.head(10))
    print("Best:", res.best_model_name)
    print("Saved:", res.best_model_path)