from __future__ import annotations
from pathlib import Path

from fil_rouge.config import PROCESSED_DIR, MODEL_STORE, DATE_COL, TARGET_COL, GROUP_COL, TEST_SIZE, RANDOM_STATE
from fil_rouge.io import read_parquet
from fil_rouge.train import benchmark_train_and_save

def main():
    df = read_parquet(PROCESSED_DIR / "train.parquet")

    leaderboard, run_dir, best_name = benchmark_train_and_save(
        df,
        model_store=MODEL_STORE,
        date_col=DATE_COL,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("\n✅ Leaderboard (top 5):")
    print(leaderboard.head(5))
    print(f"\n🏆 Best model: {best_name}")
    print(f"📦 Run saved: {run_dir}")
    print(f"⭐ Latest updated: {MODEL_STORE / 'latest'}")

if __name__ == "__main__":
    main()