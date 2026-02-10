from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_raw_olist(raw_dir: Path, raw_files: dict) -> dict[str, pd.DataFrame]:
    dfs = {}
    for key, fname in raw_files.items():
        path = raw_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing raw file: {path}")
        dfs[key] = pd.read_csv(path)
    return dfs

def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    return pd.read_parquet(path)