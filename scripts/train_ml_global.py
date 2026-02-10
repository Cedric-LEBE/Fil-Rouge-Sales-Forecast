import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fil_rouge.pipelines.ml.train_ml_global import run_train_ml_global

if __name__ == "__main__":
    run_train_ml_global()
