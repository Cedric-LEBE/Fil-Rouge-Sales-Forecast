import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fil_rouge.pipelines.data.make_dataset import run_make_dataset

if __name__ == "__main__":
    run_make_dataset()