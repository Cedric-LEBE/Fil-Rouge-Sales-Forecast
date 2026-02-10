from fil_rouge.pipelines.data.make_dataset import run_make_dataset
from fil_rouge.pipelines.ml.train_ml_global import run_train_ml_global
from fil_rouge.pipelines.ml.train_ml_region import run_train_ml_region
from fil_rouge.pipelines.ts.train_ts_region import run_train_ts_region

if __name__ == "__main__":
    run_make_dataset()
    run_train_ml_global()
    run_train_ml_region()
    run_train_ts_region()
    print("✅ ALL DONE (data -> ML -> TS)")