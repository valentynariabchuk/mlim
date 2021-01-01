import os
import pandas as pd
import numpy as np
import sklearn.metrics
import homework.h02.data as lib


def score(x, t):
    assert np.all(
        t[["user_id", "order_number"]].values
        == x[["user_id", "order_number"]].values
    )
    return sklearn.metrics.roc_auc_score(t["y"].values, x["yhat"].values)


if __name__ == "__main__":

    # read environment variables
    path_repo = os.environ.get('PATH_REPO')

    # read config
    config = lib.read_yaml(f"{path_repo}/homework/h02/config.yaml")
    path_data = config["path"]
    path_results = f"{path_data}/results"
    os.makedirs(path_results, exist_ok=True)

    # load data
    truth = pd.read_parquet(f"{path_data}/truth.parquet")
    random = pd.read_parquet(f"{path_results}/random.pt")
    heuristic = pd.read_parquet(f"{path_results}/heuristic.pt")
    lightgbm = pd.read_parquet(f"{path_results}/lightgbm.pt")

    # run benchmark
    auc_random = score(random, truth)
    auc_heuristic = score(heuristic, truth)
    auc_lightgbm = score(lightgbm, truth)

    # save results
    scores = {
        "random": float(auc_random),
        "heuristic": float(auc_heuristic),
        "lightgbm": float(auc_lightgbm),
    }
    lib.write_yaml(scores, f"{path_results}/scores.yaml")
