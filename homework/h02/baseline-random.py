import os
import pandas as pd
import numpy as np
import homework.h02.data as lib


if __name__ == "__main__":

    # read environment variables
    path_repo = os.environ.get('PATH_REPO')

    # read config
    config = lib.read_yaml(f"{path_repo}/homework/h02/config.yaml")
    path_data = config["path"]
    path_results = f"{path_data}/results"
    os.makedirs(path_results, exist_ok=True)

    # load data
    prediction_index = pd.read_parquet(f"{path_data}/prediction_index.parquet")

    # use random probabilties as prediction
    # we expect an AUC of .5 for random probabilities
    random = prediction_index.copy()
    random["yhat"] = np.random.uniform(0, 1, random.shape[0])

    # save result
    random.to_parquet(f"{path_results}/random.pt")
