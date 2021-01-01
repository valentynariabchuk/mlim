import os
import pandas as pd
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
    y_df = pd.read_parquet(f"{path_data}/processed/y_df.pt")

    # build heuristic
    # we use the last observation for `y` as our prediction
    heuristic = (
        y_df[y_df["order_number_inv"] == 1][["user_id", "y_int"]]
        .merge(prediction_index, on="user_id")
        .rename(columns={"y_int": "yhat"})
    ).sort_values("user_id")

    # save result
    heuristic.to_parquet(f"{path_results}/heuristic.pt")
