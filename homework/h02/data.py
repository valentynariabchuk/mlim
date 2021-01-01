import os
import yaml
import pandas as pd
import numpy as np


def read_yaml(x):
    with open(x, "r") as con:
        config = yaml.safe_load(con)
    return config


def write_yaml(x, f):
    with open(f, "w") as con:
        yaml.safe_dump(x, con)


if __name__ == "__main__":

    # read environment variables
    path_repo = os.environ.get("PATH_REPO")

    # read config
    config = read_yaml(f"{path_repo}/homework/h02/config.yaml")
    path_data = config["path"]
    path_processed = f"{path_data}/processed"
    os.makedirs(path_processed, exist_ok=True)

    # load data
    products = pd.read_parquet(f"{path_data}/products.parquet")
    baskets = pd.read_parquet(f"{path_data}/baskets.parquet")
    orders = pd.read_parquet(f"{path_data}/orders.parquet")
    prediction_index = pd.read_parquet(f"{path_data}/prediction_index.parquet")

    # the target is a binary variable derived from `y` (days to next order). we predict
    # days to next order so we shift `days_since_prior_order` by 1 order. we can construct
    # multiple training samples for each user. the prediction target is the last
    # observation for each shopper (where y is not available).
    #
    # we sort orders so we can simply shift by `-1` in the next step
    # first and last values for `days_since_prior_order` are NaN
    orders = orders.sort_values(["user_id", "order_number"], ascending=True)
    # verify that first `days_since_prior_order` values for all shoppers are NaN
    # and that the last `days_since_prior_order` values for all shoppers are not NaN
    # this is a prerequsite for simply shifting by -1, regardless of `user_id`
    assert np.all(orders.groupby("user_id").days_since_prior_order.head(1).isnull())
    assert np.all(orders.groupby("user_id").days_since_prior_order.tail(1).notnull())
    # define the target variable
    orders["y_int_l1"] = (orders["days_since_prior_order"] <= 14).astype(int)
    orders["y"] = orders["days_since_prior_order"].shift(-1)
    orders["y_int"] = (orders["y"] <= 14).astype(int)

    # add number of products per baskets
    n_product_order = (
        baskets.groupby("order_id")[["product_id"]].nunique().reset_index()
    )
    n_product_order.rename(columns={"product_id": "size"}, inplace=True)
    orders = orders.merge(n_product_order, on="order_id", how="left").reset_index(
        drop=True
    )

    # inverse order counter (used in building training data)
    orders["order_number_inv"] = (
        orders.groupby("user_id").order_number.transform(max) - orders["order_number"]
    )

    # shorten variable names
    orders.rename(columns={"days_since_prior_order": "dspo"}, inplace=True)

    # build training data
    # we create training/testing samples for each value of `order_number_inv`, starting
    # with 0. `order_number_inv=0` is our test set and `order_number_inv>0` is our
    # training/validation set.
    # not all shoppers have observations for a given `o`, this depends on length of their
    # order history. also note that we keep all data prior to `o`, regardless of the
    # length of the order history. this means that we use a different amount of data in
    # building features across shoppers. our reason for doing this is that we do not have
    # trending features (in this case we need to either normalize or use a constant time
    # window) and that more data yields more accurate feature values for the given
    # statistics. we avoid leakage by only using data available at each given point in
    # time (i.e., before the next order). consider replacing min/max by percentiles (e.g.,
    # 10% and 90%) to deal with outliers.
    # this preprocessing introduces three hyperparameters:
    # - the number of observations used for constructing our feature data `o`
    # - the lags used in feature computation `l`
    # - the feature set (e.g., base variables, aggregation functions)
    x_list = []
    y_list = []
    L = config["data"]["L"]
    # code below assumes that first value is the full window (and the largest lag)
    # a bit hacky... but does the job ;)
    assert L[0] > orders.order_number_inv.max()
    assert np.all(L[1:] < orders.order_number_inv.max())
    # loop through lags and build features
    for o in range(0, config["data"]["O"]):

        data_sets = []

        # loop over selected lags
        for i, l in enumerate(L):

            orders_o_l = orders[
                (orders["order_number_inv"] >= o)
                & (orders["order_number_inv"] < (o + l))
            ]

            # for first iteration, include `last` to get the latest values of the feature
            # variables
            if i == 0:
                features_o_l = orders_o_l.groupby("user_id").agg(
                    {
                        "dspo": ["last", "mean", "min", "max"],
                        "size": ["last", "mean", "min", "max"],
                        "y_int_l1": ["last", "mean"],
                    }
                )
            else:
                features_o_l = orders_o_l.groupby("user_id").agg(
                    {
                        "dspo": ["mean", "min", "max"],
                        "size": ["mean", "min", "max"],
                        "y_int_l1": ["mean"],
                    }
                )

            # flatten column names
            features_o_l.columns = ["_".join(c) for c in features_o_l.columns.values]

            # add lag to column names
            if i > 0:
                features_o_l.columns = [f"{c}_l{l}" for c in features_o_l.columns]

            data_sets.append(features_o_l)

        # merge data sets
        data_o = data_sets[0]
        for i in range(1, len(data_sets)):
            data_o = data_o.merge(data_sets[i], on="user_id")

        # additional (derived) features
        # add 1 because `dspo_mean` can be 0 (and we can't divide by 0)
        data_o["ratio_dspo"] = data_o["dspo_last"] / (1 + data_o["dspo_mean"])
        data_o["ratio_size"] = data_o["size_last"] / data_o["size_mean"]
        data_o["trend_dspo"] = data_o["dspo_mean"] - data_o[f"dspo_mean_l{min(L)}"]
        data_o["trend_size"] = data_o["size_mean"] - data_o[f"size_mean_l{min(L)}"]

        # write to list
        data_o["order_number_inv"] = o
        x_list.append(data_o)
        y_list.append(
            orders[orders["order_number_inv"] == o][
                ["user_id", "order_number_inv", "y_int"]
            ]
        )

    # data set for making final predictions
    x_predict = x_list[0].reset_index()

    # train and test dat
    x_df = pd.concat(x_list[1:]).reset_index()
    y_df = pd.concat(y_list[1:])

    # save processed data sets
    x_predict.to_parquet(f"{path_processed}/x_predict.pt")
    x_df.to_parquet(f"{path_processed}/x_df.pt")
    y_df.to_parquet(f"{path_processed}/y_df.pt")
