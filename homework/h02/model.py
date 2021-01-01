import os
import pickle
import yaml
import pandas as pd
import numpy as np
import joblib
import lightgbm
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
    x_df = pd.read_parquet(f"{path_data}/processed/x_df.pt")
    y_df = pd.read_parquet(f"{path_data}/processed/y_df.pt")
    x_predict = pd.read_parquet(f"{path_data}/processed/x_predict.pt")
    y_predict = None
    assert np.all(x_df.columns == x_predict.columns)

    # extract features from data
    x_df = x_df.set_index(["user_id", "order_number_inv"])
    features = x_df.columns
    x_predict = x_predict[features].values

    # remove NaNs
    mask_not_null = x_df["dspo_last"].notnull().values
    y_df = y_df[mask_not_null]
    x_df = x_df[mask_not_null]
    assert x_df["dspo_last"].isnull().sum() == 0

    # create validation set by split in time dimension
    mask = (x_df.reset_index().order_number_inv > 1).values

    X_val = x_df[~mask].values
    X_train = x_df[mask].values

    y_val = y_df[~mask].y_int.values
    y_train = y_df[mask].y_int.values

    # hyperparameter search we start with a random search for hyperparameters. this will
    # randomly select combinations of hyperparameters from a "grid" (not really a grid,
    # rather ranges in each hyperparameter dimension), evaluate them on the training data
    # (using cross validation), and return the values that perform the best.  note that it
    # would be better touse a split in the time dimension here as well. we can do this by
    # providing an iterable that yields indices to `cv` in
    # `sklearn.model_selection.randomizedsearchcv`. why not go ahead and try this
    # modification? does this improve the results for our hyperparameter search?
    #file_hps_results = f"{path_results}/random_search_lgbm.pickle"
    #if os.path.isfile(file_hps_results):
    #
    #    # load prior results
    #    with open(file_hps_results, "rb") as con:
    #        randomized_search = pickle.load(con)
    #
    #    # print results and extract best parameters
    #    print(randomized_search.best_score_)
    #    print(json.dumps(randomized_search.best_params_, indent=4))
    #    lgbm_parameters_star = randomized_search.best_params_
    #
    #else:
    #
    #    # run hyperparameter search
    #    param_grid = {
    #        "boosting_type": ["gbdt", "dart"],
    #        "num_leaves": range(10, 200, 5),
    #        "max_depth": range(1, 20, 1),
    #        "learning_rate": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
    #        "n_estimators": [500, 750, 1000, 1250, 1500, 2000],
    #        "objective": ["binary"],
    #        "metric": ["auc"],
    #        "bagging_fraction": np.arange(0.4, 1, 0.1),
    #        "bagging_freq": range(1, 5, 1),
    #        "feature_fraction": np.arange(0.4, 1, 0.1),
    #        "max_bin": range(10, 500, 10),
    #        "min_data_in_leaf": range(5, 50, 1),
    #    }
    #
    #    lightgbm_classifier = lightgbm.LGBMClassifier()
    #
    #    randomized_search = sklearn.model_selection.RandomizedSearchCV(
    #        estimator=lightgbm_classifier,
    #        param_distributions=param_grid,
    #        cv=5,  # here we should implement a better split strategy, see comment above
    #        n_iter=250,
    #        n_jobs=-1,
    #        scoring="roc_auc",
    #        verbose=2,
    #        random_state=501,
    #    )
    #
    #    randomized_search.fit(X_train, y_train)
    #
    #    with open(file_hps_results, "wb") as con:
    #        pickle.dump(randomized_search, con)

    # train lightgbm model
    lightgbm_classifier = lightgbm.LGBMClassifier(**config["model"])
    lightgbm_classifier.fit(
        x_df.values,
        y_df.y_int.values,
    )

    # make predictions
    pred_lightgbm = prediction_index.copy()
    pred_lightgbm["yhat"] = lightgbm_classifier.predict_proba(x_predict)[:, 1]
    assert pred_lightgbm["yhat"].isnull().sum() == 0

    # save result
    joblib.dump(lightgbm_classifier, f"{path_results}/lightgbm-model.pkl")
    pred_lightgbm.to_parquet(f"{path_results}/lightgbm.pt")
