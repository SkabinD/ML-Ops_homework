import pandas as pd
import json

with open("app/config/features.json", "r", encoding="utf-8") as f:
    config = json.load(f)

CATBOOST_FEATURES = config["catboost_features"]
LIGHTGBM_FEATURES = config["lightgbm_features"]
CATEGORICAL_FEATURES = config["categorical_features"]
ACTIVE_FEATURES = list(set(CATBOOST_FEATURES + LIGHTGBM_FEATURES))

def read_file(file=None, file_path="app/input/test_full.csv") -> set[pd.DataFrame, pd.Series]:
    global ACTIVE_FEATURES
    if file:
        data = pd.read_csv(file)
    else:
        data = pd.read_csv(file_path)
    user = data["client_id"]
    data = data[ACTIVE_FEATURES]
    return data, user

def cat_feats_proc(data: pd.DataFrame) -> pd.DataFrame:
    global CATEGORICAL_FEATURES
    data = data.copy()
    data[CATEGORICAL_FEATURES] = data[CATEGORICAL_FEATURES].fillna("unknown")
    return data

def prepare_datasets(data: pd.DataFrame) -> dict["catboost": pd.DataFrame, 
                                                 "lightgbm": pd.DataFrame]:
    global CATBOOST_FEATURES
    global LIGHTGBM_FEATURES
    X_data_catboost = data[CATBOOST_FEATURES]
    X_data_lightgbm = data[LIGHTGBM_FEATURES]
    return {"catboost": X_data_catboost, "lightgbm": X_data_lightgbm}

def run_preproc(data: pd.DataFrame) -> dict["catboost": pd.DataFrame, 
                                            "lightgbm": pd.DataFrame]:
    data = cat_feats_proc(data)
    X_data_dict = prepare_datasets(data)
    return X_data_dict
