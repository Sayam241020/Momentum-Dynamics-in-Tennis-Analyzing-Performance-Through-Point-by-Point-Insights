
import os
import pandas as pd
import joblib
import xgboost as xgb

from tennis_model_utils import (
    compute_point_winner,
    compute_hmm_momentum_match,
    train_xgboost_model,
    train_lightgbm_model
)

DATA_PATH = "./data/wimbledon_2011_cleaned.csv"
MODEL_DIR = "models/"

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = compute_point_winner(df)

    # Example: train on the first match
    mid = df['match_id'].unique()[0]
    match_data = df[df['match_id']==mid].copy()
    match_data = compute_hmm_momentum_match(match_data)

    # XGBoost
    bst, *_ = train_xgboost_model(match_data, 'momentum_p1')
    bst.save_model(os.path.join(MODEL_DIR, "xgb_model.json"))
    print("Saved XGBoost to models/xgb_model.json")

    # LightGBM
    lgbm, *_ = train_lightgbm_model(match_data, 'momentum_p1')
    joblib.dump(lgbm, os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    print("Saved LightGBM to models/lgbm_model.pkl")

if __name__ == "__main__":
    main()

