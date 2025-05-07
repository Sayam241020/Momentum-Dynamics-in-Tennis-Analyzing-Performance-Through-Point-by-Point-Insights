import sys
import joblib
import xgboost as xgb
import pandas as pd

from your_app_module import (
    compute_point_winner,
    compute_hmm_momentum_match
)

XGB_PATH = "models/xgb_model.json"
LGBM_PATH = "models/lgbm_model.pkl"

def load_models():
    xgb_model = xgb.Booster()
    xgb_model.load_model(XGB_PATH)
    lgb_model = joblib.load(LGBM_PATH)
    return xgb_model, lgb_model

def prepare_input(csv_path):
    df = pd.read_csv(csv_path)
    df = compute_point_winner(df)
    df = compute_hmm_momentum_match(df)
    return df

def infer(csv_path):
    xgb_model, lgb_model = load_models()
    df = prepare_input(csv_path)
    # take the last point's features
    sample = df.iloc[[-1]][['P1Ace','P2Ace','P1Winner','P2Winner','momentum_p1']].fillna(0)
    dmat = xgb.DMatrix(sample)
    p_xgb = xgb_model.predict(dmat)[0]
    p_lgb = lgb_model.predict_proba(sample)[:,1][0]
    print(f"XGBoost P1 win‐prob: {p_xgb:.3f}")
    print(f"LightGBM P1 win‐prob: {p_lgb:.3f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infer.py <point_by_point_csv>")
        sys.exit(1)
    infer(sys.argv[1])
