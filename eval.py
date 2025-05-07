import os
import joblib
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from your_app_module import (
    compute_point_winner,
    compute_hmm_momentum_match
)

# Paths
DATA_PATH = "data/match_points.csv"
XGB_PATH  = "models/xgb_model.json"
LGBM_PATH = "models/lgbm_model.pkl"

def load_test_match():
    """
    Loads the second distinct match in the CSV for evaluation,
    computes point winners and HMM‐momentum, and returns that DataFrame.
    """
    df = pd.read_csv(DATA_PATH)
    df = compute_point_winner(df)
    unique_ids = df['match_id'].unique()
    if len(unique_ids) < 2:
        raise ValueError("Need at least two distinct match_id values in the data.")
    test_id = unique_ids[1]
    test_df = df[df['match_id'] == test_id].copy()
    test_df = compute_hmm_momentum_match(test_df)
    return test_df

def eval_xgb(test_df):
    """
    Evaluate the native‐API XGBoost Booster on test_df.
    Prints Accuracy and AUC.
    """
    # Prepare features and labels exactly like in train
    X = test_df[['P1Ace', 'P2Ace', 'P1Winner', 'P2Winner', 'momentum_p1']].fillna(0)
    y = test_df['point_winner'].dropna().astype(int).map({1: 1, 2: 0})
    X = X.loc[y.index]

    dtest = xgb.DMatrix(X, label=y)

    model = xgb.Booster()
    model.load_model(XGB_PATH)

    # Predict probabilities
    y_proba = model.predict(dtest)
    # Convert to binary preds at 0.5 threshold
    y_pred = (y_proba > 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)

    print("=== XGBoost Evaluation ===")
    print(f"Test match_id: {test_df['match_id'].iloc[0]}")
    print(f"Num points:   {len(y)}")
    print(f"Accuracy:     {acc:.4f}")
    print(f"AUC:          {auc:.4f}")
    print()

def eval_lgb(test_df):
    """
    Evaluate the LightGBM model on test_df.
    Prints Accuracy and AUC.
    """
    # Prepare features and labels
    X = test_df[['P1Ace', 'P2Ace', 'P1Winner', 'P2Winner', 'momentum_p1']].fillna(0)
    y = test_df['point_winner'].dropna().astype(int).map({1: 1, 2: 0})
    X = X.loc[y.index]

    model = joblib.load(LGBM_PATH)

    # Predict probabilities for class 1
    y_proba = model.predict_proba(X)[:, 1]
    # Binary predictions
    y_pred = (y_proba > 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)

    print("=== LightGBM Evaluation ===")
    print(f"Test match_id: {test_df['match_id'].iloc[0]}")
    print(f"Num points:   {len(y)}")
    print(f"Accuracy:     {acc:.4f}")
    print(f"AUC:          {auc:.4f}")
    print()

def main():
    # Check that model files exist
    if not os.path.exists(XGB_PATH) or not os.path.exists(LGBM_PATH):
        print("Model weights not found in models/. Run train.py first.")
        return

    test_df = load_test_match()
    eval_xgb(test_df)
    eval_lgb(test_df)

if __name__ == "__main__":
    main()
