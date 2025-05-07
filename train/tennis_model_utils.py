import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from hmmlearn.hmm import CategoricalHMM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_point_winner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized point‐winner based on game/point score change.
    Adds a nullable Int64 column 'point_winner' with 1 or 2.
    """
    df = df.copy()
    df['point_winner'] = np.nan
    required = ['match_id', 'p1_games', 'p2_games', 'p1_score_numeric', 'p2_score_numeric']
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing column for point_winner: {c}")

    # ensure numeric
    for col in ['p1_games', 'p2_games', 'p1_score_numeric', 'p2_score_numeric']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    prev = df.shift(1)
    same = (df['match_id'] == prev['match_id'])
    p1_game_up   = same & (df['p1_games'] > prev['p1_games'])
    p1_point_up  = same & (df['p1_games'] == prev['p1_games']) & (df['p2_games'] == prev['p2_games']) & (df['p1_score_numeric'] > prev['p1_score_numeric'])
    p2_game_up   = same & (df['p2_games'] > prev['p2_games'])
    p2_point_up  = same & (df['p1_games'] == prev['p1_games']) & (df['p2_games'] == prev['p2_games']) & (df['p2_score_numeric'] > prev['p2_score_numeric'])

    df.loc[p1_game_up|p1_point_up, 'point_winner'] = 1
    df.loc[p2_game_up|p2_point_up, 'point_winner'] = 2

    df['point_winner'] = df['point_winner'].astype(float).astype('Int64')
    return df

def calculate_ema(data, beta=0.9):
    """Simple EMA, back- and forward-fills around NaNs."""
    arr = np.asarray(data, dtype=float)
    ema = np.zeros_like(arr)
    # find first valid
    valid = np.where(~np.isnan(arr))[0]
    if valid.size == 0:
        return ema
    start = valid[0]
    ema[start] = arr[start]
    for t in range(start+1, len(arr)):
        if np.isnan(arr[t]):
            ema[t] = ema[t-1]
        else:
            ema[t] = beta*ema[t-1] + (1-beta)*arr[t]
    ema[:start] = ema[start]
    return ema

def compute_hmm_momentum_match(match_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a 2-state CategoricalHMM on the sequence of point wins,
    then smooth the P1-state probability via EMA → 'momentum_p1'.
    """
    df = match_data.copy()
    if 'point_winner' not in df.columns:
        df['momentum_p1'] = 0.5
        return df

    clean = df.dropna(subset=['point_winner'])
    if len(clean) < 5:
        df['momentum_p1'] = 0.5
        return df

    obs = clean['point_winner'].astype(int).values - 1  # 0 or 1
    try:
        model = CategoricalHMM(n_components=2, n_iter=100, tol=0.01, random_state=42,
                               init_params='ste', params='ste')
        model.fit(obs.reshape(-1,1))
        probs = model.predict_proba(obs.reshape(-1,1))
        # pick the state most associated with P1 wins
        state_idx = np.argmax(model.emissionprob_[:,0])
        p1_probs = probs[:, state_idx]
        ema = calculate_ema(p1_probs, beta=0.9)
        full = pd.Series(0.5, index=df.index)
        full.loc[clean.index] = ema
        df['momentum_p1'] = full.values
    except Exception:
        logger.warning("HMM failed, using neutral momentum", exc_info=True)
        df['momentum_p1'] = 0.5

    return df

def train_xgboost_model(df_match: pd.DataFrame, momentum_feature: str):
    """
    Trains a native‐API XGBoost Booster to predict point_winner.
    Returns (Booster, X_test, y_test, y_proba_test, fpr, tpr, auc_score, acc_train, acc_val, acc_test, cm)
    """
    df = df_match.copy()
    features = ['P1Ace','P2Ace','P1Winner','P2Winner', momentum_feature]
    target = 'point_winner'
    # prepare
    X = df[features].fillna(0)
    y = df[target].dropna().astype(int).map({1:1,2:0})
    X = X.loc[y.index]
    if len(y.unique())<2:
        raise ValueError("Need both classes in target")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    if len(X_temp)<2:
        X_val, y_val = X_temp, y_temp
        X_test, y_test = X_temp.copy(), y_temp.copy()
    else:
        strat = y_temp if len(y_temp.unique())>1 else None
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                        random_state=42, stratify=strat)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    params = {'objective':'binary:logistic','eval_metric':'logloss','eta':0.1,'max_depth':6,'seed':42}
    evals = [(dtrain,'train'),(dval,'validation')]
    bst = xgb.train(params, dtrain, num_boost_round=200,
                    evals=evals, early_stopping_rounds=10, verbose_eval=False)

    # eval
    p_train = (bst.predict(dtrain)>0.5).astype(int)
    p_val   = (bst.predict(dval)>0.5).astype(int)
    p_test_proba = bst.predict(dtest)
    p_test = (p_test_proba>0.5).astype(int)

    acc_train = accuracy_score(y_train, p_train)
    acc_val   = accuracy_score(y_val, p_val)
    acc_test  = accuracy_score(y_test, p_test)

    fpr, tpr, _ = roc_curve(y_test, p_test_proba)
    auc_score = auc(fpr, tpr)
    cm = confusion_matrix(y_test, p_test)

    return bst, X_test, y_test, p_test_proba, fpr, tpr, auc_score, acc_train, acc_val, acc_test, cm

def train_lightgbm_model(df_match: pd.DataFrame, momentum_feature: str):
    """
    Trains an LGBMClassifier, returns (model, X_test, y_test, y_proba, fpr, tpr, auc_score, acc_train, acc_val, acc_test, cm)
    """
    df = df_match.copy()
    features = ['P1Ace','P2Ace','P1Winner','P2Winner', momentum_feature]
    target = 'point_winner'
    X = df[features].fillna(0)
    y = df[target].dropna().astype(int).map({1:1,2:0})
    X = X.loc[y.index]
    if len(y.unique())<2:
        raise ValueError("Need both classes in target")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    if len(X_temp)<2:
        X_val, y_val = X_temp, y_temp
        X_test, y_test = X_temp.copy(), y_temp.copy()
    else:
        strat = y_temp if len(y_temp.unique())>1 else None
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                        random_state=42, stratify=strat)

    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val,y_val)], eval_metric='logloss',
              callbacks=[lgb.early_stopping(10, verbose=False)])

    p_train = model.predict(X_train)
    p_val   = model.predict(X_val)
    p_test_proba = model.predict_proba(X_test)[:,1]
    p_test = (p_test_proba>0.5).astype(int)

    acc_train = accuracy_score(y_train, p_train)
    acc_val   = accuracy_score(y_val, p_val)
    acc_test  = accuracy_score(y_test, p_test)

    fpr, tpr, _ = roc_curve(y_test, p_test_proba)
    auc_score = auc(fpr, tpr)
    cm = confusion_matrix(y_test, p_test)

    return model, X_test, y_test, p_test_proba, fpr, tpr, auc_score, acc_train, acc_val, acc_test, cm

