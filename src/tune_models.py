"""
Hyperparameter Tuning Module.
Performs cross-validated RandomizedSearchCV on top candidate gradient boosting algorithms
(LightGBM, HistGradientBoosting, XGBoost).
Evaluates tuned models on the validation set and selects the champion model.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb

from src.data_preprocessing import prepare_preprocessed_datasets
from src.train_regression import calculate_metrics


def run_tuning(data_dict=None, n_iter=25, cv_folds=5, random_state=42):
    """
    Executes RandomizedSearchCV for top models, compares them on validation set,
    and returns the best performing tuned model.
    """
    if data_dict is None:
        data_dict = prepare_preprocessed_datasets(random_state=random_state)

    X_train = data_dict["X_train"]
    X_val = data_dict["X_val"]
    y_train = data_dict["y_train"]
    y_val = data_dict["y_val"]
    base_prep = data_dict["base_preprocessor"]

    X_train_trans = base_prep.transform(X_train)
    X_val_trans = base_prep.transform(X_val)

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # 1. LightGBM Parameter Grid
    lgb_param_dist = {
        "n_estimators": [120, 180, 250, 320],
        "learning_rate": [0.03, 0.05, 0.08, 0.1, 0.12],
        "num_leaves": [20, 31, 45, 63, 80],
        "max_depth": [-1, 6, 8, 10, 12],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_samples": [10, 20, 30, 50],
        "reg_alpha": [0.0, 0.1, 1.0, 5.0],
        "reg_lambda": [0.0, 0.1, 1.0, 5.0],
    }

    print("\n[1/3] Tuning LightGBM with RandomizedSearchCV (25 iterations, 5 folds)...")
    lgb_base = lgb.LGBMRegressor(random_state=random_state, n_jobs=-1, verbose=-1)
    lgb_search = RandomizedSearchCV(
        estimator=lgb_base,
        param_distributions=lgb_param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=kf,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    t0 = time.time()
    lgb_search.fit(X_train_trans, y_train)
    lgb_time = time.time() - t0
    best_lgb = lgb_search.best_estimator_

    # 2. HistGradientBoosting Parameter Grid
    hgb_param_dist = {
        "max_iter": [120, 180, 250, 300],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "max_leaf_nodes": [20, 31, 45, 63],
        "max_depth": [6, 8, 10, 12, None],
        "min_samples_leaf": [10, 20, 30, 50],
        "l2_regularization": [0.0, 0.1, 1.0, 5.0],
    }

    print("\n[2/3] Tuning HistGradientBoosting with RandomizedSearchCV (20 iterations, 5 folds)...")
    hgb_base = HistGradientBoostingRegressor(random_state=random_state)
    hgb_search = RandomizedSearchCV(
        estimator=hgb_base,
        param_distributions=hgb_param_dist,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=kf,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    t0 = time.time()
    hgb_search.fit(X_train_trans, y_train)
    hgb_time = time.time() - t0
    best_hgb = hgb_search.best_estimator_

    # 3. XGBoost Parameter Grid
    xgb_param_dist = {
        "n_estimators": [120, 180, 250],
        "learning_rate": [0.04, 0.07, 0.1],
        "max_depth": [5, 6, 8],
        "subsample": [0.75, 0.85, 1.0],
        "colsample_bytree": [0.75, 0.85, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_alpha": [0.0, 0.5, 2.0],
        "reg_lambda": [0.5, 1.0, 3.0],
    }

    print("\n[3/3] Tuning XGBoost with RandomizedSearchCV (15 iterations, 5 folds)...")
    xgb_base = xgb.XGBRegressor(random_state=random_state, n_jobs=-1)
    xgb_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=xgb_param_dist,
        n_iter=15,
        scoring="neg_root_mean_squared_error",
        cv=kf,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    t0 = time.time()
    xgb_search.fit(X_train_trans, y_train)
    xgb_time = time.time() - t0
    best_xgb = xgb_search.best_estimator_

    # Evaluation on Validation Set
    tuned_candidates = {
        "Tuned LightGBM": (best_lgb, lgb_search.best_params_, lgb_time),
        "Tuned HistGradientBoosting": (best_hgb, hgb_search.best_params_, hgb_time),
        "Tuned XGBoost": (best_xgb, xgb_search.best_params_, xgb_time),
    }

    comparison = []
    print("\n" + "=" * 80)
    print("VALIDATION EVALUATION OF TUNED MODELS:")
    print("=" * 80)

    for name, (est, params, t) in tuned_candidates.items():
        y_val_pred = est.predict(X_val_trans)
        y_train_pred = est.predict(X_train_trans)

        m_val = calculate_metrics(y_val, y_val_pred)
        m_train = calculate_metrics(y_train, y_train_pred)

        comparison.append({
            "Model": name,
            "Val_MAE_INR": m_val["MAE_INR"],
            "Val_RMSE_INR": m_val["RMSE_INR"],
            "Val_R2": m_val["R2_Score"],
            "Val_ExpVar": m_val["Explained_Variance"],
            "Train_MAE_INR": m_train["MAE_INR"],
            "Train_R2": m_train["R2_Score"],
            "Tuning_Time_s": round(t, 2),
            "Best_Params": params,
        })
        print(f"[{name:28s}] Val MAE: INR {m_val['MAE_INR']:>10,.2f} | Val RMSE: INR {m_val['RMSE_INR']:>10,.2f} | Val R2: {m_val['R2_Score']:.4f}")

    df_tuned = pd.DataFrame(comparison).sort_values(by="Val_RMSE_INR", ascending=True).reset_index(drop=True)

    winner_name = df_tuned.iloc[0]["Model"]
    winner_row = df_tuned.iloc[0]
    winner_est = tuned_candidates[winner_name][0]
    winner_params = tuned_candidates[winner_name][1]

    print(f"\nChampion Model Selected: {winner_name}")
    print(f"Optimal Parameters: {winner_params}")
    print(f"Validation MAE: INR {winner_row['Val_MAE_INR']:,.2f} | RMSE: INR {winner_row['Val_RMSE_INR']:,.2f} | R2: {winner_row['Val_R2']:.4f}")

    return {
        "df_tuned": df_tuned,
        "winner_name": winner_name,
        "winner_estimator": winner_est,
        "winner_params": winner_params,
        "base_preprocessor": base_prep,
        "data_dict": data_dict,
    }


if __name__ == "__main__":
    res = run_tuning()
