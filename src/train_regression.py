import os
import sys
import time
import json
import numpy as np
import pandas as pd

# Add project root to sys.path so 'src' can be imported anywhere
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR

# Conditional imports for gradient boosting packages
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from src.data_preprocessing import prepare_preprocessed_datasets


def calculate_metrics(y_true, y_pred):
    """
    Computes regression metrics with non-negative payout clipping.
    """
    y_pred_clipped = np.clip(y_pred, 0, None)
    mae = mean_absolute_error(y_true, y_pred_clipped)
    mse = mean_squared_error(y_true, y_pred_clipped)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred_clipped)
    evs = explained_variance_score(y_true, y_pred_clipped)

    return {
        "MAE_INR": round(mae, 2),
        "RMSE_INR": round(rmse, 2),
        "MSE": round(mse, 2),
        "R2_Score": round(r2, 4),
        "Explained_Variance": round(evs, 4),
    }


def get_model_definitions():
    """
    Returns a dictionary of candidate regression models and their pipeline type (scaled vs base).
    Sensible baselines configured for efficiency and accuracy.
    """
    models = {
        # Linear & Regularized (Scaled)
        "Linear Regression": {
            "model": LinearRegression(),
            "use_scaled": True,
        },
        "Ridge Regression": {
            "model": Ridge(alpha=1.0, random_state=42),
            "use_scaled": True,
        },
        "Lasso Regression": {
            "model": Lasso(alpha=1.0, max_iter=2000, random_state=42),
            "use_scaled": True,
        },
        "ElasticNet Regression": {
            "model": ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=2000, random_state=42),
            "use_scaled": True,
        },
        # Distance & Kernel Based (Scaled)
        "K-Nearest Neighbors": {
            "model": KNeighborsRegressor(n_neighbors=7, n_jobs=-1),
            "use_scaled": True,
        },
        "Linear SVR": {
            "model": LinearSVR(C=1.0, max_iter=3000, random_state=42),
            "use_scaled": True,
        },
        # Tree-Based & Ensembles (Unscaled / Base)
        "Decision Tree": {
            "model": DecisionTreeRegressor(max_depth=12, min_samples_leaf=10, random_state=42),
            "use_scaled": False,
        },
        "Random Forest": {
            "model": RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42),
            "use_scaled": False,
        },
        "Extra Trees": {
            "model": ExtraTreesRegressor(n_estimators=100, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42),
            "use_scaled": False,
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(n_estimators=120, max_depth=5, learning_rate=0.1, random_state=42),
            "use_scaled": False,
        },
        "HistGradientBoosting": {
            "model": HistGradientBoostingRegressor(max_iter=150, max_depth=8, learning_rate=0.1, random_state=42),
            "use_scaled": False,
        },
    }

    if HAS_XGBOOST:
        models["XGBoost"] = {
            "model": xgb.XGBRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            ),
            "use_scaled": False,
        }

    if HAS_LIGHTGBM:
        models["LightGBM"] = {
            "model": lgb.LGBMRegressor(
                n_estimators=150,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            "use_scaled": False,
        }

    if HAS_CATBOOST:
        models["CatBoost"] = {
            "model": cb.CatBoostRegressor(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_seed=42,
                verbose=0,
            ),
            "use_scaled": False,
        }

    return models


def benchmark_all_models(data_dict=None):
    """
    Fits and evaluates all models on train and validation sets.
    Returns comparison DataFrame, fitted models dict, and preprocessors.
    """
    if data_dict is None:
        data_dict = prepare_preprocessed_datasets()

    X_train = data_dict["X_train"]
    X_val = data_dict["X_val"]
    y_train = data_dict["y_train"]
    y_val = data_dict["y_val"]

    base_prep = data_dict["base_preprocessor"]
    scaled_prep = data_dict["scaled_preprocessor"]

    # Pre-transform training and validation features for fast fitting
    print("Transforming feature matrices...")
    X_train_base = base_prep.transform(X_train)
    X_val_base = base_prep.transform(X_val)

    X_train_scaled = scaled_prep.transform(X_train)
    X_val_scaled = scaled_prep.transform(X_val)

    models_dict = get_model_definitions()
    results = []
    fitted_models = {}

    print(f"\nBenchmarking {len(models_dict)} regression algorithms...")
    print("-" * 80)

    for name, config in models_dict.items():
        estimator = config["model"]
        use_scaled = config["use_scaled"]

        X_tr = X_train_scaled if use_scaled else X_train_base
        X_v = X_val_scaled if use_scaled else X_val_base

        t0 = time.time()
        estimator.fit(X_tr, y_train)
        fit_time = time.time() - t0

        # Predictions
        y_train_pred = estimator.predict(X_tr)
        y_val_pred = estimator.predict(X_v)

        train_m = calculate_metrics(y_train, y_train_pred)
        val_m = calculate_metrics(y_val, y_val_pred)

        fitted_models[name] = {
            "estimator": estimator,
            "use_scaled": use_scaled,
            "fit_time_sec": round(fit_time, 2),
            "train_metrics": train_m,
            "val_metrics": val_m,
        }

        results.append({
            "Algorithm": name,
            "Pipeline_Scaling": "StandardScaler" if use_scaled else "None (Tree)",
            "Val_MAE_INR": val_m["MAE_INR"],
            "Val_RMSE_INR": val_m["RMSE_INR"],
            "Val_R2": val_m["R2_Score"],
            "Val_ExpVar": val_m["Explained_Variance"],
            "Train_MAE_INR": train_m["MAE_INR"],
            "Train_R2": train_m["R2_Score"],
            "Fit_Time_s": round(fit_time, 2),
        })

        print(f"[{name:22s}] Val MAE: INR {val_m['MAE_INR']:>10,.2f} | Val RMSE: INR {val_m['RMSE_INR']:>10,.2f} | Val R2: {val_m['R2_Score']:.4f} | Time: {fit_time:.2f}s")

    df_results = pd.DataFrame(results).sort_values(by="Val_RMSE_INR", ascending=True).reset_index(drop=True)
    return df_results, fitted_models, data_dict


if __name__ == "__main__":
    df_res, fitted, data_dict = benchmark_all_models()
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON TABLE (Sorted by Validation RMSE):")
    print("=" * 80)
    print(df_res.to_string())
