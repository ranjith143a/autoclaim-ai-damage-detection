"""
Final Model Evaluation and Serialization Module.
Evaluates the tuned champion model on the untouched test set (6,000 samples).
Generates diagnostic plots (Actual vs Predicted, Residuals, Feature Importance),
performs error analysis across vehicle segments and damage counts,
and serializes the fitted pipeline and model artifacts via Joblib.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import xgboost as xgb
from src.data_preprocessing import prepare_preprocessed_datasets, TARGET_COLUMN
from src.train_regression import calculate_metrics


def run_final_evaluation_and_save():
    """
    Fits the champion XGBoost model with optimal parameters on the combined
    Train+Val data (or evaluates on Test), generates diagnostic charts,
    and serializes the pipeline and metadata.
    """
    data_dict = prepare_preprocessed_datasets(random_state=42)

    X_train = data_dict["X_train"]
    X_val = data_dict["X_val"]
    X_test = data_dict["X_test"]
    y_train = data_dict["y_train"]
    y_val = data_dict["y_val"]
    y_test = data_dict["y_test"]
    base_prep = data_dict["base_preprocessor"]
    feature_names = data_dict["feature_names"]

    # Transform
    X_train_trans = base_prep.transform(X_train)
    X_val_trans = base_prep.transform(X_val)
    X_test_trans = base_prep.transform(X_test)

    # Optimal parameters found in Phase 5
    best_params = {
        "subsample": 0.75,
        "reg_lambda": 3.0,
        "reg_alpha": 0.5,
        "n_estimators": 250,
        "min_child_weight": 5,
        "max_depth": 5,
        "learning_rate": 0.07,
        "colsample_bytree": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    print("Training Champion XGBoost model with optimal hyperparameters on training set...")
    champion_model = xgb.XGBRegressor(**best_params)
    champion_model.fit(X_train_trans, y_train)

    # Predictions with non-negative clipping
    y_train_raw = champion_model.predict(X_train_trans)
    y_val_raw = champion_model.predict(X_val_trans)
    y_test_raw = champion_model.predict(X_test_trans)

    y_train_pred = np.clip(y_train_raw, 0, None)
    y_val_pred = np.clip(y_val_raw, 0, None)
    y_test_pred = np.clip(y_test_raw, 0, None)

    train_metrics = calculate_metrics(y_train, y_train_pred)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)

    print("\n" + "=" * 80)
    print("FINAL MODEL PERFORMANCE SUMMARY (CHAMPION XGBOOST):")
    print("=" * 80)
    print(f"Training Set   (28,000 samples): MAE = INR {train_metrics['MAE_INR']:>10,.2f} | RMSE = INR {train_metrics['RMSE_INR']:>10,.2f} | R2 = {train_metrics['R2_Score']:.4f}")
    print(f"Validation Set ( 6,000 samples): MAE = INR {val_metrics['MAE_INR']:>10,.2f} | RMSE = INR {val_metrics['RMSE_INR']:>10,.2f} | R2 = {val_metrics['R2_Score']:.4f}")
    print(f"Test Set       ( 6,000 samples): MAE = INR {test_metrics['MAE_INR']:>10,.2f} | RMSE = INR {test_metrics['RMSE_INR']:>10,.2f} | R2 = {test_metrics['R2_Score']:.4f}")
    print("=" * 80)

    # Check non-negative guarantees
    neg_raw_count = (y_test_raw < 0).sum()
    print(f"Raw negative test predictions before clipping: {neg_raw_count} ({neg_raw_count/len(y_test)*100:.2f}%)")
    print(f"Min test prediction after clipping: INR {y_test_pred.min():,.2f}")

    # Create directory for plots
    plots_dir = os.path.join(PROJECT_ROOT, "models", "evaluation_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Diagnostic Plots
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # A. Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_test_pred, alpha=0.25, color="#1f77b4", edgecolors="none", s=20)
    max_val = max(y_test.max(), y_test_pred.max())
    ax1.plot([0, max_val], [0, max_val], color="crimson", linestyle="--", linewidth=2, label="Perfect Fit (y = x)")
    ax1.set_title(f"Actual vs Predicted Claim Amount (Test Set)\nR² = {test_metrics['R2_Score']:.4f}, MAE = ₹{test_metrics['MAE_INR']:,.0f}", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Actual Claim Amount (INR)", fontsize=11)
    ax1.set_ylabel("Predicted Claim Amount (INR)", fontsize=11)
    ax1.legend(loc="upper left")

    # B. Residuals Distribution
    residuals = y_test - y_test_pred
    ax2 = axes[0, 1]
    sns.histplot(residuals, bins=60, kde=True, color="#2ca02c", ax=ax2)
    ax2.axvline(0, color="crimson", linestyle="--", linewidth=1.5, label="Zero Error")
    ax2.axvline(np.mean(residuals), color="navy", linestyle=":", linewidth=1.5, label=f"Mean Error: ₹{np.mean(residuals):,.0f}")
    ax2.set_title(f"Residual Distribution (Mean = ₹{np.mean(residuals):,.0f}, Std = ₹{np.std(residuals):,.0f})", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Residual (Actual - Predicted) in INR", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.legend(loc="upper right")

    # C. Residuals vs Predicted
    ax3 = axes[1, 0]
    ax3.scatter(y_test_pred, residuals, alpha=0.25, color="#ff7f0e", edgecolors="none", s=20)
    ax3.axhline(0, color="crimson", linestyle="--", linewidth=1.5)
    ax3.set_title("Residuals vs Predicted Values (Homoscedasticity Check)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Predicted Claim Amount (INR)", fontsize=11)
    ax3.set_ylabel("Residual (Actual - Predicted) in INR", fontsize=11)

    # D. Feature Importance (Top 15)
    ax4 = axes[1, 1]
    importances = champion_model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False).head(15)
    sns.barplot(x="Importance", y="Feature", data=feat_df, palette="viridis", ax=ax4)
    ax4.set_title("Top 15 Feature Importances (XGBoost Gini Gain)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Relative Importance Score", fontsize=11)
    ax4.set_ylabel("Feature Name", fontsize=11)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "xgboost_test_evaluation.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved evaluation plots to: {plot_path}")

    # 2. Detailed Error Analysis
    test_eval_df = X_test.copy()
    test_eval_df["actual_claim"] = y_test
    test_eval_df["predicted_claim"] = y_test_pred
    test_eval_df["absolute_error"] = np.abs(y_test - y_test_pred)
    test_eval_df["error_pct"] = np.where(y_test > 0, (test_eval_df["absolute_error"] / y_test) * 100, 0)

    flag_cols = [c for c in test_eval_df.columns if c.endswith("_flag")]
    test_eval_df["total_damaged_parts"] = test_eval_df[flag_cols].sum(axis=1)

    print("\n--- Error Analysis by Vehicle Segment ---")
    seg_err = test_eval_df.groupby("vehicle_segment")["absolute_error"].agg(["count", "mean", "median", "std"])
    seg_err.columns = ["Count", "Mean_MAE_INR", "Median_MAE_INR", "Std_INR"]
    print(seg_err.to_string())

    print("\n--- Error Analysis by Repair Region ---")
    reg_err = test_eval_df.groupby("repair_region")["absolute_error"].agg(["count", "mean", "median", "std"])
    reg_err.columns = ["Count", "Mean_MAE_INR", "Median_MAE_INR", "Std_INR"]
    print(reg_err.to_string())

    print("\n--- Error Analysis by Total Damaged Components ---")
    part_err = test_eval_df.groupby("total_damaged_parts")["absolute_error"].agg(["count", "mean", "median", "std"])
    part_err.columns = ["Count", "Mean_MAE_INR", "Median_MAE_INR", "Std_INR"]
    print(part_err.to_string())

    # Zero claims analysis
    zero_mask = y_test == 0
    zero_preds = y_test_pred[zero_mask]
    print(f"\nZero Actual Claims Analysis (Count: {zero_mask.sum()}):")
    print(f"  Mean predicted on actual zero claims: INR {zero_preds.mean():,.2f}")
    print(f"  Exact zeros correctly predicted: {(zero_preds == 0).sum()} ({(zero_preds == 0).mean()*100:.2f}%)")
    print(f"  Predicted <= INR 2,000 on zero claims: {(zero_preds <= 2000).sum()} ({(zero_preds <= 2000).mean()*100:.2f}%)")

    # 3. Serialization (Phase 7)
    models_dir = os.path.join(PROJECT_ROOT, "models")
    model_pkl_path = os.path.join(models_dir, "best_claim_regressor.pkl")
    prep_pkl_path = os.path.join(models_dir, "preprocessing_pipeline.pkl")
    meta_json_path = os.path.join(models_dir, "regression_metadata.json")

    joblib.dump(champion_model, model_pkl_path)
    joblib.dump(base_prep, prep_pkl_path)

    metadata = {
        "model_name": "Tuned XGBoost Regressor",
        "algorithm": "XGBRegressor",
        "hyperparameters": best_params,
        "input_feature_columns": list(X_train.columns),
        "transformed_feature_names": feature_names,
        "num_features_raw": len(X_train.columns),
        "num_features_transformed": len(feature_names),
        "target_column": TARGET_COLUMN,
        "clipping_strategy": "Non-negative clipping: np.clip(pred, 0, None)",
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "categorical_features": ["vehicle_segment", "repair_region"],
        "top_features": feat_df.to_dict(orient="records"),
    }

    with open(meta_json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("SAVED PIPELINE AND MODEL ARTIFACTS:")
    print(f"  Regressor:              {model_pkl_path}")
    print(f"  Preprocessing Pipeline: {prep_pkl_path}")
    print(f"  Metadata JSON:          {meta_json_path}")
    print("=" * 80)

    return champion_model, base_prep, metadata


if __name__ == "__main__":
    run_final_evaluation_and_save()
