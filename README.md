# AI-Based Vehicle Damage Detection and Insurance Claim Amount Prediction
## Integrated Deep Learning (ResNet50 CNN) + Machine Learning (XGBoost Regression) System

---

## 1. System Overview & Executive Summary
This project provides a complete, end-to-end, production-grade artificial intelligence solution for automated motor insurance claim assessment. The system seamlessly fuses computer vision and predictive machine learning:

1. **Deep Learning (Multi-Label CNN):** Processes an uploaded vehicle photograph and predicts damage probabilities and binary damage classifications across **eight specific structural components** in exact order:
   - `Front-bumper` (Threshold: 0.42)
   - `Hood` (Threshold: 0.60)
   - `Headlight` (Threshold: 0.31)
   - `Front-door` (Threshold: 0.63)
   - `Back-door` (Threshold: 0.74)
   - `Back-bumper` (Threshold: 0.86)
   - `Fender` (Threshold: 0.28)
   - `Windshield` (Threshold: 0.83)
2. **Machine Learning (Predictive Regression):** Ingests the 8 CNN damage probabilities and 8 binary flags alongside vehicle metadata (vehicle age, segment, insured declared value) and policy/workshop economics (repair region tier, labor rate per hour, spare-parts price index, policy deductible) to predict the estimated **Insurance Claim Payout in Indian Rupees (INR)**.
3. **Interactive Streamlit Web Application (`app.py`):** Provides a responsive interface for real-time image uploads, interactive metadata customization, visual component progress bars, and prominent claim estimation with legal and regulatory disclaimers.

---

## 2. End-to-End Workflow Pipeline

```
  [Vehicle Image Upload] ──> [ResNet50 Multi-Label CNN] ──> [8 Damage Probabilities & Flags]
                                                                        │
  [Vehicle & Policy Details] ───────────────────────────────────────────┤
  • Age: 0–15 years                                                     │
  • Segment: Hatchback / Sedan / SUV / Luxury                           ▼
  • Insured Value (IDV in ₹)                              [Tabular Feature Construction]
  • Repair Region: Tier 1 / 2 / 3                               (23 Input Features)
  • Labor Rate (₹/hr)                                                   │
  • Parts Price Index (0.85–1.30)                                       ▼
  • Deductible: ₹ 0 / 2K / 5K / 10K                        [Fitted ColumnTransformer]
                                                                        │
                                                                        ▼
                                                           [Tuned XGBoost Regressor]
                                                                        │
                                                                        ▼
                                                          [Non-Negative Post-Clipping]
                                                                        │
                                                                        ▼
                                                       [Predicted Claim Amount in INR]
```

---

## 3. Project Directory Architecture

```
Resume project/
│
├── data/
│   └── synthetic_vehicle_claims_40000.csv     <- 40,000 claim records with CNN features & metadata
│
├── model_artifacts/
│   ├── best_multilabel_model.keras            <- Pre-trained ResNet50 multi-label CNN
│   ├── best_resnet50.keras                    <- ResNet50 checkpoint
│   ├── best_efficientnetb0.keras              <- EfficientNet-B0 checkpoint
│   ├── best_mobilenetv3.keras                 <- MobileNetV3Large checkpoint
│   └── multilabel_config.json                 <- 8 components, thresholds, and input resolution
│
├── models/
│   ├── best_claim_regressor.pkl               <- Serialized champion Tuned XGBoost model
│   ├── preprocessing_pipeline.pkl             <- Serialized fitted Scikit-learn ColumnTransformer
│   ├── regression_metadata.json               <- Hyperparameters, feature schema, and test metrics
│   └── evaluation_plots/
│       └── xgboost_test_evaluation.png        <- Diagnostic plots on 6,000 unseen test samples
│
├── notebooks/
│   ├── 01_data_analysis.ipynb                 <- Comprehensive EDA, distributions, correlations
│   ├── 02_preprocessing.ipynb                 <- Leak-free 70/15/15 split, imputer, OneHotEncoder
│   ├── 03_model_training.ipynb                <- Benchmark of 13 regression algorithms
│   ├── 04_hyperparameter_tuning.ipynb         <- 5-Fold RandomizedSearchCV on top candidates
│   └── 05_final_evaluation.ipynb              <- Unbiased test set evaluation & residual analysis
│
├── src/
│   ├── cnn_inference.py                       <- VehicleDamageCNN class with optimal thresholds
│   ├── data_preprocessing.py                  <- Data splitting, dual ColumnTransformer pipelines
│   ├── train_regression.py                    <- Benchmark suite evaluating MAE, RMSE, R2, ExpVar
│   ├── tune_models.py                         <- Cross-validated hyperparameter optimization
│   ├── evaluate_models.py                     <- Test set evaluation, diagnostic charts & export
│   └── predict_claim.py                       <- Unified end-to-end ClaimAssessmentPredictor
│
├── app.py                                     <- Interactive Streamlit web application
├── requirements.txt                           <- Dependency specifications
└── README.md                                  <- Complete system documentation
```

---

## 4. Benchmark Results & Algorithm Comparison

All models were evaluated on an identical **70% Training (28,000 records)** and **15% Validation (6,000 records)** split with non-negative payout clipping:

| Algorithm | Preprocessing Scaling | Val MAE (INR) | Val RMSE (INR) | Val R² Score | Training Time |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | None (Tree) | ₹ 9,424.92 | ₹ 14,368.81 | 0.8327 | 14.15s |
| **LightGBM** | None (Tree) | ₹ 9,456.71 | ₹ 14,391.64 | 0.8321 | 0.24s |
| **HistGradientBoosting** | None (Tree) | ₹ 9,465.67 | ₹ 14,393.99 | 0.8321 | 0.42s |
| **XGBoost (Baseline)** | None (Tree) | ₹ 9,536.95 | ₹ 14,499.90 | 0.8296 | 0.30s |
| **Random Forest** | None (Tree) | ₹ 10,044.41 | ₹ 15,474.95 | 0.8059 | 2.96s |
| **Extra Trees** | None (Tree) | ₹ 10,102.73 | ₹ 15,708.90 | 0.8000 | 0.65s |
| **Linear Regression** | StandardScaler | ₹ 10,436.49 | ₹ 15,820.86 | 0.7971 | 0.04s |
| **Ridge Regression** | StandardScaler | ₹ 10,436.60 | ₹ 15,820.89 | 0.7971 | 0.01s |
| **Lasso Regression** | StandardScaler | ₹ 10,436.71 | ₹ 15,821.19 | 0.7971 | 0.12s |
| **K-Nearest Neighbors** | StandardScaler | ₹ 10,448.97 | ₹ 16,389.64 | 0.7823 | 0.00s |
| **ElasticNet Regression** | StandardScaler | ₹ 11,247.52 | ₹ 16,518.97 | 0.7788 | 0.05s |
| **Decision Tree** | None (Tree) | ₹ 11,335.81 | ₹ 17,566.21 | 0.7499 | 0.29s |
| **Linear SVR** | StandardScaler | ₹ 20,575.03 | ₹ 31,062.52 | 0.2179 | 0.03s |

---

## 5. Hyperparameter Tuning & Final Model Selection

Top gradient boosting algorithms underwent cross-validated `RandomizedSearchCV` (5-fold CV):
- **Champion Regressor:** **Tuned XGBoost Regressor (`XGBRegressor`)**
- **Optimal Hyperparameters:**
  ```python
  {
      'n_estimators': 250,
      'learning_rate': 0.07,
      'max_depth': 5,
      'subsample': 0.75,
      'colsample_bytree': 1.0,
      'min_child_weight': 5,
      'reg_alpha': 0.5,
      'reg_lambda': 3.0,
      'random_state': 42
  }
  ```

### Performance on Untouched Test Set (6,000 Unseen Samples):
- **Mean Absolute Error (MAE):** **₹ 9,467.96 INR**
- **Root Mean Squared Error (RMSE):** **₹ 14,468.09 INR**
- **Coefficient of Determination ($R^2$):** **0.8281**
- **Explained Variance Score:** **0.8281**
- **Non-Negative Guarantee:** Enforced via post-prediction clipping at zero.

---

## 6. Installation & Execution Guide

### Step 1: Environment Setup
Ensure Python 3.10+ (recommended Anaconda) is available:
```bash
cd "D:\MLP course files\Deep learning\Resume project"
pip install -r requirements.txt
```

### Step 2: Running the Pipeline Modules (Optional Verification)
```bash
# 1. Test CNN Inference on sample image
python src/cnn_inference.py

# 2. Test Tabular Preprocessing & Leak-Free Splits
python src/data_preprocessing.py

# 3. Re-run Regression Algorithm Benchmarks
python src/train_regression.py

# 4. Re-run Hyperparameter Tuning
python src/tune_models.py

# 5. Re-evaluate on Untouched Test Set and Export Models
python src/evaluate_models.py

# 6. Test End-to-End Prediction
python src/predict_claim.py
```

### Step 3: Launching the Streamlit Web Application
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

---

## 7. Interactive Web Application Usage
1. **Upload or Select Image:** Upload any vehicle photo (damaged or undamaged), or pick one of the curated samples from the dropdown.
2. **Configure Parameters in Sidebar:**
   - Vehicle Segment: *Hatchback / Sedan / SUV / Luxury*
   - Vehicle Age: *0 to 15 years*
   - Insured Declared Value (IDV): *₹ 2,00,000 to ₹ 1,00,00,000*
   - Policy Deductible: *₹ 0 / ₹ 2,000 / ₹ 5,000 / ₹ 10,000*
   - Repair Region Tier: *Tier 1 (Metro) / Tier 2 (Urban) / Tier 3 (Semi-Urban)*
   - Labor Rate: *₹ 300 to ₹ 2,000 per hour*
   - Spare-Parts Price Index: *0.85 to 1.30*
3. **Click `🚀 Assess Damage & Predict Claim Amount`**:
   - The ResNet50 CNN calculates 8 component probabilities against calibrated decision thresholds.
   - Status badges display **DAMAGED** or **OK** alongside interactive confidence bars.
   - The Tuned XGBoost model computes the estimated net insurance claim amount in Indian Rupees (INR).
   - Summary cards display total damaged components, deductible applied, and required insurance disclaimers.

---

## 8. Regulatory & Insurance Settlement Disclaimer
> **⚠️ Notice:** This system is an AI-assisted diagnostic estimation tool designed to support claim adjusters and underwriters. Predictions are based on visible external body damage and macroeconomic workshop assumptions. Internal mechanical damage, frame structural compromise, and hidden sensor calibrations must be inspected on-site by a licensed insurance surveyor before legal claim settlement.
