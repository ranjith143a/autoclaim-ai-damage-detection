"""
Streamlit Web Application: AI-Based Vehicle Damage Detection and Insurance Claim Assessment.
Integrates Multi-Label ResNet50 CNN with Tuned XGBoost Regression Pipeline.
"""

import os
import sys
import json
from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st

# Setup Project Path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict_claim import ClaimAssessmentPredictor

# Page Configuration
st.set_page_config(
    page_title="AutoClaim AI | Vehicle Damage & Claim Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Financial/Insurtech UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .claim-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        border-radius: 12px;
        padding: 24px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
        margin-bottom: 20px;
    }
    .claim-amount {
        font-size: 2.8rem;
        font-weight: 800;
        color: #FEF08A;
        margin: 10px 0;
    }
    .status-badge-damaged {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        border: 1px solid #FCA5A5;
    }
    .status-badge-ok {
        background-color: #ECFDF5;
        color: #065F46;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        border: 1px solid #6EE7B7;
    }
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        color: #92400E;
        font-size: 0.85rem;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Deep Learning & ML Models...")
def load_predictor():
    return ClaimAssessmentPredictor()


def main():
    st.markdown('<div class="main-title">🚗 AutoClaim AI: Damage Detection & Claim Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">End-to-End Integrated System: 8-Component ResNet50 CNN + Tuned XGBoost Regression</div>', unsafe_allow_html=True)

    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure the models are trained and saved in `models/`.")
        return

    # Sidebar: Vehicle & Insurance Metadata
    st.sidebar.header("📋 Vehicle & Policy Details")
    st.sidebar.markdown("Enter vehicle characteristics and policy terms for claim estimation.")

    st.sidebar.subheader("1. Vehicle Attributes")
    vehicle_segment = st.sidebar.selectbox(
        "Vehicle Segment",
        options=["Hatchback", "Sedan", "SUV", "Luxury"],
        index=1,
        help="Vehicle category significantly impacts spare-part costs and replacement complexity."
    )

    vehicle_age = st.sidebar.slider(
        "Vehicle Age (Years)",
        min_value=0,
        max_value=15,
        value=4,
        step=1,
        help="Older vehicles experience parts depreciation and wear."
    )

    idv_defaults = {"Hatchback": 750000, "Sedan": 1100000, "SUV": 1550000, "Luxury": 3200000}
    default_idv = idv_defaults.get(vehicle_segment, 1100000)
    insured_value = st.sidebar.number_input(
        "Insured Declared Value - IDV (₹)",
        min_value=200000,
        max_value=10000000,
        value=default_idv,
        step=50000,
        format="%d"
    )

    st.sidebar.subheader("2. Policy & Deductible")
    deductible = st.sidebar.selectbox(
        "Standard Policy Deductible (₹)",
        options=[0, 2000, 5000, 10000],
        index=2,
        help="Mandatory deductible borne by policyholder before insurance payout."
    )

    st.sidebar.subheader("3. Repair & Workshop Economics")
    repair_region = st.sidebar.selectbox(
        "Repair Region",
        options=["Tier 1", "Tier 2", "Tier 3"],
        index=1,
        help="Metro (Tier 1), Urban (Tier 2), or Semi-urban (Tier 3) workshop location."
    )

    labor_rate_defaults = {"Tier 1": 900, "Tier 2": 650, "Tier 3": 480}
    labor_rate = st.sidebar.number_input(
        "Labor Rate per Hour (₹)",
        min_value=300,
        max_value=2000,
        value=labor_rate_defaults.get(repair_region, 650),
        step=50
    )

    parts_index = st.sidebar.slider(
        "Spare-Parts Price Index",
        min_value=0.85,
        max_value=1.30,
        value=1.00,
        step=0.01,
        help="Macroeconomic cost multiplier reflecting parts availability and inflation."
    )

    # Main Area Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Assessment & Estimation", "📊 Model Architecture & Evaluation", "📖 Dataset & System Info"])

    with tab1:
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.subheader("1. Vehicle Image Input")

            input_method = st.radio(
                "Choose Input Method:",
                options=["Upload Image", "Select Curated Sample"],
                horizontal=True
            )

            image_to_process = None

            if input_method == "Upload Image":
                uploaded_file = st.file_uploader(
                    "Upload accident vehicle photo (JPG, PNG)",
                    type=["jpg", "jpeg", "png"],
                    help="Upload a clear front or rear photo of the damaged or undamaged vehicle."
                )
                if uploaded_file is not None:
                    image_to_process = Image.open(uploaded_file).convert("RGB")
                    st.image(image_to_process, caption="Uploaded Image", use_container_width=True)
            else:
                sample_options = {
                    "Multi-Component Collision (Bumper + Hood + Fender + Headlight)": "Car_Parts_Damage_Dataset/Car damages dataset/File1/img/Car damages 101.png",
                    "Side Impact (Front & Back Door + Fender)": "Car_Parts_Damage_Dataset/Car damages dataset/File1/img/Car damages 102.jpg",
                    "Front Bumper Breakage": "Comprehensive_Car_Damage/F_Breakage/FB_1.jpg",
                    "Normal Vehicle (Undamaged - Front View)": "Comprehensive_Car_Damage/F_Normal/FN_1.jpg",
                    "Normal Vehicle (Undamaged - Rear View)": "Comprehensive_Car_Damage/R_Normal/RN_1.jpg",
                    "Rear Bumper Breakage": "Comprehensive_Car_Damage/R_Breakage/RB_1.jpg",
                }
                sample_choice = st.selectbox("Select Sample Image:", list(sample_options.keys()))
                sample_rel_path = sample_options[sample_choice]
                sample_full_path = os.path.join(PROJECT_ROOT, sample_rel_path)

                if os.path.exists(sample_full_path):
                    image_to_process = Image.open(sample_full_path).convert("RGB")
                    st.image(image_to_process, caption=f"Sample: {sample_choice}", use_container_width=True)
                else:
                    st.warning("Sample image file not found on disk. Please upload an image.")

            assess_btn = st.button("🚀 Assess Damage & Predict Claim Amount", type="primary", use_container_width=True)

        with col_right:
            st.subheader("2. AI Assessment Results")

            if assess_btn and image_to_process is not None:
                with st.spinner("Analyzing image features through ResNet50 CNN and running XGBoost regression..."):
                    vehicle_details = {
                        "vehicle_age_years": vehicle_age,
                        "vehicle_segment": vehicle_segment,
                        "vehicle_insured_value_inr": insured_value,
                        "repair_region": repair_region,
                        "labor_rate_per_hour_inr": labor_rate,
                        "parts_price_index": parts_index,
                        "policy_deductible_inr": deductible,
                    }

                    results = predictor.assess_claim(image_to_process, vehicle_details)

                # Prominent Claim Card
                claim_val = results["predicted_claim_inr"]
                st.markdown(f"""
                <div class="claim-card">
                    <div style="font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;">
                        Estimated Insurance Claim Payout
                    </div>
                    <div class="claim-amount">₹ {claim_val:,.0f}</div>
                    <div style="font-size: 0.95rem; opacity: 0.9;">
                        Target: {results['total_damaged_components']} Damaged Component(s) Detected
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Quick Summary Metrics
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div style="font-size: 0.8rem; color: #64748B;">VEHICLE SEGMENT</div>
                        <div style="font-size: 1.1rem; font-weight: bold; color: #1E293B;">{vehicle_segment}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div style="font-size: 0.8rem; color: #64748B;">REPAIR REGION</div>
                        <div style="font-size: 1.1rem; font-weight: bold; color: #1E293B;">{repair_region} (₹{labor_rate}/hr)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div style="font-size: 0.8rem; color: #64748B;">DEDUCTIBLE APPLIED</div>
                        <div style="font-size: 1.1rem; font-weight: bold; color: #1E293B;">- ₹{deductible:,}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Component-Level Breakdown Table & Bars
                st.write("##### 🔬 8-Component Damage Detection Breakdown")
                comp_data = []
                for item in results["cnn_component_results"]:
                    status = "DAMAGED" if item["is_damaged"] else "UNDAMAGED"
                    comp_data.append({
                        "Vehicle Component": item["component"],
                        "CNN Probability": f"{item['probability']:.1%}",
                        "Optimal Threshold": f"{item['threshold']:.1%}",
                        "Status": status,
                        "raw_prob": item["probability"]
                    })

                for item in results["cnn_component_results"]:
                    c_col1, c_col2, c_col3 = st.columns([3, 4, 2])
                    with c_col1:
                        st.write(f"**{item['component']}**")
                    with c_col2:
                        st.progress(float(item["probability"]))
                    with c_col3:
                        if item["is_damaged"]:
                            st.markdown(f'<span class="status-badge-damaged">DAMAGED ({item["probability"]:.2f})</span>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<span class="status-badge-ok">OK ({item["probability"]:.2f})</span>', unsafe_allow_html=True)

                # Legal Disclaimer
                st.markdown(f"""
                <div class="disclaimer-box">
                    <strong>⚠️ Regulatory & Insurance Notice:</strong><br>
                    {results['disclaimer']}
                </div>
                """, unsafe_allow_html=True)

            elif assess_btn and image_to_process is None:
                st.warning("Please upload a vehicle image or select a sample image first.")
            else:
                st.info("👈 Upload an image or select a sample, adjust parameters, and click **Assess Damage & Predict Claim Amount**.")

    with tab2:
        st.subheader("Model Architecture & Benchmarking Report")
        col_m1, col_m2 = st.columns([1, 1])

        with col_m1:
            st.markdown("""
            ### 1. Multi-Label CNN Architecture
            - **Base Model:** ResNet50 (pre-trained on ImageNet)
            - **Input Dimensions:** 224 × 224 × 3 RGB
            - **Output Layer:** 8 Sigmoid Units with Binary Cross-Entropy
            - **Target Components:**
              1. Front-bumper (Threshold: 0.42)
              2. Hood (Threshold: 0.60)
              3. Headlight (Threshold: 0.31)
              4. Front-door (Threshold: 0.63)
              5. Back-door (Threshold: 0.74)
              6. Back-bumper (Threshold: 0.86)
              7. Fender (Threshold: 0.28)
              8. Windshield (Threshold: 0.83)
            - **Validation Binary Accuracy:** 88.99%
            - **Exact Match Ratio:** 68.65%
            """)

        with col_m2:
            st.markdown("""
            ### 2. Machine Learning Regression Champion
            - **Algorithm:** Tuned XGBoost Regressor (`XGBRegressor`)
            - **Dataset Size:** 40,000 Synthetic Insurance Claims (70/15/15 split)
            - **Features:** 23 Input Features (8 CNN Probabilities + 8 CNN Flags + 7 Vehicle/Policy Attributes)
            - **Optimal Hyperparameters:**
              - `n_estimators`: 250
              - `learning_rate`: 0.07
              - `max_depth`: 5
              - `subsample`: 0.75
              - `colsample_bytree`: 1.0
              - `reg_lambda`: 3.0, `reg_alpha`: 0.5
            - **Untouched Test Set Metrics:**
              - **MAE:** ₹ 9,468 INR
              - **RMSE:** ₹ 14,468 INR
              - **R² Score:** 0.8281
            """)

        st.markdown("---")
        st.subheader("Diagnostic Evaluation Charts on Untouched Test Set (6,000 claims)")
        eval_plot_path = os.path.join(PROJECT_ROOT, "models", "evaluation_plots", "xgboost_test_evaluation.png")
        if os.path.exists(eval_plot_path):
            st.image(eval_plot_path, caption="Actual vs Predicted, Residuals Distribution, Homoscedasticity, and Feature Importance", use_container_width=True)
        else:
            st.info("Evaluation plots will appear here once generated.")

    with tab3:
        st.subheader("Dataset Structure & Preprocessing Pipeline")
        st.markdown("""
        ### Synthetic Insurance Claim Dataset (40,000 Records)
        The tabular dataset simulates realistic insurance claim payouts modeled as a function of component repair costs,
        labor hours by region, spare-parts price inflation, vehicle depreciation, and deductible deductions:
        
        $$\\text{Claim Amount} = \\max\\left(0, \\left(\\sum_{i=1}^8 \\text{Flag}_i \\times \\text{PartCost}_i \\times \\text{Index} + \\text{LaborHours} \\times \\text{LaborRate}\\right) - \\text{Deductible}\\right) + \\epsilon$$

        ### Preprocessing Pipeline:
        1. **Partitioning:** 70% Train (28,000), 15% Validation (6,000), 15% Test (6,000) with reproducible seed.
        2. **Imputation:** Median for continuous numericals; Most frequent for categoricals.
        3. **Encoding:** One-Hot Encoding (`drop='first'`) for `vehicle_segment` and `repair_region`.
        4. **Tree Compatibility:** Unscaled raw distributions fed directly to XGBoost to maintain optimal threshold split purity.
        5. **Non-Negative Payout Guarantee:** Clipped at zero to eliminate unreasonable negative payouts.
        """)


if __name__ == "__main__":
    main()
