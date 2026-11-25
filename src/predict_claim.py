"""
End-to-End Prediction Pipeline Module.
Combines CNN damage detection with the fitted regression pipeline
to predict insurance claim amounts from an image and vehicle metadata.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cnn_inference import VehicleDamageCNN


class ClaimAssessmentPredictor:
    """
    Unified End-to-End Inference Engine:
    Image + Vehicle & Insurance Details -> CNN Damage Detection -> Tabular Preprocessing -> Claim Amount (INR)
    """

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(PROJECT_ROOT, "models")
        
        self.model_dir = model_dir
        self.regressor_path = os.path.join(model_dir, "best_claim_regressor.pkl")
        self.pipeline_path = os.path.join(model_dir, "preprocessing_pipeline.pkl")
        self.metadata_path = os.path.join(model_dir, "regression_metadata.json")

        # Initialize CNN
        self.cnn = VehicleDamageCNN()

        # Load Regression Pipeline & Model
        self._load_regression_artifacts()

    def _load_regression_artifacts(self):
        """Loads serialized regression model, preprocessing pipeline, and metadata."""
        if not os.path.exists(self.regressor_path):
            raise FileNotFoundError(f"Trained regressor not found at: {self.regressor_path}")
        if not os.path.exists(self.pipeline_path):
            raise FileNotFoundError(f"Preprocessing pipeline not found at: {self.pipeline_path}")

        self.regressor = joblib.load(self.regressor_path)
        self.pipeline = joblib.load(self.pipeline_path)

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
            self.input_columns = self.metadata["input_feature_columns"]
        else:
            # Fallback exact 23 column order
            self.input_columns = [
                "vehicle_age_years", "vehicle_segment", "vehicle_insured_value_inr",
                "repair_region", "labor_rate_per_hour_inr", "parts_price_index",
                "policy_deductible_inr", "front_bumper_cnn_probability",
                "hood_cnn_probability", "headlight_cnn_probability",
                "front_door_cnn_probability", "back_door_cnn_probability",
                "back_bumper_cnn_probability", "fender_cnn_probability",
                "windshield_cnn_probability", "front_bumper_cnn_flag",
                "hood_cnn_flag", "headlight_cnn_flag", "front_door_cnn_flag",
                "back_door_cnn_flag", "back_bumper_cnn_flag", "fender_cnn_flag",
                "windshield_cnn_flag"
            ]

    def assess_claim(self, image_source, vehicle_details):
        """
        Executes end-to-end assessment:
        Args:
            image_source: filepath, PIL Image, or file-like object
            vehicle_details: dict containing:
                - vehicle_age_years: int
                - vehicle_segment: str ('hatchback', 'sedan', 'suv', 'luxury')
                - vehicle_insured_value_inr: float/int
                - repair_region: str ('tier_1', 'tier_2', 'tier_3')
                - labor_rate_per_hour_inr: float/int
                - parts_price_index: float
                - policy_deductible_inr: float/int

        Returns:
            dict containing:
                - predicted_claim_inr: float
                - predicted_claim_formatted: str
                - cnn_results: dict with probabilities and flags
                - damaged_components: list of damaged component names
                - total_damaged_components: int
                - input_row: pd.DataFrame
                - disclaimer: str
        """
        # 1. Run CNN Inference
        cnn_out = self.cnn.predict(image_source)
        cnn_features = cnn_out["regression_features"]

        # 2. Build Tabular Feature Row
        row_dict = {}

        # Fill metadata with lowercasing for categoricals
        row_dict["vehicle_age_years"] = int(vehicle_details.get("vehicle_age_years", 5))
        row_dict["vehicle_segment"] = str(vehicle_details.get("vehicle_segment", "sedan")).lower().strip()
        row_dict["vehicle_insured_value_inr"] = float(vehicle_details.get("vehicle_insured_value_inr", 1000000))
        row_dict["repair_region"] = str(vehicle_details.get("repair_region", "tier_2")).lower().strip().replace(" ", "_")
        row_dict["labor_rate_per_hour_inr"] = float(vehicle_details.get("labor_rate_per_hour_inr", 650))
        row_dict["parts_price_index"] = float(vehicle_details.get("parts_price_index", 1.0))
        row_dict["policy_deductible_inr"] = float(vehicle_details.get("policy_deductible_inr", 5000))

        # Fill CNN features
        for k, v in cnn_features.items():
            row_dict[k] = v

        # Construct DataFrame with exact column ordering
        input_df = pd.DataFrame([row_dict])[self.input_columns]

        # 3. Transform via fitted preprocessing pipeline
        X_trans = self.pipeline.transform(input_df)

        # 4. Predict and clip at zero
        raw_pred = float(self.regressor.predict(X_trans)[0])
        claim_inr = max(0.0, raw_pred)

        # Logic check: If all components are undamaged and deductible is high, payout is 0
        if cnn_out["total_damaged_count"] == 0 and claim_inr < 2500:
            claim_inr = 0.0

        disclaimer = (
            "This is an AI-generated estimated claim amount based on the available model "
            "and input information. It is not an official insurance settlement or guaranteed payout."
        )

        return {
            "predicted_claim_inr": round(claim_inr, 2),
            "predicted_claim_formatted": f"INR {claim_inr:,.2f}",
            "cnn_component_results": cnn_out["component_results"],
            "cnn_probabilities": cnn_out["probabilities"],
            "cnn_flags": cnn_out["flags"],
            "damaged_components": cnn_out["damaged_components_list"],
            "total_damaged_components": cnn_out["total_damaged_count"],
            "input_features": row_dict,
            "disclaimer": disclaimer,
        }


if __name__ == "__main__":
    predictor = ClaimAssessmentPredictor()
    print("Testing End-to-End Predictor with synthetic dummy vehicle image...")

    # Create a dummy image
    test_img = Image.new("RGB", (224, 224), color=(180, 50, 50))
    sample_details = {
        "vehicle_age_years": 3,
        "vehicle_segment": "SUV",
        "vehicle_insured_value_inr": 1500000,
        "repair_region": "Tier 1",
        "labor_rate_per_hour_inr": 900,
        "parts_price_index": 1.05,
        "policy_deductible_inr": 5000,
    }

    result = predictor.assess_claim(test_img, sample_details)
    print("\nResult:")
    print(f"Predicted Claim Amount: {result['predicted_claim_formatted']}")
    print(f"Total Damaged Components: {result['total_damaged_components']}")
    print("Damaged Components:", result["damaged_components"])
    print("\nComponent Probabilities:")
    for c in result["cnn_component_results"]:
        status = "DAMAGED" if c["is_damaged"] else "OK"
        print(f"  {c['component']:15s}: {c['probability']:.4f} (Thresh: {c['threshold']:.2f}) -> [{status}]")
