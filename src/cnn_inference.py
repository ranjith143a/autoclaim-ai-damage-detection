"""
CNN Inference Module for Multi-Label Vehicle Damage Detection.
Integrates the trained ResNet50 multi-label model with optimal decision thresholds.
"""

import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras


class VehicleDamageCNN:
    """
    Multi-label vehicle damage detector that predicts probability scores
    and binary damage flags across 8 vehicle components using a trained CNN.
    """

    def __init__(self, model_dir=None):
        if model_dir is None:
            # Default to model_artifacts relative to project root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, "model_artifacts")

        self.model_dir = model_dir
        self.config_path = os.path.join(model_dir, "multilabel_config.json")
        self.model_path = os.path.join(model_dir, "best_multilabel_model.keras")

        self._load_config()
        self.model = None

    def _load_config(self):
        """Loads component ordering, optimal thresholds, and input size."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

        with open(self.config_path, "r") as f:
            self.config = json.load(f)

        self.target_components = self.config["target_components"]
        self.thresholds = self.config["optimal_thresholds"]
        self.input_size = tuple(self.config.get("input_size", [224, 224, 3]))

        # Mapping to regression column names
        self.col_prob_mapping = {
            comp: f"{comp.lower().replace('-', '_')}_cnn_probability"
            for comp in self.target_components
        }
        self.col_flag_mapping = {
            comp: f"{comp.lower().replace('-', '_')}_cnn_flag"
            for comp in self.target_components
        }

    def load_model(self):
        """Loads Keras model if not already loaded."""
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")
            self.model = keras.models.load_model(self.model_path, compile=False)
        return self.model

    def preprocess_image(self, image_source):
        """
        Preprocesses an image from path, PIL Image, or bytes to normalized (1, 224, 224, 3) tensor.
        """
        if isinstance(image_source, str):
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"Image not found at: {image_source}")
            img = Image.open(image_source).convert("RGB")
        elif isinstance(image_source, Image.Image):
            img = image_source.convert("RGB")
        elif hasattr(image_source, "read"):
            img = Image.open(image_source).convert("RGB")
        else:
            raise ValueError("Unsupported image source type. Provide a filepath, PIL Image, or file-like object.")

        target_h, target_w = self.input_size[0], self.input_size[1]
        img_resized = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        tensor = np.expand_dims(arr, axis=0)
        return tensor

    def predict(self, image_source):
        """
        Predicts damage probabilities and flags for all 8 target components.
        
        Returns:
            dict containing:
                - component_results: list of dicts with component, probability, threshold, and is_damaged
                - probabilities: dict of {component: prob}
                - flags: dict of {component: flag}
                - regression_features: dict mapping exact regression dataset columns to values
                - damaged_components_list: list of damaged component names
                - total_damaged_count: count of damaged components
        """
        self.load_model()
        tensor = self.preprocess_image(image_source)

        raw_preds = self.model(tensor, training=False).numpy()[0]

        component_results = []
        probabilities = {}
        flags = {}
        regression_features = {}
        damaged_list = []

        for idx, comp in enumerate(self.target_components):
            prob = float(raw_preds[idx])
            thresh = float(self.thresholds.get(comp, 0.5))
            is_damaged = int(prob >= thresh)

            probabilities[comp] = prob
            flags[comp] = is_damaged

            prob_col = self.col_prob_mapping[comp]
            flag_col = self.col_flag_mapping[comp]
            regression_features[prob_col] = prob
            regression_features[flag_col] = is_damaged

            if is_damaged:
                damaged_list.append(comp)

            component_results.append({
                "component": comp,
                "probability": round(prob, 4),
                "threshold": round(thresh, 4),
                "is_damaged": bool(is_damaged),
                "confidence_pct": round(prob * 100, 1)
            })

        return {
            "component_results": component_results,
            "probabilities": probabilities,
            "flags": flags,
            "regression_features": regression_features,
            "damaged_components_list": damaged_list,
            "total_damaged_count": len(damaged_list)
        }


if __name__ == "__main__":
    cnn = VehicleDamageCNN()
    print("Testing CNN initialization...")
    cnn.load_model()
    print("CNN Loaded successfully!")
    print(f"Target components ({len(cnn.target_components)}):", cnn.target_components)
    print("Thresholds:", cnn.thresholds)
