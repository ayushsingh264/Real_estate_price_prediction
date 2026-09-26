import json
from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_DIR, REGISTRY_PATH
from .features import make_feature_frame


class ModelService:
    def __init__(self):
        self.model = None
        self.version = None
        self.selected_model = None
        self.residual_std = 0.0
        self.confidence_multiplier = 1.96
        self.load_latest()

    def load_latest(self):
        if not REGISTRY_PATH.exists():
            raise FileNotFoundError(
                "No trained model found. Run: python scripts/run_training.py"
            )

        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        version = registry.get("latest_version")

        entries = [
            x for x in registry.get("versions", [])
            if x["version"] == version
        ]
        if not entries:
            raise RuntimeError("Latest model is not present in the registry.")

        entry = entries[-1]
        model_path = Path(entry["model_path"])

        if not model_path.exists():
            model_path = MODEL_DIR / f"real_estate_price_{version}.joblib"

        self.model = joblib.load(model_path)
        self.version = version
        self.selected_model = entry["model"]

        metrics_path = Path(entry["metrics_path"])
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.residual_std = float(metrics.get("residual_std", 0))
            self.confidence_multiplier = float(
                metrics.get("confidence_multiplier", 1.96)
            )

    def predict(self, **kwargs):
        frame = make_feature_frame(**kwargs)
        prediction = float(self.model.predict(frame)[0])

        margin = self.confidence_multiplier * self.residual_std

        return {
            "predicted_price": round(prediction, 2),
            "confidence_interval": {
                "lower_bound": round(max(0, prediction - margin), 2),
                "upper_bound": round(prediction + margin, 2),
            },
            "model_version": self.version,
            "selected_model": self.selected_model,
        }
