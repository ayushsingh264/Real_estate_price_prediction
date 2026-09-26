import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from .config import (
    FEATURE_COLUMNS,
    HOUSE_PRICE_DATA,
    METRICS_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    REGISTRY_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    VALIDATION_SIZE,
)
from .data import load_house_price_data
from .models import build_model_pipelines

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
        "mape_percent": mape,
    }


def train_and_register_model():
    df = load_house_price_data(HOUSE_PRICE_DATA)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    val_ratio = VALIDATION_SIZE / (1 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=RANDOM_STATE,
    )

    candidates = build_model_pipelines(RANDOM_STATE)
    results = {}

    for name, pipeline in candidates.items():
        logger.info("Training %s", name)

        cv = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="neg_root_mean_squared_error",
        )
        cv_rmse = float(-cv.mean())

        pipeline.fit(X_train, y_train)
        val_pred = pipeline.predict(X_val)
        val_metrics = metrics(y_val.to_numpy(), val_pred)

        results[name] = {
            "cv_rmse": cv_rmse,
            "validation": val_metrics,
        }

        logger.info(
            "%s | CV RMSE %.2f | Validation RMSE %.2f",
            name,
            cv_rmse,
            val_metrics["rmse"],
        )

    selected_name = min(results, key=lambda n: results[n]["cv_rmse"])
    selected_pipeline = candidates[selected_name]

    # Refit selected model on train + validation.
    selected_pipeline.fit(X_train_val, y_train_val)

    test_pred = selected_pipeline.predict(X_test)
    test_metrics = metrics(y_test.to_numpy(), test_pred)

    residuals = y_test.to_numpy() - test_pred
    residual_std = float(np.std(residuals, ddof=1))

    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"real_estate_price_{version}.joblib"
    metrics_path = METRICS_DIR / f"metrics_{version}.json"

    joblib.dump(selected_pipeline, model_path)

    payload = {
        "project": "real_estate_price_prediction",
        "version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected_name,
        "rows": int(len(df)),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "splits": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "model_comparison": results,
        "test_metrics": test_metrics,
        "residual_std": residual_std,
        "confidence_multiplier": 1.96,
    }

    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    registry = {"latest_version": version, "versions": []}
    if REGISTRY_PATH.exists():
        try:
            registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    registry.setdefault("versions", [])
    registry["versions"].append({
        "version": version,
        "model": selected_name,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "created_at_utc": payload["created_at_utc"],
        "test_metrics": test_metrics,
    })
    registry["latest_version"] = version

    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print("\nTRAINING COMPLETE")
    print(f"Selected model : {selected_name}")
    print(f"Model version  : {version}")
    print(f"Test MAE       : {test_metrics['mae']:.2f}")
    print(f"Test RMSE      : {test_metrics['rmse']:.2f}")
    print(f"Test R2        : {test_metrics['r2']:.4f}")
    print(f"Test MAPE      : {test_metrics['mape_percent']:.2f}%")

    return payload


if __name__ == "__main__":
    train_and_register_model()
