import importlib.util

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from .features import create_preprocessor


def build_model_pipelines(random_state: int = 42):
    candidates = {
        "random_forest": (
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            ),
            False,
        ),
        "extra_trees": (
            ExtraTreesRegressor(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            ),
            False,
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.04,
                max_depth=3,
                loss="huber",
                random_state=random_state,
            ),
            False,
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=0.1,
                random_state=random_state,
            ),
            False,
        ),
        "mlp_regressor": (
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=1500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=random_state,
            ),
            True,
        ),
    }

    pipelines = {}
    for name, (model, scale_numeric) in candidates.items():
        pipelines[name] = Pipeline([
            ("preprocessor", create_preprocessor(scale_numeric)),
            ("model", model),
        ])

    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBRegressor

        pipelines["xgboost"] = Pipeline([
            ("preprocessor", create_preprocessor(False)),
            ("model", XGBRegressor(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=random_state,
                n_jobs=4,
            )),
        ])

    return pipelines
