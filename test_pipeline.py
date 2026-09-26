from ml_pipeline.config import HOUSE_PRICE_DATA
from ml_pipeline.data import load_house_price_data
from ml_pipeline.models import build_model_pipelines


def test_dataset():
    df = load_house_price_data(HOUSE_PRICE_DATA)
    assert len(df) == 300
    assert "Price" in df.columns


def test_models():
    models = build_model_pipelines()
    assert "random_forest" in models
    assert "extra_trees" in models
    assert "gradient_boosting" in models
    assert "mlp_regressor" in models
