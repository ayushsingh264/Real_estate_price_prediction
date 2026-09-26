from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
REGISTRY_DIR = ARTIFACTS_DIR / "registry"
REGISTRY_PATH = REGISTRY_DIR / "registry.json"

HOUSE_PRICE_DATA = RAW_DATA_DIR / "house_prices.csv"

FEATURE_COLUMNS = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Age",
    "Location",
    "Property_Type",
]
TARGET_COLUMN = "Price"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

for directory in [
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    METRICS_DIR,
    REGISTRY_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
