from pathlib import Path
import pandas as pd

from .config import FEATURE_COLUMNS, TARGET_COLUMN

REQUIRED_COLUMNS = ["Property_ID", *FEATURE_COLUMNS, TARGET_COLUMN]


def load_house_price_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return validate_house_price_data(df)


def validate_house_price_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if data.empty:
        raise ValueError("Dataset is empty.")

    if data[REQUIRED_COLUMNS].isna().any().any():
        missing = data[REQUIRED_COLUMNS].isna().sum()
        raise ValueError(f"Missing values found: {missing[missing > 0].to_dict()}")

    if data.duplicated().any():
        raise ValueError("Duplicate rows found.")

    for column in ["Area", "Bedrooms", "Bathrooms", "Age", "Price"]:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if (data["Area"] <= 0).any():
        raise ValueError("Area must be greater than zero.")
    if (data["Bedrooms"] <= 0).any():
        raise ValueError("Bedrooms must be greater than zero.")
    if (data["Bathrooms"] <= 0).any():
        raise ValueError("Bathrooms must be greater than zero.")
    if (data["Age"] < 0).any():
        raise ValueError("Age cannot be negative.")
    if (data["Price"] <= 0).any():
        raise ValueError("Price must be greater than zero.")

    return data
