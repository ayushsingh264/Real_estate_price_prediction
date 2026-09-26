import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["Area", "Bedrooms", "Bathrooms", "Age"]
CATEGORICAL_FEATURES = ["Location", "Property_Type"]


def create_preprocessor(scale_numeric: bool = False):
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def make_feature_frame(
    area: float,
    bedrooms: int,
    bathrooms: int,
    age: int,
    location: str,
    property_type: str,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age": age,
        "Location": location,
        "Property_Type": property_type,
    }])
