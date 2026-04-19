import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def build_preprocessor(num_features: list, cat_features: list) -> ColumnTransformer:
    numeric_branch = Pipeline([
        ("num_imputer", SimpleImputer(strategy="median")),
        ("num_scaler",  StandardScaler()),
    ])

    categorical_branch = Pipeline([
        ("cat_imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value=-1)),
    ])

    return ColumnTransformer(
        transformers=[
            ("numPreprocess", numeric_branch,     num_features),
            ("catPreprocess", categorical_branch, cat_features),
        ],
        remainder="drop",
    )
