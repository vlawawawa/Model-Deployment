"""Build sklearn pipelines for each candidate model.

Spaceship Titanic mixes numeric and categorical features and has missing
values everywhere, so every pipeline shares the same preprocessor:

    numeric:     median impute -> standard scale
    categorical: most-frequent impute -> one-hot encode

The preprocessor lives inside the Pipeline, so the saved model.joblib does
all imputation, scaling, and encoding at inference time. The serving code
only has to hand it a DataFrame of raw features.

Standardisation is essential for Logistic Regression and KNN (distance /
scale sensitive). XGBoost is tree-based and does not need scaling, but we
keep the same preprocessor for uniformity - the cost is negligible and the
code stays simpler.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def build_pipelines() -> dict[str, Pipeline]:
    """Return a dict of {name: pipeline} for all candidate models."""
    return {
        "logistic_regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "knn": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("clf", KNeighborsClassifier(n_neighbors=15)),
        ]),
        "xgboost": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("clf", XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_estimators=200,
            )),
        ]),
    }
