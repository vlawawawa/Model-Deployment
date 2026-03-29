import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config.config import LR_C, LR_SOLVER, LR_MAX_ITER, RANDOM_STATE
from src.features.pipeline_preprocessor import build_preprocessor


def build_spaceship_pipeline(num_features: list, cat_features: list) -> Pipeline:
    preprocessor = build_preprocessor(num_features, cat_features)

    lr = LogisticRegression(
        C=LR_C,
        solver=LR_SOLVER,
        max_iter=LR_MAX_ITER,
        random_state=RANDOM_STATE,
    )

    return Pipeline([
        ("preprocessing", preprocessor),
        ("classifier",    lr),
    ])
