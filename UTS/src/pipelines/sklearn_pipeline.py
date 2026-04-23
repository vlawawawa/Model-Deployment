import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from config.config import (
    NUM_FEATURES, CAT_FEATURES,
    RF_CRITERION, RF_MAX_DEPTH, RF_N_ESTIMATORS,
    RF_REG_MAX_DEPTH, RF_REG_N_ESTIMATORS,
    RANDOM_STATE,
)
from src.features.pipeline_preprocessor import build_preprocessor


def build_classifier_pipeline() -> Pipeline:
    preprocessor = build_preprocessor(NUM_FEATURES, CAT_FEATURES)
    clf = RandomForestClassifier(
        criterion=RF_CRITERION,
        max_depth=RF_MAX_DEPTH,
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    return Pipeline([
        ("preprocessing", preprocessor),
        ("classifier",    clf),
    ])


def build_regressor_pipeline() -> Pipeline:
    preprocessor = build_preprocessor(NUM_FEATURES, CAT_FEATURES)
    reg = RandomForestRegressor(
        max_depth=RF_REG_MAX_DEPTH,
        n_estimators=RF_REG_N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    return Pipeline([
        ("preprocessing", preprocessor),
        ("regressor",     reg),
    ])
