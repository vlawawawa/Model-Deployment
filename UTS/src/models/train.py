import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlflow
import mlflow.sklearn

from config.config import (
    RF_CRITERION, RF_MAX_DEPTH, RF_N_ESTIMATORS,
    RF_REG_MAX_DEPTH, RF_REG_N_ESTIMATORS,
    MLFLOW_TRACKING_URI, MLFLOW_EXP_CLASS, MLFLOW_EXP_REG,
    ARTIFACT_CLASSIFIER, ARTIFACT_REGRESSOR,
)
from src.utils.io import save_artifact


def _setup_mlflow(experiment_name: str) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)


def train_classifier(pipeline, x_train, y_train) -> str:
    _setup_mlflow(MLFLOW_EXP_CLASS)

    with mlflow.start_run() as run:
        mlflow.set_tag("model_type", "random_forest_classifier")
        mlflow.set_tag("dataset",    "student_placement")
        mlflow.log_param("criterion",     RF_CRITERION)
        mlflow.log_param("max_depth",     RF_MAX_DEPTH)
        mlflow.log_param("n_estimators",  RF_N_ESTIMATORS)

        pipeline.fit(x_train, y_train)

        mlflow.sklearn.log_model(pipeline, name="model",
                                 registered_model_name="placement_classifier")
        save_artifact(pipeline, ARTIFACT_CLASSIFIER)

        print(f"Classifier trained. Run ID: {run.info.run_id}")
        return run.info.run_id


def train_regressor(pipeline, x_train, y_train) -> str:
    _setup_mlflow(MLFLOW_EXP_REG)

    with mlflow.start_run() as run:
        mlflow.set_tag("model_type", "random_forest_regressor")
        mlflow.set_tag("dataset",    "student_placement_placed_only")
        mlflow.log_param("max_depth",    RF_REG_MAX_DEPTH)
        mlflow.log_param("n_estimators", RF_REG_N_ESTIMATORS)

        pipeline.fit(x_train, y_train)

        mlflow.sklearn.log_model(pipeline, name="model",
                                 registered_model_name="salary_regressor")
        save_artifact(pipeline, ARTIFACT_REGRESSOR)

        print(f"Regressor trained. Run ID: {run.info.run_id}")
        return run.info.run_id
