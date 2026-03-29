import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlflow
import mlflow.sklearn

from config.config import (
    LR_C, LR_SOLVER, LR_MAX_ITER,
    MLFLOW_TRACKING_URI, MLFLOW_EXP_PIPELINE,
    ARTIFACT_PIPELINE,
)
from src.utils.io import save_artifact


def _setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXP_PIPELINE)


def train_pipeline(pipeline, x_train, y_train) -> str:
    _setup_mlflow()

    with mlflow.start_run() as run:
        mlflow.set_tag("model_type", "logistic_regression")
        mlflow.set_tag("dataset",    "spaceship_titanic")

        mlflow.log_param("C",        LR_C)
        mlflow.log_param("solver",   LR_SOLVER)
        mlflow.log_param("max_iter", LR_MAX_ITER)

        pipeline.fit(x_train, y_train)

        mlflow.sklearn.log_model(pipeline, name="model",
                                 registered_model_name="spaceship_lr")
        save_artifact(pipeline, ARTIFACT_PIPELINE)

        print(f"Pipeline trained. Run ID: {run.info.run_id}")
        return run.info.run_id
