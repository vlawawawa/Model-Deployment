import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
import numpy as np

from config.config import MLFLOW_TRACKING_URI


def evaluate_classifier(x_test, y_test, run_id: str) -> tuple:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    preds = model.predict(x_test)
    acc   = accuracy_score(y_test, preds)
    prec  = precision_score(y_test, preds, average="macro")
    rec   = recall_score(y_test, preds, average="macro")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)

    print(f"[Classifier] Accuracy={acc:.3f} | Precision={prec:.3f} | Recall={rec:.3f}")
    return acc, prec, rec


def evaluate_regressor(x_test, y_test, run_id: str) -> tuple:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    preds = model.predict(x_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("mae",  mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2",   r2)

    print(f"[Regressor]  MAE={mae:.3f} | RMSE={rmse:.3f} | R²={r2:.3f}")
    return mae, rmse, r2
