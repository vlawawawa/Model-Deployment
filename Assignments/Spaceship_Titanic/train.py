"""
Spaceship Titanic – Step 3: Training
Trains an optimised Logistic Regression and logs to MLflow.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib


def train(train_set):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Spaceship-Titanic-Pipeline")

    X_train = train_set.drop("Transported", axis=1)
    y_train = train_set["Transported"]

    # Hyperparameter search
    param_grid = {
        "C":        [0.01, 0.1, 1, 10],
        "penalty":  ["l1", "l2"],
        "solver":   ["liblinear", "saga"],
        "max_iter": [500, 1000],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid, cv=cv, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    with mlflow.start_run() as run:
        mlflow.set_tag("model_type", "logistic_regression")
        mlflow.set_tag("dataset", "spaceship_titanic")

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("cv_accuracy", grid.best_score_)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            registered_model_name="spaceship_lr"
        )
        joblib.dump(best_model, "artifacts/model.pkl")

        print(f"Model trained. Best CV accuracy: {grid.best_score_:.4f}")
        print(f"Best params: {grid.best_params_}")
        print(f"Run ID: {run.info.run_id}")
        print("Model saved to artifacts/model.pkl")
        return run.info.run_id


if __name__ == "__main__":
    train(None)