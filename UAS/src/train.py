"""
train.py
OOP training pipeline with MLflow experiment tracking.

Run:
    python src/train.py --data data_C.csv

What it does
------------
1. Loads raw data, applies DataCleaner.
2. Splits BY Customer_ID (GroupShuffleSplit) to prevent the same customer
   appearing in train and test -> avoids leakage / inflated scores.
3. Trains several models (LogReg, RandomForest, XGBoost, LightGBM) inside a
   full sklearn Pipeline (clean -> preprocess -> model).
4. Logs params, macro-F1, accuracy, and the model to MLflow for every run.
5. Selects the best model by macro-F1, refits, and writes:
       models/model.pkl, models/label_encoder.pkl, models/metadata.json

Classes
-------
ModelTrainer : owns the train/select loop and MLflow logging.
Evaluator    : computes and packages metrics for one fitted model.
"""
from __future__ import annotations
import argparse
import json
import os
import warnings
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import mlflow
import mlflow.sklearn

# Allow running as `python src/train.py` from repo root.
try:
    from preprocessing import (DataCleaner, PreprocessorBuilder,
                               TARGET, ID_COLUMNS)
except ImportError:  # when imported as package
    from src.preprocessing import (DataCleaner, PreprocessorBuilder,
                                   TARGET, ID_COLUMNS)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


@dataclass
class Evaluator:
    """Computes and stores evaluation metrics for a fitted pipeline."""
    labels: list[str]

    def evaluate(self, pipeline, X_test, y_test) -> dict:
        pred = pipeline.predict(X_test)
        return {
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
            "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
            "report": classification_report(
                y_test, pred, target_names=self.labels, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }


@dataclass
class ModelTrainer:
    """Trains and compares candidate models with MLflow tracking."""
    data_path: str
    model_dir: str = "models"
    experiment_name: str = "credit_score_classification"
    test_size: float = 0.2
    candidates: dict = field(default_factory=dict)

    def __post_init__(self):
        os.makedirs(self.model_dir, exist_ok=True)
        if not self.candidates:
            self.candidates = {
                "LogisticRegression": LogisticRegression(
                    max_iter=1000, class_weight="balanced"),
                "RandomForest": RandomForestClassifier(
                    n_estimators=200, class_weight="balanced",
                    n_jobs=-1, random_state=RANDOM_STATE),
                "XGBoost": XGBClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.1,
                    n_jobs=-1, eval_metric="mlogloss",
                    random_state=RANDOM_STATE),
                "LightGBM": LGBMClassifier(
                    n_estimators=300, max_depth=6, class_weight="balanced",
                    n_jobs=-1, random_state=RANDOM_STATE, verbose=-1),
            }

    # ----- data -----------------------------------------------------------
    def _load_split(self):
        raw = pd.read_csv(self.data_path, index_col=0)
        groups = raw["Customer_ID"].values

        # Target is taken from RAW (never cleaned/dropped). Features stay RAW
        # here -- DataCleaner runs *inside* the pipeline so the exact same
        # cleaning is serialized into the pickle and re-run at inference.
        y = raw[TARGET]
        X = raw.drop(columns=[TARGET])

        splitter = GroupShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=RANDOM_STATE)
        tr_idx, te_idx = next(splitter.split(X, y, groups))

        overlap = len(set(groups[tr_idx]) & set(groups[te_idx]))
        assert overlap == 0, f"Customer leakage: {overlap} shared IDs"

        self.label_encoder = LabelEncoder()
        y_tr = self.label_encoder.fit_transform(y.iloc[tr_idx])
        y_te = self.label_encoder.transform(y.iloc[te_idx])
        # Cleaned copy ONLY for inferring numeric/categorical column lists.
        cleaned_for_cols = DataCleaner().transform(raw.drop(columns=[TARGET]))
        return (X.iloc[tr_idx], X.iloc[te_idx], y_tr, y_te, cleaned_for_cols)

    # ----- training loop --------------------------------------------------
    def run(self) -> dict:
        X_tr, X_te, y_tr, y_te, cleaned_cols = self._load_split()

        # Infer numeric/categorical columns from a CLEANED frame.
        prep_builder = PreprocessorBuilder.from_cleaned_frame(cleaned_cols)
        evaluator = Evaluator(labels=list(self.label_encoder.classes_))

        mlflow.set_experiment(self.experiment_name)
        leaderboard = {}
        best_name, best_f1, best_pipeline = None, -1.0, None

        for name, estimator in self.candidates.items():
            with mlflow.start_run(run_name=name):
                pipeline = Pipeline([
                    ("clean", DataCleaner()),
                    ("preprocess", prep_builder.build()),
                    ("model", estimator),
                ])
                pipeline.fit(X_tr, y_tr)
                metrics = evaluator.evaluate(pipeline, X_te, y_te)

                mlflow.log_param("model_type", name)
                mlflow.log_params({
                    f"hp_{k}": v for k, v in estimator.get_params().items()
                    if isinstance(v, (int, float, str, bool))})
                mlflow.log_metric("accuracy", metrics["accuracy"])
                mlflow.log_metric("macro_f1", metrics["macro_f1"])
                mlflow.log_metric("weighted_f1", metrics["weighted_f1"])
                mlflow.sklearn.log_model(pipeline, name="model")

                leaderboard[name] = metrics["macro_f1"]
                print(f"[{name:18s}] macro-F1={metrics['macro_f1']:.4f} "
                      f"acc={metrics['accuracy']:.4f}")

                if metrics["macro_f1"] > best_f1:
                    best_name, best_f1 = name, metrics["macro_f1"]
                    best_pipeline, best_metrics = pipeline, metrics

        # ----- persist the best -------------------------------------------
        joblib.dump(best_pipeline, os.path.join(self.model_dir, "model.pkl"))
        joblib.dump(self.label_encoder,
                    os.path.join(self.model_dir, "label_encoder.pkl"))
        metadata = {
            "best_model": best_name,
            "macro_f1": best_f1,
            "classes": list(self.label_encoder.classes_),
            "leaderboard": leaderboard,
            "feature_columns": list(cleaned_cols.columns),
            "raw_input_columns": [c for c in pd.read_csv(
                self.data_path, index_col=0, nrows=1).columns
                if c not in (ID_COLUMNS + [TARGET])],
        }
        with open(os.path.join(self.model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nBEST: {best_name}  macro-F1={best_f1:.4f}")
        print(best_metrics["report"])
        return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_C.csv")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    ModelTrainer(data_path=args.data, model_dir=args.model_dir).run()


if __name__ == "__main__":
    main()
