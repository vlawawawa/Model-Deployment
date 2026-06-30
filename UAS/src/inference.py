"""
inference.py
OOP inference service that loads the pickled pipeline and serves predictions.

The pickle already contains DataCleaner -> preprocessing -> model, so inference
only needs to feed a raw-shaped DataFrame in. This guarantees the cleaning that
ran in training runs identically here (no train/serve skew).

Classes
-------
InferencePipeline : loads artifacts once, exposes predict() and predict_proba().

CLI:
    python src/inference.py --input sample.csv
"""
from __future__ import annotations
import argparse
import json
import os
import sys

# Ensure the directory containing preprocessing.py (this file's dir) is on the
# path so the pickled DataCleaner class resolves no matter the working dir.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import preprocessing  # noqa: F401  (registers DataCleaner for unpickling)


class InferencePipeline:
    """Loads the trained pipeline + label encoder and produces predictions."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.pipeline = joblib.load(os.path.join(model_dir, "model.pkl"))
        self.label_encoder = joblib.load(
            os.path.join(model_dir, "label_encoder.pkl"))
        with open(os.path.join(model_dir, "metadata.json")) as f:
            self.metadata = json.load(f)
        self.feature_columns = self.metadata["feature_columns"]
        self.classes = self.metadata["classes"]

    def _coerce(self, data) -> pd.DataFrame:
        """Accept a dict (one record) or DataFrame; return a DataFrame."""
        if isinstance(data, dict):
            return pd.DataFrame([data])
        if isinstance(data, pd.DataFrame):
            return data.copy()
        raise TypeError("data must be a dict or pandas DataFrame")

    def predict(self, data) -> list[str]:
        """Return predicted class label(s) as strings."""
        df = self._coerce(data)
        encoded = self.pipeline.predict(df)
        return self.label_encoder.inverse_transform(encoded).tolist()

    def predict_proba(self, data) -> list[dict]:
        """Return per-class probability dict(s)."""
        df = self._coerce(data)
        probs = self.pipeline.predict_proba(df)
        return [dict(zip(self.classes, row.round(4).tolist())) for row in probs]

    def predict_with_confidence(self, data) -> list[dict]:
        """Convenience: label + confidence + full distribution per record."""
        labels = self.predict(data)
        proba = self.predict_proba(data)
        out = []
        for label, dist in zip(labels, proba):
            out.append({
                "prediction": label,
                "confidence": float(dist[label]),
                "probabilities": dist,
            })
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="CSV file shaped like the raw training data")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    pipe = InferencePipeline(model_dir=args.model_dir)
    df = pd.read_csv(args.input, index_col=0) if "," in open(args.input).readline() else pd.read_csv(args.input)
    results = pipe.predict_with_confidence(df)
    for i, r in enumerate(results):
        print(f"Row {i}: {r['prediction']} "
              f"(confidence={r['confidence']:.3f})")


if __name__ == "__main__":
    main()
