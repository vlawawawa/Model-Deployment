"""SageMaker inference entry point for the Spaceship Titanic pipeline.

The model.joblib is a full sklearn Pipeline (preprocessor + classifier),
so this file's only job is to translate between HTTP bodies and the
DataFrame shape the pipeline expects.

Request format (JSON):
    {
      "instances": [
        {
          "PassengerId": "0013_01",
          "HomePlanet": "Earth",
          "CryoSleep": true,
          "Cabin": "G/3/S",
          "Destination": "TRAPPIST-1e",
          "Age": 27.0,
          "VIP": false,
          "RoomService": 0.0,
          "FoodCourt": 0.0,
          "ShoppingMall": 0.0,
          "Spa": 0.0,
          "VRDeck": 0.0,
          "Name": "Nelly Carsoning"
        }
      ]
    }

PassengerId and Name are accepted but ignored. Missing fields are allowed -
the pipeline's imputers fill them at scoring time.

Four functions form the SageMaker contract:
    model_fn   - load model from disk (called once per container)
    input_fn   - parse request body (called per request)
    predict_fn - run inference (called per request)
    output_fn  - serialize response (called per request)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

# data.py sits next to this file in the deployed source bundle.
from data import CLASS_NAMES, prepare_features


JSON_CONTENT_TYPE = "application/json"


def model_fn(model_dir: str):
    """Load the pickled sklearn Pipeline from model.joblib."""
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, request_content_type: str) -> pd.DataFrame:
    """Parse incoming JSON and turn it into a model-ready DataFrame."""
    if request_content_type != JSON_CONTENT_TYPE:
        raise ValueError(f"Unsupported content type: {request_content_type}")

    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")

    payload = json.loads(request_body)
    instances = payload["instances"]
    raw = pd.DataFrame(instances)
    return prepare_features(raw)


def predict_fn(input_data: pd.DataFrame, pipeline) -> dict:
    """Run inference. Returns probabilities, predicted class IDs, and labels."""
    probs = pipeline.predict_proba(input_data)
    class_ids = np.argmax(probs, axis=1)
    labels = [CLASS_NAMES[int(i)] for i in class_ids]
    return {
        "probabilities": probs.tolist(),
        "predictions": class_ids.tolist(),
        "labels": labels,
    }


def output_fn(prediction: dict, accept_content_type: str):
    """Serialize the prediction dict for the response body."""
    if accept_content_type == JSON_CONTENT_TYPE:
        return json.dumps(prediction), JSON_CONTENT_TYPE
    raise ValueError(f"Unsupported accept type: {accept_content_type}")
