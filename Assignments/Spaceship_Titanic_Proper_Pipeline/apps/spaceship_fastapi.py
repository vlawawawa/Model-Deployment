"""
Run with: uvicorn apps.spaceship_fastapi:app --reload
Then test at: http://127.0.0.1:8000/docs

NOTE: If spaceship_pipeline.pkl does not exist yet, the full training
      pipeline is run automatically on first startup so the server
      always starts cleanly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from config.config import ARTIFACT_PIPELINE
from src.utils.io import load_artifact
from src.data.loader import feature_engineering


# Auto-train pipeline if pkl is missing

if not ARTIFACT_PIPELINE.exists():
    print("[spaceship_fastapi] No pipeline found – running training pipeline...")
    from config.config import NUM_FEATURES, CAT_FEATURES
    from src.data.loader import ingest_data, load_frame, split_features_target, split_train_test
    from src.pipelines.sklearn_pipeline import build_spaceship_pipeline
    from src.utils.io import save_artifact

    ingest_data()
    df = load_frame()
    X, y = split_features_target(df)
    x_train, _, y_train, _ = split_train_test(X, y)

    pipeline = build_spaceship_pipeline(NUM_FEATURES, CAT_FEATURES)
    pipeline.fit(x_train, y_train)
    save_artifact(pipeline, ARTIFACT_PIPELINE)
    print(f"[spaceship_fastapi] Pipeline trained and saved to {ARTIFACT_PIPELINE}")


# Load pipeline

pipeline = load_artifact(ARTIFACT_PIPELINE)

app = FastAPI(title="Spaceship Titanic Prediction API")


# Schema 

class PassengerFeatures(BaseModel):
    PassengerId:  str   = "6767_67"
    HomePlanet:   str   = "Earth"
    CryoSleep:    bool  = False
    Cabin:        str   = "B/67/P"
    Destination:  str   = "TRAPPIST-1e"
    Age:          int = 67
    VIP:          bool  = False
    RoomService:  int = 0.0
    FoodCourt:    int = 0.0
    ShoppingMall: int = 0.0
    Spa:          int = 0.0
    VRDeck:       int = 0.0
    Name:         str   = "Pria Solo"


# Endpoints

@app.get("/")
def root():
    return {"message": "Welcome to the Spaceship Titanic Prediction API"}


@app.post("/predict")
def predict(passenger: PassengerFeatures):
    df = pd.DataFrame([passenger.model_dump()])
    df = feature_engineering(df)

    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0][1]

    return {
        "transported":  bool(prediction),
        "probability":  round(float(probability), 4),
        "message":      "TRANSPORTED" if prediction else "NOT TRANSPORTED",
    }
