"""
Run with: uvicorn apps.B_fastapi:app --reload
Then test at: http://127.0.0.1:8000/docs

NOTE: If .pkl artifacts do not exist, the training pipeline runs automatically.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from config.config import ARTIFACT_CLASSIFIER, ARTIFACT_REGRESSOR
from src.utils.io import load_artifact


# Auto-train if pkl is missing 

def _auto_train():
    from src.data.loader import ingest_data, get_classification_data, get_regression_data
    from src.pipelines.sklearn_pipeline import build_classifier_pipeline, build_regressor_pipeline
    from src.utils.io import save_artifact

    print("[B_fastapi] No artifacts found – running training pipeline...")
    ingest_data()

    x_train_c, _, y_train_c, _ = get_classification_data()
    clf = build_classifier_pipeline()
    clf.fit(x_train_c, y_train_c)
    save_artifact(clf, ARTIFACT_CLASSIFIER)

    x_train_r, _, y_train_r, _ = get_regression_data()
    reg = build_regressor_pipeline()
    reg.fit(x_train_r, y_train_r)
    save_artifact(reg, ARTIFACT_REGRESSOR)
    print("[B_fastapi] Models trained and saved.")


if not ARTIFACT_CLASSIFIER.exists() or not ARTIFACT_REGRESSOR.exists():
    _auto_train()


# Load models 

classifier = load_artifact(ARTIFACT_CLASSIFIER)
regressor  = load_artifact(ARTIFACT_REGRESSOR)

app = FastAPI(
    title="Student Placement Prediction API",
    description="Predicts student placement status (classification) and salary package (regression).",
    version="1.0.0",
)


# Input schema 

class StudentFeatures(BaseModel):
    gender:                     str   = "Male"
    ssc_percentage:             int   = 70
    hsc_percentage:             int   = 72
    degree_percentage:          int   = 72
    cgpa:                       float = 7.5
    entrance_exam_score:        int   = 65
    technical_skill_score:      int   = 65
    soft_skill_score:           int   = 65
    internship_count:           int   = 1
    live_projects:              int   = 1
    work_experience_months:     int   = 6
    certifications:             int   = 2
    attendance_percentage:      int   = 85
    backlogs:                   int   = 0
    extracurricular_activities: str   = "Yes"


# Endpoints 

@app.get("/")
def root():
    return {"message": "Welcome to the Student Placement Prediction API"}


@app.post("/predict/placement")
def predict_placement(student: StudentFeatures):
    """
    Predict whether a student will be placed (classification).
    Returns placed (bool) and probability.
    """
    df     = pd.DataFrame([student.model_dump()])
    pred   = int(classifier.predict(df)[0])
    prob   = float(classifier.predict_proba(df)[0][1])

    return {
        "placed":      bool(pred),
        "probability": round(prob, 4),
        "message":     "Student WILL be placed" if pred else "Student will NOT be placed",
    }


@app.post("/predict/salary")
def predict_salary(student: StudentFeatures):
    """
    Predict expected salary package in LPA (regression).
    Best called when student is predicted to be placed.
    """
    df     = pd.DataFrame([student.model_dump()])
    salary = float(regressor.predict(df)[0])
    salary = max(0.0, round(salary, 2))

    return {
        "salary_lpa": salary,
        "message":    f"Predicted salary package: {salary:.2f} LPA",
    }


@app.post("/predict/full")
def predict_full(student: StudentFeatures):
    """
    Combined endpoint: returns both placement prediction and salary estimate.
    """
    df       = pd.DataFrame([student.model_dump()])
    placed   = bool(classifier.predict(df)[0])
    prob     = float(classifier.predict_proba(df)[0][1])
    salary   = float(regressor.predict(df)[0]) if placed else 0.0
    salary   = max(0.0, round(salary, 2))

    return {
        "placed":      placed,
        "probability": round(prob, 4),
        "salary_lpa":  salary,
        "summary":     (
            f"Student WILL be placed with estimated salary {salary:.2f} LPA"
            if placed else
            "Student will NOT be placed"
        ),
    }
