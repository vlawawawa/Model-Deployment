from pathlib import Path

# Paths 
BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_RAW_DIR  = BASE_DIR / "data" / "raw"
DATA_ING_DIR  = BASE_DIR / "data" / "ingested"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

ARTIFACT_CLASSIFIER = ARTIFACTS_DIR / "placement_classifier.pkl"
ARTIFACT_REGRESSOR  = ARTIFACTS_DIR / "salary_regressor.pkl"

# Columns 
DROP_COLS          = ["student_id"]
TARGET_CLASS       = "placement_status"
TARGET_REG         = "salary_package_lpa"

NUM_FEATURES = [
    "ssc_percentage", "hsc_percentage", "degree_percentage", "cgpa",
    "entrance_exam_score", "technical_skill_score", "soft_skill_score",
    "internship_count", "live_projects", "work_experience_months",
    "certifications", "attendance_percentage", "backlogs",
]
CAT_FEATURES = ["gender", "extracurricular_activities"]

# Hyperparameters 
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# Classifier – Random Forest
RF_CRITERION = "gini"
RF_MAX_DEPTH = 8
RF_N_ESTIMATORS = 100

# Regressor – Random Forest
RF_REG_MAX_DEPTH    = 8
RF_REG_N_ESTIMATORS = 100

# MLflow 
MLFLOW_TRACKING_URI  = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
MLFLOW_EXP_CLASS     = "Student Placement Classification"
MLFLOW_EXP_REG       = "Student Salary Regression"

# Threshold 
ACCURACY_THRESHOLD = 0.75
