import tomllib
from pathlib import Path

# Load TOML 
BASE_DIR   = Path(__file__).resolve().parent.parent
_TOML_FILE = Path(__file__).resolve().parent / "config.toml"

with open(_TOML_FILE, "rb") as f:
    _cfg = tomllib.load(f)

# Paths (built from BASE_DIR + toml values) 
DATA_RAW_DIR  = BASE_DIR / "data" / "raw"
DATA_ING_DIR  = BASE_DIR / "data" / "ingested"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

ARTIFACT_CLASSIFIER = ARTIFACTS_DIR / _cfg["artifacts"]["classifier"]
ARTIFACT_REGRESSOR  = ARTIFACTS_DIR / _cfg["artifacts"]["regressor"]

# Data columns 
DROP_COLS    = _cfg["data"]["drop_cols"]
TARGET_CLASS = _cfg["data"]["target_class"]
TARGET_REG   = _cfg["data"]["target_reg"]
NUM_FEATURES = _cfg["data"]["num_features"]
CAT_FEATURES = _cfg["data"]["cat_features"]

# Training 
RANDOM_STATE       = _cfg["training"]["random_state"]
TEST_SIZE          = _cfg["training"]["test_size"]
ACCURACY_THRESHOLD = _cfg["training"]["accuracy_threshold"]

# Classifier hyperparameters 
RF_CRITERION    = _cfg["model"]["classifier"]["criterion"]
RF_MAX_DEPTH    = _cfg["model"]["classifier"]["max_depth"]
RF_N_ESTIMATORS = _cfg["model"]["classifier"]["n_estimators"]

# Regressor hyperparameters 
RF_REG_MAX_DEPTH    = _cfg["model"]["regressor"]["max_depth"]
RF_REG_N_ESTIMATORS = _cfg["model"]["regressor"]["n_estimators"]

# MLflow 
MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
MLFLOW_EXP_CLASS    = _cfg["mlflow"]["experiment_classifier"]
MLFLOW_EXP_REG      = _cfg["mlflow"]["experiment_regressor"]