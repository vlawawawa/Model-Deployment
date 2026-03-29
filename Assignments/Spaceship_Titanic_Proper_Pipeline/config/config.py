from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_RAW_DIR  = BASE_DIR / "data" / "raw"
DATA_ING_DIR  = BASE_DIR / "data" / "ingested"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

ARTIFACT_PIPELINE = ARTIFACTS_DIR / "spaceship_pipeline.pkl"

# ── Columns ────────────────────────────────────────────────────────────────
TARGET_COL = "Transported"
DROP_COLS  = ["PassengerId", "Cabin", "Name"]

NUM_FEATURES = [
    "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "Cabin_num", "Group_size", "Solo", "Family_size",
    "TotalSpending", "HasSpending", "NoSpending",
    "Age_missing", "CryoSleep_missing",
    "RoomService_ratio", "FoodCourt_ratio", "ShoppingMall_ratio",
    "Spa_ratio", "VRDeck_ratio",
]

CAT_FEATURES = [
    "HomePlanet", "CryoSleep", "Destination", "VIP",
    "Deck", "Side", "Age_group",
]

SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# ── Hyperparameters ────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2

LR_C       = 10
LR_SOLVER  = "liblinear"
LR_MAX_ITER = 1000

# ── MLflow ─────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
MLFLOW_EXP_PIPELINE = "Spaceship Titanic"

# ── Threshold ──────────────────────────────────────────────────────────────
ACCURACY_THRESHOLD = 0.75
