"""Load and prepare the Spaceship Titanic dataset."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_TRAIN_PATH = "train.csv"

# Columns we drop because they are identifiers, not signal.
DROP_COLUMNS = ["PassengerId", "Name"]

# Columns produced by prepare_features and consumed by the model.
NUMERIC_FEATURES = [
    "Age",
    "CabinNum",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
]

CATEGORICAL_FEATURES = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "Deck",
    "Side",
]

FEATURE_NAMES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN = "Transported"
CLASS_NAMES = ["not_transported", "transported"]


RAW_INPUT_COLUMNS = [
    "HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP",
    "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
]


def split_cabin(df: pd.DataFrame) -> pd.DataFrame:
    """Split the Cabin string 'Deck/Num/Side' into three columns.

    Missing cabins propagate as NaN in all three resulting columns; the
    imputer in the pipeline fills them. Output dtypes are deliberately
    plain numpy (object + float64) - the newer pandas extension dtypes
    use pd.NA which sklearn's downstream validators cannot cast.
    """
    out = df.copy()
    # Convert to plain object so the string accessor never returns pd.NA.
    cabin = out["Cabin"].astype(object)
    parts = cabin.where(cabin.notna(), "").astype(str).str.split("/", expand=True)
    valid = cabin.notna()

    deck = parts[0] if 0 in parts.columns else pd.Series([""] * len(out), index=out.index)
    num = parts[1] if 1 in parts.columns else pd.Series([""] * len(out), index=out.index)
    side = parts[2] if 2 in parts.columns else pd.Series([""] * len(out), index=out.index)

    out["Deck"] = deck.where(valid, np.nan).astype(object)
    out["Side"] = side.where(valid, np.nan).astype(object)
    out["CabinNum"] = pd.to_numeric(num.where(valid, np.nan), errors="coerce").astype("float64")
    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a raw Spaceship Titanic frame into model-ready features.

    Used both at training time (after loading train.csv) and at inference
    time (after parsing the request body). Keeps the train/serve contract
    in one place so they cannot drift apart.

    Robust to partial payloads: any raw column missing from `df` is filled
    with NaN, then the pipeline's imputers handle it at scoring time.
    """
    df = df.copy()
    for col in RAW_INPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = split_cabin(df)
    out = df[FEATURE_NAMES].copy()
    # Normalise categoricals: bools / pandas StringArray / mixed types all
    # become plain Python strings with np.nan for missing. sklearn's
    # SimpleImputer chokes on pandas' pd.NA ("boolean value of NA is
    # ambiguous"), and the OneHotEncoder needs consistent labels across
    # the True/False values that may arrive as either bool or "True"/"False".
    for col in CATEGORICAL_FEATURES:
        s = out[col]
        s = s.astype(object)
        s = s.where(pd.notna(s), np.nan)
        mask = pd.notna(s)
        s.loc[mask] = s.loc[mask].astype(str)
        out[col] = s
    return out


def load_dataset(path: str = DEFAULT_TRAIN_PATH) -> pd.DataFrame:
    """Read the Kaggle train.csv file."""
    return pd.read_csv(path)


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split. Returns (X_train, X_test, y_train, y_test)."""
    X = prepare_features(df)
    y = df[TARGET_COLUMN].astype(int)  # bool -> 0/1
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
