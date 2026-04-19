import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import (
    DATA_RAW_DIR, DATA_ING_DIR,
    DROP_COLS, TARGET_CLASS, TARGET_REG,
    RANDOM_STATE, TEST_SIZE,
)

DATA_ING_DIR.mkdir(parents=True, exist_ok=True)


def ingest_data() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_ING_DIR.mkdir(parents=True, exist_ok=True)

    raw_file = DATA_RAW_DIR / "B.csv"
    assert raw_file.exists(), f"B.csv not found at {raw_file}"

    df = pd.read_csv(raw_file)
    assert not df.empty, "Dataset is empty"

    out_file = DATA_ING_DIR / "B.csv"
    df.to_csv(out_file, index=False)
    print(f"Data ingested: {raw_file.name} → {out_file}")
    print(f"Shape: {df.shape}")
    print(f"Placement rate: {df[TARGET_CLASS].mean():.1%}")


def load_frame() -> pd.DataFrame:
    path = DATA_ING_DIR / "B.csv"
    return pd.read_csv(path)


# Classification split 

def get_classification_data():
    df = load_frame().drop(columns=DROP_COLS + [TARGET_REG])
    X  = df.drop(columns=[TARGET_CLASS])
    y  = df[TARGET_CLASS]
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y)


# Regression split (placed students only) 

def get_regression_data():
    df = load_frame()
    df = df[df[TARGET_CLASS] == 1].drop(columns=DROP_COLS + [TARGET_CLASS])
    X  = df.drop(columns=[TARGET_REG])
    y  = df[TARGET_REG]
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE)
