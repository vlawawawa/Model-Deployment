import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import (
    DATA_RAW_DIR, DATA_ING_DIR,
    TARGET_COL, DROP_COLS, SPENDING_COLS,
    RANDOM_STATE, TEST_SIZE,
)


# ── Ingestion ──────────────────────────────────────────────────────────────

def ingest_data() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_ING_DIR.mkdir(parents=True, exist_ok=True)

    raw_file = DATA_RAW_DIR / "train.csv"
    assert raw_file.exists(), f"train.csv not found at {raw_file}"

    df = pd.read_csv(raw_file)
    assert not df.empty, "Dataset is empty"

    out_file = DATA_ING_DIR / "train.csv"
    df.to_csv(out_file, index=False)
    print(f"Data ingested: {raw_file.name} → {out_file}")
    print(f"Shape: {df.shape}")


# ── Feature Engineering ────────────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Cabin → Deck, Cabin_num, Side
    df["Deck"]      = df["Cabin"].apply(lambda x: x.split("/")[0] if pd.notna(x) else None)
    df["Cabin_num"] = df["Cabin"].apply(lambda x: float(x.split("/")[1]) if pd.notna(x) else None)
    df["Side"]      = df["Cabin"].apply(lambda x: x.split("/")[2] if pd.notna(x) else None)

    # PassengerId → Group_size, Solo
    df["Group"]      = df["PassengerId"].apply(lambda x: x.split("_")[0])
    df["Group_size"] = df.groupby("Group")["Group"].transform("count")
    df["Solo"]       = (df["Group_size"] == 1).astype(int)

    # Name → Family_size
    df["LastName"]    = df["Name"].apply(lambda x: x.split()[-1] if pd.notna(x) else "Unknown")
    df["Family_size"] = df.groupby("LastName")["LastName"].transform("count")

    # Spending features
    df["TotalSpending"] = df[SPENDING_COLS].sum(axis=1)
    df["HasSpending"]   = (df["TotalSpending"] > 0).astype(int)
    df["NoSpending"]    = (df["TotalSpending"] == 0).astype(int)
    for col in SPENDING_COLS:
        df[f"{col}_ratio"] = df[col] / (df["TotalSpending"] + 1)

    # Age group
    df["Age_group"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 30, 50, 100],
        labels=["Child", "Teen", "Young_Adult", "Adult", "Senior"],
    ).astype(object)

    # Missing-value flags
    df["Age_missing"]       = df["Age"].isna().astype(int)
    df["CryoSleep_missing"] = df["CryoSleep"].isna().astype(int)

    # Cast bool columns to string for OrdinalEncoder
    for col in ["CryoSleep", "VIP"]:
        df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else None)

    return df


# ── Load + Split ───────────────────────────────────────────────────────────

def load_frame() -> pd.DataFrame:
    path = DATA_ING_DIR / "train.csv"
    df = pd.read_csv(path)
    return feature_engineering(df)


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
    y = df[TARGET_COL].astype(int)
    return X, y


def split_train_test(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
