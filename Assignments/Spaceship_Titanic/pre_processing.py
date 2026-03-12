"""
Spaceship Titanic – Step 2: Preprocessing
Reads ingested data, engineers features, splits, encodes, and saves the preprocessor artifact.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

CATEGORICAL_COLS = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Age_group"]

NUMERICAL_COLS = [
    "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "Cabin_num", "Group_size", "Solo", "Family_size", "TotalSpending",
    "HasSpending", "NoSpending", "Age_missing", "CryoSleep_missing",
    "RoomService_ratio", "FoodCourt_ratio", "ShoppingMall_ratio",
    "Spa_ratio", "VRDeck_ratio",
]


def feature_engineering(df):
    df = df.copy()

    # Cabin → Deck, Cabin_num, Side
    df["Deck"]      = df["Cabin"].apply(lambda x: x.split("/")[0] if pd.notna(x) else "Unknown")
    df["Cabin_num"] = df["Cabin"].apply(lambda x: float(x.split("/")[1]) if pd.notna(x) else -1.0)
    df["Side"]      = df["Cabin"].apply(lambda x: x.split("/")[2] if pd.notna(x) else "Unknown")

    # PassengerId → Group, Group_size, Solo
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
        df["Age"], bins=[0, 12, 18, 30, 50, 100],
        labels=["Child", "Teen", "Young_Adult", "Adult", "Senior"]
    ).astype(str)

    # Missing-value indicators
    df["Age_missing"]       = df["Age"].isna().astype(int)
    df["CryoSleep_missing"] = df["CryoSleep"].isna().astype(int)

    return df


def preprocess():
    os.makedirs("artifacts", exist_ok=True)

    df = pd.read_csv("ingested/train.csv")
    df = feature_engineering(df)

    # Fill missing values
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("Unknown")

    num_medians = {col: df[col].median() for col in NUMERICAL_COLS if col in df.columns}
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(num_medians[col])

    # Encode categoricals
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    feature_columns = CATEGORICAL_COLS + NUMERICAL_COLS
    X = df[feature_columns]
    y = df["Transported"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save preprocessor artifacts
    preprocessor = {
        "label_encoders":  label_encoders,
        "num_medians":     num_medians,
        "feature_columns": feature_columns,
    }
    joblib.dump(preprocessor, "artifacts/preprocessor.pkl")

    train_set = X_train.copy()
    train_set["Transported"] = y_train.values

    test_set = X_test.copy()
    test_set["Transported"] = y_test.values

    print(f"Preprocessing done. Train: {X_train.shape}, Test: {X_test.shape}")
    print("Preprocessor saved to artifacts/preprocessor.pkl")
    return train_set, test_set


if __name__ == "__main__":
    preprocess()