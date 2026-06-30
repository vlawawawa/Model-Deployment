"""
preprocessing.py
OOP preprocessing for the Credit Score classification pipeline.

Classes
-------
DataCleaner          : sklearn-compatible transformer that fixes the dirty raw
                       columns (trailing underscores, placeholder junk, text
                       'X Years and Y Months', impossible-value outliers).
PreprocessorBuilder  : constructs the numeric/categorical ColumnTransformer.

The cleaner is a transformer (not done inline) so the *exact* same cleaning is
applied at training time and at inference time from the pickle. This is what
prevents train/serve skew.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Columns that are identifiers / leakage and must never reach the model.
ID_COLUMNS = ["ID", "Customer_ID", "Name", "SSN", "Month"]

# Columns stored as strings with trailing "_" garbage -> need numeric coercion.
NUMERIC_STRING_COLUMNS = [
    "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly",
    "Monthly_Balance",
]

# Plausible ranges; anything outside is injected noise -> set to NaN, then impute.
CLIP_RANGES = {
    "Age": (0, 100), "Annual_Income": (0, 300_000), "Num_Bank_Accounts": (0, 20),
    "Num_Credit_Card": (0, 20), "Interest_Rate": (0, 50), "Num_of_Loan": (0, 20),
    "Num_of_Delayed_Payment": (0, 50), "Num_Credit_Inquiries": (0, 50),
}

# Known placeholder / corrupt category tokens in this dataset.
PLACEHOLDER_TOKENS = ["_______", "!@9#%8", "_", "__10000__", "NM", "#F%$D@*&8"]

TARGET = "Credit_Score"


def _parse_credit_history_age(value) -> float:
    """'9 Years and 8 Months' -> 116.0 (total months)."""
    if pd.isna(value):
        return np.nan
    years = re.search(r"(\d+)\s*Year", str(value))
    months = re.search(r"(\d+)\s*Month", str(value))
    return (int(years.group(1)) if years else 0) * 12 + \
           (int(months.group(1)) if months else 0)


class DataCleaner(BaseEstimator, TransformerMixin):
    """Cleans raw credit data into a numeric/categorical-ready frame.

    Stateless w.r.t. fitted parameters (it only applies deterministic rules),
    but implemented as a transformer so it lives inside the pickle and runs
    identically in training and inference.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Text credit-history-age -> months.
        if "Credit_History_Age" in df.columns:
            df["Credit_History_Age"] = df["Credit_History_Age"].apply(
                _parse_credit_history_age)

        # 2. Strip trailing underscores, coerce to numeric.
        for col in NUMERIC_STRING_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace("_", "", regex=False),
                    errors="coerce")

        # 3. Clip impossible numeric values to NaN.
        for col, (lo, hi) in CLIP_RANGES.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

        # 4. Replace placeholder tokens in categoricals with NaN.
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].replace(PLACEHOLDER_TOKENS, np.nan)
        df = df.infer_objects(copy=False)

        # 5. Type_of_Loan is messy free text -> reduce to a binary "has detail".
        if "Type_of_Loan" in df.columns:
            df["Type_of_Loan"] = df["Type_of_Loan"].notna().astype(int)

        # 6. Drop identifier / leakage columns if present.
        df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns],
                     errors="ignore")
        return df


class PreprocessorBuilder:
    """Builds the imputation + scaling + one-hot ColumnTransformer.

    Column lists are inferred from a cleaned sample so the inference service
    does not need to hardcode them.
    """

    def __init__(self, numeric_cols: list[str], categorical_cols: list[str]):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

    @classmethod
    def from_cleaned_frame(cls, df_clean: pd.DataFrame) -> "PreprocessorBuilder":
        feats = df_clean.drop(columns=[TARGET], errors="ignore")
        numeric = feats.select_dtypes(include=np.number).columns.tolist()
        categorical = feats.select_dtypes(include="object").columns.tolist()
        return cls(numeric, categorical)

    def build(self) -> ColumnTransformer:
        numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        categorical_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        return ColumnTransformer([
            ("num", numeric_pipe, self.numeric_cols),
            ("cat", categorical_pipe, self.categorical_cols),
        ])
