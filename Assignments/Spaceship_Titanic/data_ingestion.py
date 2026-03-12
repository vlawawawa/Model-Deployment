"""
Spaceship Titanic – Step 1: Data Ingestion
Reads raw train.csv and saves it to the ingested/ folder.
"""

from pathlib import Path
import pandas as pd

BASE_DIR     = Path(__file__).parent
INGESTED_DIR = BASE_DIR / "ingested"
INPUT_FILE   = BASE_DIR / "train.csv"
OUTPUT_FILE  = INGESTED_DIR / "train.csv"


def ingest_data():
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    assert not df.empty, "Dataset is empty"

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Data ingested: {INPUT_FILE} → {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum().to_string()}")


if __name__ == "__main__":
    ingest_data()