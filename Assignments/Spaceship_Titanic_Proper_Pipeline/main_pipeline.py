import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import ACCURACY_THRESHOLD, NUM_FEATURES, CAT_FEATURES
from src.data.loader import ingest_data, load_frame, split_features_target, split_train_test
from src.pipelines.sklearn_pipeline import build_spaceship_pipeline
from src.models.train import train_pipeline
from src.models.evaluate import evaluate


def main():
    print("=" * 50)
    print("Spaceship Titanic – sklearn Pipeline")
    print("=" * 50)

    print("\nStep 1: Data Ingestion")
    ingest_data()

    print("\nStep 2: Load and Split")
    df = load_frame()
    X, y = split_features_target(df)
    x_train, x_test, y_train, y_test = split_train_test(X, y)

    print("\nStep 3: Build and Train Pipeline")
    pipeline = build_spaceship_pipeline(NUM_FEATURES, CAT_FEATURES)
    run_id = train_pipeline(pipeline, x_train, y_train)

    print("\nStep 4: Evaluation")
    accuracy, precision, recall = evaluate(x_test, y_test, run_id)

    print("\n" + "=" * 50)
    if accuracy >= ACCURACY_THRESHOLD:
        print(f"Model APPROVED (accuracy={accuracy:.3f})")
    else:
        print(f"Model REJECTED (accuracy={accuracy:.3f} < threshold={ACCURACY_THRESHOLD})")


if __name__ == "__main__":
    main()
