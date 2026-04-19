import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import ACCURACY_THRESHOLD
from src.data.loader import ingest_data, get_classification_data, get_regression_data
from src.pipelines.sklearn_pipeline import build_classifier_pipeline, build_regressor_pipeline
from src.models.train import train_classifier, train_regressor
from src.models.evaluate import evaluate_classifier, evaluate_regressor


def main():
    print("=" * 55)
    print("  Student Placement – ML Pipeline")
    print("=" * 55)

    # Step 1 
    print("\nStep 1: Data Ingestion")
    ingest_data()

    # Step 2 & 3: Classification 
    print("\nStep 2: Classification – Placement Status")
    x_train_c, x_test_c, y_train_c, y_test_c = get_classification_data()
    clf_pipeline = build_classifier_pipeline()
    run_id_c = train_classifier(clf_pipeline, x_train_c, y_train_c)

    print("\nStep 3: Evaluate Classifier")
    acc, prec, rec = evaluate_classifier(x_test_c, y_test_c, run_id_c)

    # Step 4 & 5: Regression 
    print("\nStep 4: Regression – Salary Prediction (placed students only)")
    x_train_r, x_test_r, y_train_r, y_test_r = get_regression_data()
    reg_pipeline = build_regressor_pipeline()
    run_id_r = train_regressor(reg_pipeline, x_train_r, y_train_r)

    print("\nStep 5: Evaluate Regressor")
    mae, rmse, r2 = evaluate_regressor(x_test_r, y_test_r, run_id_r)

    # Summary 
    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  Classifier → Accuracy: {acc:.3f}  {'APPROVED' if acc >= ACCURACY_THRESHOLD else 'REJECTED'}")
    print(f"  Regressor  → R²: {r2:.3f}  |  MAE: {mae:.3f} LPA  |  RMSE: {rmse:.3f} LPA")
    print("=" * 55)


if __name__ == "__main__":
    main()
