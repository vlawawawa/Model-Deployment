"""
sagemaker_entry.py
SageMaker training entry point. SageMaker passes data and output dirs via
environment variables; this adapts our ModelTrainer to those paths.

Placed in aws_source/ alongside a copy of the src/ modules so the container
has everything it needs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import ModelTrainer  # the OOP trainer, copied into aws_source/


def main():
    # SageMaker conventions:
    input_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    data_path = os.path.join(input_dir, "data_C.csv")
    print(f"Training on {data_path}, writing model to {model_dir}")

    # MLflow logs locally inside the job; for a managed tracking server point
    # MLFLOW_TRACKING_URI at an MLflow server or SageMaker experiments.
    trainer = ModelTrainer(data_path=data_path, model_dir=model_dir)
    trainer.run()
    print("Training complete. Artifacts in", model_dir)


if __name__ == "__main__":
    main()
