"""
sagemaker_train.py
Cloud training pipeline on AWS SageMaker.

This launches the SAME OOP training code (src/train.py) as a SageMaker training
job using the SKLearn framework container. The model artifact is written to S3.

Prerequisites
-------------
- aws cli configured (`aws configure`) with an IAM user/role that can use
  SageMaker and read/write the S3 bucket.
- A SageMaker execution role ARN (create one in IAM with
  AmazonSageMakerFullAccess + S3 access).
- boto3 + sagemaker installed locally:  pip install boto3 sagemaker

Run:
    python aws/sagemaker_train.py \
        --bucket my-credit-bucket \
        --role arn:aws:iam::<ACCOUNT_ID>:role/<SageMakerRole>
"""
import argparse
import os

import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--role", required=True, help="SageMaker execution role ARN")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--instance-type", default="ml.m5.large")
    args = p.parse_args()

    boto3.setup_default_session(region_name=args.region)
    sess = sagemaker.Session()

    # 1. Upload the raw dataset to S3.
    data_s3 = sess.upload_data(
        path="data_C.csv",
        bucket=args.bucket,
        key_prefix="credit-score/input")
    print("Uploaded data to", data_s3)

    # 2. Define the training job. entry_point reuses our training logic; the
    #    SKLearn container installs requirements.txt from source_dir.
    estimator = SKLearn(
        entry_point="sagemaker_entry.py",
        source_dir="aws_source",          # contains entry + src + requirements
        role=args.role,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version="1.2-1",
        base_job_name="credit-score",
        hyperparameters={},
        output_path=f"s3://{args.bucket}/credit-score/model",
    )

    # 3. Launch. SageMaker spins up the instance, runs training, ships the
    #    model.tar.gz (model.pkl + label_encoder.pkl + metadata.json) to S3.
    estimator.fit({"train": data_s3})
    print("Model artifact:", estimator.model_data)


if __name__ == "__main__":
    main()
