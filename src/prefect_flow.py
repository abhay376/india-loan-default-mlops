"""
Prefect Retraining Flow
========================
Automates the full retraining pipeline (preprocess → train) using Prefect.
Can be run manually or scheduled as a Prefect deployment.

Usage:
    python src/prefect_flow.py
"""

import os
import sys
from datetime import datetime

from prefect import flow, task, get_run_logger

# Add project root to path so imports work from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import preprocess
from src.train_model import train


@task(name="preprocess_data", retries=2, retry_delay_seconds=30)
def preprocess_task():
    """Task: Run the full preprocessing pipeline."""
    logger = get_run_logger()
    logger.info("Starting data preprocessing...")
    X, y, feature_names = preprocess()
    logger.info(
        f"Preprocessing complete — {X.shape[0]} samples, "
        f"{X.shape[1]} features"
    )
    return X, y, feature_names


@task(name="train_models", retries=1, retry_delay_seconds=60)
def train_task():
    """Task: Train all models, log to MLflow, save the best."""
    logger = get_run_logger()
    logger.info("Starting model training (3 models with MLflow)...")
    best_model, best_name, results_df = train()
    logger.info(f"Training complete — Best model: {best_name}")
    return best_model, best_name, results_df


@task(name="log_summary")
def log_summary_task(best_name: str, results_df):
    """Task: Print a summary of the retraining run."""
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info("  RETRAINING PIPELINE SUMMARY")
    logger.info(f"  Timestamp   : {datetime.now().isoformat()}")
    logger.info(f"  Best Model  : {best_name}")
    logger.info("=" * 60)
    logger.info(f"\n{results_df.to_string(index=False)}")


@flow(name="loan_default_retraining", log_prints=True)
def retraining_flow():
    """
    End-to-end retraining flow:
      1. Preprocess raw CSV data
      2. Train GradientBoosting, RandomForest & XGBoost
      3. Log results and save the best model
    """
    # Step 1 — Preprocess
    X, y, feature_names = preprocess_task()

    # Step 2 — Train
    best_model, best_name, results_df = train_task()

    # Step 3 — Summary
    log_summary_task(best_name, results_df)

    print(f"\n✅ Retraining pipeline completed successfully!")
    return best_model


# ──────────────────────────────────────────────
if __name__ == "__main__":
    retraining_flow()
