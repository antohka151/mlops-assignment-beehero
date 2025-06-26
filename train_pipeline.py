#!/usr/bin/env python3
"""
Training pipeline entry point for the Colony Strength Classifier.
This script orchestrates the entire training process from data loading to model evaluation.
"""
import argparse
import mlflow
from datetime import datetime
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

from src.config.schema import PipelineConfig
from src.data.loader.component import DataLoader
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

def parse_args():
    # ... (no changes needed here)
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Colony Strength Classifier")
    parser.add_argument(
        "--config", 
        type=str, 
        default="src/config/config.yaml",
        help="Path to the pipeline configuration YAML file"
    )
    parser.add_argument(
        "--tracking-uri", 
        type=str, 
        help="MLflow tracking URI (overrides config)"
    )
    return parser.parse_args()


def main():
    """Main entry point for the training pipeline."""
    args = parse_args()
    config = PipelineConfig.from_yaml(args.config)
    if args.tracking_uri:
        config.mlflow.tracking_uri = args.tracking_uri

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.mlflow.run_name.replace("{timestamp}", timestamp)

    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"Started MLflow run: {run.info.run_id} ({run_name})")
        mlflow.log_dict(config.model_dump(), "config.json")

        # --- Step 1: Data Loading and Splitting ---
        logger.info("--- Loading and Splitting Data ---")
        data_loader = DataLoader(config.data_loader)
        raw_data = data_loader.load_data()
        X = raw_data.drop(columns=[config.training.target_column])
        y = raw_data[config.training.target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.training.test_size,
            random_state=config.training.random_state,
            stratify=y if config.training.stratify else None
        )
        logger.info(f"Data split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # --- Step 2: Train Model ---
        logger.info("--- Training Model ---")
        model = train_model(X_train, y_train, config)

        # --- Step 3: Log Metadata ---
        logger.info("--- Logging Metadata ---")
        try:
            preprocessor = model._sklearn_pipeline['preprocessor']
            metadata = preprocessor.get_metadata()
            mlflow.log_dict([m.model_dump() for m in metadata], "preprocessor_metadata.json")
        except Exception as e:
            logger.warning(f"Could not log preprocessor metadata: {e}")

        # --- Step 4: Evaluate Model ---
        logger.info("--- Evaluating Model ---")
        evaluation_results = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            metrics_to_compute=config.evaluation.metrics
        )
        logger.info(f"Evaluation results: {evaluation_results['metrics']}")
        mlflow.log_metrics(evaluation_results["metrics"])
        for name, content in evaluation_results["artifacts"].items():
            mlflow.log_dict(content, name)

        # --- Step 5: Log Final Model ---
        logger.info("--- Logging Final Model to MLflow ---")
        input_sample = X_test.head()
        signature = infer_signature(
            model_input=input_sample,
            model_output=model.predict(None, input_sample)
        )

        mlflow.pyfunc.log_model(
            name="model",
            python_model=model,
            code_paths=["src"],
            registered_model_name=config.mlflow.registered_model_name,
            signature=signature,
            input_example=input_sample
        )

        logger.info(f"Pipeline run completed successfully. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise