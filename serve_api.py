#!/usr/bin/env python3
"""
API server entry point for the Colony Strength Classifier.
This script starts the FastAPI server to serve model predictions.
"""
import argparse
import os
import uvicorn
from src.config.schema import MLflowConfig
from src.api.service import ModelService
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start Colony Strength Classifier API")
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="Host to bind the API server to"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind the API server to"
    )
    parser.add_argument(
        "--model-uri", 
        type=str, 
        help="URI to the MLflow model. If not provided, MODEL_URI environment variable is used."
    )
    parser.add_argument(
        "--tracking-uri", 
        type=str, 
        help="MLflow tracking URI. If not provided, MLFLOW_TRACKING_URI environment variable is used."
    )
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="Enable auto-reload during development"
    )
    return parser.parse_args()


def main():
    """Main entry point for the API server."""
    args = parse_args()
    
    # Set environment variables for the model service
    if args.model_uri:
        os.environ["MODEL_URI"] = args.model_uri
    
    if args.tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri
    
    # Log startup information
    logger.info(f"Starting API server on {args.host}:{args.port}")
    if args.model_uri:
        logger.info(f"Using model URI: {args.model_uri}")
    if args.tracking_uri:
        logger.info(f"Using MLflow tracking URI: {args.tracking_uri}")
    
    # Start the FastAPI server using uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"API server failed with error: {str(e)}", exc_info=True)
        raise
