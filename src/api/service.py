# src/api/service.py
"""
Model prediction service that loads models from MLflow and serves predictions.
"""
import os
import mlflow
from typing import Optional
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelService:
    """
    Service for loading and serving predictions from MLflow models.
    
    This class handles model loading and inference, allowing the API endpoints
    to remain lean and focused on request/response handling.
    """
    def __init__(
        self, 
        model_uri: Optional[str] = None,
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize the model service.
        
        Args:
            model_uri: URI to the MLflow model. If None, will use environment variable.
            tracking_uri: URI to the MLflow tracking server. If None, will use environment variable.
        """
        self.model = None
        self.model_uri = model_uri or os.environ.get("MODEL_URI")
        tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"Set MLflow tracking URI to {tracking_uri}")
        
        if self.model_uri:
            self._load_model()
        else:
            logger.warning("No model URI provided. Model needs to be loaded before predictions.")
    
    def _load_model(self):
        """Load the model from MLflow."""
        try:
            logger.info(f"Loading model from {self.model_uri}")
            self.model = mlflow.pyfunc.load_model(self.model_uri)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate predictions for the input data.
        
        Args:
            data: DataFrame with feature columns required by the model
            
        Returns:
            A pandas Series containing predictions
        """
        if self.model is None:
            if not self.model_uri:
                raise ValueError("No model URI provided. Cannot load model.")
            self._load_model()
        
        try:
            # The model's predict method returns a Series with the index preserved
            return self.model.predict(data)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
