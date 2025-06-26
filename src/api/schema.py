# src/api/schema.py
"""
Defines request and response schemas for the API.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd


class PredictionRequest(BaseModel):
    """
    Schema for prediction request data.
    
    The API expects a list of records (instances), where each record is a 
    dictionary of feature values for a single sample.
    """
    instances: List[Dict[str, Any]] = Field(
        ..., 
        description="List of records to predict"
    )

    class Config:
        schema_extra = {
            "example": {
                "instances": [
                    {
                        "temperature": 24.5,
                        "humidity": 65.0,
                        "light_intensity": 800,
                        "vibration": 0.015,
                        "sound_frequency": 250
                    }
                ]
            }
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the request data to a pandas DataFrame for model input."""
        return pd.DataFrame(self.instances)


class PredictionResponse(BaseModel):
    """
    Schema for prediction response data.
    
    The API returns predictions along with the input record ID, 
    which helps clients match predictions with their input data.
    """
    predictions: List[str] = Field(
        ..., 
        description="Predicted colony strength categories"
    )
    record_ids: List[int] = Field(
        ..., 
        description="Record identifiers from the original input, preserving order"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": ["Strong"],
                "record_ids": [0]
            }
        }
