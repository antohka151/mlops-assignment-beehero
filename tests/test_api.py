# tests/test_api.py

import json
import pytest
from fastapi.testclient import TestClient
import pandas as pd
from unittest.mock import patch, MagicMock

from src.api.main import app
from src.api.schema import PredictionRequest


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as client:
        yield client


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "endpoints" in response.json()


@patch("src.api.main.model_service")
def test_predict_endpoint_success(mock_model_service, client):
    """Test the prediction endpoint with a successful prediction."""
    # Mock the model service to return predictions
    mock_predictions = pd.Series(["Strong", "Weak"], index=[0, 1])
    mock_model_service.predict.return_value = mock_predictions
    
    # Create a prediction request
    request_data = {
        "instances": [
            {
                "temperature": 24.5,
                "humidity": 65.0,
                "light_intensity": 800,
                "vibration": 0.015,
                "sound_frequency": 250
            },
            {
                "temperature": 20.1,
                "humidity": 70.5,
                "light_intensity": 600,
                "vibration": 0.025,
                "sound_frequency": 300
            }
        ]
    }
    
    # Make the prediction request
    response = client.post("/predict", json=request_data)
    
    # Check the response
    assert response.status_code == 200
    response_data = response.json()
    assert "predictions" in response_data
    assert "record_ids" in response_data
    assert response_data["predictions"] == ["Strong", "Weak"]
    assert response_data["record_ids"] == [0, 1]
    
    # Verify the model service was called correctly
    mock_model_service.predict.assert_called_once()
    # We can't directly compare DataFrames since they're different objects,
    # but we can verify the shape
    args, _ = mock_model_service.predict.call_args
    assert len(args[0]) == 2  # 2 rows in the DataFrame


@patch("src.api.main.model_service")
def test_predict_endpoint_model_error(mock_model_service, client):
    """Test the prediction endpoint when the model fails."""
    # Mock the model service to raise an exception
    mock_model_service.predict.side_effect = Exception("Model prediction failed")
    
    # Create a prediction request
    request_data = {
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
    
    # Make the prediction request
    response = client.post("/predict", json=request_data)
    
    # Check the response (should be a 500 Internal Server Error)
    assert response.status_code == 500
    assert "Prediction failed" in response.json()["detail"]
