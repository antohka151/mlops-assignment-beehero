# src/api/main.py
"""
FastAPI endpoint for model inference.
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import traceback

from src.api.service import ModelService
from src.api.schema import PredictionRequest, PredictionResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize the FastAPI app
app = FastAPI(
    title="Colony Strength Predictor API",
    description="API for predicting bee colony strength",
    version="0.1.0",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model service
model_service = ModelService()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
async def root():
    """Root endpoint that returns basic API information."""
    return {
        "message": "Colony Strength Predictor API",
        "version": "0.1.0",
        "endpoints": {
            "/predict": "Make predictions with the model",
            "/health": "Check API health"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions with the model.
    
    This endpoint accepts a list of feature records and returns predictions
    for each record along with their indices to maintain data alignment.
    """
    try:
        # Convert request data to DataFrame
        input_df = request.to_dataframe()
        
        if input_df.empty:
            raise HTTPException(status_code=400, detail="No instances provided")
            
        logger.info(f"Making predictions for {len(input_df)} instances")
        
        # Get predictions
        predictions = model_service.predict(input_df)
        
        # Create response with predictions and record IDs
        # Note: predictions may have fewer records than input if outliers were removed
        return PredictionResponse(
            predictions=predictions.tolist(),
            record_ids=predictions.index.tolist()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
