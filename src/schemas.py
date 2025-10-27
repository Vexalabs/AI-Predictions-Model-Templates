"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, date

class PredictionRequest(BaseModel):
    """Request model for stock prediction"""
    name: str = Field(..., description="Stock symbol (e.g., AAPL, MSFT)")
    date: Optional[str] = Field(None, description="Date for prediction (YYYY-MM-DD format). If not provided, uses current date")
    current_price: Optional[float] = Field(None, description="Current stock price. If not provided, fetches automatically")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "AAPL",
                "date": "2025-04-10",
                "current_price": 202.38
            }
        }

class Forecast(BaseModel):
    """Individual forecast data point"""
    timestamp: str = Field(..., description="Forecast timestamp")
    forecast_index: int = Field(..., description="Forecast sequence number")
    price: float = Field(..., description="Predicted price")
    pct_change: float = Field(..., description="Percentage change from current price")
    direction: str = Field(..., description="Price direction (UP/DOWN)")

class Prediction(BaseModel):
    """Prediction model information"""
    name: str = Field(..., description="Model name")
    description: str = Field(..., description="Model description")
    horizon: int = Field(..., description="Prediction horizon in days")
    frequency: int = Field(..., description="Prediction frequency")
    stock_name: str = Field(..., description="Stock symbol")
    forecasts: List[Forecast] = Field(..., description="List of forecasts")

class PredictionResponse(BaseModel):
    """Response model for stock prediction"""
    prediction_timestamp: str = Field(..., description="When prediction was made")
    predictions: List[Prediction] = Field(..., description="List of predictions")
    market_context: Optional[dict] = Field(None, description="Market context information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction_timestamp": "2025-08-12T07:41:42.236533Z",
                "predictions": [
                    {
                        "name": "S&P 500 Prophet Model",
                        "description": "Stock price forecast with 1-day horizon using Facebook Prophet",
                        "horizon": 1,
                        "frequency": 1,
                        "stock_name": "AAPL",
                        "forecasts": [
                            {
                                "timestamp": "2025-08-11T00:00:00",
                                "forecast_index": 1,
                                "price": 637.01,
                                "pct_change": -0.03,
                                "direction": "DOWN"
                            }
                        ]
                    }
                ],
                "market_context": {
                    "current_price": 202.38,
                    "market_status": "CLOSED",
                    "prediction_type": "next trading day's close",
                    "target_date": "2025-10-07"
                }
            }
        }

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    last_trained_at: Optional[str] = Field(None, description="Last training timestamp")
    timestamp: str = Field(..., description="Current timestamp")
