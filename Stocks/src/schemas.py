from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field



class TimeFramePrediction(BaseModel):
    price_prediction: float = Field(..., description="Predicted stock price")
    price_up_down: str = Field(..., description="Direction of price movement: 'up' or 'down'")
    percentage_change: float = Field(..., description="Expected percentage change in price")


# Request Schema for /predict
class PredictionRequest(BaseModel):
    name: str = Field(..., description="Name of the stock (e.g., TSLA)")
    date: datetime = Field(default_factory=datetime.now, description="Timestamp of prediction")
    open: float = Field(..., description="most recent open price of the stock")
    high: float = Field(..., description="most recent close price of the stock")
    low: float = Field(..., description="most recent high price of the stock")
    close: float = Field(..., description="most recent close price of the stock")
    volume: float = Field(..., description="most resent volume of the stock")

# Response Schema for /predict
class PredictionResponse(BaseModel):
    four_hours: TimeFramePrediction = Field(
        ..., description="Predicted price and change for the next 4 hours"
    )
    twenty_four_hours: TimeFramePrediction = Field(
        ..., description="Predicted price and change for the next 24 hours"
    )
    two_days: TimeFramePrediction = Field(
        ..., description="Predicted price and change for the next 2 days"
    )
    seven_days: TimeFramePrediction = Field(
        ..., description="Predicted price and change for the next 7 days"
    )
