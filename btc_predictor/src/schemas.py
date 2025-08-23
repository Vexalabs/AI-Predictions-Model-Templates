from pydantic import BaseModel
from typing import Literal


class PredictionRequest(BaseModel):
    symbol: str = "BTC-USD"
    interval: str = "1d"


class HorizonPrediction(BaseModel):
    price_prediction: float
    price_up_down: Literal["up", "down"]
    percentage_change: float


class PredictionResponse(BaseModel):
    _4_hours: HorizonPrediction
    _24_hours: HorizonPrediction
    _2_days: HorizonPrediction
    _7_days: HorizonPrediction
