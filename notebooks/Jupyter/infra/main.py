from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from datetime import datetime, timedelta
from model.model import CryptoModel
import pandas as pd

app = FastAPI()

# --- Pydantic Models ---
class CryptoData(BaseModel):
    coin_name: str
    volume: float
    daily_high: float
    daily_low: float
    open: float
    close: float

class Forecast(BaseModel):
    timestamp: datetime
    forecast_index: int
    price: float
    pct_change: float
    direction: str

class PredictionResponse(BaseModel):
    name: str
    description: str
    horizon: int
    frequency: int
    coin_name: str
    current_price: float
    forecasts: List[Forecast]

# --- Model Training ---
# Initialize and train the model with some dummy time-series data
model = CryptoModel()
data = {'day': range(10), 'price': [100 + i*2 for i in range(10)]}
df = pd.DataFrame(data)
model.train(df)

# --- API Endpoint ---
@app.post("/predict", response_model=PredictionResponse)
def predict(data: CryptoData):
    """
    Predicts the cryptocurrency price for the next 7 days.
    """
    # Use the model to get predictions
    predictions = model.predict(current_price=data.close, days=7)

    # Format the response
    forecasts = []
    for i, price in enumerate(predictions):
        if data.close > 0:
            pct_change = ((price - data.close) / data.close) * 100
        else:
            pct_change = 0
        direction = "UP" if price > data.close else "DOWN"
        forecasts.append(
            Forecast(
                timestamp=datetime.now() + timedelta(days=i),
                forecast_index=i,
                price=price,
                pct_change=pct_change,
                direction=direction,
            )
        )

    return PredictionResponse(
        name="Crypto Price Predictor",
        description="A simple model to predict cryptocurrency prices.",
        horizon=7,
        frequency=1,
        coin_name=data.coin_name,
        current_price=data.close,
        forecasts=forecasts,
    )
