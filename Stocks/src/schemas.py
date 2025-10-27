from typing import Optional
from pydantic import BaseModel, Field


# These schemas are used to ensure the request and response formats for the API endpoints are well-defined.
# They help in validating the data structure and types for incoming requests and outgoing responses.


# Request Schema for /predict
class PredictionRequest(BaseModel):
    name: str = Field(..., description="Name of the stock")
    date: str = Field(..., description="Date of the prediction in YYYY-MM-DD format")
    current_price: float = Field(..., description="Current price of the stock")


# Lets correct the schema for the prediction response.
# here is a sample response structure based on the provided code snippet:
# sample_response = {
#   "prediction_timestamp":"2025-08-12T07:41:42.236533Z",
#   "predictions":[
#     {
#       "name":"Apple GPT", # Name your name
#       "description":"Stock price forecast with 7-day horizon.", # Describe your model
#       "horizon":7,
#       "frequency":1,
#       "stock_name":"S&P 500", # Which index/stock are you predicting?
#       "forecasts":[
#       {"timestamp":"2025-08-11T00:00:00","forecast_index":1,"price":637.0087458737507,"pct_change":-0.03,"direction":"DOWN"},{"timestamp":"2025-08-12T00:00:00","forecast_index":2,"price":628.8247751970202,"pct_change":-1.31,"direction":"DOWN"},{"timestamp":"2025-08-13T00:00:00","forecast_index":3,"price":640.4971446347408,"pct_change":0.52,"direction":"UP"},{"timestamp":"2025-08-14T00:00:00","forecast_index":4,"price":638.1062217748986,"pct_change":0.15,"direction":"UP"},{"timestamp":"2025-08-15T00:00:00","forecast_index":5,"price":640.8362960967352,"pct_change":0.57,"direction":"UP"},{"timestamp":"2025-08-18T00:00:00","forecast_index":6,"price":639.421582551462,"pct_change":0.35,"direction":"UP"},{"timestamp":"2025-08-19T00:00:00","forecast_index":7,"price":642.4778955136717,"pct_change":0.83,"direction":"UP"}
#       ]
#     }
#   ]
# }


class Forecast(BaseModel):
    timestamp: str = Field(..., description="Timestamp of the forecast")
    forecast_index: int = Field(..., description="Index of the forecast")
    price: float = Field(..., description="Predicted stock price")
    pct_change: float = Field(
        ..., description="Percentage change from the previous price"
    )
    direction: str = Field(..., description="Direction of the price change (UP/DOWN)")


class Prediction(BaseModel):
    name: str = Field(..., description="Name of the model")
    description: str = Field(..., description="Description of the model")
    horizon: int = Field(..., description="Forecast horizon in days")
    frequency: int = Field(..., description="Frequency of predictions")
    stock_name: str = Field(
        ..., description="Name of the stock or index being predicted"
    )
    forecasts: list[Forecast] = Field(
        ..., description="List of forecasted prices with timestamps and changes"
    )


class PredictionResponse(BaseModel):
    prediction_timestamp: str = Field(..., description="Timestamp of the prediction")
    predictions: list[Prediction] = Field(
        ..., description="List of predictions with details"
    )
