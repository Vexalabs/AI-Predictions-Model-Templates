from typing import Optional
from pydantic import BaseModel, Field


# These schemas are used to ensure the request and response formats for the API endpoints are well-defined.
# They help in validating the data structure and types for incoming requests and outgoing responses.


# Request Schema for /predict
class PredictionRequest(BaseModel):
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

# Response Schema for /predict
class PredictionResponse(BaseModel):
    predicted_close: float
