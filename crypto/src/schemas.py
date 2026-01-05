from pydantic import BaseModel

class PredictionRequest(BaseModel):
    ticker: str
    days: int

class PredictionResponse(BaseModel):
    ticker: str
    prediction: float
    confidence: float
