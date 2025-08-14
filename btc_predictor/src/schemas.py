from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Raw model prediction (0 or 1)")
    prediction_label: str = Field(..., description="'Up' if BTC is predicted to rise, else 'Down'")
    message: str = Field(..., description="Human-readable prediction message")
