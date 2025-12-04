from fastapi import FastAPI
from .schemas import PredictionRequest, PredictionResponse
from .model import model

app = FastAPI(
    title="Crypto Prediction Model API",
    description="A sample API for crypto price predictions.",
    version="1.0.0",
)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Prediction endpoint.
    Delegates logic to the CryptoModel class.
    """
    return model.predict(request)

@app.get("/")
async def root():
    return {"message": "Welcome to the Crypto Prediction Model API!"}
