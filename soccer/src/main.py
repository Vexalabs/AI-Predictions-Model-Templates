from fastapi import FastAPI
from .schemas import PredictionRequest, PredictionResponse
from .model import model

app = FastAPI(
    title="Soccer Match Prediction Model API",
    description="A sample API for soccer match outcome predictions.",
    version="1.0.0",
)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Prediction endpoint.
    Delegates logic to the SoccerModel class.
    """
    results = model.predict(request.matches)
    return PredictionResponse(predictions=results)

@app.get("/")
async def root():
    return {"message": "Welcome to the Soccer Match Prediction Model API!"}
