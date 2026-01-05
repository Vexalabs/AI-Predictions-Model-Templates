from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="Soccer Match Prediction Model API",
    description="A sample API for soccer match outcome predictions.",
    version="1.0.0",
)

class Match(BaseModel):
    home_team: str
    away_team: str
    league: str

class PredictionRequest(BaseModel):
    matches: List[Match]

class Prediction(BaseModel):
    match: Match
    outcome: str # e.g., "HOME_WIN", "AWAY_WIN", "DRAW"
    probability: float

class PredictionResponse(BaseModel):
    predictions: List[Prediction]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    A dummy prediction endpoint for soccer matches.
    """
    predictions = []
    for match in request.matches:
        # Dummy logic
        if len(match.home_team) > len(match.away_team):
            outcome = "HOME_WIN"
            prob = 0.6
        elif len(match.away_team) > len(match.home_team):
            outcome = "AWAY_WIN"
            prob = 0.5
        else:
            outcome = "DRAW"
            prob = 0.3

        predictions.append(Prediction(
            match=match,
            outcome=outcome,
            probability=prob
        ))

    return PredictionResponse(predictions=predictions)

@app.get("/")
async def root():
    return {"message": "Welcome to the Soccer Match Prediction Model API!"}
