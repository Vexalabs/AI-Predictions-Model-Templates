from pydantic import BaseModel
from typing import List

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
