from typing import Optional
from pydantic import BaseModel, Field


# These schemas are used to ensure the request and response formats for the API endpoints are well-defined.
# They help in validating the data structure and types for incoming requests and outgoing responses.


# Request Schema for /predict
class PredictionRequest(BaseModel):
    home_team: str = Field(..., description="Name of the home team")
    away_team: str = Field(..., description="Name of the away team")
    home_team_odds_avg: float = Field(
        ..., description="Average odds for the home team to win"
    )
    away_team_odds_avg: float = Field(
        ..., description="Average odds for the away team to win"
    )


# Response Schema for /predict
class PredictionResponse(BaseModel):
    winner: Optional[str] = Field(
        None, description="Predicted winning team (can be null)"
    )
    winner_confidence_pct: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Confidence percentage for the winner prediction (can be null)",
    )
    winner_best_bet_odds: Optional[float] = Field(
        None, description="Best available odds for the winning team (can be null)"
    )
    over_under: Optional[str] = Field(
        None,
        description="Predicted over/under outcome (e.g., 'Over 2.5', 'Under 2.5', can be null)",
    )
    over_under_confidence_pct: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Confidence percentage for the over/under prediction (can be null)",
    )
    over_under_best_bet_odds: Optional[float] = Field(
        None, description="Best available odds for the over/under bet (can be null)"
    )
    spread: Optional[str] = Field(
        None, description="Predicted spread outcome (e.g., 'Team A -1.5', can be null)"
    )
    spread_confidence_pct: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Confidence percentage for the spread prediction (can be null)",
    )
    spread_best_bet_odds: Optional[float] = Field(
        None, description="Best available odds for the spread bet (can be null)"
    )
