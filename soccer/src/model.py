from typing import List
from .schemas import Match, Prediction

class SoccerModel:
    """
    Sample soccer prediction model logic.
    Replace this class with your actual ML model loading and inference code.
    """
    
    def predict(self, matches: List[Match]) -> List[Prediction]:
        predictions = []
        for match in matches:
            # Dummy logic - Replace with real inference
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
        return predictions

# Singleton instance for easy import
model = SoccerModel()
