from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_read_root():
    """
    Verify the health check/root endpoint exists.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_contract():
    """
    Verify the /predict endpoint adheres to the Soccer Model Contract.
    """
    payload = {
        "matches": [
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "league": "Premier League"
            },
            {
                "home_team": "Real Madrid",
                "away_team": "Barcelona",
                "league": "La Liga"
            }
        ]
    }
    
    response = client.post("/predict", json=payload)
    
    # 1. Check Status Code
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    data = response.json()
    
    # 2. Check Output Schema Structure
    assert "predictions" in data, "Response missing 'predictions' list"
    assert isinstance(data["predictions"], list), "'predictions' must be a list"
    assert len(data["predictions"]) == 2, "Expected 2 predictions for 2 input matches"
    
    # 3. Check Individual Prediction Schema
    first_pred = data["predictions"][0]
    
    assert "match" in first_pred, "Prediction missing 'match' object"
    assert "outcome" in first_pred, "Prediction missing 'outcome' field"
    assert "probability" in first_pred, "Prediction missing 'probability' field"
    
    # 4. Check Data Types
    assert isinstance(first_pred["outcome"], str), "'outcome' must be a string"
    assert isinstance(first_pred["probability"], float), "'probability' must be a float"
    
    # 5. Check Logic (Basic sanity check)
    assert first_pred["match"]["home_team"] == "Arsenal"
    valid_outcomes = ["HOME_WIN", "AWAY_WIN", "DRAW"]
    assert first_pred["outcome"] in valid_outcomes, f"Outcome must be one of {valid_outcomes}"

def test_predict_empty_list():
    """
    Verify the API handles an empty list of matches correctly.
    """
    payload = {"matches": []}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predictions"] == []
