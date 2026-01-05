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
    Verify the /predict endpoint adheres to the Crypto Model Contract.
    
    Contract:
    - Input: {"ticker": str, "days": int}
    - Output: {"ticker": str, "prediction": float, "confidence": float}
    """
    payload = {
        "ticker": "BTC",
        "days": 7
    }
    
    response = client.post("/predict", json=payload)
    
    # 1. Check Status Code
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    data = response.json()
    
    # 2. Check Output Schema Fields
    assert "ticker" in data, "Response missing 'ticker' field"
    assert "prediction" in data, "Response missing 'prediction' field"
    assert "confidence" in data, "Response missing 'confidence' field"
    
    # 3. Check Data Types
    assert isinstance(data["ticker"], str), "'ticker' must be a string"
    assert isinstance(data["prediction"], float), "'prediction' must be a float"
    assert isinstance(data["confidence"], float), "'confidence' must be a float"
    
    # 4. Check Logic (Basic sanity check)
    assert data["ticker"] == "BTC"
    assert 0 <= data["confidence"] <= 1.0, "Confidence must be between 0 and 1"

def test_predict_invalid_input():
    """
    Verify the API handles invalid input correctly (422 Unprocessable Entity).
    """
    # Missing 'days'
    payload = {
        "ticker": "BTC"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
