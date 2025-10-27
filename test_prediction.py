import requests
import json

# Test prediction - FIXED URL
url = "http://localhost:8000/api/v1/predict"  # Added /api/v1/
data = {
    "name": "AAPL",
    "date": "2025-04-10",
    "current_price": 202.38
}

try:
    print("Making prediction request to:", url)
    response = requests.post(url, json=data)
    print("Status Code:", response.status_code)
    print("Response:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print("Error:", e)
