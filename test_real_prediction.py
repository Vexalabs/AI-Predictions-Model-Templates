import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_market_status():
    """Test market status endpoint"""
    response = requests.get(f"{BASE_URL}/api/v1/market-status")
    print("Market Status:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_available_symbols():
    """Test symbols endpoint"""
    response = requests.get(f"{BASE_URL}/api/v1/symbols")
    data = response.json()
    print("Available Symbols:")
    print(f"Total: {data['total_count']} stocks")
    print("First 10 symbols:", data['symbols'][:10])
    print("Last updated:", data['last_updated'])
    print()

def test_prediction_with_current_date():
    """Test prediction using current date (no date specified)"""
    prediction_data = {
        "name": "AAPL",
        "current_price": 202.38
        # No date specified - will use current date
    }
    
    print("Testing prediction with current date...")
    print("Request data:", json.dumps(prediction_data, indent=2))
    
    response = requests.post(
        f"{BASE_URL}/api/v1/predict",
        headers={"Content-Type": "application/json"},
        json=prediction_data
    )
    
    print("Status Code:", response.status_code)
    if response.status_code == 200:
        result = response.json()
        print("Prediction Result:")
        print(json.dumps(result, indent=2))
        
        # Extract key information
        forecast = result['predictions'][0]['forecasts'][0]
        print(f"\n📊 PREDICTION SUMMARY:")
        print(f"Stock: {result['predictions'][0]['stock_name']}")
        print(f"Current Price: ${prediction_data['current_price']}")
        print(f"Predicted Price: ${forecast['price']}")
        print(f"Predicted Change: {forecast['pct_change']:+.2f}%")
        print(f"Direction: {forecast['direction']}")
        print(f"Prediction Date: {forecast['timestamp']}")
    else:
        print("Error:", response.text)
    print()

def test_prediction_with_specific_date():
    """Test prediction with specific date"""
    prediction_data = {
        "name": "MSFT",
        "date": "2025-04-10",
        "current_price": 350.25
    }
    
    print("Testing prediction with specific date...")
    print("Request data:", json.dumps(prediction_data, indent=2))
    
    response = requests.post(
        f"{BASE_URL}/api/v1/predict",
        headers={"Content-Type": "application/json"},
        json=prediction_data
    )
    
    print("Status Code:", response.status_code)
    if response.status_code == 200:
        result = response.json()
        forecast = result['predictions'][0]['forecasts'][0]
        print(f"\n📊 PREDICTION SUMMARY:")
        print(f"Stock: {result['predictions'][0]['stock_name']}")
        print(f"Current Price: ${prediction_data['current_price']}")
        print(f"Predicted Price: ${forecast['price']}")
        print(f"Predicted Change: {forecast['pct_change']:+.2f}%")
        print(f"Direction: {forecast['direction']}")
    else:
        print("Error:", response.text)
    print()

if __name__ == "__main__":
    print("🧪 Testing S&P 500 Prediction API - Real Trading Scenario")
    print("=" * 60)
    
    try:
        # Test market status
        test_market_status()
        
        # Test available symbols
        test_available_symbols()
        
        # Test prediction with current date (for Monday's close)
        test_prediction_with_current_date()
        
        # Test prediction with specific date
        test_prediction_with_specific_date()
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")
