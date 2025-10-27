import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_production_gui():
    """Test the production GUI"""
    print("🌐 Production GUI Available at:")
    print(f"   {BASE_URL}/docs")
    print("   This is the Swagger UI - interactive API documentation")
    print()

def test_market_status():
    """Test market status endpoint"""
    response = requests.get(f"{BASE_URL}/api/v1/market-status")
    print("📊 Market Status:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_current_price():
    """Test current price endpoint"""
    response = requests.get(f"{BASE_URL}/api/v1/current-price/AAPL")
    print("💰 Current AAPL Price:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_prediction_auto_price():
    """Test prediction with automatic price fetching"""
    prediction_data = {
        "name": "AAPL"
        # No current_price - will fetch automatically
        # No date - will use current date
    }
    
    print("🤖 Testing prediction with automatic price fetching...")
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
        market_context = result.get('market_context', {})
        
        print(f"\n📊 PRODUCTION PREDICTION SUMMARY:")
        print(f"Stock: {result['predictions'][0]['stock_name']}")
        print(f"Current Price: ${market_context.get('current_price', 'N/A')}")
        print(f"Predicted Price: ${forecast['price']}")
        print(f"Predicted Change: {forecast['pct_change']:+.2f}%")
        print(f"Direction: {forecast['direction']}")
        print(f"Market Status: {market_context.get('market_status', 'N/A')}")
        print(f"Prediction Type: {market_context.get('prediction_type', 'N/A')}")
        print(f"Target Date: {market_context.get('target_date', 'N/A')}")
    else:
        print("Error:", response.text)
    print()

if __name__ == "__main__":
    print("🚀 Testing Production-Ready S&P 500 Prediction API")
    print("=" * 60)
    
    try:
        # Test production GUI
        test_production_gui()
        
        # Test market status
        test_market_status()
        
        # Test current price
        test_current_price()
        
        # Test prediction with automatic price fetching
        test_prediction_auto_price()
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")
