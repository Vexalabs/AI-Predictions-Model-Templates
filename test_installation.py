#!/usr/bin/env python3
"""
Test script to verify the S&P 500 Daily Predictor installation
"""

import sys
import pandas as pd
import numpy as np
from prophet import Prophet
import json
from datetime import datetime, timedelta

def test_installation():
    """Test if all required libraries are properly installed"""
    print("Testing S&P 500 Daily Predictor Installation")
    print("=" * 50)
    
    try:
        # Test pandas
        print("✓ Testing pandas...")
        df = pd.DataFrame({'date': [datetime.now()], 'value': [100]})
        print(f"  Pandas version: {pd.__version__}")
        
        # Test numpy
        print("✓ Testing numpy...")
        arr = np.array([1, 2, 3, 4, 5])
        print(f"  Numpy version: {np.__version__}")
        
        # Test Prophet
        print("✓ Testing Prophet...")
        model = Prophet()
        print("  Prophet model created successfully")
        
        # Test JSON
        print("✓ Testing JSON...")
        test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
        json_str = json.dumps(test_data)
        print("  JSON serialization working")
        
        print("\n🎉 All tests passed! Installation is successful.")
        print("\nYou can now run the S&P 500 Daily Predictor:")
        print("  python sp500_daily_predictor.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_installation()
    sys.exit(0 if success else 1)
