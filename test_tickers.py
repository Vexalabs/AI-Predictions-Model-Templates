#!/usr/bin/env python3
"""
Test script to verify S&P 500 ticker fetching
"""

import requests
import pandas as pd
from io import StringIO

def test_sp500_tickers():
    """Test fetching S&P 500 tickers"""
    print("Testing S&P 500 Ticker Fetching")
    print("=" * 40)
    
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Try slickcharts first
        try:
            print("Trying SlickCharts...")
            response = requests.get('https://www.slickcharts.com/sp500', headers=headers)
            if response.status_code == 200:
                tables = pd.read_html(StringIO(response.text))
                if tables:
                    df = tables[0]
                    tickers = df['Symbol'].str.replace('.B', '').tolist()
                    tickers = list(dict.fromkeys(tickers))
                    print(f"✓ Successfully fetched {len(tickers)} tickers from SlickCharts")
                    print(f"First 10 tickers: {tickers[:10]}")
                    return tickers
        except Exception as e:
            print(f"SlickCharts failed: {e}")

        # Fallback to Wikipedia
        print("Falling back to Wikipedia...")
        response = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers)
        if response.status_code == 200:
            tables = pd.read_html(StringIO(response.text))
            tickers = tables[0]['Symbol'].tolist()
            print(f"✓ Successfully fetched {len(tickers)} tickers from Wikipedia")
            print(f"First 10 tickers: {tickers[:10]}")
            return tickers
        
        print("❌ Failed to fetch tickers")
        return []
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

if __name__ == "__main__":
    tickers = test_sp500_tickers()
    if tickers:
        print(f"\n🎉 Success! Found {len(tickers)} S&P 500 tickers")
    else:
        print("\n❌ Failed to fetch tickers")

