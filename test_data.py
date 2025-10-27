import yfinance as yf
import pandas as pd

# Test if yfinance can fetch AAPL data
try:
    print("Testing yfinance data fetch...")
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="3mo", interval="1d")
    print(f"✅ Successfully fetched {len(data)} days of data")
    print(f"Latest close price: ${data['Close'].iloc[-1]:.2f}")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
except Exception as e:
    print(f"❌ Error fetching data: {e}")
