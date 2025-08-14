import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from typing import Dict


class BTCModel:
    MODEL_PATH = "src/models/rf.pkl"
    SCALER_PATH = "src/models/scaler.pkl"
    FEATURES_PATH = "src/models/feature_cols.pkl"

    def __init__(self):
        self.model = joblib.load(self.MODEL_PATH)
        self.scaler = joblib.load(self.SCALER_PATH)
        self.feature_cols = joblib.load(self.FEATURES_PATH)

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['month'] = pd.to_datetime(df['Date']).dt.month
        df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
        df['open-close'] = df['Open'] - df['Close']
        df['low-high'] = df['Low'] - df['High']
        df['vol_change'] = df['Volume'].pct_change().fillna(0)
        df['lag_return'] = df['Close'].pct_change().fillna(0)

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-9)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow

        df['sma_fast'] = df['Close'].rolling(window=12, min_periods=1).mean()

        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        ma20 = df['Close'].rolling(window=20, min_periods=1).mean()
        std20 = df['Close'].rolling(window=20, min_periods=1).std()
        ma20 = pd.Series(ma20.values.flatten(), index=df.index)
        std20 = pd.Series(std20.values.flatten(), index=df.index)

        df['bb_high'] = ma20 + 2 * std20
        df['bb_low'] = ma20 - 2 * std20

        bb_high = pd.Series(df['bb_high'].values.flatten(), index=df.index)
        bb_low = pd.Series(df['bb_low'].values.flatten(), index=df.index)
        close = pd.Series(df['Close'].values.flatten(), index=df.index)

        df['bb_pos'] = (close - bb_low) / (bb_high - bb_low + 1e-9)

        df['atr_14'] = (df['High'] - df['Low']).rolling(window=14, min_periods=1).mean()

        df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

        # df = df.dropna()
        return df

    def predict(self) -> Dict:
        df = yf.download('BTC-USD', period='30d', interval='1d', progress=False)
        if df.empty:
            raise ValueError("Failed to download BTC-USD data")
        df.reset_index(inplace=True)

        ticker = yf.Ticker("BTC-USD")
        live_price = float(ticker.info["regularMarketPrice"])


        
        today_row = {
            "Date": pd.Timestamp.today(),
            "Open": live_price,
            "High": live_price,
            "Low": live_price,
            "Close": live_price,
            "Adj Close": live_price,
            "Volume": df["Volume"].iloc[-1]  
        }
        df = pd.concat([df, pd.DataFrame([today_row])], ignore_index=True)

        
        df = self.create_features(df)

        
        last_row = df.iloc[-1][self.feature_cols]
        if last_row.isnull().any():
            last_row = last_row.fillna(df.iloc[-2][self.feature_cols])

        latest_scaled = self.scaler.transform(last_row.values.reshape(1, -1))

        pred = self.model.predict(latest_scaled)[0]


        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(latest_scaled)[0]
            confidence = float(probs[pred]) * 100
        else:
            confidence = None

        return {
            "24_hours": {
                "price_prediction": live_price,
                "price_up_down": "up" if pred == 1 else "down",
                "percentage_change": confidence
            }
        }
