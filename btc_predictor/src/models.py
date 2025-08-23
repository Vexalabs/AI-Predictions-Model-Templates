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
        return df

    def predict(self, symbol: str = "BTC-USD", interval: str = "1d") -> Dict:
       
        df = yf.download(symbol, period="3y", interval=interval, progress=False)
        if df.empty:
            raise ValueError(f"Failed to download {symbol} data for features")
        df.reset_index(inplace=True)
        df = self.create_features(df)

        try:
            ticker = yf.Ticker(symbol)
            live_price = float(
                ticker.info.get("regularMarketPrice", float(df["Close"].iloc[-1]))
            )
        except Exception:
            live_price = float(df["Close"].iloc[-1])

        latest_data = df.iloc[-1][self.feature_cols].values.reshape(1, -1)
        latest_scaled = self.scaler.transform(latest_data)
        pred = int(self.model.predict(latest_scaled)[0])          
        direction = "up" if pred == 1 else "down"

        confidence: float | None = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(latest_scaled)[0]
            confidence = float(probs[pred]) * 100.0

        payload = {
            "price": round(live_price, 2),
            "price_up_down": direction,
            "percentage_change": None if confidence is None else round(confidence, 2),
        }

        return {
            "24_hours": payload,
        }
