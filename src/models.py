"""
S&P 500 Stock Prediction Model using Facebook Prophet
Aligned with original sp500_daily_predictor.py parameters
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import logging
from typing import Optional, Dict, Any, List
import asyncio
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class StockPredictionModel:
    """
    S&P 500 Stock Prediction Model using Facebook Prophet
    Aligned with original sp500_daily_predictor.py parameters
    """
    
    def __init__(self):
        self.is_initialized = False
        self.last_trained_at = None
        
        # ALIGNED WITH ORIGINAL: Use same parameters
        self.training_days = 30  # Use last 30 days for training (ORIGINAL)
        self.prediction_horizon = 1  # Predict next 1 day only (ORIGINAL)
        
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
        
        logger.info("Stock Prediction Model initialized with original parameters")
    
    async def initialize(self):
        """Initialize the model (async for FastAPI compatibility)"""
        try:
            # Pre-load some popular stocks for faster response
            popular_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            
            logger.info("Pre-loading popular stocks...")
            for symbol in popular_symbols:
                try:
                    await self._preload_stock_data(symbol)
                except Exception as e:
                    logger.warning(f"Failed to pre-load {symbol}: {e}")
            
            self.is_initialized = True
            self.last_trained_at = datetime.now().isoformat()
            logger.info("Model initialization complete")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
    
    async def _preload_stock_data(self, symbol: str):
        """Pre-load stock data for faster predictions"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._fetch_stock_data, symbol)
        except Exception as e:
            logger.error(f"Failed to preload data for {symbol}: {e}")
    
    def _fetch_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical stock data using yfinance (REPLACEMENT FOR IBKR)"""
        try:
            cache_key = f"{symbol}_data"
            current_time = datetime.now().timestamp()
            
            # Check cache first
            if cache_key in self.cache:
                timestamp, data = self.cache[cache_key]
                if current_time - timestamp <= self.cache_duration:
                    logger.info(f"Using cached data for {symbol}")
                    return data
            
            logger.info(f"Fetching data for {symbol}")
            
            # ALIGNED WITH ORIGINAL: Request more calendar days to get enough trading days
            # For 30 trading days, we need approximately 90 calendar days
            calendar_days_needed = 90  # Request 90 calendar days to get ~60 trading days, then trim to 30
            
            # Download data using yfinance (REPLACEMENT FOR IBKR)
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo", interval="1d")  # 3 months to get ~90 calendar days
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Process the data
            df = data.reset_index()
            df.columns = df.columns.str.lower()
            df['date'] = pd.to_datetime(df['date'])
            
            # Add technical indicators (ALIGNED WITH ORIGINAL)
            df = self._add_technical_indicators(df)
            
            # ALIGNED WITH ORIGINAL: Take only the last 30 trading days AFTER technical indicators
            if len(df) > self.training_days:
                df = df.tail(self.training_days).reset_index(drop=True)
                logger.info(f"Trimmed to last {self.training_days} trading days after technical indicators")
            
            # Cache the data
            self.cache[cache_key] = (current_time, df)
            
            logger.info(f"Successfully fetched {len(df)} days of data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame (ALIGNED WITH ORIGINAL)"""
        try:
            # ALIGNED WITH ORIGINAL: Simple Moving Averages (minimal periods for 30-day training)
            df['sma_3'] = df['close'].rolling(window=3).mean()
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_10'] = df['close'].rolling(window=10).mean()
            
            # ALIGNED WITH ORIGINAL: Exponential Moving Averages (minimal periods for 30-day training)
            df['ema_3'] = df['close'].ewm(span=3).mean()
            df['ema_5'] = df['close'].ewm(span=5).mean()
            
            # ALIGNED WITH ORIGINAL: MACD (simplified for 30-day training)
            df['macd'] = df['ema_3'] - df['ema_5']
            
            # ALIGNED WITH ORIGINAL: RSI (minimal window for 30-day training)
            df['rsi'] = self._calculate_rsi(df['close'], 5)  # ORIGINAL uses period=5
            
            # ALIGNED WITH ORIGINAL: Bollinger Bands (minimal window for 30-day training)
            df['bb_middle'] = df['close'].rolling(window=5).mean()
            bb_std = df['close'].rolling(window=5).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # ALIGNED WITH ORIGINAL: Volume indicators (minimal window for 30-day training)
            df['volume_sma'] = df['volume'].rolling(window=5).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # ALIGNED WITH ORIGINAL: Price change indicators (minimal window for 30-day training)
            df['price_change'] = df['close'].pct_change()
            df['volatility'] = df['price_change'].rolling(window=5).std()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 5) -> pd.Series:
        """Calculate RSI indicator (ALIGNED WITH ORIGINAL: period=5)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _prepare_prophet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Prophet model (ALIGNED WITH ORIGINAL)"""
        try:
            prophet_df = pd.DataFrame()
            # FIX: Remove timezone from datetime column
            prophet_df['ds'] = df['date'].dt.tz_localize(None)  # Remove timezone
            prophet_df['y'] = df['close']  # ALIGNED WITH ORIGINAL: predict 'close' price
            
            # ALIGNED WITH ORIGINAL: Add additional regressors (updated for 30-day training with minimal indicators)
            prophet_df['sma_3'] = df['sma_3']
            prophet_df['sma_5'] = df['sma_5']
            prophet_df['sma_10'] = df['sma_10']
            prophet_df['ema_3'] = df['ema_3']
            prophet_df['ema_5'] = df['ema_5']
            prophet_df['rsi'] = df['rsi']
            prophet_df['macd'] = df['macd']
            prophet_df['volume_ratio'] = df['volume_ratio']
            prophet_df['volatility'] = df['volatility']
            
            # ALIGNED WITH ORIGINAL: Add day of week and month as features
            prophet_df['day_of_week'] = prophet_df['ds'].dt.dayofweek
            prophet_df['month'] = prophet_df['ds'].dt.month
            prophet_df['quarter'] = prophet_df['ds'].dt.quarter
            
            # Remove rows with NaN values
            initial_rows = len(prophet_df)
            prophet_df = prophet_df.dropna()
            final_rows = len(prophet_df)
            
            logger.info(f"Prophet data preparation: {initial_rows} -> {final_rows} rows after removing NaN")
            
            if final_rows < 10:
                logger.warning(f"Very few rows remaining ({final_rows}), this may cause prediction issues")
            
            return prophet_df
            
        except Exception as e:
            logger.error(f"Error preparing Prophet data: {e}")
            return pd.DataFrame()
    
    async def predict(self, symbol: str, current_price: float, prediction_date: datetime) -> Optional[Dict[str, Any]]:
        """
        Generate stock price prediction (ALIGNED WITH ORIGINAL)
        
        Args:
            symbol: Stock symbol
            current_price: Current stock price
            prediction_date: Date for prediction
            
        Returns:
            Dictionary with prediction results
        """
        try:
            logger.info(f"Generating prediction for {symbol}")
            
            # Fetch historical data
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, self._fetch_stock_data, symbol)
            
            if df is None or df.empty:
                logger.error(f"No data available for {symbol}")
                return None
            
            # ALIGNED WITH ORIGINAL: Check minimum data requirement
            if len(df) < 20:  # ORIGINAL uses 20 as minimum
                logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                return None
            
            # Prepare Prophet data
            prophet_df = self._prepare_prophet_data(df)
            
            # ALIGNED WITH ORIGINAL: Check minimum Prophet data requirement
            if len(prophet_df) < 15:  # ORIGINAL uses 15 as minimum
                logger.warning(f"Insufficient data for Prophet prediction: {len(prophet_df)} rows")
                return None
            
            # ALIGNED WITH ORIGINAL: Generate future dates (next 1 trading day only)
            future_dates = self._generate_future_dates(prediction_date, self.prediction_horizon)
            
            # ALIGNED WITH ORIGINAL: Create Prophet model with same settings
            model = Prophet(
                daily_seasonality=False,  # Daily seasonality not relevant for daily data
                weekly_seasonality=True,   # Weekly patterns (weekday effects)
                yearly_seasonality=False,  # Disabled for short training period
                changepoint_prior_scale=0.05,  # Allow for trend changes
                seasonality_prior_scale=10.0,  # Strong seasonality
                holidays_prior_scale=10.0,     # Holiday effects
                seasonality_mode='multiplicative'  # Multiplicative seasonality for financial data
            )
            
            # ALIGNED WITH ORIGINAL: Add regressors (updated for 30-day training with minimal indicators)
            model.add_regressor('sma_3')
            model.add_regressor('sma_5')
            model.add_regressor('sma_10')
            model.add_regressor('ema_3')
            model.add_regressor('ema_5')
            model.add_regressor('rsi')
            model.add_regressor('macd')
            model.add_regressor('volume_ratio')
            model.add_regressor('volatility')
            model.add_regressor('day_of_week')
            model.add_regressor('month')
            model.add_regressor('quarter')
            
            # Fit the model
            model.fit(prophet_df)
            
            # Prepare future dataframe
            future = pd.DataFrame({'ds': future_dates})
            
            # ALIGNED WITH ORIGINAL: Add regressors for future dates (updated for 30-day training with minimal indicators)
            last_sma_3 = df['sma_3'].iloc[-1]
            last_sma_5 = df['sma_5'].iloc[-1]
            last_sma_10 = df['sma_10'].iloc[-1]
            last_ema_3 = df['ema_3'].iloc[-1]
            last_ema_5 = df['ema_5'].iloc[-1]
            last_rsi = df['rsi'].iloc[-1]
            last_macd = df['macd'].iloc[-1]
            last_volume_ratio = df['volume_ratio'].iloc[-1]
            last_volatility = df['volatility'].iloc[-1]
            
            future['sma_3'] = last_sma_3
            future['sma_5'] = last_sma_5
            future['sma_10'] = last_sma_10
            future['ema_3'] = last_ema_3
            future['ema_5'] = last_ema_5
            future['rsi'] = last_rsi
            future['macd'] = last_macd
            future['volume_ratio'] = last_volume_ratio
            future['volatility'] = last_volatility
            future['day_of_week'] = future['ds'].dt.dayofweek
            future['month'] = future['ds'].dt.month
            future['quarter'] = future['ds'].dt.quarter
            
            # Generate forecast
            forecast = model.predict(future)
            predicted_values = forecast['yhat'].values
            
            # ALIGNED WITH ORIGINAL: Apply reasonable bounds based on historical volatility
            last_value = df['close'].iloc[-1]
            historical_volatility = df['close'].pct_change().std()
            
            # ALIGNED WITH ORIGINAL: Apply bounds (more conservative for daily predictions)
            bounded_predictions = []
            for i, pred_value in enumerate(predicted_values):
                # ALIGNED WITH ORIGINAL: Conservative bounds for daily predictions
                max_change = min(0.05, historical_volatility * 2)  # Max 5% change per day
                min_value = last_value * (1 - max_change)
                max_value = last_value * (1 + max_change)
                bounded_pred = np.clip(pred_value, min_value, max_value)
                bounded_predictions.append(bounded_pred)
            
            # Format response (ALIGNED WITH ORIGINAL: single day prediction)
            forecasts = []
            for i, (date, price) in enumerate(zip(future_dates, bounded_predictions)):
                pct_change = ((price - current_price) / current_price) * 100
                forecasts.append({
                    "timestamp": date.strftime("%Y-%m-%dT00:00:00"),
                    "forecast_index": i + 1,
                    "price": round(price, 2),
                    "pct_change": round(pct_change, 2),
                    "direction": "UP" if pct_change > 0 else "DOWN"
                })
            
            prediction_result = {
                "prediction_timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "predictions": [
                    {
                        "name": "S&P 500 Prophet Model",
                        "description": f"Stock price forecast with {self.prediction_horizon}-day horizon using Facebook Prophet",
                        "horizon": self.prediction_horizon,  # ALIGNED: 1 day
                        "frequency": 1,
                        "stock_name": symbol,
                        "forecasts": forecasts
                    }
                ]
            }
            
            logger.info(f"Successfully generated prediction for {symbol}")
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error generating prediction for {symbol}: {e}")
            return None
    
    def _generate_future_dates(self, start_date: datetime, horizon: int) -> List[datetime]:
        """Generate future trading dates (ALIGNED WITH ORIGINAL)"""
        future_dates = []
        current_date = start_date
        
        for i in range(horizon):
            # ALIGNED WITH ORIGINAL: Skip weekends
            while current_date.weekday() >= 5:  # Saturday=5, Sunday=6
                current_date += timedelta(days=1)
            future_dates.append(current_date)
            current_date += timedelta(days=1)
        
        return future_dates

# Global model instance
_model_instance = None

def get_prediction_model() -> StockPredictionModel:
    """Get the global prediction model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = StockPredictionModel()
    return _model_instance
