#!/usr/bin/env python3
"""
S&P 500 Daily OHLC Prediction Script (Standalone Version)
=========================================================

This script fetches historical daily data for S&P 500 using yfinance
and uses Facebook Prophet to predict next day's OHLC values.

This version doesn't require Interactive Brokers API and can run standalone.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import os
import json
from datetime import datetime, timedelta
import pytz
import time
from pathlib import Path
import logging
import yfinance as yf
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sp500_daily_predictions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SP500DailyPredictorStandalone:
    """
    S&P 500 Daily OHLC Predictor using Facebook Prophet (Standalone Version)
    Uses yfinance for data fetching instead of Interactive Brokers API
    """
    
    def __init__(self):
        # Prediction settings
        self.symbol = "SPY"  # S&P 500 ETF as proxy
        self.prediction_horizon = 7  # Predict next 7 days
        self.training_days = 252  # 1 year of trading days
        
        # Cache for historical data
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
        
        # Output directory
        self.output_dir = Path("sp500_predictions")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("S&P 500 Daily Predictor (Standalone) initialized")
    
    def fetch_daily_historical_data(self, symbol="SPY", days=252):
        """
        Fetch daily historical data for S&P 500 using yfinance
        
        Args:
            symbol: Stock symbol (default: SPY for S&P 500)
            days: Number of days to fetch (default: 252 = 1 year)
        
        Returns:
            pandas DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_daily_{days}"
        current_time = time.time()
        
        # Check cache first
        if cache_key in self.cache:
            timestamp, data = self.cache[cache_key]
            if current_time - timestamp <= self.cache_duration:
                logger.info(f"Using cached daily data for {symbol}")
                return data
        
        logger.info(f"Fetching {days} days of historical data for {symbol}")
        
        try:
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 30)  # Add buffer for weekends/holidays
            
            # Fetch data using yfinance
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d',
                auto_adjust=True,
                prepost=False,
                threads=True
            )
            
            if df.empty:
                logger.error(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Rename columns to match expected format
            df.columns = df.columns.str.lower()
            df = df.rename(columns={'date': 'date'})
            
            # Ensure we have the required columns
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns. Available: {df.columns.tolist()}")
                return pd.DataFrame()
            
            # Take only the last 'days' records
            df = df.tail(days).reset_index(drop=True)
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            # Cache the data
            self.cache[cache_key] = (current_time, df)
            
            logger.info(f"Fetched {len(df)} daily records for {symbol}")
            logger.info(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price change indicators
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=20).std()
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_prophet_data(self, df, metric='close'):
        """
        Prepare data for Prophet model
        
        Args:
            df: DataFrame with historical data
            metric: Which metric to predict (open, high, low, close, volume)
            
        Returns:
            DataFrame formatted for Prophet
        """
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = df['date']
        prophet_df['y'] = df[metric]
        
        # Add additional regressors
        prophet_df['sma_20'] = df['sma_20']
        prophet_df['sma_50'] = df['sma_50']
        prophet_df['rsi'] = df['rsi']
        prophet_df['macd'] = df['macd']
        prophet_df['volume_ratio'] = df['volume_ratio']
        prophet_df['volatility'] = df['volatility']
        
        # Add day of week and month as features
        prophet_df['day_of_week'] = prophet_df['ds'].dt.dayofweek
        prophet_df['month'] = prophet_df['ds'].dt.month
        prophet_df['quarter'] = prophet_df['ds'].dt.quarter
        prophet_df['is_month_end'] = prophet_df['ds'].dt.is_month_end.astype(int)
        prophet_df['is_quarter_end'] = prophet_df['ds'].dt.is_quarter_end.astype(int)
        
        # Remove rows with NaN values
        prophet_df = prophet_df.dropna()
        
        return prophet_df
    
    def predict_daily_ohlc(self, df, horizon=7):
        """
        Predict next N days of OHLC data using Prophet
        
        Args:
            df: Historical DataFrame
            horizon: Number of days to predict
            
        Returns:
            Dictionary with predictions for each metric
        """
        predictions = {}
        metrics = ['open', 'high', 'low', 'close', 'volume']
        
        logger.info(f"Starting daily OHLC prediction for {horizon} days")
        
        # Get last trading day for reference
        last_day = {
            'date': df['date'].iloc[-1],
            'open': df['open'].iloc[-1],
            'high': df['high'].iloc[-1],
            'low': df['low'].iloc[-1],
            'close': df['close'].iloc[-1],
            'volume': df['volume'].iloc[-1]
        }
        
        logger.info(f"Last trading day: {last_day['date'].date()}")
        logger.info(f"Last close price: ${last_day['close']:.2f}")
        
        # Generate future dates (next N trading days)
        future_dates = []
        current_date = last_day['date'] + timedelta(days=1)
        
        for i in range(horizon):
            # Skip weekends
            while current_date.weekday() >= 5:  # Saturday=5, Sunday=6
                current_date += timedelta(days=1)
            future_dates.append(current_date)
            current_date += timedelta(days=1)
        
        logger.info(f"Predicting for dates: {[d.date() for d in future_dates]}")
        
        # Predict each metric
        for metric in metrics:
            logger.info(f"Predicting {metric}...")
            
            try:
                # Prepare Prophet data
                prophet_df = self.prepare_prophet_data(df, metric)
                
                if len(prophet_df) < 50:  # Need minimum data points
                    logger.warning(f"Insufficient data for {metric} prediction")
                    predictions[metric] = [last_day[metric]] * horizon
                    continue
                
                # Initialize Prophet model with appropriate settings for daily data
                model = Prophet(
                    daily_seasonality=False,  # Daily seasonality not relevant for daily data
                    weekly_seasonality=True,   # Weekly patterns (weekday effects)
                    yearly_seasonality=True,   # Annual patterns
                    changepoint_prior_scale=0.05,  # Allow for trend changes
                    seasonality_prior_scale=10.0,  # Strong seasonality
                    holidays_prior_scale=10.0,     # Holiday effects
                    seasonality_mode='multiplicative'  # Multiplicative seasonality for financial data
                )
                
                # Add regressors
                model.add_regressor('sma_20')
                model.add_regressor('sma_50')
                model.add_regressor('rsi')
                model.add_regressor('macd')
                model.add_regressor('volume_ratio')
                model.add_regressor('volatility')
                model.add_regressor('day_of_week')
                model.add_regressor('month')
                model.add_regressor('quarter')
                model.add_regressor('is_month_end')
                model.add_regressor('is_quarter_end')
                
                # Fit the model
                model.fit(prophet_df)
                
                # Prepare future dataframe
                future = pd.DataFrame({'ds': future_dates})
                
                # Add regressors for future dates
                # For future dates, we'll use the last known values or simple projections
                last_sma_20 = df['sma_20'].iloc[-1]
                last_sma_50 = df['sma_50'].iloc[-1]
                last_rsi = df['rsi'].iloc[-1]
                last_macd = df['macd'].iloc[-1]
                last_volume_ratio = df['volume_ratio'].iloc[-1]
                last_volatility = df['volatility'].iloc[-1]
                
                future['sma_20'] = last_sma_20
                future['sma_50'] = last_sma_50
                future['rsi'] = last_rsi
                future['macd'] = last_macd
                future['volume_ratio'] = last_volume_ratio
                future['volatility'] = last_volatility
                future['day_of_week'] = future['ds'].dt.dayofweek
                future['month'] = future['ds'].dt.month
                future['quarter'] = future['ds'].dt.quarter
                future['is_month_end'] = future['ds'].dt.is_month_end.astype(int)
                future['is_quarter_end'] = future['ds'].dt.is_quarter_end.astype(int)
                
                # Generate forecast
                forecast = model.predict(future)
                predicted_values = forecast['yhat'].values
                
                # Apply reasonable bounds based on historical volatility
                last_value = last_day[metric]
                historical_volatility = df[metric].pct_change().std()
                
                # Apply bounds (more conservative for longer horizons)
                bounded_predictions = []
                for i, pred_value in enumerate(predicted_values):
                    # Increase max change for longer predictions
                    max_change = min(0.10, historical_volatility * 3 * (1 + i * 0.1))
                    min_value = last_value * (1 - max_change)
                    max_value = last_value * (1 + max_change)
                    bounded_pred = np.clip(pred_value, min_value, max_value)
                    bounded_predictions.append(bounded_pred)
                
                predictions[metric] = bounded_predictions
                
                logger.info(f"{metric} prediction complete: {bounded_predictions}")
                
            except Exception as e:
                logger.error(f"Error predicting {metric}: {e}")
                # Fallback to last known value
                predictions[metric] = [last_day[metric]] * horizon
        
        return predictions, future_dates
    
    def format_predictions_json(self, predictions, future_dates, metadata):
        """
        Format predictions into JSON structure matching the expected API format
        
        Args:
            predictions: Dictionary with OHLCV predictions
            future_dates: List of future dates
            metadata: Additional metadata
            
        Returns:
            JSON-formatted prediction data
        """
        forecasts = []
        
        for i, date in enumerate(future_dates):
            forecast = {
                "timestamp": date.strftime("%Y-%m-%dT00:00:00"),
                "forecast_index": i + 1,
                "open": round(predictions['open'][i], 2),
                "high": round(predictions['high'][i], 2),
                "low": round(predictions['low'][i], 2),
                "close": round(predictions['close'][i], 2),
                "volume": int(predictions['volume'][i]),
                "pct_change": round(((predictions['close'][i] - metadata['last_close']) / metadata['last_close']) * 100, 2),
                "direction": "UP" if predictions['close'][i] > metadata['last_close'] else "DOWN"
            }
            forecasts.append(forecast)
        
        prediction_data = {
            "prediction_timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "predictions": [
                {
                    "name": "S&P 500 Daily Prophet Model",
                    "description": "Daily OHLC prediction using Facebook Prophet with technical indicators",
                    "horizon": len(future_dates),
                    "frequency": 1,
                    "stock_name": "S&P 500",
                    "forecasts": forecasts
                }
            ],
            "metadata": {
                "model_type": "Prophet",
                "training_days": metadata['training_days'],
                "last_training_date": metadata['last_training_date'],
                "prediction_horizon": len(future_dates),
                "symbol": self.symbol,
                "data_source": "yfinance"
            }
        }
        
        return prediction_data
    
    def save_predictions(self, prediction_data, filename=None):
        """
        Save predictions to JSON file
        
        Args:
            prediction_data: JSON-formatted prediction data
            filename: Optional custom filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sp500_daily_predictions_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(prediction_data, f, indent=2)
        
        logger.info(f"Predictions saved to {filepath}")
        return filepath
    
    def run_prediction(self):
        """
        Main method to run the daily prediction process
        """
        try:
            logger.info("Starting S&P 500 daily prediction process")
            
            # Fetch historical data
            df = self.fetch_daily_historical_data(self.symbol, self.training_days)
            
            if df.empty:
                logger.error("Failed to fetch historical data")
                return None
            
            # Generate predictions
            predictions, future_dates = self.predict_daily_ohlc(df, self.prediction_horizon)
            
            # Prepare metadata
            metadata = {
                'last_close': df['close'].iloc[-1],
                'training_days': len(df),
                'last_training_date': df['date'].iloc[-1].strftime("%Y-%m-%d")
            }
            
            # Format as JSON
            prediction_data = self.format_predictions_json(predictions, future_dates, metadata)
            
            # Save predictions
            filepath = self.save_predictions(prediction_data)
            
            # Print summary
            logger.info("Prediction Summary:")
            logger.info(f"Last close: ${metadata['last_close']:.2f}")
            logger.info(f"Predicted next close: ${predictions['close'][0]:.2f}")
            logger.info(f"Predicted change: {((predictions['close'][0] - metadata['last_close']) / metadata['last_close']) * 100:.2f}%")
            
            return prediction_data
            
        except Exception as e:
            logger.error(f"Error in prediction process: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

def main():
    """
    Main function to run the S&P 500 daily predictor
    """
    print("S&P 500 Daily OHLC Predictor (Standalone)")
    print("=" * 50)
    
    # Initialize predictor
    predictor = SP500DailyPredictorStandalone()
    
    # Run prediction
    result = predictor.run_prediction()
    
    if result:
        print("\nPrediction completed successfully!")
        print(f"Results saved to: {predictor.output_dir}")
        
        # Print first few predictions
        print("\nFirst 3 days predictions:")
        for i, forecast in enumerate(result['predictions'][0]['forecasts'][:3]):
            print(f"Day {i+1} ({forecast['timestamp'][:10]}): "
                  f"O:${forecast['open']:.2f} H:${forecast['high']:.2f} "
                  f"L:${forecast['low']:.2f} C:${forecast['close']:.2f} "
                  f"({forecast['pct_change']:+.2f}%)")
    else:
        print("Prediction failed. Check logs for details.")

if __name__ == "__main__":
    main()
