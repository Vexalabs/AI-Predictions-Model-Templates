#!/usr/bin/env python3
"""
S&P 500 Daily OHLC Prediction Script
====================================

This script fetches historical daily data for S&P 500 and uses Facebook Prophet
to predict next day's OHLC (Open, High, Low, Close) values.

Based on the existing hourly prediction logic from live_ibapi_V9.8.py
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
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData
from typing import List
import threading
import logging
import requests

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

class SP500DailyPredictor(EWrapper, EClient):
    """
    S&P 500 Daily OHLC Predictor using Facebook Prophet
    Predicts OHLC for ALL S&P 500 stocks
    """
    
    def __init__(self):
        EClient.__init__(self, self)
        
        # Data storage
        self.historical_data_store = {}
        self.historical_data_ready = threading.Event()
        self._active_symbol = None
        self._req_id_counter = 0
        
        # Prediction settings
        self.prediction_horizon = 1  # Predict next 1 day only
        self.training_days = 30  # Use last 30 days for training
        
        # Cache for historical data
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
        
        # Output directory
        self.output_dir = Path("sp500_predictions")
        self.output_dir.mkdir(exist_ok=True)
        
        # S&P 500 tickers
        self.sp500_tickers = []
        
        # Connection status
        self.connected = False
        
        logger.info("S&P 500 Daily Predictor initialized")
    
    def get_new_req_id(self):
        """Generate unique request ID"""
        self._req_id_counter += 1
        return self._req_id_counter
    
    def connect_to_tws(self, host='127.0.0.1', port=7497, client_id=None, max_attempts=3):
        """Connect to IBKR TWS using native ibapi with retry logic and timeout"""
        if client_id is None:
            # Generate unique client ID based on current time
            client_id = int(time.time()) % 1000
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"Attempting to connect to IBKR (Attempt {attempt + 1}/{max_attempts})")
                logger.info(f"Using client ID: {client_id}")
                self.connect(host, port, client_id)

                # Start the event loop thread
                thread = threading.Thread(target=self.run, daemon=True)
                thread.start()

                # Wait for connection to establish
                timeout = 10  # seconds
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if self.isConnected():
                        self.connected = True
                        logger.info("Successfully connected to TWS")
                        return True
                    time.sleep(1)

                # Timeout handling
                logger.warning("Connection timeout. Disconnecting and retrying...")
                self.disconnect()
                time.sleep(2)

            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2)

        logger.error("Failed to connect after multiple attempts")
        return False
    
    def disconnect_from_tws(self):
        """Disconnect from TWS"""
        if self.connected:
            self.disconnect()
            self.connected = False
            logger.info("Disconnected from TWS")
    
    def get_sp500_tickers(self):
        """Get S&P 500 tickers ordered by market cap weight using slickcharts.com"""
        try:
            # Add headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Try slickcharts first
            try:
                response = requests.get('https://www.slickcharts.com/sp500', headers=headers)
                if response.status_code == 200:
                    # Fix the pandas warning by using StringIO
                    from io import StringIO
                    tables = pd.read_html(StringIO(response.text))
                    if tables:
                        df = tables[0]
                        # Clean and extract symbols
                        tickers = df['Symbol'].str.replace('.B', '').tolist()
                        # Remove any duplicates while preserving order
                        tickers = list(dict.fromkeys(tickers))
                        logger.info(f"Successfully fetched {len(tickers)} tickers from SlickCharts")
                        return tickers
            except Exception as e:
                logger.warning(f"SlickCharts fetch failed: {e}")

            # Fallback to Wikipedia with headers
            logger.info("Falling back to Wikipedia...")
            response = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers)
            if response.status_code == 200:
                # Fix the pandas warning by using StringIO
                from io import StringIO
                tables = pd.read_html(StringIO(response.text))
                tickers = tables[0]['Symbol'].tolist()
                logger.info(f"Successfully fetched {len(tickers)} tickers from Wikipedia")
                return tickers
            
            raise Exception(f"Failed to fetch data: Status code {response.status_code}")

        except Exception as e:
            logger.error(f"Error fetching S&P 500 tickers: {e}")
            return []

    def get_stock_contract(self, symbol: str) -> Contract:
        """Get contract for individual stock"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract
    
    def fetch_daily_historical_data(self, symbol, days=252):
        """
        Fetch daily historical data for individual stock
        
        Args:
            symbol: Stock symbol
            days: Number of days to fetch (default: 252 = 1 year)
        
        Returns:
            List of BarData objects
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
        
        # Reset data store
        self.historical_data_store[symbol] = []
        self.historical_data_ready.clear()
        self._active_symbol = symbol
        
        # Get contract
        contract = self.get_stock_contract(symbol)
        req_id = self.get_new_req_id()
        
        # Calculate end time (use current date for most recent data)
        now = datetime.now(pytz.timezone('US/Eastern'))
        # Always use current date to get the most recent data available
        end_time = now
        
        # Format end time
        utc_time = end_time.astimezone(pytz.UTC)
        formatted_end_time = utc_time.strftime('%Y%m%d-%H:%M:%S')
        
        try:
            # Request historical data (request more calendar days to get enough trading days)
            # For 30 trading days, we need approximately 90 calendar days (accounting for weekends and significant NaN losses)
            calendar_days_needed = 90  # Request 90 calendar days to get ~60 trading days, then trim to 30
            logger.info(f"Requesting {calendar_days_needed} calendar days to get ~60 trading days")
            
            self.reqHistoricalData(
                reqId=req_id,
                contract=contract,
                endDateTime=formatted_end_time,
                durationStr=f"{calendar_days_needed} D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=1,  # Regular trading hours only
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )
            
            logger.info("Historical data request sent")
            
            # Wait for response
            if self.historical_data_ready.wait(timeout=30):
                bars = self.historical_data_store.get(symbol, [])
                if bars:
                    logger.info(f"Received {len(bars)} daily bars for {symbol}")
                    # Cache the data
                    self.cache[cache_key] = (current_time, bars)
                    return bars
                else:
                    logger.warning("No bars received")
                    return []
            else:
                logger.error("Timeout waiting for historical data")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []
    
    def historicalData(self, reqId: int, bar: BarData):
        """Callback for historical data"""
        if self._active_symbol:
            self.historical_data_store[self._active_symbol].append(bar)
    
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """Callback when historical data is complete"""
        logger.info(f"Historical data complete for request {reqId}")
        self.historical_data_ready.set()
    
    def build_daily_dataframe(self, bars):
        """
        Convert historical bars to pandas DataFrame
        
        Args:
            bars: List of BarData objects
            
        Returns:
            pandas DataFrame with OHLCV data
        """
        if not bars:
            return pd.DataFrame()
        
        data = []
        for bar in bars:
            data.append({
                'date': pd.to_datetime(bar.date),
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume)
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('date').reset_index(drop=True)
        
        # Add technical indicators FIRST (on all available data)
        df = self.add_technical_indicators(df)
        
        # Take only the last 30 trading days AFTER technical indicators
        if len(df) > self.training_days:
            df = df.tail(self.training_days).reset_index(drop=True)
            logger.info(f"Trimmed to last {self.training_days} trading days after technical indicators")
        
        logger.info(f"Built DataFrame with {len(df)} daily records")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the DataFrame (optimized for 30-day training)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        # Simple Moving Averages (minimal periods for 30-day training)
        df['sma_3'] = df['close'].rolling(window=3).mean()
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        
        # Exponential Moving Averages (minimal periods for 30-day training)
        df['ema_3'] = df['close'].ewm(span=3).mean()
        df['ema_5'] = df['close'].ewm(span=5).mean()
        
        # MACD (simplified for 30-day training)
        df['macd'] = df['ema_3'] - df['ema_5']
        
        # RSI (minimal window for 30-day training)
        df['rsi'] = self.calculate_rsi(df['close'], 5)
        
        # Bollinger Bands (minimal window for 30-day training)
        df['bb_middle'] = df['close'].rolling(window=5).mean()
        bb_std = df['close'].rolling(window=5).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume indicators (minimal window for 30-day training)
        df['volume_sma'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price change indicators (minimal window for 30-day training)
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=5).std()
        
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
        
        # Add additional regressors (updated for 30-day training with minimal indicators)
        prophet_df['sma_3'] = df['sma_3']
        prophet_df['sma_5'] = df['sma_5']
        prophet_df['sma_10'] = df['sma_10']
        prophet_df['ema_3'] = df['ema_3']
        prophet_df['ema_5'] = df['ema_5']
        prophet_df['rsi'] = df['rsi']
        prophet_df['macd'] = df['macd']
        prophet_df['volume_ratio'] = df['volume_ratio']
        prophet_df['volatility'] = df['volatility']
        
        # Add day of week and month as features
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
    
    def predict_daily_close(self, df, horizon=1):
        """
        Predict next day's CLOSE price using Prophet
        
        Args:
            df: Historical DataFrame
            horizon: Number of days to predict (default: 1)
            
        Returns:
            Dictionary with close price predictions and future dates
        """
        logger.info(f"Starting daily CLOSE prediction for {horizon} days")
        
        # Get last trading day for reference
        last_day = {
            'date': df['date'].iloc[-1],
            'close': df['close'].iloc[-1]
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
        
        try:
            # Prepare Prophet data for CLOSE price only
            prophet_df = self.prepare_prophet_data(df, 'close')
            
            if len(prophet_df) < 15:  # Reduced minimum for 30-day training with optimized indicators
                logger.warning(f"Insufficient data for close prediction: {len(prophet_df)} rows")
                predictions = [last_day['close']] * horizon
            else:
                # Initialize Prophet model with appropriate settings for daily data
                model = Prophet(
                    daily_seasonality=False,  # Daily seasonality not relevant for daily data
                    weekly_seasonality=True,   # Weekly patterns (weekday effects)
                    yearly_seasonality=False,  # Disabled for short training period
                    changepoint_prior_scale=0.05,  # Allow for trend changes
                    seasonality_prior_scale=10.0,  # Strong seasonality
                    holidays_prior_scale=10.0,     # Holiday effects
                    seasonality_mode='multiplicative'  # Multiplicative seasonality for financial data
                )
                
                # Add regressors (updated for 30-day training with minimal indicators)
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
                
                # Add regressors for future dates (updated for 30-day training with minimal indicators)
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
                
                # Apply reasonable bounds based on historical volatility
                last_value = last_day['close']
                historical_volatility = df['close'].pct_change().std()
                
                # Apply bounds (more conservative for daily predictions)
                bounded_predictions = []
                for i, pred_value in enumerate(predicted_values):
                    # Conservative bounds for daily predictions
                    max_change = min(0.05, historical_volatility * 2)  # Max 5% change per day
                    min_value = last_value * (1 - max_change)
                    max_value = last_value * (1 + max_change)
                    bounded_pred = np.clip(pred_value, min_value, max_value)
                    bounded_predictions.append(bounded_pred)
                
                predictions = bounded_predictions
                
                logger.info(f"Close prediction complete: {predictions}")
                
        except Exception as e:
            logger.error(f"Error predicting close: {e}")
            # Fallback to last known value
            predictions = [last_day['close']] * horizon
        
        return {'close': predictions}, future_dates
    
    def fix_ohlc_logic(self, predictions):
        """
        Fix OHLC predictions to ensure logical relationships:
        Low <= Open,Close <= High
        """
        fixed_predictions = {}
        
        for i in range(len(predictions['close'])):
            open_price = predictions['open'][i]
            high_price = predictions['high'][i]
            low_price = predictions['low'][i]
            close_price = predictions['close'][i]
            
            # Ensure Low <= Open,Close <= High
            low_price = min(low_price, open_price, close_price)
            high_price = max(high_price, open_price, close_price)
            
            # Store fixed values
            if i == 0:
                fixed_predictions['open'] = [open_price]
                fixed_predictions['high'] = [high_price]
                fixed_predictions['low'] = [low_price]
                fixed_predictions['close'] = [close_price]
                fixed_predictions['volume'] = [predictions['volume'][i]]
            else:
                fixed_predictions['open'].append(open_price)
                fixed_predictions['high'].append(high_price)
                fixed_predictions['low'].append(low_price)
                fixed_predictions['close'].append(close_price)
                fixed_predictions['volume'].append(predictions['volume'][i])
        
        logger.info("OHLC logic fixed - ensuring Low <= Open,Close <= High")
        return fixed_predictions
    
    def format_predictions_json(self, predictions, future_dates, metadata):
        """
        Format predictions into JSON structure matching the expected API format
        
        Args:
            predictions: Dictionary with close price predictions
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
                "close": round(predictions['close'][i], 2),
                "pct_change": round(((predictions['close'][i] - metadata['last_close']) / metadata['last_close']) * 100, 2),
                "direction": "UP" if predictions['close'][i] > metadata['last_close'] else "DOWN"
            }
            forecasts.append(forecast)
        
        prediction_data = {
            "prediction_timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "predictions": [
                {
                    "name": f"{metadata['symbol']} Daily Close Prediction",
                    "description": f"Daily close price prediction for {metadata['symbol']} using Facebook Prophet with technical indicators",
                    "horizon": len(future_dates),
                    "frequency": 1,
                    "stock_name": metadata['symbol'],
                    "forecasts": forecasts
                }
            ],
            "metadata": {
                "model_type": "Prophet",
                "symbol": metadata['symbol'],
                "training_days": metadata['training_days'],
                "last_training_date": metadata['last_training_date'],
                "prediction_horizon": len(future_dates),
                "data_source": "Interactive Brokers API"
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
    
    def predict_single_stock(self, symbol):
        """
        Predict OHLC for a single stock
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with prediction data or None if failed
        """
        try:
            logger.info(f"Predicting {symbol}...")
            
            # Fetch historical data
            bars = self.fetch_daily_historical_data(symbol, self.training_days)
            
            if not bars:
                logger.warning(f"No data for {symbol}")
                return None
            
            # Build DataFrame
            df = self.build_daily_dataframe(bars)
            
            if df.empty or len(df) < 20:  # Reduced minimum for 30-day training
                logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                return None
            
            # Generate predictions
            predictions, future_dates = self.predict_daily_close(df, self.prediction_horizon)
            
            # Prepare metadata
            metadata = {
                'symbol': symbol,
                'last_close': df['close'].iloc[-1],
                'training_days': len(df),
                'last_training_date': df['date'].iloc[-1].strftime("%Y-%m-%d")
            }
            
            # Format as JSON
            prediction_data = self.format_predictions_json(predictions, future_dates, metadata)
            
            logger.info(f"[SUCCESS] {symbol} prediction complete: ${metadata['last_close']:.2f} -> ${predictions['close'][0]:.2f}")
            
            return prediction_data
            
        except Exception as e:
            logger.error(f"Error predicting {symbol}: {e}")
            return None

    def run_prediction(self, max_stocks=None):
        """
        Main method to run the daily prediction process for all S&P 500 stocks
        
        Args:
            max_stocks: Maximum number of stocks to predict (None for all)
        """
        try:
            logger.info("Starting S&P 500 daily prediction process for all stocks")
            
            # Connect to TWS first
            if not self.connect_to_tws():
                logger.error("Failed to connect to TWS. Please ensure TWS/IB Gateway is running.")
                return None
            
            # Get S&P 500 tickers
            self.sp500_tickers = self.get_sp500_tickers()
            
            if not self.sp500_tickers:
                logger.error("Failed to fetch S&P 500 tickers")
                return None
            
            # Limit number of stocks if specified
            if max_stocks:
                self.sp500_tickers = self.sp500_tickers[:max_stocks]
                logger.info(f"Limited to first {max_stocks} stocks")
            
            logger.info(f"Predicting OHLC for {len(self.sp500_tickers)} S&P 500 stocks")
            
            # Store all predictions
            all_predictions = []
            successful_predictions = 0
            failed_predictions = 0
            
            # Predict each stock
            for i, symbol in enumerate(self.sp500_tickers, 1):
                logger.info(f"Processing {symbol} ({i}/{len(self.sp500_tickers)})")
                
                prediction = self.predict_single_stock(symbol)
                
                if prediction:
                    all_predictions.append(prediction)
                    successful_predictions += 1
                else:
                    failed_predictions += 1
                
                # Add small delay to avoid overwhelming the API
                time.sleep(0.1)
            
            # Combine all predictions into a single JSON structure
            combined_predictions = {
                "prediction_timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "total_stocks": len(self.sp500_tickers),
                "successful_predictions": successful_predictions,
                "failed_predictions": failed_predictions,
                "predictions": all_predictions,
                "metadata": {
                    "model_type": "Prophet",
                    "prediction_horizon": self.prediction_horizon,
                    "training_days": self.training_days,
                    "data_source": "Interactive Brokers API"
                }
            }
            
            # Save combined predictions
            filepath = self.save_predictions(combined_predictions, "sp500_all_stocks_predictions.json")
            
            # Print summary
            logger.info("=" * 60)
            logger.info("PREDICTION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total stocks processed: {len(self.sp500_tickers)}")
            logger.info(f"Successful predictions: {successful_predictions}")
            logger.info(f"Failed predictions: {failed_predictions}")
            logger.info(f"Success rate: {(successful_predictions/len(self.sp500_tickers)*100):.1f}%")
            logger.info(f"Results saved to: {filepath}")
            
            # Disconnect from TWS
            self.disconnect_from_tws()
            
            return combined_predictions
            
        except Exception as e:
            logger.error(f"Error in prediction process: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Disconnect from TWS on error
            self.disconnect_from_tws()
            return None

def main():
    """
    Main function to run the S&P 500 daily predictor for all stocks
    """
    print("S&P 500 Daily Close Price Predictor - All Stocks")
    print("Training: Last 30 days | Prediction: Next 1 day")
    print("=" * 60)
    
    # Initialize predictor
    predictor = SP500DailyPredictor()
    
    # Ask user for number of stocks to process (for testing)
    try:
        max_stocks_input = input("Enter max number of stocks to process (or press Enter for all): ").strip()
        max_stocks = int(max_stocks_input) if max_stocks_input else None
    except ValueError:
        max_stocks = None
    
    # Run prediction
    result = predictor.run_prediction(max_stocks)
    
    if result:
        print("\n🎉 Prediction completed successfully!")
        print(f"Results saved to: {predictor.output_dir}")
        print(f"Total stocks processed: {result['total_stocks']}")
        print(f"Successful predictions: {result['successful_predictions']}")
        print(f"Failed predictions: {result['failed_predictions']}")
        
        # Print sample predictions
        if result['predictions']:
            print("\nSample predictions (first 3 stocks):")
            for i, stock_prediction in enumerate(result['predictions'][:3]):
                symbol = stock_prediction['metadata']['symbol']
                forecast = stock_prediction['predictions'][0]['forecasts'][0]
                print(f"{symbol}: ${forecast['close']:.2f} ({forecast['pct_change']:+.2f}%)")
    else:
        print("❌ Prediction failed. Check logs for details.")

if __name__ == "__main__":
    main()
