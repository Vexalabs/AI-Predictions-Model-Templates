from ib_insync import *
import pandas as pd
import numpy as np
from prophet import Prophet
import os
from datetime import datetime, timedelta
import pytz
import signal
import sys
import time
from tqdm import tqdm
import json
from pathlib import Path
import requests
from functools import lru_cache
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
import threading
import time
from ibapi.contract import Contract
from ibapi.common import BarData
from typing import List
from ibapi.common import ListOfHistoricalTick, ListOfHistoricalTickLast
from sklearn.linear_model import LinearRegression


class Logger(object):
    def __init__(self, log_dir):
        self.terminal = sys.stdout
        # Create filename with current date and time
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output_logs directory inside log_dir if it doesn't exist
        output_logs_dir = os.path.join(log_dir, 'output_logs')
        if not os.path.exists(output_logs_dir):
            os.makedirs(output_logs_dir)
            
        # Store log file in output_logs directory
        log_file = os.path.join(output_logs_dir, f"trading_output_{current_time}.txt")
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate writing to file

    def flush(self):
        # needed for python 3 compatibility
        pass

# Add this right after your imports, before the main trading logic
def initialize_logging():
    # Use the same directory as your CSV files
    log_dir = "trade_logs"  # Main directory for all logs
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Create output_logs directory inside trade_logs
    output_logs_dir = os.path.join(log_dir, 'output_logs')
    if not os.path.exists(output_logs_dir):
        os.makedirs(output_logs_dir)
        
    sys.stdout = Logger(log_dir)

class IBApiClient(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.nextOrderId = None
        self.market_data = {}     # symbol -> {'bid': x, 'ask': y, 'last': z}
        self.ticker_id_map = {}   # ticker_id -> symbol
        self.req_id_counter = 1000
        self.exit_orders = set()
        self.historical_data_store = {}
        self.historical_data_ready = threading.Event()
        self.historical_ticks_store = {}
        self.historical_ticks_ready = {}
        self.account_summary_data = []
        self.account_summary_ready = threading.Event()
        self.positions_list = []
        self.positions_ready = threading.Event()

        # Add trade tracking
        self.trades_store = {}  # orderId -> trade_info
        self.trades_list = []   # List of all trades for compatibility
        self.trades_ready = threading.Event()

        self.connect_to_tws()
        if not self.isConnected():
            raise ConnectionError("Failed to connect to IBKR")
        print("Successfully connected to IBKR")
        

        # Start exit order monitoring
        self.monitor_exit_orders()

        self.symbols = self.get_sp500_tickers()  # Get all S&P 500 symbols
        self.profit_targets = [0.01]  # Single profit target of 1%
        self.positions = {}  # Keep this for tracking active positions
        self.trades_df = pd.DataFrame()
        self.active_tickers = {}  # Add this for tracking market data subscriptions
        self.market_data_subscriptions = {}  # Add this for tracking subscriptions
        
        # Cache configuration
        self.cache = {
            'hourly': {},    # For 1-hour resolution data
            'daily': {},     # For daily resolution data
            'market': {},    # For real-time market data (short cache)
            'delta': {}      # For cumulative delta data
        }
        
        self.cache_durations = {
            'hourly': 300,   # 5 minutes
            'daily': 1800,   # 30 minutes
            'market': 10,    # 10 seconds
            'delta': 300     # 5 minutes for cumulative delta (increased from 60 seconds)
        }
        
        self.debug = False  # Add debug attribute
        
        # SMA Filter Configuration
        self.sma_period = 50  # 50-period SMA
        self.sma_margin_threshold = 5.0  # Current price must be within ±5% of SMA50 (hard gate)
        # ADJUSTMENT GUIDE:
        # - More strict: Use 3.0% or 2.0% (fewer trades, tighter range around SMA)
        # - More lenient: Use 7.0% or 10.0% (more trades, wider range around SMA)
        # - Current setting: 5.0% (balanced hard gate approach)
        self.sma_duration = '60 D'  # Get 60 days of data to ensure 50-period SMA calculation
    
    

    # Let's inject manual OCO handling outside `place_order` into your class.
    # We'll add a function `monitor_exit_orders` that runs in the background and cancels the opposite order when one is filled.
    # Then, we'll slightly modify `place_order` to register orders with a tracking system and return their metadata.

    # Insert this method into the LiveTrading class:
    def monitor_exit_orders(self):
        """Monitor positions since bracket orders handle TP/SL automatically"""
        # This will be handled by the orderStatus and execDetails callbacks
        # No need for ib_insync event handlers
        pass

    def sync_positions(self):
        """Synchronize local positions dictionary with IB positions"""
        try:
            # FIX: Use get_positions() instead of self.positions()
            ib_positions = self.get_positions()
            
            # Remove positions that no longer exist in IB
            for symbol in list(self.positions.keys()):
                if symbol not in ib_positions:
                    print(f"Position {symbol} no longer in IB, removing from local tracking")
                    self.positions.pop(symbol, None)
            
            # Update position sizes from IB
            for symbol, pos in self.positions.items():
                if symbol in ib_positions:
                    pos['current_shares'] = ib_positions[symbol]['position']
                    
            # Log any discrepancies
            for symbol, ib_pos in ib_positions.items():
                if symbol not in self.positions:
                    print(f"Warning: Found IB position for {symbol} not in local tracking")
                    
        except Exception as e:
            print(f"Error syncing positions: {e}")

    def calculate_current_profit(self, symbol, position):
        """Calculate current profit percentage for a position"""
        try:
            # FIX: Use get_positions() instead of self.positions()
            ib_positions = self.get_positions()
            ib_position = ib_positions.get(symbol)
            if not ib_position:
                return 0.0
                
            contract = self.get_contract(symbol)
            ticker = self.reqMktData_ibapi(contract)
            time.sleep(0.1)
            
            # Ensure we have numeric values
            entry_price = float(position.get('entry_price', 0))
            current_price = float(ticker['last']) if ticker['last'] else 0.0
            
            if current_price > 0 and entry_price > 0:
                return ((current_price - entry_price) / entry_price) * 100
            return 0.0
        except Exception as e:
            print(f"Error calculating profit for {symbol}: {e}")
            return 0.0

    def process_exit(self, symbol, exit_price, exit_type):
        """Enhanced process_exit method to ensure proper CSV logging"""
        try:
            if symbol not in self.positions:
                print(f"Warning: No position data found for {symbol}")
                return

            position = self.positions[symbol]
            entry_price = position.get('entry_price', 0)
            shares = position.get('shares', 0)
            
            # Calculate P&L
            pnl = (exit_price - entry_price) * shares
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price else 0
            
            print(f"\nLogging exit for {symbol}:")
            print(f"Exit Type: {exit_type}")
            print(f"Exit Price: ${exit_price:.2f}")
            print(f"P/L: ${pnl:.2f} ({pnl_pct:.2f}%)")

            # Update the trade log CSV
            today = datetime.now().strftime('%Y%m%d')
            filename = f"trade_logs/trades_{today}.csv"
            
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                # Find the matching entry row that hasn't been closed
                mask = (df['Symbol'] == symbol) & (df['Exit_Time'].isna())
                if len(df[mask]) > 0:
                    idx = df[mask].index[0]
                    df.loc[idx, 'Exit_Time'] = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S')
                    df.loc[idx, 'Exit_Price'] = exit_price
                    df.loc[idx, 'Exit_Value'] = exit_price * shares
                    df.loc[idx, 'P/L'] = pnl
                    df.loc[idx, 'P/L %'] = pnl_pct
                    df.loc[idx, 'Exit_Reason'] = exit_type
                    df.to_csv(filename, index=False)
                    print(f"Successfully updated exit information in {filename}")
                else:
                    print(f"Warning: No matching entry found for {symbol} in {filename}")
            else:
                print(f"Warning: Trade log file {filename} not found")

            # Clean up position
            self.positions.pop(symbol, None)
            
        except Exception as e:
            print(f"Error processing exit for {symbol}: {e}")
            import traceback
            print(traceback.format_exc())

    def connect_to_tws(self, host='127.0.0.1', port=7497, client_id=None, max_attempts=3):
        """Connect to IBKR TWS using native ibapi with retry logic and timeout"""
        if client_id is None:
            # Generate unique client ID based on current time
            client_id = int(time.time()) % 1000
        
        for attempt in range(max_attempts):
            try:
                print(f"\nAttempting to connect to IBKR (Attempt {attempt + 1}/{max_attempts})")
                print(f"Using client ID: {client_id}")
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
                        print("[TWS] Successfully connected to TWS")
                        return True
                    time.sleep(1)

                # Timeout handling
                print("[TWS] Connection timeout. Disconnecting and retrying...")
                self.disconnect()
                time.sleep(2)

            except Exception as e:
                print(f"[TWS] Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2)

        print("[TWS] Failed to connect after multiple attempts")
        return False
            
    def get_contract(self, symbol: str) -> Contract:
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract
        
    def get_cached_historical_data(self, symbol, resolution, duration_str, end_time=None):
        """
        Get historical data with enhanced caching, retry mechanism, and rate limiting
        """
        # Rate limiting - add delay between requests
        if hasattr(self, '_last_request_time'):
            time_since_last = time.time() - self._last_request_time
            if time_since_last < 0.5:  # 500ms minimum between requests
                sleep_time = 0.5 - time_since_last
                print(f"Rate limiting: waiting {sleep_time:.2f}s before next request")
                time.sleep(sleep_time)
        
        self._last_request_time = time.time()
        
        cache = self.cache[resolution]
        cache_duration = self.cache_durations[resolution]
        cache_key = f"{symbol}_{duration_str}_{end_time if end_time else 'current'}"
        current_time = time.time()

        if cache_key in cache:
            timestamp, data = cache[cache_key]
            if current_time - timestamp <= cache_duration:
                print(f"Using cached {resolution} data for {symbol}")
                return data

        # Enhanced retry mechanism
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                print(f"Fetching historical data for {symbol} (attempt {attempt + 1}/{max_retries})")
                
                # Reset store and event
                self.historical_data_store[symbol] = []
                self.historical_data_ready.clear()
                self._active_symbol = symbol

                # Format end time in UTC format
                if end_time:
                    utc_time = end_time.astimezone(pytz.UTC)
                    formatted_end_time = utc_time.strftime('%Y%m%d-%H:%M:%S')
                else:
                    utc_now = datetime.now(pytz.UTC)
                    formatted_end_time = utc_now.strftime('%Y%m%d-%H:%M:%S')

                # Build request
                contract = self.get_contract(symbol)
                req_id = self.get_new_req_id()
                
                # Use proper bar size for hourly data
                if resolution == "daily":
                    bar_size = "1 day"
                else:  # hourly
                    bar_size = "1 hour"
                    # Request more days to ensure we get at least 30 hourly bars
                    # Account for weekends and holidays by requesting 8 days
                    duration_str = "8 D" 

                try:
                    self.reqHistoricalData(
                        reqId=req_id,
                        contract=contract,
                        endDateTime=formatted_end_time,
                        durationStr=duration_str,
                        barSizeSetting=bar_size,
                        whatToShow="TRADES",
                        useRTH=1,  # Regular trading hours only
                        formatDate=1,
                        keepUpToDate=False,
                        chartOptions=[]
                    )
                    print("Historical data request sent successfully")
                except Exception as e:
                    print(f"Error sending historical data request: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        return []

                print("Waiting for historical data response...")
                
                try:
                    if self.historical_data_ready.wait(timeout=20):  # Increased timeout
                        bars = self.historical_data_store.get(symbol, [])
                        if bars:
                            print(f"Historical data response received for {symbol}: {len(bars)} bars")
                            cache[cache_key] = (current_time, bars)
                            return bars
                        else:
                            print(f"[WARNING] No bars received for {symbol} (attempt {attempt + 1}/{max_retries})")
                    else:
                        print(f"[WARNING] Timeout fetching historical data for {symbol} (attempt {attempt + 1}/{max_retries})")
                        
                except Exception as e:
                    print(f"[ERROR] Exception waiting for historical data for {symbol}: {e}")
                
                # Add delay between retries
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    
            except Exception as e:
                print(f"[ERROR] Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
        
        print(f"[ERROR] All attempts failed for {symbol}")
        return []

    def historicalData(self, reqId: int, bar: BarData):
        """Handle historical data bars"""
        #print(f"historicalData callback called for reqId: {reqId}")
        if hasattr(self, '_active_symbol'):
            symbol = self._active_symbol
            if symbol not in self.historical_data_store:
                self.historical_data_store[symbol] = []
            self.historical_data_store[symbol].append(bar)
            #print(f"Received bar for {symbol}: {bar.date}")

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """Called when historical data request is complete"""
        #print(f"historicalDataEnd callback called for reqId: {reqId}")
        #print(f"[{self._active_symbol}] Historical data fetch complete. Start: {start}, End: {end}")
        self.historical_data_ready.set()
        #print("Historical data ready event set")
    
    def build_dataframe_from_bars(self, symbol):
        bars = self.historical_data_store.get(symbol, [])
        if not bars:
            return pd.DataFrame()

        # Filter to only include hourly bars (minute = 0)
        hourly_bars = []
        for b in bars:
            # Parse the date string (format: '20250627  19:00:00')
            date_str = b.date.strip()
            if '  ' in date_str:  # Handle double spaces
                date_str = date_str.replace('  ', ' ')
            
            try:
                # Parse the time and check if it's on the hour (minute = 0)
                bar_time = datetime.strptime(date_str, '%Y%m%d %H:%M:%S')
                if bar_time.minute == 0:  # Only include bars that start on the hour
                    # Convert to Eastern time
                    utc_time = pytz.UTC.localize(bar_time)
                    eastern_time = utc_time.astimezone(pytz.timezone('US/Eastern'))
                    hourly_bars.append({
                        'date': eastern_time,
                        'open': b.open,
                        'high': b.high,
                        'low': b.low,
                        'close': b.close,
                        'volume': b.volume
                    })
            except Exception as e:
                print(f"Error parsing bar time: {e}")
                continue

        if not hourly_bars:
            print(f"No hourly bars found for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(hourly_bars)
        
        # Sort by date to ensure chronological order
        df = df.sort_values('date').reset_index(drop=True)
        
        # Remove duplicates if any
        df = df.drop_duplicates(subset=['date']).reset_index(drop=True)
        
        print(f"Built DataFrame with {len(df)} hourly bars for {symbol}")
        
        # Debug: Show date range
        if len(df) > 0:
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df


    def fetch_rolling_hourly_data(self, symbol):
        """Fetch last 30 hours of data with caching"""
        try:
            end_time = datetime.now(pytz.timezone('US/Eastern'))
            bars = self.get_cached_historical_data(
                symbol=symbol,
                resolution='hourly',
                duration_str='8 D',
                end_time=end_time
            )
            
            if not bars:
                print(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Debug print
            #print(f"\nProcessing data for {symbol}:")
            #print(f"Number of bars received: {len(bars)}")
            
            # Process data with explicit error checking
            try:
                df = self.build_dataframe_from_bars(symbol)
                
                # Debug print
                print(f"DataFrame created with shape: {df.shape}")
                print(f"Columns: {df.columns.tolist()}")
                
                # Verify we have data before taking tail
                if len(df) == 0:
                    print(f"Empty DataFrame for {symbol}")
                    return pd.DataFrame()
                
                # Ensure we have at least 30 bars for training
                if len(df) < 30:
                    print(f"WARNING: Only {len(df)} bars available for {symbol}, need 30 for proper training")
                    print(f"Requesting more historical data...")
                    
                    # Try requesting more data
                    bars = self.get_cached_historical_data(
                        symbol=symbol,
                        resolution='hourly',
                        duration_str='15 D',  # Try 15 days
                        end_time=end_time
                    )
                    
                    if bars:
                        df = self.build_dataframe_from_bars(symbol)
                        if len(df) < 30:
                            print(f"ERROR: Still only {len(df)} bars available for {symbol}. Skipping.")
                            return pd.DataFrame()
                
                # Take last 30 hours and calculate indicators
                df = df.tail(30)
                print(df)
                
                # Debug print
                print(f"After tail(30) shape: {df.shape}")
                
                # Calculate fast stochastic
                df['fast_stoc'] = self.calculate_fast_stochastic(df)
                
                return df
                
            except Exception as e:
                print(f"Error processing DataFrame for {symbol}: {e}")
                import traceback
                print(traceback.format_exc())
                return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching hourly data for {symbol}: {e}")
            import traceback
            print(traceback.format_exc())
            return pd.DataFrame()

    def get_fresh_market_data(self, symbol):
        """Get fresh real-time market data (bid/ask prices) - NO CACHING for frequent data"""
        try:
            MAX_SUBSCRIPTIONS = 100  # IBKR's limit
            
            # Check if we're near the subscription limit
            if len(self.active_tickers) >= MAX_SUBSCRIPTIONS - 5:  # Buffer of 5
                print(f"Approaching subscription limit. Cleaning up...")
                self.cleanup_market_data()
            
            # Get fresh market data (no caching)
            contract = self.get_contract(symbol)
            if not contract:
                print(f"Could not create contract for {symbol}")
                return None
            
            ticker = self.reqMktData_ibapi(contract)
            
            if ticker:
                # Track subscription (but don't cache the data)
                self.active_tickers[symbol] = ticker
                return ticker
            else:
                print(f"No market data received for {symbol}")
                return None
            
        except Exception as e:
            print(f"Error getting fresh market data for {symbol}: {e}")
            # Return a fallback structure
            return {
                'bid': None, 'ask': None, 'last': None, 'close': None,
                'bidSize': 0, 'askSize': 0, 'lastSize': 0
            }

    def get_cached_market_data(self, symbol):
        """Get real-time market data with subscription management - CACHED for less frequent data"""
        try:
            MAX_SUBSCRIPTIONS = 100  # IBKR's limit
            
            # Check if we're near the subscription limit
            if len(self.active_tickers) >= MAX_SUBSCRIPTIONS - 5:  # Buffer of 5
                print(f"Approaching subscription limit. Cleaning up...")
                self.cleanup_market_data()
            
            # Initialize cache if not exists
            if not hasattr(self, 'cache') or 'market' not in self.cache:
                self.cache = {
                    'hourly': {},    # For 1-hour resolution data
                    'daily': {},     # For daily resolution data
                    'market': {},    # For real-time market data (short cache)
                    'delta': {}      # For cumulative delta data
                }
                self.cache_durations = {
                    'hourly': 300,   # 5 minutes
                    'daily': 1800,   # 30 minutes
                    'market': 10,    # 10 seconds
                    'delta': 300     # 5 minutes for cumulative delta
                }
            
            cache = self.cache['market']
            cache_duration = self.cache_durations['market']
            current_time = time.time()
            
            # Return cached data if valid
            if symbol in cache:
                timestamp, data = cache[symbol]
                if current_time - timestamp <= cache_duration:
                    return data
            
            # Get fresh market data
            contract = self.get_contract(symbol)
            if not contract:
                print(f"Could not create contract for {symbol}")
                return None
            
            ticker = self.reqMktData_ibapi(contract)
            
            if ticker:
                # Track subscription
                self.active_tickers[symbol] = ticker
                
                # Cache the data
                cache[symbol] = (current_time, ticker)
                
                return ticker
            else:
                print(f"No market data received for {symbol}")
                return None
            
        except Exception as e:
            print(f"Error getting market data for {symbol}: {e}")
            # Return a fallback structure
            return {
                'bid': None, 'ask': None, 'last': None, 'close': None,
                'bidSize': 0, 'askSize': 0, 'lastSize': 0
            }

    def calculate_indicators(self, df):
        """Calculate fast stochastic and custom factors"""
        # Fast Stochastic
        k_period = 14
        d_period = 3
        
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        df['fast_stoc'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        
        # Custom Factors (using predicted values for current bar)
        df['alpha'] = (df['high'] - df['open']) / df['open']
        df['beta'] = (df['close'] - df['low']) / df['low']
        df['gamma'] = (df['close'] - df['open']) / df['open']
        df['delta'] = df['volume'].pct_change()
        
        return df
        
    def prepare_prophet_data(self, df):
        """Prepare data for Prophet model"""
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = df['date']
        prophet_df['y'] = df['close']
        return prophet_df
        
    def predict_next_hour(self, df, target_time):
        """Generate OHLCV predictions for next hour using Prophet"""
        try:
            predictions = {}
            metrics = ['open', 'high', 'low', 'close', 'volume']
            
            # Create a copy of the dataframe to avoid modifying original
            df_copy = df.copy()
            
            # Convert timezone-aware dates to timezone-naive
            df_copy['date'] = df_copy['date'].dt.tz_localize(None)
            target_prediction_time = pd.to_datetime(target_time).tz_localize(None)
            
            print("\nProphet Model Analysis:")
            print("-" * 40)
            print(f"Current Trading Day: {target_prediction_time.date()}")
            print(f"Training Window:")
            print(f"  First Hour: {df_copy['date'].iloc[0]}")
            print(f"  Last Hour: {df_copy['date'].iloc[-1]}")
            print(f"Target Prediction: {target_prediction_time}")
            
            # Store last hour's actual values for comparison
            last_hour = {
                'open': df_copy['open'].iloc[-1],
                'high': df_copy['high'].iloc[-1],
                'low': df_copy['low'].iloc[-1],
                'close': df_copy['close'].iloc[-1],
                'volume': df_copy['volume'].iloc[-1]
            }
            
            for metric in metrics:
                # Prepare data for Prophet
                prophet_df = pd.DataFrame({
                    'ds': df_copy['date'],  # Now using timezone-naive dates
                    'y': df_copy[metric]
                })
                
                # Add time features
                prophet_df['hour'] = prophet_df['ds'].dt.hour
                prophet_df['minute'] = prophet_df['ds'].dt.minute
                
                # Initialize Prophet model
                model = Prophet(
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    changepoint_prior_scale=0.01
                )
                
                model.add_regressor('hour')
                model.add_regressor('minute')
                model.fit(prophet_df)
                
                # Prepare future dataframe
                future = pd.DataFrame({'ds': [target_prediction_time]})
                future['hour'] = future['ds'].dt.hour
                future['minute'] = future['ds'].dt.minute
                
                # Generate forecast
                forecast = model.predict(future)
                predicted_value = forecast.iloc[-1]['yhat']
                
                # Apply bounds
                last_value = df_copy[metric].iloc[-1]
                max_change = 0.05
                min_value = last_value * (1 - max_change)
                max_value = last_value * (1 + max_change)
                predictions[metric] = np.clip(predicted_value, min_value, max_value)
            
            # Print comparison metrics
            print("\nMetrics Comparison:")
            print("-" * 40)
            print(f"{'Metric':<10} {'Last Hour':<15} {'Prediction':<15} {'Change %':<10}")
            print("-" * 40)
            for metric in metrics:
                last_val = last_hour[metric]
                pred_val = predictions[metric]
                change_pct = ((pred_val - last_val) / last_val) * 100
                
                if metric == 'volume':
                    print(f"{metric:<10} {last_val:,.0f} {pred_val:,.0f} {change_pct:>9.2f}%")
                else:
                    print(f"{metric:<10} ${last_val:<14.2f} ${pred_val:<14.2f} {change_pct:>9.2f}%")
            
            return predictions
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

    def predict_next_6_hours(self, df, target_time):
        """Generate OHLCV predictions for next 6 hours using Prophet"""
        try:
            predictions = {}
            metrics = ['open', 'high', 'low', 'close', 'volume']
            
            # Create a copy of the dataframe to avoid modifying original
            df_copy = df.copy()
            
            # Convert timezone-aware dates to timezone-naive
            df_copy['date'] = df_copy['date'].dt.tz_localize(None)
            target_prediction_time = pd.to_datetime(target_time).tz_localize(None)
            
            print("\n6-Hour Prophet Model Analysis:")
            print("-" * 40)
            print(f"Current Trading Day: {target_prediction_time.date()}")
            print(f"Training Window:")
            print(f"  First Hour: {df_copy['date'].iloc[0]}")
            print(f"  Last Hour: {df_copy['date'].iloc[-1]}")
            print(f"Target Prediction Start: {target_prediction_time}")
            
            # Store last hour's actual values for comparison
            last_hour = {
                'open': df_copy['open'].iloc[-1],
                'high': df_copy['high'].iloc[-1],
                'low': df_copy['low'].iloc[-1],
                'close': df_copy['close'].iloc[-1],
                'volume': df_copy['volume'].iloc[-1]
            }
            
            # Generate 6 future timestamps (hourly)
            future_times = []
            for i in range(6):
                future_time = target_prediction_time + pd.Timedelta(hours=i+1)
                future_times.append(future_time)
            
            # Initialize predictions structure
            for metric in metrics:
                predictions[metric] = []
            
            # Generate predictions for each of the 6 hours
            for hour_idx, future_time in enumerate(future_times):
                print(f"\nPredicting Hour {hour_idx + 1}: {future_time}")
                
                for metric in metrics:
                    # Prepare data for Prophet
                    prophet_df = pd.DataFrame({
                        'ds': df_copy['date'],
                        'y': df_copy[metric]
                    })
                    
                    # Add time features
                    prophet_df['hour'] = prophet_df['ds'].dt.hour
                    prophet_df['minute'] = prophet_df['ds'].dt.minute
                    
                    # Initialize Prophet model
                    model = Prophet(
                        daily_seasonality=True,
                        weekly_seasonality=True,
                        yearly_seasonality=False,
                        changepoint_prior_scale=0.01
                    )
                    
                    model.add_regressor('hour')
                    model.add_regressor('minute')
                    model.fit(prophet_df)
                    
                    # Prepare future dataframe
                    future = pd.DataFrame({'ds': [future_time]})
                    future['hour'] = future['ds'].dt.hour
                    future['minute'] = future['ds'].dt.minute
                    
                    # Generate forecast
                    forecast = model.predict(future)
                    predicted_value = forecast.iloc[-1]['yhat']
                    
                    # Apply bounds (more conservative for longer predictions)
                    last_value = df_copy[metric].iloc[-1]
                    # Increase max change for longer predictions (hourly decay)
                    max_change = 0.05 * (1 + hour_idx * 0.1)  # 5%, 5.5%, 6%, 6.5%, 7%, 7.5%
                    min_value = last_value * (1 - max_change)
                    max_value = last_value * (1 + max_change)
                    predicted_value = np.clip(predicted_value, min_value, max_value)
                    
                    predictions[metric].append(predicted_value)
            
            # Add the 6-hour price array for trend analysis
            predictions['next_6h_prices'] = predictions['close']
            
            # Print 6-hour summary
            print("\n6-Hour Predictions Summary:")
            print("-" * 40)
            print(f"{'Hour':<6} {'Open':<12} {'High':<12} {'Low':<12} {'Close':<12} {'Volume':<12}")
            print("-" * 40)
            
            for hour_idx in range(6):
                open_val = predictions['open'][hour_idx]
                high_val = predictions['high'][hour_idx]
                low_val = predictions['low'][hour_idx]
                close_val = predictions['close'][hour_idx]
                volume_val = predictions['volume'][hour_idx]
                
                print(f"{hour_idx+1:<6} ${open_val:<11.2f} ${high_val:<11.2f} ${low_val:<11.2f} ${close_val:<11.2f} {volume_val:>11,.0f}")
            
            # Calculate trend metrics
            current_price = last_hour['close']
            price_6h_ahead = predictions['close'][5]  # 6th hour
            price_change_6h = price_6h_ahead - current_price
            price_change_percent_6h = (price_change_6h / current_price) * 100
            
            print(f"\nTrend Analysis:")
            print(f"  Current Price: ${current_price:.2f}")
            print(f"  6-Hour Ahead Price: ${price_6h_ahead:.2f}")
            print(f"  6-Hour Price Change: ${price_change_6h:.2f} ({price_change_percent_6h:.2f}%)")
            
            return predictions
            
        except Exception as e:
            print(f"Error in 6-hour prediction: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

    def calculate_fast_stochastic(self, df, period=14):
        """Calculate Fast Stochastic with data validation
        
        Args:
            df: DataFrame containing OHLC data
            period: Stochastic period (default 14)
        Returns:
            Series containing Fast Stochastic values or None if validation fails
        """
        try:
            # Check if we have enough data points
            if len(df) < period:
                print(f"Warning: Insufficient data for Fast Stochastic calculation. Need {period} points, got {len(df)}")
                return None
            
            # Check for NaN values in required columns
            if df[['low', 'high', 'close']].isnull().any().any():
                print("Warning: NaN values detected in price data")
                return None
            
            # Calculate Fast Stochastic if data is valid
            lowest_low = df['low'].rolling(window=period).min()
            highest_high = df['high'].rolling(window=period).max()
            fast_stoc = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
            
            # Fill any remaining NaN values that might occur at the start of the series
            #fast_stoc = fast_stoc.fillna(method='bfill')
            fast_stoc = fast_stoc.bfill()
            
            return fast_stoc
            
        except Exception as e:
            print(f"Error calculating Fast Stochastic: {e}")
            return None

    def calculate_predicted_stochastic(self, predictions, df, period=14):
        """Calculate Fast Stochastic for predicted values using last 13 actual + 1 predicted point
        
        Args:
            predictions: Dictionary containing predicted OHLCV values (6-hour format)
            df: Historical DataFrame for last 13 points
            period: Stochastic period (default 14)
        Returns:
            float: Fast Stochastic value for predicted bar (clamped to 0-100)
        """
        try:
            # Validate input data
            if len(df) < period - 1:
                print(f"Warning: Insufficient historical data for predicted stochastic. Need {period-1} points, got {len(df)}")
                return None
            
            # For 6-hour predictions, we need to extract the 1-hour prediction for FS calculation
            # The predictions now contain arrays for each metric, we need the first hour (index 0)
            if isinstance(predictions['high'], list) and len(predictions['high']) > 0:
                # 6-hour prediction format - extract 1-hour prediction
                one_hour_prediction = {
                    'high': predictions['high'][0],  # First hour prediction
                    'low': predictions['low'][0],    # First hour prediction
                    'close': predictions['close'][0], # First hour prediction
                    'open': predictions['open'][0],   # First hour prediction
                    'volume': predictions['volume'][0] # First hour prediction
                }
                print(f"Using 1-hour prediction for Fast Stochastic calculation: Close=${one_hour_prediction['close']:.2f}")
            else:
                # Fallback to old format (single hour prediction)
                one_hour_prediction = predictions
                print(f"Using legacy prediction format for Fast Stochastic calculation")
            
            # Validate predicted values
            required_keys = ['high', 'low', 'close']
            if not all(key in one_hour_prediction for key in required_keys):
                print("Warning: Missing required OHLC values in predictions")
                return None
            
            # Normalize predictions to ensure they're realistic
            normalized_predictions = self.validate_and_normalize_predictions(one_hour_prediction, df)
            
            # Get the last 13 points from historical data
            last_13_high = df['high'].tail(period-1).values
            last_13_low = df['low'].tail(period-1).values
            
            # Create 14-point arrays including the normalized predicted values
            high_values = np.append(last_13_high, normalized_predictions['high'])
            low_values = np.append(last_13_low, normalized_predictions['low'])
            
            # Calculate highest high and lowest low over the 14-point window
            highest_high = np.max(high_values)
            lowest_low = np.min(low_values)
            
            # Check for division by zero
            if highest_high == lowest_low:
                print("Warning: Highest high equals lowest low, cannot calculate stochastic")
                return 50.0  # Return neutral value
            
            # Calculate Fast Stochastic using normalized predicted close
            fast_stoc = 100 * (normalized_predictions['close'] - lowest_low) / (highest_high - lowest_low)
            
            # Clamp the result to 0-100 range
            fast_stoc = np.clip(fast_stoc, 0.0, 100.0)
            
            # Debug logging (uncomment if needed)
            #print("\nPredicted Stochastic Calculation Details:")
            #print(f"Window Size: {period}")
            #print(f"1-Hour Predicted High: {one_hour_prediction['high']:.2f}")
            #print(f"1-Hour Predicted Low: {one_hour_prediction['low']:.2f}")
            #print(f"1-Hour Predicted Close: {one_hour_prediction['close']:.2f}")
            #print(f"Normalized Predicted High: {normalized_predictions['high']:.2f}")
            #print(f"Normalized Predicted Low: {normalized_predictions['low']:.2f}")
            #print(f"Normalized Predicted Close: {normalized_predictions['close']:.2f}")
            #print(f"Highest High in window: {highest_high:.2f}")
            #print(f"Lowest Low in window: {lowest_low:.2f}")
            #print(f"Resulting Fast Stochastic: {fast_stoc:.2f}")
            
            return fast_stoc
            
        except Exception as e:
            print(f"Error calculating predicted Fast Stochastic: {e}")
            return None

    def get_cached_cumulative_delta(self, symbol, lookback_minutes=60):
        """Enhanced Cumulative Delta calculation with proper tick-by-tick analysis"""
        try:
            current_time = datetime.now(pytz.timezone('US/Eastern'))
            cache_key = f"{symbol}_cd_{lookback_minutes}"
            
            # Check cache first
            if (cache_key in self.cache['delta'] and 
                (current_time - self.cache['delta'][cache_key]['timestamp']).total_seconds() < self.cache_durations['delta']):
                return self.cache['delta'][cache_key]['value']
            
            # Clean up existing subscriptions
            self.cleanup_market_data()
            
            print(f"\nCalculating enhanced Cumulative Delta for {symbol}")
            
            # Get historical trades
            trades = self.get_historical_ticks_for_cd(symbol, lookback_minutes)
            
            if not trades:
                print(f"No trades found for {symbol}")
                return None
            
            # Enhanced CD calculation with proper flow analysis
            cumulative_delta = 0
            buy_volume = 0
            sell_volume = 0
            total_volume = 0
            
            # Sort trades by timestamp for proper sequence
            sorted_trades = sorted(trades, key=lambda x: x.time if hasattr(x, 'time') else 0)
            
            if len(sorted_trades) < 2:
                print(f"Insufficient trades for CD calculation: {len(sorted_trades)}")
                return None
            
            # Calculate initial reference price (first trade)
            ref_price = sorted_trades[0].price
            
            # Process trades with enhanced flow detection
            for i, trade in enumerate(sorted_trades):
                if not hasattr(trade, 'price') or not hasattr(trade, 'size'):
                    continue
                
                price = trade.price
                size = trade.size
                total_volume += size
                
                # Enhanced direction determination
                # Method 1: Price movement relative to reference
                price_change = price - ref_price
                
                # Method 2: Tick-by-tick momentum
                if i > 0:
                    prev_price = sorted_trades[i-1].price
                    tick_momentum = price - prev_price
                else:
                    tick_momentum = 0
                
                # Method 3: Volume-weighted price analysis
                # If price is higher than reference and volume is significant, likely buying
                # If price is lower than reference and volume is significant, likely selling
                
                # Determine trade direction using multiple criteria
                is_buy = False
                
                # Primary: Price momentum
                if price_change > 0.001:  # Price up by more than 0.1%
                    is_buy = True
                elif price_change < -0.001:  # Price down by more than 0.1%
                    is_buy = False
                else:
                    # Secondary: Tick momentum
                    if tick_momentum > 0:
                        is_buy = True
                    elif tick_momentum < 0:
                        is_buy = False
                    else:
                        # Tertiary: Volume-weighted decision
                        # For small price changes, use volume as tiebreaker
                        if size > 100:  # Significant volume
                            is_buy = price > ref_price
                        else:
                            # For small volumes, assume neutral
                            continue
                
                # Update cumulative delta
                if is_buy:
                    cumulative_delta += size
                    buy_volume += size
                else:
                    cumulative_delta -= size
                    sell_volume += size
                
                # Update reference price (exponential moving average)
                ref_price = ref_price * 0.7 + price * 0.3
            
            # Calculate CD ratio for normalization
            cd_ratio = cumulative_delta / total_volume if total_volume > 0 else 0
            
            # Print enhanced analysis
            print(f"\nEnhanced Cumulative Delta Analysis for {symbol}:")
            print(f"Total Trades Analyzed: {len(sorted_trades)}")
            print(f"Total Volume: {total_volume}")
            print(f"Buy Volume: {buy_volume}")
            print(f"Sell Volume: {sell_volume}")
            print(f"Net Cumulative Delta: {cumulative_delta}")
            print(f"CD Ratio: {cd_ratio:.4f}")
            buy_sell_ratio_str = f"{buy_volume/sell_volume:.2f}" if sell_volume > 0 else "N/A"
            print(f"Buy/Sell Ratio: {buy_sell_ratio_str}")
            
            # Cache the result
            self.cache['delta'][cache_key] = {
                'timestamp': current_time,
                'value': cumulative_delta
            }
            
            print(f"Cached CD for {symbol}: {cumulative_delta}")
            
            # Clean up
            self.cleanup_market_data()
            
            return cumulative_delta
            
        except Exception as e:
            print(f"Error calculating Cumulative Delta for {symbol}: {e}")
            self.cleanup_market_data()
            return None

    def calculate_slopes(self, values, periods=3):
        """Calculate slope of recent values using linear regression"""
        try:
            if len(values) < periods:
                print(f"Not enough values for slope calculation. Need {periods}, got {len(values)}")
                return 0
            
            x = np.arange(periods)
            y = np.array(values[-periods:])
            print(f"\nSlope calculation debug:")
            print(f"X values: {x}")
            print(f"Y values: {y}")
            
            slope, _ = np.polyfit(x, y, 1)
            print(f"Calculated slope: {slope}")
            return slope
            
        except Exception as e:
            print(f"Error calculating slope: {e}")
            return 0

    def check_entry_conditions(self, df, predictions, symbol):
        """Enhanced entry conditions with 7 required filters - Fixed 2% allocation when all pass
        
        Required Filters (ALL must pass):
        1. SMA Filter: Current price within ±5% of 50-period SMA (Hard Gate)
        2. Buffer Filter: Current price above 5-day low (Price Floor Check)
        3. FS20 Filter: Current FS value 20-90 inclusive (Prevent oversold and overbought conditions)
        4. CD Signal: Cumulative Delta >= 0
        5. OFI Signal: Order Flow Imbalance >= 0.05
        6. FS Momentum Signal: Current FS > Previous FS (Upward Momentum)
        7. FS Prediction Signal: Predicted FS > Current FS (Continued Upward Momentum)
        
        Position Size: Fixed 2% allocation when all filters pass
        """
        try:
            # ===== SMA FILTER - FIRST FILTER (NEW) =====
            # Filter 1: Current price must be within ±5% of 50-period SMA (Hard Gate)
            # OPTIMIZATION: Current market price is fetched once here and reused throughout all filters
            print(f"\n[SMA FILTER - FIRST FILTER] for {symbol}:")
            
            # Get daily data for SMA calculation (need at least sma_period days)
            daily_bars = self.get_cached_historical_data(
                symbol=symbol,
                resolution='daily',
                duration_str=self.sma_duration  # Get 60 days to ensure we have 50 for SMA
            )
            
            if not daily_bars or len(daily_bars) < self.sma_period:
                print(f"[{symbol}] SMA Filter: Insufficient daily data for {self.sma_period}-period SMA calculation")
                print(f"  Required: {self.sma_period} days, Available: {len(daily_bars) if daily_bars else 0} days")
                return False, None
            
            # Calculate SMA using daily close prices
            daily_closes = [bar.close for bar in daily_bars[-self.sma_period:]]  # Last sma_period days
            sma_50 = sum(daily_closes) / len(daily_closes)
            
            # Get current market price (fetched once and reused throughout all filters for consistency)
            ticker = self.get_fresh_market_data(symbol)
            if not ticker or not ticker.get('bid') or not ticker.get('ask'):
                print(f"[{symbol}] SMA Filter: No valid market data available, using last close price as fallback")
                current_price = df['close'].iloc[-1] if not df.empty else None
                if not current_price:
                    print(f"[{symbol}] SMA Filter: No valid price data available")
                    return False, None
            else:
                current_price = ticker.get('ask') if ticker.get('ask') else ticker.get('last')
                if not current_price:
                    current_price = ticker.get('last') if ticker.get('last') else df['close'].iloc[-1]
            
            if not current_price:
                print(f"[{symbol}] SMA Filter: No valid ask price available")
                return False, None
            
            # Calculate price difference from SMA and percentage
            price_diff_from_sma = current_price - sma_50
            price_diff_percent_from_sma = (price_diff_from_sma / sma_50) * 100
            
            # Use configurable margin threshold (currently set to 5% in class initialization)
            # Hard gate: current price must be within ±5% of SMA50
            # You can adjust self.sma_margin_threshold to make filter more/less strict
            sma_filter_passed = abs(price_diff_percent_from_sma) <= self.sma_margin_threshold
            
            print(f"  {self.sma_period}-Period SMA (Daily Close): ${sma_50:.2f}")
            print(f"  Current Market Price: ${current_price:.2f}")
            print(f"  Price Difference from SMA: ${price_diff_from_sma:.2f} ({price_diff_percent_from_sma:.2f}%)")
            print(f"  SMA Margin Threshold: ±{self.sma_margin_threshold}% (hard gate around SMA)")
            print(f"  SMA Filter Passed: {sma_filter_passed}")
            
            # Apply SMA filter - FIRST FILTER (Hard Gate)
            if not sma_filter_passed:
                print(f"[{symbol}] [X] SMA FILTER FAILED: Current price (${current_price:.2f}) is outside ±{self.sma_margin_threshold}% range of SMA (${sma_50:.2f})")
                print(f"  Required: within ±{self.sma_margin_threshold}%, Actual: {price_diff_percent_from_sma:.2f}%")
                print(f"  Skipping trade - price outside SMA hard gate range")
                return False, None
            
            print(f"[{symbol}] [OK] SMA FILTER PASSED - Price {price_diff_percent_from_sma:.2f}% from SMA (within ±{self.sma_margin_threshold}% hard gate)")
            # ===== END SMA FILTER =====
            
            # ===== 6-HOUR PRICE TREND PREDICTION - SECOND FILTER (PAUSED) =====
            # 6-HOUR TREND GATE IS CURRENTLY PAUSED - NOT USED FOR ENTRY SIGNAL GENERATION
            print(f"\n[6-HOUR TREND ANALYSIS - SECOND FILTER (PAUSED)] for {symbol}:")
            print(f"  [INFO] 6-HOUR TREND GATE IS CURRENTLY PAUSED - NOT USED FOR ENTRY DECISIONS")
            
            # Set default values for 6-hour trend analysis (not used for filtering)
            price_trend_gate = True  # Always pass when paused
            price_change_6h = 0.0
            price_change_percent_6h = 0.0
            price_6h_ahead = current_price  # Use current price as placeholder
            trend_threshold = 0.5  # Keep for reference
            
            print(f"  Current Market Price: ${current_price:.2f}")
            print(f"  6-Hour Ahead Price: N/A (Gate Paused)")
            print(f"  6-Hour Price Change: N/A (Gate Paused)")
            print(f"  Trend Threshold: {trend_threshold}% (Not Applied)")
            print(f"  6-Hour Price Trend Gate: {price_trend_gate} [PAUSED]")
            print(f"  [INFO] 6-HOUR TREND GATE BYPASSED - Continuing to next filter")
            # ===== END 6-HOUR TREND GATE (PAUSED) =====
            
            # ===== BUFFER FILTER - THIRD FILTER =====
            # Filter 3: Current price should be above 5-day low (Price Floor Check)
            # This is the THIRD filter that must pass after SMA and trend gates
            
            # Get 5-day low for buffer calculation
            bars = self.get_cached_historical_data(
                symbol=symbol,
                resolution='daily',
                duration_str='5 D'
            )
            
            if not bars or len(bars) < 5:
                print(f"[{symbol}] Buffer Filter: Insufficient daily data for 5-day low calculation")
                return False, None
                
            lowest_5d_price = min(bar.low for bar in bars)
            
            # Use the current_price from SMA filter section (already fetched and validated)
            # No need to fetch market data again - reuse current_price as expected_fill
            expected_fill = current_price
            
            # Check if current price is above 5-day low (Price Floor Check)
            entry_above_5d_low = expected_fill > lowest_5d_price
            
            print(f"\n[BUFFER FILTER - THIRD FILTER] for {symbol}:")
            print(f"  5-Day Lowest Price: ${lowest_5d_price:.2f}")
            print(f"  Expected Fill Price: ${expected_fill:.2f}")
            print(f"  Entry Above 5-Day Low: {entry_above_5d_low}")
            
            # Apply buffer filter - THIRD FILTER (Price Floor Check)
            if not entry_above_5d_low:
                print(f"[{symbol}] [X] BUFFER FILTER FAILED: Entry price (${expected_fill:.2f}) is below 5-day low (${lowest_5d_price:.2f})")
                print(f"  Skipping trade - price below recent support level")
                return False, None
            
            print(f"[{symbol}] [OK] BUFFER FILTER PASSED - Entry above 5-day low (Price Floor Check)")
            # ===== END BUFFER FILTER =====
            
            # ===== FS20 FILTER - FOURTH FILTER =====
            # Filter 4: Current FS value must be 20-90 inclusive (Prevent oversold and overbought conditions)
            print(f"\n[FS20 FILTER - FOURTH FILTER] for {symbol}:")
            
            # Get current bar data for FS calculation
            if df.empty or len(df) < 1:
                print(f"[{symbol}] FS20 Filter: No historical data available")
                return False, None
            
            current_fs_for_filter = df['fast_stoc'].iloc[-1]
            fs20_filter_passed = 20 <= current_fs_for_filter <= 90
            
            print(f"  Current FS: {current_fs_for_filter:.2f} (Threshold: 20-90 inclusive)")
            print(f"  FS20 Filter: {fs20_filter_passed}")
            
            # Apply FS20 filter - FOURTH FILTER
            if not fs20_filter_passed:
                print(f"[{symbol}] [X] FS20 FILTER FAILED: Current FS ({current_fs_for_filter:.2f}) not in range 20-90 inclusive")
                return False, None
            
            print(f"[{symbol}] [OK] FS20 FILTER PASSED - Current FS ({current_fs_for_filter:.2f}) in range 20-90 inclusive")
            # ===== END FS20 FILTER =====
            
            # Get actual previous bar data (iloc[-2] = previous bar)
            previous_bar = {
                'date': df['date'].iloc[-2],
                'open': df['open'].iloc[-2],
                'high': df['high'].iloc[-2],
                'low': df['low'].iloc[-2],
                'close': df['close'].iloc[-2],
                'volume': df['volume'].iloc[-2],
                'fast_stoc': df['fast_stoc'].iloc[-2]
            }
            
            # Get current bar data (iloc[-1] = current bar)
            current_bar_actual = {
                'date': df['date'].iloc[-1],
                'open': df['open'].iloc[-1],
                'high': df['high'].iloc[-1],
                'low': df['low'].iloc[-1],
                'close': df['close'].iloc[-1],
                'volume': df['volume'].iloc[-1],
                'fast_stoc': df['fast_stoc'].iloc[-1]
            }
            
            # Calculate predicted stochastic (using only 1-hour prediction as before)
            # Note: This function should still use only 1-hour prediction for FS calculation
            predicted_stoch = self.calculate_predicted_stochastic(predictions, df)
            
            # Extract 1-hour prediction values for predicted_bar (first hour of 6-hour predictions)
            if isinstance(predictions['open'], list) and len(predictions['open']) > 0:
                # 6-hour prediction format - use first hour
                predicted_bar = {
                    'open': predictions['open'][0],    # First hour prediction
                    'high': predictions['high'][0],    # First hour prediction
                    'low': predictions['low'][0],      # First hour prediction
                    'close': predictions['close'][0],  # First hour prediction
                    'volume': predictions['volume'][0], # First hour prediction
                    'fast_stoc': predicted_stoch
                }
            else:
                # Fallback to old format (single hour prediction)
                predicted_bar = {
                    'open': predictions['open'],
                    'high': predictions['high'],
                    'low': predictions['low'],
                    'close': predictions['close'],
                    'volume': predictions['volume'],
                    'fast_stoc': predicted_stoch
                }
            
            # ===== CORE FILTERS (CD & CD SLOPE) =====
            # CD: Cumulative Delta >= 0 (Primary Filter)
            cd = self.get_cached_cumulative_delta(symbol, lookback_minutes=60)
            cd_signal = cd is not None and cd >= 0
            
            # Update CD history with the current value
            if cd is not None:
                self.update_cd_history(symbol, cd)
            
            # CD Slope: CD slope > 0 (Primary Filter)
            recent_cd_values = self.get_recent_cd_values(symbol, lookback=3)
            if len(recent_cd_values) >= 2:
                cd_slope = self.calculate_slopes(recent_cd_values, 2)
                cd_slope_signal = cd_slope > 0  # CD slope increasing
                cd_slope_available = True
            else:
                cd_slope = None
                cd_slope_signal = False  # Default to False if insufficient data
                cd_slope_available = False
            
            # ===== CONFIRMATION FILTERS (Required Filters) =====
            # OFI: Order Flow Imbalance >= 0.05
            ofi = self.calculate_ofi(symbol)
            ofi_signal = ofi is not None and ofi >= 0.05  # Require some minimum positive OFI
            
            # ===== FAST STOCHASTIC MOMENTUM FILTERS (Required Filters) =====
            # FS Momentum: Current FS > Previous FS (Upward Momentum)
            current_fs = current_bar_actual['fast_stoc']
            previous_fs = previous_bar['fast_stoc']
            fs_momentum_signal = current_fs > previous_fs  # Current FS > Previous FS
            
            # FS Prediction: Predicted FS > Current FS (Continued Upward Momentum)
            predicted_fs = predicted_bar['fast_stoc']
            fs_prediction_signal = predicted_fs > current_fs  # Predicted FS > Current FS
            
            # ===== ALL REQUIRED FILTERS LOGIC =====
            # All 7 filters must pass for entry signal:
            # 1. SMA Filter (already checked)
            # 2. Entry above 5d low (already checked) 
            # 3. FS20 Filter (already checked - range 20-90)
            # 4. CD Signal >= 0
            # 5. OFI Signal >= 0.05
            # 6. FS Momentum Signal: Current FS > Previous FS
            # 7. FS Prediction Signal: Predicted FS > Current FS
            
            all_filters_passed = cd_signal and ofi_signal and fs_momentum_signal and fs_prediction_signal
            
            if not all_filters_passed:
                print(f"[{symbol}] [X] REQUIRED FILTERS FAILED:")
                print(f"  CD Signal: {cd_signal} (CD: {cd})")
                ofi_str = f"{ofi:.3f}" if ofi is not None else "N/A"
                print(f"  OFI Signal: {ofi_signal} (OFI: {ofi_str})")
                print(f"  FS Momentum Signal: {fs_momentum_signal} (Current FS: {current_fs:.2f} > Previous FS: {previous_fs:.2f})")
                print(f"  FS Prediction Signal: {fs_prediction_signal} (Predicted FS: {predicted_fs:.2f} > Current FS: {current_fs:.2f})")
                return False, None
            
            print(f"[{symbol}] [OK] ALL REQUIRED FILTERS PASSED:")
            print(f"  CD: {cd:.2f}, OFI: {ofi:.3f}, FS Momentum: {current_fs:.2f}>{previous_fs:.2f}, FS Prediction: {predicted_fs:.2f}>{current_fs:.2f}")
            
            # ===== SIMPLIFIED POSITION SIZING LOGIC =====
            # Fixed 2% allocation when all 7 filters pass
            position_size_percent = 2.0  # Fixed 2% allocation
            position_size_reason = "ALL_FILTERS_PASSED"
            
            # ===== FINAL ENTRY SIGNAL =====
            # Entry signal is TRUE if ALL 7 filters pass:
            # 1. SMA Filter (FIRST)
            # 2. 6-Hour Trend Gate (SECOND - PAUSED - Always passes)
            # 3. Buffer Filter (THIRD)
            # 4. FS20 Filter (FOURTH)
            # 5. CD Signal >= 0 (FIFTH)
            # 6. OFI Signal >= 0.05 (SIXTH)
            # 7. FS Momentum Signal: Current FS > Previous FS (SEVENTH)
            # 8. FS Prediction Signal: Predicted FS > Current FS (EIGHTH)
            entry_signal = sma_filter_passed and price_trend_gate and entry_above_5d_low and fs20_filter_passed and all_filters_passed
            
            # ===== DETAILED ANALYSIS PRINTING =====
            print("\n" + "="*60)
            print(f"ENHANCED ALPHA EXTRACTION ANALYSIS FOR {symbol} (7 REQUIRED FILTERS - FIXED 2% ALLOCATION)")
            print("="*60)
            
            print("\n[SMA FILTER - FIRST FILTER]:")
            print(f"  {self.sma_period}-Period SMA (Daily Close): ${sma_50:.2f}")
            print(f"  Current Market Price: ${current_price:.2f}")
            print(f"  Price Difference from SMA: ${price_diff_from_sma:.2f} ({price_diff_percent_from_sma:.2f}%)")
            print(f"  SMA Margin Threshold: ±{self.sma_margin_threshold}% (hard gate around SMA)")
            print(f"  SMA Filter: {sma_filter_passed} [OK] PASSED")
            
            print("\n[6-HOUR TREND GATE - SECOND FILTER (PAUSED)]:")
            print(f"  Current Market Price (from SMA Filter): ${current_price:.2f}")
            print(f"  6-Hour Ahead Price: N/A (Gate Paused)")
            print(f"  6-Hour Price Change: N/A (Gate Paused)")
            print(f"  Trend Threshold: {trend_threshold}% (Not Applied)")
            print(f"  6-Hour Price Trend Gate: {price_trend_gate} [PAUSED]")
            
            print("\n[BUFFER FILTER - THIRD FILTER]:")
            print(f"  5-Day Lowest Price: ${lowest_5d_price:.2f}")
            print(f"  Expected Fill Price (Same as Current from SMA): ${expected_fill:.2f}")
            print(f"  Entry Above 5-Day Low: {entry_above_5d_low} [OK] PASSED")
            
            print("\n[REQUIRED FILTERS (ALL MUST PASS)]:")
            print(f"  CD Value: {cd:.2f} (Target: >= 0)")
            print(f"  CD Signal: {cd_signal}")
            print(f"  OFI Value: {ofi:.3f} (Target: >= 0.05)")
            print(f"  OFI Signal: {ofi_signal}")
            print(f"  FS Momentum: Current FS {current_fs:.2f} > Previous FS {previous_fs:.2f}: {fs_momentum_signal}")
            print(f"  FS Prediction: Predicted FS {predicted_fs:.2f} > Current FS {current_fs:.2f}: {fs_prediction_signal}")
            print(f"  All Required Filters Passed: {all_filters_passed}")
            
            print("\n[POSITION SIZING]:")
            print(f"  Position Size: {position_size_percent}% (Fixed)")
            print(f"  Position Size Reason: {position_size_reason}")
            
            print("\n[SIGNAL BREAKDOWN]:")
            print(f"  1. SMA Filter: {sma_filter_passed}")
            print(f"  2. 6-Hour Trend Gate: {price_trend_gate} [PAUSED]")
            print(f"  3. Buffer Filter: {entry_above_5d_low}")
            print(f"  4. FS20 Filter: {fs20_filter_passed}")
            print(f"  5. CD Signal: {cd_signal}")
            print(f"  6. OFI Signal: {ofi_signal}")
            print(f"  7. FS Momentum Signal: {fs_momentum_signal}")
            print(f"  8. FS Prediction Signal: {fs_prediction_signal}")
            print(f"  FINAL ENTRY SIGNAL: {entry_signal}")
            
            print("="*60)
            
            # Store comprehensive signal data for logging
            signal_data = {
                'entry_fast_stoch': current_bar_actual['fast_stoc'],
                'prev_fast_stoch': previous_bar['fast_stoc'],
                'fast_stoch_delta': current_fs - previous_fs,  # FS momentum calculation
                'cumulative_delta': cd,
                'cd_slope': cd_slope if cd_slope_available else 0,
                'ofi': ofi,
                'price_action': True,  # Simplified - not used as filter
                'price_slope': 0,  # Simplified - not used as filter
                'volume_ratio': 1,  # Simplified - not used as filter
                'all_required_filters_passed': all_filters_passed,  # Store core filters result (CD+OFI+Delta+FS)
                'entry_signal': entry_signal,  # Store final entry signal result (all filters)
                'buffer_good': entry_above_5d_low,  # Store buffer filter result
                'buffer_threshold': lowest_5d_price,  # Store 5-day low
                'expected_fill': expected_fill,  # Store expected fill price
                'lowest_5d_price': lowest_5d_price,  # Store 5-day low
                'position_size_percent': position_size_percent,  # Store position size
                'position_size_reason': position_size_reason,  # Store position size reason
                'cd_signal': cd_signal,  # Store CD signal
                'ofi_signal': ofi_signal,  # Store OFI signal
                'fs_momentum_signal': fs_momentum_signal,  # Store FS momentum signal
                'fs_prediction_signal': fs_prediction_signal,  # Store FS prediction signal
                # NEW: SMA Filter Data
                'sma_filter_passed': sma_filter_passed,  # Store SMA filter result
                # NEW: FS20 Filter Data
                'fs20_filter_passed': fs20_filter_passed,  # Store FS20 filter result
                'current_fs_for_filter': current_fs_for_filter,  # Store current FS value used in FS20 filter
                'previous_fs': previous_fs,  # Store previous FS value
                'sma_50': sma_50,  # Store 50-period SMA value
                'price_diff_from_sma': price_diff_from_sma,  # Store price difference from SMA
                'price_diff_percent_from_sma': price_diff_percent_from_sma,  # Store price difference % from SMA
                'sma_margin_threshold': self.sma_margin_threshold,  # Store SMA margin threshold
                # NEW: 6-Hour Trend Data (PAUSED)
                'price_trend_gate': price_trend_gate,  # Store 6-hour trend gate result (paused)
                'price_change_6h': price_change_6h,  # Store 6-hour price change (0.0 when paused)
                'price_change_percent_6h': price_change_percent_6h,  # Store 6-hour price change % (0.0 when paused)
                'price_6h_ahead': price_6h_ahead,  # Store 6-hour ahead price (current_price when paused)
                'trend_threshold': trend_threshold,  # Store trend threshold used (for reference only)
                'trend_gate_status': 'PAUSED'  # Indicate that trend gate is paused
            }
            
            return entry_signal, signal_data
            
        except Exception as e:
            print(f"Error checking entry conditions: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False, None
            
    def calculate_ofi(self, symbol):
        """Enhanced Order Flow Imbalance (OFI) calculation with slope analysis"""
        try:
            # Clean up existing subscriptions before requesting new data
            self.cleanup_market_data()
            
            contract = self.get_contract(symbol)
            ticker = self.reqMktData_ibapi(contract)
            time.sleep(2)  # Wait for data to arrive
            
            if not ticker or not ticker.get('bid') or not ticker.get('ask'):
                print(f"No market data available for {symbol}")
                self.cleanup_market_data()
                return None
            
            # Enhanced OFI calculation
            bid_size = ticker.get('bidSize', 0)
            ask_size = ticker.get('askSize', 0)
            last_size = ticker.get('lastSize', 0)
            last_price = ticker.get('last', 0)
            bid_price = ticker.get('bid', 0)
            ask_price = ticker.get('ask', 0)
            
            # Enhanced last trade pressure analysis
            if last_size > 0 and last_price > 0:
                # More sophisticated pressure determination
                mid_price = (bid_price + ask_price) / 2 if bid_price > 0 and ask_price > 0 else last_price
                
                if last_price >= ask_price:
                    bid_size += last_size  # Strong buying pressure
                elif last_price <= bid_price:
                    ask_size += last_size  # Strong selling pressure
                elif last_price > mid_price:
                    bid_size += last_size * 0.5  # Moderate buying pressure
                    ask_size += last_size * 0.5
                else:
                    ask_size += last_size * 0.5  # Moderate selling pressure
                    bid_size += last_size * 0.5

            # Calculate OFI ratio (-1.0 to +1.0)
            total_size = bid_size + ask_size
            if total_size > 0:
                ofi_ratio = (bid_size - ask_size) / total_size
            else:
                ofi_ratio = 0

            # Enhanced OFI analysis with slope considerations
            ofi_signal_strength = "Neutral"
            if ofi_ratio > 0.1:
                ofi_signal_strength = "Strong Buy"
            elif ofi_ratio > 0.05:
                ofi_signal_strength = "Moderate Buy"
            elif ofi_ratio > 0.01:
                ofi_signal_strength = "Weak Buy"
            elif ofi_ratio < -0.1:
                ofi_signal_strength = "Strong Sell"
            elif ofi_ratio < -0.05:
                ofi_signal_strength = "Moderate Sell"
            elif ofi_ratio < -0.01:
                ofi_signal_strength = "Weak Sell"

            print(f"\nEnhanced OFI Analysis for {symbol}:")
            print(f"Bid Size: {bid_size}")
            print(f"Ask Size: {ask_size}")
            print(f"Last Size: {last_size}")
            print(f"Last Price: ${last_price:.2f}")
            print(f"Bid Price: ${bid_price:.2f}")
            print(f"Ask Price: ${ask_price:.2f}")
            print(f"OFI Ratio: {ofi_ratio:.4f}")
            print(f"Signal Strength: {ofi_signal_strength}")
            print(f"Total Size: {total_size}")
            bid_ask_ratio_str = f"{bid_size/ask_size:.2f}" if ask_size > 0 else "N/A"
            print(f"Bid/Ask Ratio: {bid_ask_ratio_str}")

            # Clean up before returning
            self.cleanup_market_data()
            
            return ofi_ratio

        except Exception as e:
            print(f"Error calculating OFI for {symbol}: {e}")
            self.cleanup_market_data()
            return None
        
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range (ATR)"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            prev_close = close.shift(1)
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = tr.rolling(window=period).mean()
            
            print(f"\nATR Calculation Details:")
            print(f"Period: {period}")
            print(f"Latest ATR: {atr.iloc[-1]:.2f}")
            print(f"ATR Range: {atr.min():.2f} - {atr.max():.2f}")
            
            return atr
            
        except Exception as e:
            print(f"Error calculating ATR: {e}")
            return None
    
    def place_order(self, contract):
        """Place bracket order with ATR-based or fallback percentage-based TP/SL"""
        try:
            # FIX: Use get_positions() method instead of self.positions()
            current_positions = self.get_positions()
            if len(current_positions) >= 100:
                print(f"[{contract.symbol}] Skipping: Max 100 positions reached (Current: {len(current_positions)})")
                return None

            # FIX: Use reqAccountSummary instead of accountSummary
            self.account_summary_data = []
            self.account_summary_ready = threading.Event()
            self.reqAccountSummary(1, "All", "NetLiquidation,AvailableFunds,BuyingPower,GrossPositionValue,TotalCashValue,RealizedPnL,UnrealizedPnL")
            
            if not self.account_summary_ready.wait(timeout=10):
                print("Timeout waiting for account summary")
                return None
            
            # Extract account values
            total_portfolio_value = 0.0
            available_funds = 0.0
            for item in self.account_summary_data:
                if item['tag'] == 'NetLiquidation':
                    total_portfolio_value = float(item['value'])
                elif item['tag'] == 'AvailableFunds':
                    available_funds = float(item['value'])
            
            # Get position size from check_entry_conditions (if available)
            position_size_percent = 2.0  # Default to 2% if not available
            if contract.symbol in self.positions and 'position_size_percent' in self.positions[contract.symbol]:
                position_size_percent = self.positions[contract.symbol]['position_size_percent']
                position_size_reason = self.positions[contract.symbol].get('position_size_reason', 'CORE_ONLY')
                print(f"Using position size from check_entry_conditions: {position_size_percent}% ({position_size_reason})")
            else:
                print(f"Using default position size: {position_size_percent}% (CORE_ONLY)")
            
            max_position_value = total_portfolio_value * (position_size_percent / 100)
            
            print(f"Total Portfolio Value: ${total_portfolio_value:,.2f}")
            print(f"Available Funds: ${available_funds:,.2f}")
            print(f"Max Position Value ({position_size_percent}%): ${max_position_value:,.2f}")

            if max_position_value > available_funds:
                print(f"\nInsufficient funds for {contract.symbol}:")
                print(f"Required Funds: ${max_position_value:,.2f}")
                print(f"Available Funds: ${available_funds:,.2f}")
                return None

            # Get market data (FRESH DATA - NO CACHING for bid/ask prices)
            ticker = self.get_fresh_market_data(contract.symbol)
            if not ticker or not ticker.get('bid') or not ticker.get('ask'):
                print(f"[{contract.symbol}] No valid market data available, using last close price as fallback")
                # Try to get historical data for fallback price
                try:
                    hist_data = self.get_cached_historical_data(contract.symbol, '1 hour', '1 D')
                    if hist_data and len(hist_data) > 0:
                        expected_fill = hist_data[-1].close
                        print(f"[{contract.symbol}] Using historical close price: ${expected_fill:.2f}")
                    else:
                        print(f"[{contract.symbol}] No historical data available for fallback")
                        return None
                except Exception as e:
                    print(f"[{contract.symbol}] Error getting fallback price: {e}")
                    return None
            else:
                expected_fill = ticker.get('ask') if ticker.get('ask') else ticker.get('last')
                if not expected_fill:
                    expected_fill = ticker.get('last') if ticker.get('last') else self.df['close'].iloc[-1]

            if not expected_fill:
                print(f"[{contract.symbol}] No valid ask price available")
                return None

            print(f"\nEntry Analysis for {contract.symbol}:")
            if ticker and ticker.get('ask'):
                print(f"Ask Price: ${ticker.get('ask'):.2f}")
            if ticker and ticker.get('last'):
                print(f"Last Price: ${ticker.get('last'):.2f}")
            print(f"Expected Fill: ${expected_fill:.2f}")
            
            # Note: Buffer filter is now handled in check_entry_conditions function as the first filter

            # COMMENTED OUT: Upside potential filter - 6-hour trend gate is currently paused in check_entry_conditions
            # if contract.symbol in self.positions and 'predictions' in self.positions[contract.symbol]:
            #     predicted_high = self.positions[contract.symbol]['predictions']['high']
            #     upside_percent = ((predicted_high - expected_fill) / expected_fill) * 100

            #     print(f"\nUpside Analysis for {contract.symbol}:")
            #     print(f"Expected Fill: ${expected_fill:.2f}")
            #     print(f"Predicted High: ${predicted_high:.2f}")
            #     print(f"Potential Upside: {upside_percent:.2f}%")

            #     if upside_percent <= 0.5:
            #         print(f"[{contract.symbol}] Skipping: Insufficient upside potential ({upside_percent:.2f}% < 0.7%)")
            #         return None
            #     else:
            #         print(f"Sufficient upside potential detected: {upside_percent:.2f}%")

            # Calculate shares
            shares = int(round(max_position_value / expected_fill, 0))
            MAX_ORDER_SIZE = 500
            shares = min(shares, MAX_ORDER_SIZE)

            # Validate expected fill
            if expected_fill is None or expected_fill == 0 or np.isnan(expected_fill):
                print(f"[{contract.symbol}] Invalid fill price: {expected_fill}. Skipping.")
                return None

            # Calculate TP/SL
            tp_price = round(expected_fill * 1.013, 2)  # 1.3%
            sl_price = round(expected_fill * 0.991, 2)  # -0.9%
            tp_percent = 1.3
            sl_percent = -0.9
            
            print(f"Take Profit: ${tp_price:.2f}")
            print(f"Stop Loss: ${sl_price:.2f}")

            if np.isnan(tp_price) or np.isnan(sl_price):
                print(f"[{contract.symbol}] Invalid TP/SL prices calculated. Skipping.")
                return None

            # Convert to ibapi order placement logic
            if all(not np.isnan(x) for x in [expected_fill, tp_price, sl_price]):
                # Create orders using ibapi
                from ibapi.order import Order
                
                parent_order_id = self.get_new_req_id()
                take_profit_order_id = parent_order_id + 1
                stop_loss_order_id = parent_order_id + 2
                
                # Create orders using ibapi - simplified like test script
                from ibapi.order import Order
                
                parent = Order()
                parent.orderId = parent_order_id
                parent.action = "BUY"
                parent.totalQuantity = shares
                parent.orderType = "MKT"
                parent.transmit = False
                parent.tif = "GTC"
                parent.eTradeOnly = False
                parent.firmQuoteOnly = False

                takeProfit = Order()
                takeProfit.orderId = take_profit_order_id
                takeProfit.action = "SELL"
                takeProfit.totalQuantity = shares
                takeProfit.orderType = "LMT"
                takeProfit.lmtPrice = tp_price
                takeProfit.transmit = False
                takeProfit.tif = "GTC"
                takeProfit.parentId = parent_order_id
                takeProfit.eTradeOnly = False
                takeProfit.firmQuoteOnly = False

                stopLoss = Order()
                stopLoss.orderId = stop_loss_order_id
                stopLoss.action = "SELL"
                stopLoss.totalQuantity = shares
                stopLoss.orderType = "STP"
                stopLoss.auxPrice = sl_price
                stopLoss.tif = "GTC"
                stopLoss.transmit = True
                stopLoss.parentId = parent_order_id
                stopLoss.eTradeOnly = False
                stopLoss.firmQuoteOnly = False

                # Initialize order status tracking
                self.pending_orders = {parent_order_id, take_profit_order_id, stop_loss_order_id}
                self.accepted_orders = set()
                
                print(f"\nSubmitting orders for {contract.symbol}:")
                print(f"Parent Order ID: {parent_order_id}")
                print(f"TP Order ID: {take_profit_order_id}")
                print(f"SL Order ID: {stop_loss_order_id}")
                
                # Place all orders
                self.placeOrder(parent_order_id, contract, parent)
                self.placeOrder(take_profit_order_id, contract, takeProfit)
                self.placeOrder(stop_loss_order_id, contract, stopLoss)

                # Add orders to trade tracking system
                temp_status = self.create_order_status_object(
                    status='Submitted',
                    filled=0,
                    remaining=shares,
                    avgFillPrice=0.0
                )
                
                self.add_trade(parent_order_id, contract, parent, temp_status)
                self.add_trade(take_profit_order_id, contract, takeProfit, temp_status)
                self.add_trade(stop_loss_order_id, contract, stopLoss, temp_status)

                # Wait for order status updates (max 2 seconds)
                timeout = 2
                start_time = time.time()
                while len(self.accepted_orders) < 3 and (time.time() - start_time) < timeout:
                    time.sleep(0.1)
                
                # Enhanced order submission verification
                order_ids = [parent_order_id, take_profit_order_id, stop_loss_order_id]
                submission_success, verified_orders = self.verify_order_submission(contract.symbol, order_ids, timeout=3)
                
                # Check if all orders were accepted
                if submission_success:
                    print(f"[{contract.symbol}] All orders accepted successfully!")
                    
                    # Store position data
                    if contract.symbol not in self.positions:
                        self.positions[contract.symbol] = {}
                        
                    self.positions[contract.symbol].update({
                        'entry_time': datetime.now(pytz.timezone('US/Eastern')),
                        'entry_price': expected_fill,
                        'shares': shares,
                        'parent_orderId': parent_order_id,
                        'tp_orderId': take_profit_order_id,
                        'sl_orderId': stop_loss_order_id,
                        'tp_percent': tp_percent,
                        'sl_percent': sl_percent,
                        'buffer_good': self.positions[contract.symbol].get('buffer_good', True),  # Get from check_entry_conditions
                        'buffer_threshold': self.positions[contract.symbol].get('buffer_threshold', 0),  # Get from check_entry_conditions
                        'expected_fill': self.positions[contract.symbol].get('expected_fill', expected_fill),  # Get from check_entry_conditions
                        'lowest_5d_price': self.positions[contract.symbol].get('lowest_5d_price', 0)  # Get from check_entry_conditions
                    })

                    # Log the trade
                    self.log_trade(contract.symbol, 'ENTRY', None)
                    return parent_order_id
                else:
                    print(f"[{contract.symbol}] Order submission verification failed!")
                    # Print order health status for debugging
                    health_status = self.check_order_health(contract.symbol)
                    print(f"Order health status: {health_status}")
                    return None
            else:
                print(f"[{contract.symbol}] Final validation failed. Skipping order placement.")
                return None

        except Exception as e:
            print(f"[ERROR] Failed placing order for {contract.symbol}: {e}")
            import traceback
            print(traceback.format_exc())
            return None
      
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
                        print(f"\nSuccessfully fetched {len(tickers)} tickers from SlickCharts")
                        return tickers
            except Exception as e:
                print(f"SlickCharts fetch failed: {e}")

            # Fallback to Wikipedia with headers
            print("\nFalling back to Wikipedia...")
            response = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers)
            if response.status_code == 200:
                # Fix the pandas warning by using StringIO
                from io import StringIO
                tables = pd.read_html(StringIO(response.text))
                tickers = tables[0]['Symbol'].tolist()
                print(f"Successfully fetched {len(tickers)} tickers from Wikipedia")
                return tickers
            
            raise Exception(f"Failed to fetch data: Status code {response.status_code}")

        except Exception as e:
            print(f"Error fetching S&P 500 tickers: {e}")
            return []
    '''
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
                    tables = pd.read_html(response.text)
                    if tables:
                        df = tables[0]
                        # Clean and extract symbols
                        tickers = df['Symbol'].str.replace('.B', '').tolist()
                        # Remove any duplicates while preserving order
                        tickers = list(dict.fromkeys(tickers))
                        print(f"\nSuccessfully fetched {len(tickers)} tickers from SlickCharts")
                        return tickers
            except Exception as e:
                print(f"SlickCharts fetch failed: {e}")

            # Fallback to Wikipedia with headers
            print("\nFalling back to Wikipedia...")
            response = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers)
            if response.status_code == 200:
                tables = pd.read_html(response.text)
                tickers = tables[0]['Symbol'].tolist()
                print(f"Successfully fetched {len(tickers)} tickers from Wikipedia")
                return tickers
            
            raise Exception(f"Failed to fetch data: Status code {response.status_code}")

        except Exception as e:
            print(f"Error fetching S&P 500 tickers: {e}")
            return []
        '''
    def initialize_trading(self):
        """Initialize trading parameters and get S&P 500 tickers"""
        self.symbols = self.get_sp500_tickers()
        print(f"\nLoaded {len(self.symbols)} S&P 500 symbols")
        
        # Define symbols to exclude from analysis
        self.excluded_symbols = {'BRK'}  # Add more symbols here as needed
        print(f"Excluded symbols: {self.excluded_symbols}")
        
        # Filter out excluded symbols
        original_count = len(self.symbols)
        self.symbols = [symbol for symbol in self.symbols if symbol not in self.excluded_symbols]
        print(f"Filtered out {original_count - len(self.symbols)} excluded symbols")
        print(f"Final symbol count: {len(self.symbols)}")
        
        # Start with empty positions (no historical loading)
        self.positions = {}
        print("Starting with empty positions (no historical loading)")
        
        # Initialize cache
        self.cache = {
            'hourly': {},
            'daily': {},
            'delta': {}
        }
        
        # Cache durations (in seconds)
        self.cache_durations = {
            'hourly': 300,  # 5 minutes
            'daily': 3600,  # 1 hour
            'delta': 300    # 5 minutes
        }
        
        # Initialize historical data storage
        self.historical_data_store = {}
        self.historical_data_ready = threading.Event()
        
        # Initialize historical ticks storage
        self.historical_ticks_store = {}
        self.historical_ticks_ready = {}
        
        print("Trading environment initialized successfully")
    
    # Historical position loading removed - starting fresh each time

    def is_market_hours(self, current_time):
        """Check if current time is within market hours (9:30 AM - 4:00 PM ET)"""
        try:
            # Convert time to US/Eastern timezone if it's not already
            if current_time.tzinfo != pytz.timezone('US/Eastern'):
                current_time = current_time.astimezone(pytz.timezone('US/Eastern'))
            
            # Check if it's a weekday
            if current_time.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                print(f"Market closed: Weekend ({current_time.strftime('%A')})")
                return False
            
            # Define market hours (9:30 AM - 4:00 PM ET)
            market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            
            # Check if within market hours
            is_market_open = market_open <= current_time <= market_close
            
            if not is_market_open:
                current_time_str = current_time.strftime('%H:%M:%S ET')
                if current_time < market_open:
                    print(f"Market not yet open. Current time: {current_time_str}")
                    minutes_to_open = int((market_open - current_time).total_seconds() / 60)
                    print(f"Market opens in {minutes_to_open} minutes")
                else:
                    print(f"Market closed. Current time: {current_time_str}")
                    
                return False
            
            return True
            
        except Exception as e:
            print(f"Error checking market hours: {e}")
            return False

    def run_live(self):
        """Main live trading loop for all S&P 500 stocks"""
        try:
            initialize_logging()  # Initialize logging at the start
            self.initialize_trading()
            
            # Start fresh with empty positions
            previous_positions = {}
            last_processed_time = None
            last_cache_cleanup = time.time()
            last_market_data_cleanup = time.time()
            
            # Enhanced data fetching with subscription management
            symbols_per_batch = 50  # Process symbols in batches
            current_batch = 0
            subscription_cleanup_counter = 0
            
            while True:
                market_time = datetime.now(pytz.timezone('US/Eastern'))
                
                # Cleanup cache and market data subscriptions every minute
                try:
                    if time.time() - last_cache_cleanup > 60:
                        self.cleanup_cache()
                        self.cleanup_market_data()
                        self.clear_old_trades()  # Clean up old trades
                        last_cache_cleanup = time.time()
                        last_market_data_cleanup = time.time()
                        
                        # No historical position loading - keeping current positions
                except TypeError as e:
                    print(f"TypeError in cache cleanup comparison: {e}")
                
                # Only trade during market hours
                if not self.is_market_hours(market_time):
                    print("\nOutside market hours, waiting...")
                    print(f"Current time: {market_time.strftime('%Y-%m-%d %H:%M:%S')} ET")
                    time.sleep(60)
                    continue
                
                # Get a test symbol to verify data availability
                test_symbol = self.symbols[0]
                test_df = self.fetch_rolling_hourly_data(test_symbol)
                
                if not test_df.empty:
                    # Print all today's data points
                    today = market_time.date()
                    today_data = test_df[pd.to_datetime(test_df['date']).dt.date == today]
                    
                    print("\nToday's data points for", test_symbol)
                    print("=" * 50)
                    for idx, row in today_data.iterrows():
                        data_time = pd.to_datetime(row['date'])
                        print(f"Time: {data_time.strftime('%H:%M')}, Open: {row['open']:.2f}, Close: {row['close']:.2f}")
                    print("=" * 50)
                    
                    latest_data_time = pd.to_datetime(test_df['date'].iloc[-1])
                    current_time = datetime.now(pytz.timezone('US/Eastern'))
                    
                    print(f"\nCurrent time: {current_time.strftime('%H:%M')}")
                    print(f"Latest data time: {latest_data_time.strftime('%H:%M')}")
                    
                    # SMART CHECK: Look for market open data (9:30 AM) or accept current data if market is open
                    today_data_times = pd.to_datetime(today_data['date'])
                    
                    if len(today_data_times) > 0:
                        # Check if we have 9:30 AM data point
                        market_open_today = None
                        for data_time in today_data_times:
                            if data_time.hour == 9 and data_time.minute == 30:
                                market_open_today = data_time
                                break
                        
                        if market_open_today is None:
                            # If no 9:30 AM data, check if market is currently open
                            if self.is_market_hours(current_time):
                                print("No 9:30 AM data found, but market is open. Proceeding with current data.")
                            else:
                                print("9:30 AM data point not found and market is closed, waiting...")
                                time.sleep(60)
                                continue
                        else:
                            print(f"Found 9:30 AM data point: {market_open_today.strftime('%H:%M')}")
                        
                        # Check hourly intervals (look for consistent 1-hour gaps)
                        if len(today_data_times) > 1:
                            time_diffs = [(today_data_times.iloc[i] - today_data_times.iloc[i-1]).total_seconds()/3600 
                                        for i in range(1, len(today_data_times))]
                            
                            print(f"Time differences between data points: {time_diffs}")
                            
                            # Check if differences are exactly 1 hour (strict check for hourly data)
                            if not all(diff == 1.0 for diff in time_diffs):
                                print("Data points do not have consistent hourly intervals (need exactly 1 hour)")
                                print(f"Expected: 1.0 hours, Got: {time_diffs}")
                                time.sleep(60)
                                continue
                        
                        print("Valid hourly intervals found - proceeding with analysis")
                    
                    # If we're here, we have valid hourly data to process
                    target_time = latest_data_time
                    
                    # Check if we have already processed this data point
                    if last_processed_time and target_time <= last_processed_time:
                        print(f"Already processed {target_time.strftime('%H:%M')} data, waiting for next hour...")
                        time.sleep(60)
                        continue
                    
                    # Enhanced data fetching with batching and subscription management
                    symbols_per_batch = 50  # Process symbols in batches
                    batch_delay = 5  # seconds between batches
                    symbol_delay = 1  # seconds between symbols within a batch
                    
                    # Calculate total number of batches needed
                    total_batches = (len(self.symbols) + symbols_per_batch - 1) // symbols_per_batch
                    
                    # Calculate batch range
                    start_idx = current_batch * symbols_per_batch
                    end_idx = min(start_idx + symbols_per_batch, len(self.symbols))
                    current_batch_symbols = self.symbols[start_idx:end_idx]
                    
                    print(f"\nProcessing batch {current_batch + 1}/{total_batches}")
                    print(f"Symbols {start_idx + 1}-{end_idx} of {len(self.symbols)}")
                    print(f"Processing {len(current_batch_symbols)} symbols with {symbol_delay}s delay between each...")
                    
                    processed_count = 0
                    for symbol in current_batch_symbols:
                        try:
                            # Add delay between symbols to prevent API overload
                            if processed_count > 0:
                                print(f"Waiting {symbol_delay}s before next symbol...")
                                time.sleep(symbol_delay)
                            
                            processed_count += 1
                            print(f"\nProcessing symbol {start_idx + processed_count}/{len(self.symbols)}: {symbol}")
                            
                            # Check if we already have a position in this symbol
                            if symbol in self.positions:
                                print(f"[{symbol}] Already have position, skipping...")
                                continue
                            
                            # STEP 2: Get hourly data for detailed analysis
                            try:
                                df = self.fetch_rolling_hourly_data(symbol)
                                if df.empty:
                                    print(f"[{symbol}] No data available, skipping...")
                                    continue
                            except Exception as e:
                                if "No security definition" in str(e) or "Error 200" in str(e):
                                    print(f"[{symbol}] Security definition not found, skipping...")
                                    continue
                                else:
                                    print(f"[{symbol}] Error fetching data: {e}")
                                    continue
                            
                            # Verify we have the latest data
                            latest_data_time = pd.to_datetime(df['date'].iloc[-1])
                            if latest_data_time < target_time:
                                print(f"[{symbol}] Skipping: Latest data ({latest_data_time.strftime('%H:%M')}) not current")
                                continue
                            
                            # STEP 3: Generate 6-hour predictions for trend analysis
                            next_target = target_time + timedelta(hours=1)
                            predictions = self.predict_next_6_hours(df, next_target)
                            if not predictions:
                                print(f"[{symbol}] No 6-hour predictions available, skipping...")
                                continue
                            
                            # Now receiving both signal and stoch_data
                            signal, stoch_data = self.check_entry_conditions(df, predictions, symbol)
                            
                            if signal and stoch_data:
                                print(f"\n[+] Entry signal generated for {symbol}")
                                print("Stoch data received:", stoch_data)  # Debug print
                                
                                contract = self.get_contract(symbol)
                                
                                if 2==2: #shares > 0:
                                    # Store both predictions and enhanced signal data
                                    self.positions[symbol] = {
                                        'predictions': predictions,
                                        # Core Fast Stochastic Data
                                        'entry_fast_stoch': stoch_data.get('entry_fast_stoch', 0),
                                        'prev_fast_stoch': stoch_data.get('prev_fast_stoch', 0),
                                        'fast_stoch_delta': stoch_data.get('fast_stoch_delta', 0),
                                        
                                        # Enhanced Flow Analysis Data
                                        'cumulative_delta': stoch_data.get('cumulative_delta', 0),
                                        'cd_slope': stoch_data.get('cd_slope', 0),
                                        'ofi': stoch_data.get('ofi', 0),
                                        
                                        # Beta Analysis Data
                                        'beta': stoch_data.get('beta', False),
                                        
                                        # Price Action Data
                                        'price_action': stoch_data.get('price_action', True),
                                        'price_slope': stoch_data.get('price_slope', 0),
                                        'volume_ratio': stoch_data.get('volume_ratio', 1),
                                        
                                        # Conviction Overlay Data
                                        'core_flow_signals': stoch_data.get('all_required_filters_passed', False),
                                        'conviction_signals': stoch_data.get('conviction_signals', True),
                                        'technical_signals': stoch_data.get('technical_signals', False),
                                        
                                        # Buffer Filter Data (NEW - First Filter)
                                        'buffer_good': stoch_data.get('buffer_good', False),
                                        'buffer_threshold': stoch_data.get('buffer_threshold', 0),
                                        'expected_fill': stoch_data.get('expected_fill', 0),
                                        'lowest_5d_price': stoch_data.get('lowest_5d_price', 0),
                                        
                                        # Position Sizing Data (NEW)
                                        'position_size_percent': stoch_data.get('position_size_percent', 2.5),
                                        'position_size_reason': stoch_data.get('position_size_reason', 'CORE_ONLY'),
                                        'confirmation_filters_count': stoch_data.get('confirmation_filters_count', 0),
                                        
                                        # Core Filter Data (NEW)
                                        'cd_signal': stoch_data.get('cd_signal', False),
                                        'cd_slope_signal': stoch_data.get('cd_slope_signal', False),
                                        'ofi_signal': stoch_data.get('ofi_signal', False),
                                        'fs_momentum_signal': stoch_data.get('fs_momentum_signal', False),
                                        'fs_prediction_signal': stoch_data.get('fs_prediction_signal', False),
                                        
                                        # SMA Filter Data (NEW)
                                        'sma_filter_passed': stoch_data.get('sma_filter_passed', False),
                                        'sma_50': stoch_data.get('sma_50', 0),
                                        'price_diff_from_sma': stoch_data.get('price_diff_from_sma', 0),
                                        'price_diff_percent_from_sma': stoch_data.get('price_diff_percent_from_sma', 0),
                                        'sma_margin_threshold': stoch_data.get('sma_margin_threshold', 5.0),
                                        
                                        # FS20 Filter Data (NEW)
                                        'fs20_filter_passed': stoch_data.get('fs20_filter_passed', False),
                                        'current_fs_for_filter': stoch_data.get('current_fs_for_filter', 0),
                                        
                                        # 6-Hour Trend Data (PAUSED)
                                        'price_trend_gate': stoch_data.get('price_trend_gate', True),
                                        'price_change_6h': stoch_data.get('price_change_6h', 0.0),
                                        'price_change_percent_6h': stoch_data.get('price_change_percent_6h', 0.0),
                                        'price_6h_ahead': stoch_data.get('price_6h_ahead', 0),
                                        'trend_threshold': stoch_data.get('trend_threshold', 0.5),
                                        'trend_gate_status': stoch_data.get('trend_gate_status', 'PAUSED'),
                                        
                                        # Trade Management
                                        'tp_percent': 1.3,  # Store TP percentage
                                        'sl_percent': -0.9   # Store SL percentage
                                    }
                                    print(f"Stored position data for {symbol}:", self.positions[symbol])  # Debug print
                                    trade = self.place_order(contract)
                                    
                                    if trade:
                                        self.send_notification(f"Entered {symbol} position at market")
                            else:
                                print(f"[-] No signal generated for {symbol}")
                        except Exception as e:
                            print(f"[ERROR] Failed to process {symbol}: {e}")
                            continue
                    
                    # Batch completed - move to next batch or reset
                    current_batch += 1
                    if current_batch >= total_batches:
                        print(f"\nCompleted all {len(self.symbols)} symbols in {total_batches} batches. Checking for new hour data...")
                        print(f"Current positions: {len(self.positions)}")
                        
                        # Check if we have new hour data available
                        test_df = self.fetch_rolling_hourly_data(test_symbol)
                        if not test_df.empty:
                            latest_data_time = pd.to_datetime(test_df['date'].iloc[-1])
                            if latest_data_time > target_time:
                                print(f"New hour data available: {latest_data_time.strftime('%H:%M')} (previous: {target_time.strftime('%H:%M')})")
                                target_time = latest_data_time
                                current_batch = 0  # Reset for next cycle
                                last_processed_time = target_time
                                print(f"Reset batch counter to 0 for new hour data")
                            else:
                                print(f"No new hour data available. Latest: {latest_data_time.strftime('%H:%M')}, Target: {target_time.strftime('%H:%M')}")
                                print("Waiting for next hour data...")
                                time.sleep(60)
                                continue
                        else:
                            print("No test data available, waiting...")
                            time.sleep(60)
                            continue
                    else:
                        print(f"\nBatch {current_batch}/{total_batches} completed. Waiting {batch_delay}s before next batch...")
                        print(f"Progress: {end_idx}/{len(self.symbols)} symbols processed")
                        print(f"Current positions: {len(self.positions)}")
                        time.sleep(batch_delay)
                    
                    # Clean up subscriptions periodically
                    subscription_cleanup_counter += 1
                    if subscription_cleanup_counter >= 10:  # Every 10 batches
                        print("Cleaning up market data subscriptions...")
                        self.cleanup_market_data()
                        subscription_cleanup_counter = 0
                    
                    # Log account summary at end of scan
                    self.log_account_summary()
                
                # Check for new data every minute
                time.sleep(60)
                
        except Exception as e:
            print(f"Error in live trading: {e}")
            self.disconnect()  # Changed from self.ib.disconnect()

    def send_notification(self, message):
        """Send notification (implement your preferred notification method)"""
        print(f"\nNOTIFICATION: {message}")
        # Add your notification method here (email, SMS, etc.)

    def log_trade(self, symbol, action, trade=None):
        """Log trade details to single daily CSV file"""
        try:
            # Keep all existing debug prints
            print("\nDebug - Position data for logging:")
            print(f"Symbol: {symbol}")
            print(f"Position data: {self.positions[symbol]}")
            print(f"Stoch values:")
            print(f"  Current: {self.positions[symbol].get('entry_fast_stoch')}")
            print(f"  Previous: {self.positions[symbol].get('prev_fast_stoch')}")
            print(f"  Delta: {self.positions[symbol].get('fast_stoch_delta')}")
            print(f"Volume/Delta values:")
            print(f"  Cumulative Delta: {self.positions[symbol].get('cumulative_delta', 0)}")
            print(f"  ATR: {self.positions[symbol].get('atr', None)}")

            # Keep existing directory creation and filename
            os.makedirs('trade_logs', exist_ok=True)
            today = datetime.now().strftime('%Y%m%d')
            filename = f"trade_logs/trades_{today}.csv"
            print(f"\nAttempting to log trade to: {filename}")
            
            if action == 'ENTRY':
                # Get the actual fill price from the trade object
                filled_price = None
                if trade and isinstance(trade, dict):
                    filled_price = trade.get('avgFillPrice')
                elif trade and hasattr(trade, 'orderStatus') and trade.orderStatus.avgFillPrice:
                    filled_price = float(trade.orderStatus.avgFillPrice)
                
                # If no fill price from trade, try to get from positions dictionary
                if not filled_price or filled_price <= 0:
                    filled_price = self.positions[symbol].get('entry_price')
                    
                # If still no valid price, try to get current market price
                if not filled_price or filled_price <= 0:
                    current_price = self.get_current_price(symbol)
                    if current_price and current_price > 0:
                        filled_price = current_price
                
                print(f"Entry Price being logged: ${filled_price:.2f}")
                
                if not filled_price or filled_price <= 0:
                    print(f"[ERROR] No valid entry price found for {symbol}")
                    #return
                # Add type conversion for numeric values only
                #filled_price = float(trade.orderStatus.avgFillPrice if trade else self.positions[symbol]['entry_price'])
                # Use stored shares from position data
                shares = int(self.positions[symbol].get('shares', 0))
                
                print(f"Entry trade details for {symbol}:")
                print(f"Filled Price: ${filled_price}")
                print(f"Shares: {shares}")
                
                # Enhanced trade_data structure with all new fields
                trade_data = {
                    'Timestamp': [datetime.now(pytz.timezone('US/Eastern'))],
                    'Symbol': [symbol],
                    'Entry_Time': [datetime.now(pytz.timezone('US/Eastern'))],
                    'Entry_Price': [filled_price],
                    'Shares': [shares],
                    'Entry_Value': [filled_price * shares],
                    'Parent_OrderId': [self.positions[symbol].get('parent_orderId', 0)],
                    'TP_OrderId': [self.positions[symbol].get('tp_orderId', 0)],
                    'SL_OrderId': [self.positions[symbol].get('sl_orderId', 0)],
                    
                    # Core Fast Stochastic Data
                    'Current_Fast_Stoch': [self.positions[symbol].get('entry_fast_stoch', 0)],
                    'Previous_Fast_Stoch': [self.positions[symbol].get('prev_fast_stoch', 0)],
                    'Fast_Stoch_Delta': [self.positions[symbol].get('fast_stoch_delta', 0)],
                    
                    # Enhanced Flow Analysis Data
                    'Cumulative_Delta': [self.positions[symbol].get('cumulative_delta', 0)],
                    'CD_Slope': [self.positions[symbol].get('cd_slope', 0)],
                    'OFI': [self.positions[symbol].get('ofi', 0)],
                    
                    # Beta Analysis Data
                    'Beta': [self.positions[symbol].get('beta', False)],
                    
                    # Price Action Data
                    'Price_Action': [self.positions[symbol].get('price_action', False)],
                    'Price_Slope': [self.positions[symbol].get('price_slope', 0)],
                    'Volume_Ratio': [self.positions[symbol].get('volume_ratio', 1)],
                    
                    # Conviction Overlay Data
                    'Core_Flow_Signals': [self.positions[symbol].get('all_required_filters_passed', False)],
                    'Conviction_Signals': [self.positions[symbol].get('conviction_signals', False)],
                    'Technical_Signals': [self.positions[symbol].get('technical_signals', False)],
                    
                    # Legacy Fields
                    'ATR': [self.positions[symbol].get('atr', None)],
                    'Predicted_Open': [self.positions[symbol].get('predictions', {}).get('open', 0)],
                    'Predicted_High': [self.positions[symbol].get('predictions', {}).get('high', 0)],
                    'Predicted_Low': [self.positions[symbol].get('predictions', {}).get('low', 0)],
                    'Predicted_Close': [self.positions[symbol].get('predictions', {}).get('close', 0)],
                    'Predicted_Volume': [self.positions[symbol].get('predictions', {}).get('volume', 0)],
                    
                    # Buffer Filter Data (NEW - First Filter)
                    'Buffer_Good': [self.positions[symbol].get('buffer_good', False)],
                    'Buffer_Threshold': [self.positions[symbol].get('buffer_threshold', 0)],
                    'Expected_Fill': [self.positions[symbol].get('expected_fill', filled_price)],
                    'Lowest_5D_Price': [self.positions[symbol].get('lowest_5d_price', 0)],
                    
                    # Position Sizing Data (NEW)
                    'Position_Size_Percent': [self.positions[symbol].get('position_size_percent', 2.5)],
                    'Position_Size_Reason': [self.positions[symbol].get('position_size_reason', 'CORE_ONLY')],
                    'Confirmation_Filters_Count': [self.positions[symbol].get('confirmation_filters_count', 0)],
                    
                    # Core Filter Data (NEW)
                    'CD_Signal': [self.positions[symbol].get('cd_signal', False)],
                    'CD_Slope_Signal': [self.positions[symbol].get('cd_slope_signal', False)],
                    'OFI_Signal': [self.positions[symbol].get('ofi_signal', False)],
                    'FS_Momentum_Signal': [self.positions[symbol].get('fs_momentum_signal', False)],
                    'FS_Prediction_Signal': [self.positions[symbol].get('fs_prediction_signal', False)],
                    
                    # SMA Filter Data (NEW)
                    'SMA_Filter_Passed': [self.positions[symbol].get('sma_filter_passed', False)],
                    'SMA_50': [self.positions[symbol].get('sma_50', 0)],
                    'Price_Diff_From_SMA': [self.positions[symbol].get('price_diff_from_sma', 0)],
                    'Price_Diff_Percent_From_SMA': [self.positions[symbol].get('price_diff_percent_from_sma', 0)],
                    'SMA_Margin_Threshold': [self.positions[symbol].get('sma_margin_threshold', 5.0)],
                    
                    # FS20 Filter Data (NEW)
                    'FS20_Filter_Passed': [self.positions[symbol].get('fs20_filter_passed', False)],
                    'Current_FS_For_Filter': [self.positions[symbol].get('current_fs_for_filter', 0)],
                    
                    # 6-Hour Trend Data (PAUSED)
                    'Price_Trend_Gate': [self.positions[symbol].get('price_trend_gate', True)],
                    'Price_Change_6H': [self.positions[symbol].get('price_change_6h', 0.0)],
                    'Price_Change_Percent_6H': [self.positions[symbol].get('price_change_percent_6h', 0.0)],
                    'Price_6H_Ahead': [self.positions[symbol].get('price_6h_ahead', 0)],
                    'Trend_Threshold': [self.positions[symbol].get('trend_threshold', 0.5)],
                    'Trend_Gate_Status': [self.positions[symbol].get('trend_gate_status', 'PAUSED')],
                    
                    'Exit_Time': [''],
                    'Exit_Price': [''],
                    'Exit_Value': [''],
                    'P/L': [''],
                    'P/L %': [''],
                    'Exit_Reason': [''],
                    'TP_Percent': [self.positions[symbol].get('tp_percent', 1.3)],
                    'SL_Percent': [self.positions[symbol].get('sl_percent', -0.9)]
                }
                
                self.positions[symbol]['entry_price'] = filled_price
                
            else:  # EXIT
                print(f"Processing exit for {symbol}")
                if os.path.exists(filename):
                    existing_df = pd.read_csv(filename)
                    mask = (existing_df['Symbol'] == symbol) & (existing_df['Exit_Time'].isna())
                    if len(existing_df[mask]) > 0:
                        idx = existing_df[mask].index[0]
                        
                        # Get filled price from trade object
                        filled_price = None
                        if trade and isinstance(trade, dict):
                            filled_price = trade.get('avgFillPrice', 0)
                        elif trade and hasattr(trade, 'orderStatus'):
                            filled_price = float(trade.orderStatus.avgFillPrice if trade.orderStatus.avgFillPrice else 0)
                        else:
                            filled_price = self.positions[symbol].get('exit_price', 0)
                        
                        shares = float(existing_df.loc[idx, 'Shares'])  # Convert to float for calculation
                        entry_price = float(existing_df.loc[idx, 'Entry_Price'])
                        
                        # Calculate P/L with numeric types
                        pnl = (filled_price - entry_price) * shares
                        pnl_pct = ((filled_price - entry_price) / entry_price) * 100 if entry_price != 0 else 0
                        
                        print(f"Exit details for {symbol}:")
                        print(f"Exit Price: ${filled_price}")
                        print(f"P/L: ${pnl:.2f} ({pnl_pct:.2f}%)")
                        
                        # Keep all existing exit information updates
                        existing_df.loc[idx, 'Exit_Time'] = datetime.now(pytz.timezone('US/Eastern'))
                        existing_df.loc[idx, 'Exit_Price'] = filled_price
                        existing_df.loc[idx, 'Exit_Value'] = filled_price * shares
                        existing_df.loc[idx, 'P/L'] = pnl
                        existing_df.loc[idx, 'P/L %'] = pnl_pct
                        existing_df.loc[idx, 'Exit_Reason'] = self.positions[symbol].get('exit_reason', 'UNKNOWN EXIT')
                        
                        existing_df.to_csv(filename, index=False)
                        print(f"Successfully updated exit information in {filename}")
                        return
                    else:
                        print(f"Warning: Could not find open entry for {symbol} in {filename}")
                else:
                    print(f"Warning: No trade log file exists for today: {filename}")
                return
            
            # Keep existing DataFrame creation and saving logic
            trade_df = pd.DataFrame(trade_data)
            if not os.path.exists(filename):
                trade_df.to_csv(filename, index=False)
                print(f"Created new trade log file: {filename}")
            else:
                trade_df.to_csv(filename, mode='a', header=False, index=False)
                print(f"Appended entry to existing file: {filename}")
            
            print(f"Retrieved delta value: {self.positions[symbol].get('fast_stoch_delta')}")
            print(f"Retrieved ATR value: {self.positions[symbol].get('atr')}")
            print(f"Successfully logged trade for {symbol}")
            
        except Exception as e:
            print(f"Error logging trade: {e}")
            import traceback
            print(traceback.format_exc())

    def shutdown(self):
        """Clean shutdown of the trading system"""
        try:
            print("\nInitiating shutdown sequence...")
            
            # Cleanup market data subscriptions
            print("Cleaning up market data subscriptions...")
            self.cleanup_market_data()
            
            # Disconnect from IB
            if self.isConnected():
                print("Disconnecting from IBKR...")
                self.disconnect()
            
            print("Shutdown complete")
            
        except Exception as e:
            print(f"Error during shutdown: {e}")

    def log_account_summary(self):
        """Log daily account metrics to JSON file"""
        try:
            # Create directory structure
            base_dir = Path('trade_logs')
            account_dir = base_dir / 'account_summaries'
            account_dir.mkdir(parents=True, exist_ok=True)
            
            # Get current date for filename
            current_date = datetime.now().strftime('%Y%m%d')
            filename = account_dir / f'account_summary_{current_date}.json'
            
            # FIX: Use reqAccountSummary instead of accountSummary
            self.account_summary_data = []
            self.account_summary_ready = threading.Event()
            
            # Request account summary
            self.reqAccountSummary(1, "All", "NetLiquidation,AvailableFunds,BuyingPower,GrossPositionValue,TotalCashValue,RealizedPnL,UnrealizedPnL")
            
            # Wait for response
            if not self.account_summary_ready.wait(timeout=10):
                print("Timeout waiting for account summary")
                return
            
            # Extract relevant metrics
            metrics = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'NetLiquidation': 0.0,
                'AvailableFunds': 0.0,
                'BuyingPower': 0.0,
                'GrossPositionValue': 0.0,
                'TotalCashValue': 0.0,
                'RealizedPnL': 0.0,
                'UnrealizedPnL': 0.0
            }
            
            # Parse account summary data
            for item in self.account_summary_data:
                if item['tag'] in metrics:
                    metrics[item['tag']] = float(item['value'])
            
            # Load existing data or create new file
            if filename.exists():
                with open(filename, 'r') as f:
                    daily_data = json.load(f)
                    daily_data['hourly_snapshots'].append(metrics)
            else:
                daily_data = {
                    'date': current_date,
                    'hourly_snapshots': [metrics]
                }
            
            # Save updated data
            with open(filename, 'w') as f:
                json.dump(daily_data, f, indent=4)
            
            print("\nAccount summary logged successfully")
            print(f"Net Liquidation: ${metrics['NetLiquidation']:,.2f}")
            print(f"Realized P/L: ${metrics['RealizedPnL']:,.2f}")
            print(f"Unrealized P/L: ${metrics['UnrealizedPnL']:,.2f}")
            
        except Exception as e:
            print(f"Error logging account summary: {e}")

    # Add these callback methods
    def accountSummary(self, reqId, account, tag, value, currency):
        """Callback for account summary data"""
        self.account_summary_data.append({
            'reqId': reqId,
            'account': account,
            'tag': tag,
            'value': value,
            'currency': currency
        })

    def accountSummaryEnd(self, reqId):
        """Called when account summary request is complete"""
        self.account_summary_ready.set()

    def update_trade_log(self, symbol, exit_time, exit_price, exit_value, pnl, pnl_percent, exit_reason):
        """Update trade log with exit information - searches across all CSV files for the entry"""
        try:
            # Search for the entry across all CSV files in trade_logs directory
            trade_logs_dir = 'trade_logs'
            if not os.path.exists(trade_logs_dir):
                print(f"Trade logs directory {trade_logs_dir} not found")
                return
            
            # Get all CSV files in the directory
            csv_files = [f for f in os.listdir(trade_logs_dir) if f.startswith('trades_') and f.endswith('.csv')]
            csv_files.sort()  # Sort to check oldest files first
            
            entry_found = False
            for csv_file in csv_files:
                log_file = os.path.join(trade_logs_dir, csv_file)
                try:
                    # Try to read CSV with error handling
                    try:
                        df = pd.read_csv(log_file)
                    except pd.errors.ParserError as e:
                        print(f"Skipping malformed CSV {csv_file}: {e}")
                        continue
                    
                    # Check if required columns exist
                    required_columns = ['Symbol', 'Exit_Time']
                    if not all(col in df.columns for col in required_columns):
                        print(f"Skipping {csv_file}: Missing required columns")
                        continue
                    
                    # Find entries for this symbol without exit info
                    mask = (df['Symbol'] == symbol) & (df['Exit_Time'].isna())
                    if mask.any():
                        # Update all open entries for this symbol
                        for idx in df[mask].index:
                            df.loc[idx, 'Exit_Time'] = exit_time
                            df.loc[idx, 'Exit_Price'] = exit_price
                            df.loc[idx, 'Exit_Value'] = exit_value
                            df.loc[idx, 'P/L'] = pnl
                            df.loc[idx, 'P/L %'] = pnl_percent
                            df.loc[idx, 'Exit_Reason'] = exit_reason
                        
                        # Save updated log
                        df.to_csv(log_file, index=False)
                        print(f"Updated trade log for {symbol} exit in {csv_file}")
                        entry_found = True
                        
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
                    continue
            
            if not entry_found:
                print(f"Warning: No open entry found for {symbol} in any trade log file")
            
        except Exception as e:
            print(f"Error updating trade log: {e}")
            import traceback
            print(traceback.format_exc())

    def check_and_log_exits(self, current_positions, previous_positions):
        """Check for closed positions and update trade logs with proper exit reason detection"""
        # This method is now only called when we have actual evidence of exits
        # from handle_order_fill, not from position comparison
        for symbol in previous_positions:
            if symbol not in current_positions:
                # Position was closed, get trade info from IB
                trades = self.get_trades()
                closed_trades = [t for t in trades if t.get('symbol') == symbol and t.get('status') == 'Filled']
                
                if closed_trades:
                    latest_trade = closed_trades[-1]
                    exit_time = datetime.now(pytz.timezone('US/Eastern'))
                    exit_price = latest_trade.get('avgFillPrice', 0)
                    exit_value = exit_price * abs(latest_trade.get('filled', 0))
                    entry_value = float(previous_positions[symbol]['position']) * float(previous_positions[symbol]['avgCost'])
                    pnl = exit_value - entry_value
                    pnl_percent = (pnl / entry_value) * 100 if entry_value != 0 else 0
                    
                    # Check if we have stored exit information from handle_order_fill
                    exit_reason = None
                    if symbol in self.positions and 'exit_reason' in self.positions[symbol]:
                        exit_reason = self.positions[symbol].get('exit_reason', 'UNKNOWN')
                        print(f"Using stored exit reason for {symbol}: {exit_reason}")
                    else:
                        # Determine exit reason based on order ID and price - use previous_positions data
                        exit_reason = self.determine_exit_reason_from_trade(symbol, latest_trade, exit_price, previous_positions[symbol])
                        print(f"Determined exit reason for {symbol}: {exit_reason}")
                    
                    # Only update CSV if we have a valid exit reason (TAKE PROFIT or STOP LOSS)
                    if exit_reason in ['TAKE PROFIT', 'STOP LOSS']:
                        # Update the trade log with proper exit reason
                        self.update_trade_log(
                            symbol=symbol,
                            exit_time=exit_time,
                            exit_price=exit_price,
                            exit_value=exit_value,
                            pnl=pnl,
                            pnl_percent=pnl_percent,
                            exit_reason=exit_reason
                        )
                        
                        print(f"\nPosition closed for {symbol}")
                        print(f"Exit Price: ${exit_price:.2f}")
                        print(f"P/L: ${pnl:.2f} ({pnl_percent:.2f}%)")
                        print(f"Exit Reason: {exit_reason}")
                        
                        # Clean up position data after CSV update
                        if symbol in self.positions:
                            del self.positions[symbol]
                    else:
                        print(f"Warning: Position {symbol} appears closed but no valid exit reason found. Skipping CSV update.")
                        print(f"Detected exit reason: {exit_reason}")
                        print(f"This position will be tracked until a valid TP/SL exit is detected.")

    def determine_exit_reason_from_trade(self, symbol, latest_trade, exit_price, position_data):
        """Determine exit reason based on order ID and price using position data from previous_positions"""
        try:
            # Get order IDs from the position data (from previous_positions)
            tp_order_id = position_data.get('tp_orderId')
            sl_order_id = position_data.get('sl_orderId')
            entry_price = position_data.get('entry_price', 0)
            tp_percent = position_data.get('tp_percent', 1.3)
            sl_percent = position_data.get('sl_percent', -0.9)
            
            # Calculate expected TP and SL prices
            expected_tp_price = entry_price * (1 + tp_percent/100)
            expected_sl_price = entry_price * (1 + sl_percent/100)
            
            # Check if this was a TP or SL order based on order ID
            trade_order_id = latest_trade.get('orderId')
            if trade_order_id == tp_order_id:
                return 'TAKE PROFIT'
            elif trade_order_id == sl_order_id:
                return 'STOP LOSS'
            else:
                # If order ID doesn't match, check by price proximity
                tp_diff = abs(exit_price - expected_tp_price)
                sl_diff = abs(exit_price - expected_sl_price)
                
                if tp_diff < sl_diff and exit_price >= expected_tp_price:
                    return 'TAKE PROFIT'
                elif sl_diff < tp_diff and exit_price <= expected_sl_price:
                    return 'STOP LOSS'
                else:
                    return 'MANUAL EXIT'  # Fallback for unknown exits
        
        except Exception as e:
            print(f"Error determining exit reason for {symbol}: {e}")
            return 'UNKNOWN EXIT'

    def determine_exit_reason(self, symbol, latest_trade, exit_price):
        """Legacy method - kept for compatibility but now calls the new method"""
        try:
            # Try to get position data from self.positions first
            if symbol in self.positions:
                position_data = self.positions[symbol]
                return self.determine_exit_reason_from_trade(symbol, latest_trade, exit_price, position_data)
            
            # If not in self.positions, try to get from CSV file
            return self.determine_exit_reason_from_csv(symbol, latest_trade, exit_price)
            
        except Exception as e:
            print(f"Error determining exit reason for {symbol}: {e}")
            return 'UNKNOWN EXIT'

    def determine_exit_reason_from_csv(self, symbol, latest_trade, exit_price):
        """Determine exit reason by reading from CSV file when position data is not available"""
        try:
            # Look for the symbol in today's CSV file
            today = datetime.now().strftime('%Y%m%d')
            filename = f"trade_logs/trades_{today}.csv"
            
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                # Find the entry for this symbol that hasn't been closed
                mask = (df['Symbol'] == symbol) & (df['Exit_Time'].isna())
                if len(df[mask]) > 0:
                    row = df[mask].iloc[0]
                    entry_price = float(row['Entry_Price'])
                    tp_percent = float(row.get('TP_Percent', 1.3))
                    sl_percent = float(row.get('SL_Percent', -0.9))
                    
                    # Calculate expected TP and SL prices
                    expected_tp_price = entry_price * (1 + tp_percent/100)
                    expected_sl_price = entry_price * (1 + sl_percent/100)
                    
                    # Check by price proximity
                    tp_diff = abs(exit_price - expected_tp_price)
                    sl_diff = abs(exit_price - expected_sl_price)
                    
                    if tp_diff < sl_diff and exit_price >= expected_tp_price:
                        return 'TAKE PROFIT'
                    elif sl_diff < tp_diff and exit_price <= expected_sl_price:
                        return 'STOP LOSS'
                    else:
                        return 'MANUAL EXIT'
            
            return 'UNKNOWN EXIT'
            
        except Exception as e:
            print(f"Error determining exit reason from CSV for {symbol}: {e}")
            return 'UNKNOWN EXIT'

    def cleanup_cache(self):
        """Remove expired cache entries"""
        try:
            current_time = datetime.now(pytz.timezone('US/Eastern'))
            current_timestamp = time.time()  # Also get current timestamp for float comparisons
            
            total_entries = 0
            expired_count = 0
            
            for resolution, cache in self.cache.items():
                duration_seconds = self.cache_durations[resolution]
                expired_keys = []
                
                for key, data in cache.items():
                    total_entries += 1
                    
                    try:
                        # Handle different cache structures
                        if isinstance(data, dict) and 'timestamp' in data:
                            # New structure: {'timestamp': timestamp, 'value': value}
                            if isinstance(data['timestamp'], datetime):
                                # Timestamp is datetime object
                                if (current_time - data['timestamp']).total_seconds() > duration_seconds:
                                    expired_keys.append(key)
                            elif isinstance(data['timestamp'], (int, float)):
                                # Timestamp is float/int (Unix timestamp)
                                if (current_timestamp - data['timestamp']) > duration_seconds:
                                    expired_keys.append(key)
                            else:
                                # Unknown timestamp type - log for debugging
                                print(f"Warning: Unknown timestamp type {type(data['timestamp'])} for key {key} in {resolution} cache")
                        elif isinstance(data, tuple) and len(data) == 2:
                            # Old structure: (timestamp, value)
                            timestamp, _ = data
                            if isinstance(timestamp, datetime):
                                # Timestamp is datetime object
                                if (current_time - timestamp).total_seconds() > duration_seconds:
                                    expired_keys.append(key)
                            elif isinstance(timestamp, (int, float)):
                                # Timestamp is float/int (Unix timestamp)
                                if (current_timestamp - timestamp) > duration_seconds:
                                    expired_keys.append(key)
                            else:
                                # Unknown timestamp type - log for debugging
                                print(f"Warning: Unknown timestamp type {type(timestamp)} for key {key} in {resolution} cache")
                        else:
                            # Unknown data structure - log for debugging
                            print(f"Warning: Unknown cache data structure {type(data)} for key {key} in {resolution} cache")
                            
                    except Exception as entry_error:
                        print(f"Error processing cache entry {key} in {resolution}: {entry_error}")
                        # Remove problematic entries
                        expired_keys.append(key)
                
                # Remove expired entries
                for key in expired_keys:
                    cache.pop(key, None)
                    expired_count += 1
            
            if expired_count > 0:
                print(f"Cache cleanup: Removed {expired_count}/{total_entries} expired entries")
                    
        except Exception as e:
            print(f"Error in cleanup_cache: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

    def get_current_price(self, symbol):
        """Helper function to get current price with fallback mechanisms"""
        try:
            # Try market data first
            contract = self.get_contract(symbol)
            ticker = self.reqMktData_ibapi(contract)
            time.sleep(0.1)
            
            # Try different price sources
            if ticker and ticker.get('last') and ticker.get('last') > 0:
                return ticker.get('last')
            elif ticker and ticker.get('close') and ticker.get('close') > 0:
                return ticker.get('close')
            elif ticker and ticker.get('bid') and ticker.get('ask'):
                return (ticker.get('bid') + ticker.get('ask')) / 2
            
            # Fallback to historical data
            print(f"Market data not available for {symbol}, trying historical data...")
            bars = self.get_cached_historical_data(
                symbol=symbol,
                resolution='daily',
                duration_str='1 D'
            )
            
            if bars and len(bars) > 0:
                latest_bar = bars[-1]
                return latest_bar.close
            
            print(f"Could not get price for {symbol}")
            return None
            
        except Exception as e:
            print(f"Error getting current price for {symbol}: {e}")
            return None


    def cleanup_market_data(self):
        """Clean up all existing market data subscriptions"""
        try:
            # Store original stdout
            original_stdout = sys.stdout

            # Temporarily redirect stdout to devnull to suppress prints
            with open(os.devnull, 'w') as devnull:
                sys.stdout = devnull

            # Cancel all market data subscriptions using ibapi
            for ticker_id, symbol in self.ticker_id_map.items():
                try:
                    self.cancelMktData(ticker_id)
                except:
                    pass
            
            # Clear tracking dictionaries
            self.active_tickers.clear()
            self.ticker_id_map.clear()
            self.market_data.clear()
            
            # Restore stdout
            sys.stdout = original_stdout

        except Exception as e:
            print(f"Error cleaning up market data: {e}")
    


    '''        
    def cleanup_market_data(self):
        """Clean up market data subscriptions"""
        try:
            # Store original stdout
            original_stdout = sys.stdout
            
            # Temporarily redirect stdout to devnull to suppress prints
            with open(os.devnull, 'w') as devnull:
                sys.stdout = devnull
                
                # Perform the cleanup
                for contract in list(self.market_data_subscriptions.keys()):
                    try:
                        self.ib.cancelMktData(contract)
                        self.market_data_subscriptions.pop(contract, None)
                    except:
                        pass
                        
            # Restore stdout
            sys.stdout = original_stdout
            
        except Exception as e:
            print(f"Error in cleanup_market_data: {e}")
    '''

    def update_cd_history(self, symbol, cd_value):
        """Update the CD history for a symbol"""
        try:
            cache_key = f"{symbol}_cd_history"
            current_time = datetime.now(pytz.timezone('US/Eastern'))
            
            # Initialize history if not exists
            if cache_key not in self.cache['delta']:
                self.cache['delta'][cache_key] = {
                    'values': [],
                    'last_update': current_time - timedelta(minutes=5)  # Force first update
                }
            
            # Add the new CD value
            if cd_value is not None:
                self.cache['delta'][cache_key]['values'].append(cd_value)
                self.cache['delta'][cache_key]['last_update'] = current_time
                
                # Keep only recent values (last 10)
                self.cache['delta'][cache_key]['values'] = self.cache['delta'][cache_key]['values'][-10:]
                
        except Exception as e:
            print(f"Error updating CD history: {e}")

    def get_recent_cd_values(self, symbol, lookback=3):
        """Track recent Cumulative Delta values"""
        try:
            cache_key = f"{symbol}_cd_history"
            current_time = datetime.now(pytz.timezone('US/Eastern'))
            
            # Initialize history if not exists
            if cache_key not in self.cache['delta']:
                self.cache['delta'][cache_key] = {
                    'values': [],
                    'last_update': current_time - timedelta(minutes=5)  # Force first update
                }
            
            # Only update if enough time has passed (e.g., 1 minute)
            time_diff = (current_time - self.cache['delta'][cache_key]['last_update']).total_seconds()
            if time_diff >= 60:  # Update every minute
                # Don't call get_cached_cumulative_delta here - it should be called from check_entry_conditions
                # and the value should be passed to this method
                pass
            
            print(f"\nCD Debug for {symbol}:")
            print(f"History values: {self.cache['delta'][cache_key]['values']}")
            print(f"Time since last update: {time_diff:.2f} seconds")
            
            return self.cache['delta'][cache_key]['values']
            
        except Exception as e:
            print(f"Error getting recent CD values: {e}")
            return []

    def historicalTicks(self, reqId: int, ticks: ListOfHistoricalTick, done: bool):
        """Handler for historical tick data when whatToShow=TRADES"""
        if reqId not in self.historical_ticks_store:
            self.historical_ticks_store[reqId] = []
        self.historical_ticks_store[reqId].extend(ticks)
        if done:
            self.historical_ticks_ready[reqId].set()

    def historicalTicksLast(self, reqId: int, ticks: ListOfHistoricalTickLast, done: bool):
        """Handler for historical tick data when whatToShow=TRADES"""
        if reqId not in self.historical_ticks_store:
            self.historical_ticks_store[reqId] = []
        self.historical_ticks_store[reqId].extend(ticks)
        if done:
            self.historical_ticks_ready[reqId].set()

    # Add these methods to your IBApiClient class:

    def reqMktData_ibapi(self, contract):
        """ibapi equivalent of ib.reqMktData with improved error handling"""
        try:
            req_id = self.get_new_req_id()
            
            # Initialize market data structure
            if contract.symbol not in self.market_data:
                self.market_data[contract.symbol] = {
                    'bid': None, 'ask': None, 'last': None, 'close': None,
                    'bidSize': 0, 'askSize': 0, 'lastSize': 0
                }
            
            self.ticker_id_map[req_id] = contract.symbol
            
            # Request market data
            self.reqMktData(req_id, contract, "", False, False, [])
            
            # Wait for data with timeout
            timeout = 2  # Reduced timeout to 2 seconds
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                data = self.market_data[contract.symbol]
                # Check if we have at least some data
                if any(data.get(key) is not None for key in ['bid', 'ask', 'last', 'close']):
                    break
                time.sleep(0.1)
            
            # Return the market data even if incomplete
            return self.market_data[contract.symbol]
            
        except Exception as e:
            print(f"Error requesting market data for {contract.symbol}: {e}")
            # Return fallback data structure
            return {
                'bid': None, 'ask': None, 'last': None, 'close': None,
                'bidSize': 0, 'askSize': 0, 'lastSize': 0
            }

    def tickPrice(self, tickerId: int, field: int, price: float, attrib):
        """Handle tick price updates"""
        symbol = self.ticker_id_map.get(tickerId)
        if symbol and symbol in self.market_data:
            if field == 1:  # BID
                self.market_data[symbol]['bid'] = price
            elif field == 2:  # ASK
                self.market_data[symbol]['ask'] = price
            elif field == 4:  # LAST
                self.market_data[symbol]['last'] = price
            elif field == 9:  # CLOSE
                self.market_data[symbol]['close'] = price

    def tickSize(self, tickerId: int, field: int, size: int):
        """Handle tick size updates"""
        symbol = self.ticker_id_map.get(tickerId)
        if symbol and symbol in self.market_data:
            if field == 0:  # BID_SIZE
                self.market_data[symbol]['bidSize'] = size
            elif field == 3:  # ASK_SIZE
                self.market_data[symbol]['askSize'] = size
            elif field == 5:  # LAST_SIZE
                self.market_data[symbol]['lastSize'] = size

    # Simple order status handler (add this method)
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        """Handle order status updates"""
        print(f"[OrderStatus] OrderID: {orderId} | Status: {status} | Filled: {filled} | Remaining: {remaining}")
        
        # Update trade tracking
        self.update_trade_status(orderId, status, filled, remaining, avgFillPrice, lastFillPrice, whyHeld, mktCapPrice)
        
        # Track accepted orders for place_order function
        if hasattr(self, 'pending_orders') and orderId in self.pending_orders:
            if status in ['Submitted', 'PreSubmitted']:
                self.accepted_orders.add(orderId)
                print(f"[{orderId}] Order accepted with status: {status}")
            elif status in ['Cancelled', 'Inactive']:
                print(f"[{orderId}] Order rejected with status: {status}")
        
        # Handle fills for position monitoring
        if status == 'Filled' and filled > 0:
            self.handle_order_fill(orderId, status, filled, avgFillPrice, parentId)

    def execDetails(self, reqId: int, contract: Contract, execution):
        """Handle execution details"""
        print(f"[Execution] Symbol: {contract.symbol} | OrderID: {execution.orderId} | "
              f"Shares: {execution.shares} | Price: {execution.price} | Side: {execution.side}")
        
        # Handle execution for position monitoring
        self.handle_execution(contract.symbol, execution.orderId, execution.shares, execution.price, execution.side)

    def handle_order_fill(self, orderId, status, filled, avgFillPrice, parentId):
        """Handle order fills for position monitoring"""
        try:
            # Check if this fill is for one of our tracked orders
            for symbol, pos in list(self.positions.items()):
                if orderId in [pos.get('parent_orderId'), pos.get('tp_orderId'), pos.get('sl_orderId')]:
                    # Determine which order type was filled
                    order_type = None
                    if orderId == pos.get('parent_orderId'):
                        order_type = 'ENTRY'
                    elif orderId == pos.get('tp_orderId'):
                        order_type = 'TAKE PROFIT'
                    elif orderId == pos.get('sl_orderId'):
                        order_type = 'STOP LOSS'

                    print(f"\n{'='*50}")
                    print(f"Fill detected for {symbol}:")
                    print(f"Order Type: {order_type}")
                    print(f"Order ID: {orderId}")
                    print(f"Fill Price: ${avgFillPrice:.2f}")
                    print(f"Fill Quantity: {filled}")
                    print(f"Fill Time: {datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Initialize or update peak price for trailing stop
                    if order_type == 'ENTRY':
                        pos['peak_price'] = avgFillPrice
                        print(f"Peak price initialized: ${pos['peak_price']:.2f}")
                    
                    # For exit orders, just log the fill but don't update CSV here
                    # The CSV will be updated by check_and_log_exits when position is detected as closed
                    if order_type in ['TAKE PROFIT', 'STOP LOSS']:
                        entry_price = pos.get('entry_price', 0)
                        pnl = (avgFillPrice - entry_price) * filled
                        pnl_pct = ((avgFillPrice - entry_price) / entry_price) * 100
                        print(f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
                        print(f"Exit Reason: {order_type}")
                        
                        # Store exit information in position data for later CSV update
                        pos['exit_price'] = avgFillPrice
                        pos['exit_reason'] = order_type
                        pos['exit_time'] = datetime.now(pytz.timezone('US/Eastern'))
                    
                    print(f"{'='*50}\n")
                    break
                    
        except Exception as e:
            print(f"[ERROR] Failed in order fill handling: {e}")
            import traceback
            print(traceback.format_exc())

    def handle_execution(self, symbol, orderId, shares, price, side):
        """Handle execution details for position monitoring"""
        try:
            # This can be used for additional execution tracking if needed
            print(f"Execution: {symbol} {side} {shares} shares @ ${price:.2f}")
            
        except Exception as e:
            print(f"[ERROR] Failed in execution handling: {e}")

    # Add this method to your IBApiClient class
    def get_new_req_id(self):
        """Get a new request ID with improved timeout handling"""
        # Initialize if not set
        if not hasattr(self, 'req_id_counter'):
            self.req_id_counter = 1000
        
        # Add timeout to prevent infinite hang
        timeout = 10  # 10 seconds timeout
        start_time = time.time()
        
        # Wait for nextOrderId if not available
        while self.nextOrderId is None:
            if time.time() - start_time > timeout:
                print("Timeout waiting for nextOrderId from TWS, using fallback")
                # Use a fallback ID
                self.req_id_counter += 1
                return self.req_id_counter
            time.sleep(0.1)

        # Update counter and return new ID
        self.req_id_counter = max(self.req_id_counter + 1, self.nextOrderId)
        return self.req_id_counter

    # Add this callback method to handle nextValidId from TWS
    def nextValidId(self, orderId: int):
        """Called by TWS to provide the next valid order ID"""
        print(f"Received nextValidId: {orderId}")
        self.nextOrderId = orderId
    
    def connectAck(self):
        """Called when connection is established"""
        print("Connection acknowledged by TWS")

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        """Handle error messages from TWS"""
        # Filter out common warnings that don't affect functionality
        if errorCode in [2104, 2106, 2158]:  # Market data farm connection messages
            return  # These are informational, not errors
        
        if errorCode == 2176:  # Fractional share size warning
            print(f"Warning: {errorString} (This is normal and doesn't affect trading)")
            return
            
        if errorCode == 354:  # Market data not subscribed
            print(f"Market data warning for reqId {reqId}: {errorString}")
            return
            
        if errorCode == 300:  # Can't find EId
            print(f"Market data cleanup for reqId {reqId}: {errorString}")
            return
            
        # Log other errors
        print(f"Error {errorCode} for reqId {reqId}: {errorString}")

    def get_positions(self):
        """Get current positions using ibapi reqPositions()"""
        self.positions_list = []
        self.positions_ready = threading.Event()
        self.reqPositions()
        
        if self.positions_ready.wait(timeout=5):
            return {pos['symbol']: pos for pos in self.positions_list}
        else:
            print("Timeout waiting for positions")
            return {}

    def position(self, account, contract, pos, avgCost):
        """Callback for position updates"""
        if pos != 0:
            self.positions_list.append({
                'symbol': contract.symbol,
                'contract': contract,
                'position': pos,
                'avgCost': avgCost,
                'account': account
            })

    def positionEnd(self):
        """Called when all positions have been received"""
        self.positions_ready.set()

    def get_trades(self):
        """Get current trades using ibapi - equivalent to self.ib.trades()"""
        return self.trades_list

    def get_trade_by_symbol(self, symbol):
        """Get trade information for a specific symbol"""
        return [trade for trade in self.trades_list if trade.get('symbol') == symbol]

    def get_filled_trades(self, symbol=None):
        """Get filled trades, optionally filtered by symbol"""
        filled_trades = [trade for trade in self.trades_list if trade.get('status') == 'Filled']
        if symbol:
            filled_trades = [trade for trade in filled_trades if trade.get('symbol') == symbol]
        return filled_trades

    def get_sell_trades(self, symbol=None):
        """Get sell trades, optionally filtered by symbol"""
        sell_trades = [trade for trade in self.trades_list if trade.get('action') == 'SELL']
        if symbol:
            sell_trades = [trade for trade in sell_trades if trade.get('symbol') == symbol]
        return sell_trades

    def add_trade(self, orderId, contract, order, orderStatus):
        """Add or update trade information"""
        # Handle both dictionary and object status
        if isinstance(orderStatus, dict):
            status_dict = orderStatus
        else:
            # Assume it's an object with attributes
            status_dict = {
                'status': getattr(orderStatus, 'status', 'Unknown'),
                'filled': getattr(orderStatus, 'filled', 0),
                'remaining': getattr(orderStatus, 'remaining', 0),
                'avgFillPrice': getattr(orderStatus, 'avgFillPrice', 0.0),
                'lastFillPrice': getattr(orderStatus, 'lastFillPrice', 0.0),
                'whyHeld': getattr(orderStatus, 'whyHeld', ''),
                'mktCapPrice': getattr(orderStatus, 'mktCapPrice', 0.0)
            }
        
        trade_info = {
            'orderId': orderId,
            'symbol': contract.symbol,
            'contract': contract,
            'order': order,
            'orderStatus': orderStatus,
            'action': order.action,
            'status': status_dict['status'],
            'filled': status_dict['filled'],
            'remaining': status_dict['remaining'],
            'avgFillPrice': status_dict['avgFillPrice'],
            'lastFillPrice': status_dict['lastFillPrice'],
            'whyHeld': status_dict['whyHeld'],
            'mktCapPrice': status_dict['mktCapPrice']
        }
        
        self.trades_store[orderId] = trade_info
        
        # Update trades_list for compatibility with existing code
        existing_trade = next((t for t in self.trades_list if t.get('orderId') == orderId), None)
        if existing_trade:
            # Update existing trade
            existing_trade.update(trade_info)
        else:
            # Add new trade
            self.trades_list.append(trade_info)
        
        print(f"Trade updated: {contract.symbol} {order.action} {status_dict['status']}")

    def update_trade_status(self, orderId, status, filled, remaining, avgFillPrice, lastFillPrice, whyHeld, mktCapPrice):
        """Update trade status when orderStatus callback is received"""
        if orderId in self.trades_store:
            trade_info = self.trades_store[orderId]
            trade_info.update({
                'status': status,
                'filled': filled,
                'remaining': remaining,
                'avgFillPrice': avgFillPrice,
                'lastFillPrice': lastFillPrice,
                'whyHeld': whyHeld,
                'mktCapPrice': mktCapPrice
            })
            
            # Update in trades_list as well
            for trade in self.trades_list:
                if trade.get('orderId') == orderId:
                    trade.update(trade_info)
                    break
            
            print(f"Trade status updated: OrderID {orderId} - {status}")

    def clear_old_trades(self, days_to_keep=7):
        """Clear old trades to prevent memory buildup"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            self.trades_list = [trade for trade in self.trades_list 
                              if 'timestamp' not in trade or 
                              pd.to_datetime(trade['timestamp']) > cutoff_date]
            print(f"Cleared old trades, keeping {len(self.trades_list)} recent trades")
        except Exception as e:
            print(f"Error clearing old trades: {e}")

    def print_current_trades(self):
        """Debug method to print current trades"""
        print(f"\nCurrent trades ({len(self.trades_list)} total):")
        for trade in self.trades_list:
            print(f"  {trade.get('symbol', 'N/A')} {trade.get('action', 'N/A')} "
                  f"Status: {trade.get('status', 'N/A')} "
                  f"Filled: {trade.get('filled', 0)} "
                  f"Price: ${trade.get('avgFillPrice', 0):.2f}")

    def create_order_status_object(self, status='Submitted', filled=0, remaining=0, 
                                 avgFillPrice=0.0, lastFillPrice=0.0, whyHeld='', mktCapPrice=0.0):
        """Create a simple order status object for tracking"""
        class SimpleOrderStatus:
            def __init__(self, status, filled, remaining, avgFillPrice, lastFillPrice, whyHeld, mktCapPrice):
                self.status = status
                self.filled = filled
                self.remaining = remaining
                self.avgFillPrice = avgFillPrice
                self.lastFillPrice = lastFillPrice
                self.whyHeld = whyHeld
                self.mktCapPrice = mktCapPrice
        
        return SimpleOrderStatus(status, filled, remaining, avgFillPrice, lastFillPrice, whyHeld, mktCapPrice)

    def verify_order_submission(self, symbol, order_ids, timeout=5):
        """Enhanced order submission verification with detailed status tracking"""
        print(f"\nVerifying order submission for {symbol}...")
        print(f"Order IDs to verify: {order_ids}")
        
        start_time = time.time()
        verified_orders = set()
        rejected_orders = set()
        
        while time.time() - start_time < timeout:
            # Check each order status
            for order_id in order_ids:
                if order_id in self.accepted_orders and order_id not in verified_orders:
                    verified_orders.add(order_id)
                    print(f"[+] Order {order_id} verified as submitted")
                
                # Check for rejected orders
                trade = self.trades_store.get(order_id)
                if trade and trade.get('status') in ['Cancelled', 'Inactive', 'Error']:
                    rejected_orders.add(order_id)
                    print(f"[-] Order {order_id} rejected with status: {trade.get('status')}")
            
            # Check if all orders are processed
            if len(verified_orders) + len(rejected_orders) == len(order_ids):
                break
                
            time.sleep(0.1)
        
        # Final verification
        if len(verified_orders) == len(order_ids):
            print(f"[+] All {len(order_ids)} orders for {symbol} successfully submitted!")
            return True, verified_orders
        elif len(rejected_orders) > 0:
            print(f"[-] Some orders for {symbol} were rejected: {rejected_orders}")
            return False, rejected_orders
        else:
            print(f"[!] Timeout waiting for order status updates for {symbol}")
            return False, set()

    def get_order_status_summary(self, symbol=None):
        """Get a summary of all order statuses"""
        if symbol:
            trades = [t for t in self.trades_list if t.get('symbol') == symbol]
        else:
            trades = self.trades_list
        
        status_summary = {}
        for trade in trades:
            order_id = trade.get('orderId')
            status = trade.get('status', 'Unknown')
            action = trade.get('action', 'Unknown')
            filled = trade.get('filled', 0)
            remaining = trade.get('remaining', 0)
            
            status_summary[order_id] = {
                'symbol': trade.get('symbol'),
                'action': action,
                'status': status,
                'filled': filled,
                'remaining': remaining,
                'avgFillPrice': trade.get('avgFillPrice', 0)
            }
        
        return status_summary

    def check_order_health(self, symbol):
        """Check the health status of all orders for a symbol"""
        trades = self.get_trade_by_symbol(symbol)
        if not trades:
            return "No orders found"
        
        health_status = {
            'total_orders': len(trades),
            'submitted': 0,
            'filled': 0,
            'cancelled': 0,
            'pending': 0,
            'errors': 0
        }
        
        for trade in trades:
            status = trade.get('status', 'Unknown')
            if status in ['Submitted', 'PreSubmitted']:
                health_status['submitted'] += 1
            elif status == 'Filled':
                health_status['filled'] += 1
            elif status in ['Cancelled', 'Inactive']:
                health_status['cancelled'] += 1
            elif status in ['PendingSubmit', 'PendingCancel']:
                health_status['pending'] += 1
            else:
                health_status['errors'] += 1
        
        return health_status

    def get_historical_ticks_for_cd(self, symbol, lookback_minutes=60):
        """Get historical ticks for accurate CD calculation using ibapi with enhanced retry mechanism"""
        try:
            # Rate limiting for tick requests
            if hasattr(self, '_last_tick_request_time'):
                time_since_last = time.time() - self._last_tick_request_time
                if time_since_last < 1.0:  # 1 second minimum between tick requests
                    sleep_time = 1.0 - time_since_last
                    print(f"Rate limiting tick requests: waiting {sleep_time:.2f}s")
                    time.sleep(sleep_time)
            
            self._last_tick_request_time = time.time()
            
            # Enhanced retry mechanism for tick requests
            max_retries = 2
            retry_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    print(f"Requesting historical ticks for {symbol} (attempt {attempt + 1}/{max_retries})")
                    
                    contract = self.get_contract(symbol)
                    current_time = datetime.now(pytz.timezone('US/Eastern'))
                    start_time = current_time - timedelta(minutes=lookback_minutes)
                    
                    # Initialize storage for this request
                    req_id = self.get_new_req_id()
                    self.historical_ticks_store[req_id] = []
                    self.historical_ticks_ready[req_id] = threading.Event()
                    
                    # Format times as strings with proper timezone - use UTC format
                    start_time_utc = start_time.astimezone(pytz.UTC)
                    current_time_utc = current_time.astimezone(pytz.UTC)
                    start_time_str = start_time_utc.strftime('%Y%m%d-%H:%M:%S')
                    current_time_str = current_time_utc.strftime('%Y%m%d-%H:%M:%S')
                    
                    print(f"Requesting historical ticks for {symbol} from {start_time_str} to {current_time_str}")
                    
                    # Request historical ticks using ibapi
                    self.reqHistoricalTicks(
                        reqId=req_id,
                        contract=contract,
                        startDateTime=start_time_str,
                        endDateTime=current_time_str,
                        numberOfTicks=1000,
                        whatToShow='TRADES',
                        useRth=0,  # 0 for False, 1 for True
                        ignoreSize=False,
                        miscOptions=[]
                    )
                    
                    # Wait for data with timeout
                    if self.historical_ticks_ready[req_id].wait(timeout=15):  # Increased timeout
                        trades = self.historical_ticks_store[req_id]
                        
                        # Clean up
                        if req_id in self.historical_ticks_store:
                            del self.historical_ticks_store[req_id]
                        if req_id in self.historical_ticks_ready:
                            del self.historical_ticks_ready[req_id]
                        
                        if trades:
                            print(f"Received {len(trades)} trades for {symbol}")
                            return trades
                        else:
                            print(f"[WARNING] No trades received for {symbol} (attempt {attempt + 1}/{max_retries})")
                    else:
                        print(f"[WARNING] Timeout waiting for historical ticks for {symbol} (attempt {attempt + 1}/{max_retries})")
                    
                    # Add delay between retries
                    if attempt < max_retries - 1:
                        print(f"Retrying tick request in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        
                except Exception as e:
                    print(f"[ERROR] Attempt {attempt + 1} failed for tick request {symbol}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
            
            print(f"[ERROR] All tick request attempts failed for {symbol}")
            return None
            
        except Exception as e:
            print(f"Error getting historical ticks for {symbol}: {e}")
            return None

    def calculate_cd_from_ticks(self, symbol, trades):
        """Calculate CD from historical ticks data - REMOVED: now done directly in main method"""
        # This method is no longer needed as CD calculation is done directly in get_cached_cumulative_delta
        # to exactly match the ib_insync implementation
        pass

    def fix_existing_csv_exit_reasons(self, date_str=None):
        """Manually fix exit reasons in existing CSV file based on actual exit prices vs expected TP/SL"""
        try:
            if date_str is None:
                date_str = datetime.now().strftime('%Y%m%d')
            
            filename = f"trade_logs/trades_{date_str}.csv"
            if not os.path.exists(filename):
                print(f"CSV file {filename} not found")
                return
            
            print(f"Analyzing and fixing exit reasons in {filename}")
            df = pd.read_csv(filename)
            
            # Find rows that have exit information but wrong exit reasons
            fixed_count = 0
            for idx, row in df.iterrows():
                # Skip rows without exit information
                if pd.isna(row['Exit_Time']) or pd.isna(row['Exit_Price']):
                    continue
                
                # Skip rows that already have correct exit reasons
                if row['Exit_Reason'] in ['TAKE PROFIT', 'STOP LOSS']:
                    continue
                
                # Calculate expected TP and SL prices
                entry_price = float(row['Entry_Price'])
                tp_percent = float(row.get('TP_Percent', 1.3))
                sl_percent = float(row.get('SL_Percent', -0.9))
                
                expected_tp_price = entry_price * (1 + tp_percent/100)
                expected_sl_price = entry_price * (1 + sl_percent/100)
                exit_price = float(row['Exit_Price'])
                
                # Determine correct exit reason based on price proximity
                tp_diff = abs(exit_price - expected_tp_price)
                sl_diff = abs(exit_price - expected_sl_price)
                
                old_reason = row['Exit_Reason']
                new_reason = None
                
                if tp_diff < sl_diff and exit_price >= expected_tp_price:
                    new_reason = 'TAKE PROFIT'
                elif sl_diff < tp_diff and exit_price <= expected_sl_price:
                    new_reason = 'STOP LOSS'
                else:
                    new_reason = 'MANUAL EXIT'
                
                if new_reason != old_reason:
                    df.loc[idx, 'Exit_Reason'] = new_reason
                    fixed_count += 1
                    print(f"Fixed {row['Symbol']}: {old_reason} -> {new_reason} "
                          f"(Entry: ${entry_price:.2f}, Exit: ${exit_price:.2f}, "
                          f"Expected TP: ${expected_tp_price:.2f}, Expected SL: ${expected_sl_price:.2f})")
            
            if fixed_count > 0:
                df.to_csv(filename, index=False)
                print(f"Fixed {fixed_count} exit reasons in {filename}")
            else:
                print("No exit reasons needed fixing")
                
        except Exception as e:
            print(f"Error fixing CSV exit reasons: {e}")
            import traceback
            print(traceback.format_exc())

    def analyze_csv_exit_reasons(self, date_str=None):
        """Analyze exit reasons in CSV file to help debug issues"""
        try:
            if date_str is None:
                date_str = datetime.now().strftime('%Y%m%d')
            
            filename = f"trade_logs/trades_{date_str}.csv"
            if not os.path.exists(filename):
                print(f"CSV file {filename} not found")
                return
            
            print(f"Analyzing exit reasons in {filename}")
            df = pd.read_csv(filename)
            
            # Count exit reasons
            exit_reasons = df['Exit_Reason'].value_counts()
            print("\nExit Reason Analysis:")
            print("-" * 40)
            for reason, count in exit_reasons.items():
                print(f"{reason}: {count}")
            
            # Analyze specific cases
            print("\nDetailed Analysis:")
            print("-" * 40)
            for idx, row in df.iterrows():
                if not pd.isna(row['Exit_Time']) and not pd.isna(row['Exit_Price']):
                    entry_price = float(row['Entry_Price'])
                    exit_price = float(row['Exit_Price'])
                    tp_percent = float(row.get('TP_Percent', 1.3))
                    sl_percent = float(row.get('SL_Percent', -0.9))
                    
                    expected_tp_price = entry_price * (1 + tp_percent/100)
                    expected_sl_price = entry_price * (1 + sl_percent/100)
                    
                    tp_diff = abs(exit_price - expected_tp_price)
                    sl_diff = abs(exit_price - expected_sl_price)
                    
                    print(f"{row['Symbol']}: Entry=${entry_price:.2f}, Exit=${exit_price:.2f}, "
                          f"Expected TP=${expected_tp_price:.2f}, Expected SL=${expected_sl_price:.2f}, "
                          f"Current Reason='{row['Exit_Reason']}', "
                          f"Should be: {'TAKE PROFIT' if tp_diff < sl_diff and exit_price >= expected_tp_price else 'STOP LOSS' if sl_diff < tp_diff and exit_price <= expected_sl_price else 'MANUAL EXIT'}")
                
        except Exception as e:
            print(f"Error analyzing CSV: {e}")
            import traceback
            print(traceback.format_exc())

    def validate_and_normalize_predictions(self, predictions, df, lookback_periods=20):
        """Validate and normalize predicted OHLC values to ensure they're realistic
        
        Args:
            predictions: Dictionary containing predicted OHLCV values
            df: Historical DataFrame for validation
            lookback_periods: Number of historical periods to use for validation
        Returns:
            dict: Normalized predictions with realistic values
        """
        try:
            if len(df) < lookback_periods:
                lookback_periods = len(df)
            
            # Get recent historical data for validation
            recent_df = df.tail(lookback_periods)
            
            # Calculate historical statistics
            historical_high_max = recent_df['high'].max()
            historical_low_min = recent_df['low'].min()
            historical_close_mean = recent_df['close'].mean()
            historical_close_std = recent_df['close'].std()
            historical_range = historical_high_max - historical_low_min
            
            # Define reasonable bounds (within 2 standard deviations)
            max_allowed_deviation = historical_close_std * 2
            max_allowed_high = historical_high_max + max_allowed_deviation
            min_allowed_low = historical_low_min - max_allowed_deviation
            
            # Normalize predicted values
            normalized_predictions = {}
            
            # Ensure high >= low
            if predictions['high'] < predictions['low']:
                predictions['high'], predictions['low'] = predictions['low'], predictions['high']
            
            # Clamp high and low to reasonable bounds
            normalized_predictions['high'] = np.clip(predictions['high'], historical_low_min, max_allowed_high)
            normalized_predictions['low'] = np.clip(predictions['low'], min_allowed_low, historical_high_max)
            
            # Ensure low <= high after clamping
            if normalized_predictions['low'] > normalized_predictions['high']:
                normalized_predictions['low'] = normalized_predictions['high']
            
            # Clamp close to be between low and high
            normalized_predictions['close'] = np.clip(
                predictions['close'], 
                normalized_predictions['low'], 
                normalized_predictions['high']
            )
            
            # Add volume if present
            if 'volume' in predictions:
                historical_volume_mean = recent_df['volume'].mean()
                historical_volume_std = recent_df['volume'].std()
                max_allowed_volume = historical_volume_mean + (historical_volume_std * 2)
                normalized_predictions['volume'] = np.clip(predictions['volume'], 0, max_allowed_volume)
            
            # Log normalization if values were significantly changed
            original_values = [predictions.get('high', 0), predictions.get('low', 0), predictions.get('close', 0)]
            normalized_values = [normalized_predictions['high'], normalized_predictions['low'], normalized_predictions['close']]
            
            if any(abs(orig - norm) > historical_close_std * 0.5 for orig, norm in zip(original_values, normalized_values)):
                print(f"Warning: Predicted values were normalized:")
                print(f"  Original: High={predictions.get('high', 0):.2f}, Low={predictions.get('low', 0):.2f}, Close={predictions.get('close', 0):.2f}")
                print(f"  Normalized: High={normalized_predictions['high']:.2f}, Low={normalized_predictions['low']:.2f}, Close={normalized_predictions['close']:.2f}")
            
            return normalized_predictions
            
        except Exception as e:
            print(f"Error validating predictions: {e}")
            return predictions  # Return original predictions if validation fails
        
    def has_upward_trend(self, df, threshold=0.001):
        """Check if the trend of closing prices is upward using linear regression."""
        close_prices = df['close'].values
        time_index = np.arange(len(close_prices)).reshape(-1, 1)
        
        model = LinearRegression().fit(time_index, close_prices)
        slope = model.coef_[0]
        
        print(f"Trend slope: {slope:.5f}")
        
        return slope > threshold  # e.g. threshold = 0.001 means 0.1% increase per hour

    def has_daily_upward_trend(self, symbol, threshold=0.001):
        """Check if the daily trend is upward using daily OHLC data."""
        try:
            # Get daily data for trend analysis
            bars = self.get_cached_historical_data(
                symbol=symbol,
                resolution='daily',
                duration_str='30 D'  # Get 30 days of daily data
            )
            
            if not bars or len(bars) < 10:  # Need at least 10 days for trend analysis
                print(f"[{symbol}] Insufficient daily data for trend analysis")
                return False
            
            # Convert bars to dataframe
            daily_data = []
            for bar in bars:
                daily_data.append({
                    'date': pd.to_datetime(bar.date),
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })
            
            df = pd.DataFrame(daily_data)
            df = df.sort_values('date').reset_index(drop=True)
            
            # Use closing prices for trend analysis
            close_prices = df['close'].values
            time_index = np.arange(len(close_prices)).reshape(-1, 1)
            
            model = LinearRegression().fit(time_index, close_prices)
            slope = model.coef_[0]
            
            # Calculate percentage slope (normalize by average price)
            avg_price = np.mean(close_prices)
            slope_percent = (slope / avg_price) * 100
            
            print(f"[{symbol}] Daily trend slope: {slope:.5f} ({slope_percent:.3f}% per day)")
            
            # Check if slope is positive and above threshold
            is_upward = slope_percent > threshold
            
            if is_upward:
                print(f"[{symbol}] + Daily upward trend detected ({slope_percent:.3f}% per day)")
            else:
                print(f"[{symbol}] - No daily upward trend ({slope_percent:.3f}% per day)")
            
            return is_upward
            
        except Exception as e:
            print(f"[{symbol}] Error checking daily trend: {e}")
            return False


def main():
    trader = IBApiClient()
    
    def signal_handler(signum, frame):
        print("\nShutdown signal received...")
        trader.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        trader.run_live()
    except KeyboardInterrupt:
        trader.shutdown()
    except Exception as e:
        print(f"Error in main: {e}")
        trader.shutdown()

if __name__ == "__main__":
    main()
