import os
import time
import random
from typing import Dict, Any
import datetime
import logging
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()



class AIModel:
    def __init__(self, api_key=None, symbol="AAPL"):
        self.api_key = os.getenv("ALPHAVANTAGE_API_KEY", "demo")
        self.symbol = symbol
        self.model = None
        self.last_trained_at = None
        self.model_path = f"model_{self.symbol}.pkl"
        self.metrics_path = f"metrics_{self.symbol}.json"
        self.metrics = {"mae": None, "mse": None, "r2": None}
        self._load_model()  # Attempt to load an initial model

    def _load_model(self):
        """
        Load a pre-trained model and metrics from disk if available.
        """
        logger.info("INFO: Attempting to load AI model...")
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.last_trained_at = datetime.datetime.fromtimestamp(os.path.getmtime(self.model_path)).isoformat()
            logger.info(f"INFO: AI model loaded from {self.model_path}.")
            # Load metrics if available
            import json
            if os.path.exists(self.metrics_path):
                try:
                    with open(self.metrics_path, "r") as f:
                        self.metrics = json.load(f)
                except Exception as e:
                    logger.error(f"ERROR: Failed to load metrics: {e}")
        else:
            logger.info("INFO: No trained model found. Please retrain.")

    def _fetch_data(self):
        """
        Fetch historical stock data from Alpha Vantage.
        """
        url = (
            f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={self.symbol}&outputsize=full&apikey={self.api_key}&datatype=csv"
        )
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception("Failed to fetch data from Alpha Vantage.")
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        return df

    def _preprocess(self, df):
        """
        Preprocess the Alpha Vantage data for ML model (predict next day's close).
        Converts string columns to float and extracts features.
        """
        # Log columns for debugging
        logger.info(f"Columns in fetched DataFrame: {df.head()}")
        # Find the correct date column
        date_col = None
        for candidate in ["timestamp", "date"]:
            if candidate in df.columns:
                date_col = candidate
                break
        if not date_col:
            raise Exception(f"No date column found in DataFrame. Columns: {df.columns.tolist()}")
        df = df.sort_values(date_col)
        # Convert relevant columns to float
        for col in ["open", "high", "low", "close", "adjusted_close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
            else:
                raise Exception(f"Column '{col}' not found in DataFrame. Columns: {df.columns.tolist()}")
        df["target"] = df["close"].shift(-1)
        df = df.dropna()
        X = df[["open", "high", "low", "close", "volume"]]
        y = df["target"]
        return X, y

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the next day's close price given features.
        """
        logger.info(f"INFO: Making prediction with data: {data}")
        if self.model is None:
            raise Exception("Model not trained.")
        # Expecting data to have keys: open, high, low, close, volume
        X = pd.DataFrame([data])
        prediction = self.model.predict(X)[0]
        return {"predicted_close": prediction}

    def retrain(self):
        """
        Retrain the AI model using Alpha Vantage data and scikit-learn, and evaluate metrics.
        Persists metrics to disk for future retrieval.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        logger.info("INFO: Starting AI model retraining process...")
        try:
            logger.info("INFO: Fetching historical stock data from Alpha Vantage...")
            df = self._fetch_data()
            logger.info("INFO: Preprocessing data...")
            X, y = self._preprocess(df)
            logger.info("INFO: Splitting data for evaluation...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logger.info("INFO: Training new model version...")
            model = LinearRegression()
            model.fit(X_train, y_train)
            self.model = model
            self.last_trained_at = datetime.datetime.now().isoformat()
            joblib.dump(model, self.model_path)
            # Evaluate
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            self.metrics = {"mae": mae, "mse": mse, "r2": r2}
            # Persist metrics to disk
            import json
            try:
                with open(self.metrics_path, "w") as f:
                    json.dump(self.metrics, f)
            except Exception as e:
                logger.error(f"ERROR: Failed to save metrics: {e}")
            logger.info(f"Model evaluation - MAE: {mae}, MSE: {mse}, R2: {r2}")
            logger.info(f"INFO: Model retrained and saved to {self.model_path}.")
            return {
                "status": "success",
                "message": "Model retrained and loaded successfully.",
                "last_trained_at": self.last_trained_at,
                "mae": mae,
                "mse": mse,
                "r2": r2
            }
        except Exception as e:
            logger.error(f"ERROR: Model retraining failed: {e}")
            return {"status": "error", "message": f"Model retraining failed: {str(e)}"}

    def get_metrics(self):
        """
        Return the latest evaluation metrics for the model.
        """
        return self.metrics
