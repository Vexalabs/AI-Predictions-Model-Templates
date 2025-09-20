import time
import random
from typing import Dict, Any
import datetime
import logging

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIModel:
    def __init__(self):
        self.model: DecisionTreeRegressor | None = None
        self.last_trained_at: str | None = None
        self._load_model()  # Attempt to load initial model

    def _load_model(self):
        """
        Placeholder for loading a pre-trained model from disk or a cloud storage.
        This would load your actual ML model artifact.
        """
        logger.info("INFO: Attempting to load AI model...")
        # Initialize with default Decision Tree parameters
        self.model = DecisionTreeRegressor(
            max_depth=7, #tree depth = 7 if we want more the algorithm will be slower but must more accurate
            random_state=44,
            min_samples_split=3,
            min_samples_leaf=1
        )
        self.last_trained_at = str(datetime.datetime.now()).split(".")[0]
        logger.info("INFO: AI model initialized (DecisionTreeRegressor).")

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Makes predictions using the Decision Tree model.
        """

        logger.info(f"INFO: Making prediction with data: {data}")

        if not self.model:
            raise ValueError("Model is not loaded or trained.")

        current_price = data["current_price"]
        horizon = 7
        today = datetime.datetime.strptime(data["date"], "%Y-%m-%d")

        
        forecasts = []
        prev_price = current_price
        
        
        day_offsets = np.array([[i] for i in range(1, horizon + 1)])
        
        # Predict prices for all days at once
        predicted_prices = self.model.predict(day_offsets)
        
        for i in range(horizon):
            predicted_price = float(predicted_prices[i])
            pct_change = ((predicted_price - prev_price) / prev_price) * 100
         
            if pct_change>=0:
                direction ="UP"
            else:
                direction = "DOWN"

            forecasts.append(
                {
                    "timestamp": (today + datetime.timedelta(days=i+1)).strftime("%Y-%m-%dT00:00:00"),
                    "forecast_index": i + 1,
                    "price": round(predicted_price, 2),  
                    "pct_change": round(pct_change, 2),
                    "direction": direction,
                }
            )
            prev_price = predicted_price

        return {
            "prediction_timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "predictions": [
                {
                    "name": "Jad Matta",  # Changed to match example
                    "description": "Stock price forecast with 7-day horizon.",  
                    "horizon": 7,
                    "frequency": 1,
                    "stock_name": data["name"], 
                    "forecasts": forecasts,
                }
            ],
        }

    def retrain(self):
        """
        Retrains the Decision Tree model with new data.
        """
        logger.info("INFO: Starting AI model retraining process...")
        try:
           
            np.random.seed(44) 
            
            
            X_train = np.array([[i] for i in range(1, 366)])
            
         
            base_trend = 100 + np.arange(1, 366) * 0.5  # Upward trend in year
            seasonal_effect = 10 * np.sin(np.arange(1, 366) * 0.25)  # yearly pattern
            noise = np.random.normal(0, 2, 100)  # Random noise
            
            y_train = base_trend + seasonal_effect + noise

            # Split data for training and validation
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

            logger.info("INFO: Training DecisionTreeRegressor model...")
            
            # Train the Decision Tree model
            self.model.fit(X_train_split, y_train_split)
            
            # Evaluate the model
            y_pred = self.model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            logger.info(f"INFO: Model evaluation - MSE: {mse:.4f}, R²: {r2:.4f}")

            self.last_trained_at = str(datetime.datetime.now()).split(".")[0]
            logger.info(
                f"INFO: Model retraining complete. New model trained at {self.last_trained_at}"
            )

            return {
                "status": "success",
                "message": "Model retrained and loaded successfully.",
                "last_trained_at": self.last_trained_at,
                "evaluation_metrics": {
                    "mse": round(mse, 2),
                    "r2_score": round(r2, 2)
                }
            }
        except Exception as e:
            logger.error(f"ERROR: Model retraining failed: {e}")
            return {"status": "error", "message": f"Model retraining failed: {str(e)}"}