import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

class CryptoModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, df: pd.DataFrame):
        """
        Trains the model on the given dataframe.
        """
        X = df[['day']]
        y = df['price']
        self.model.fit(X, y)

    def predict(self, current_price: float, days: int = 7):
        """
        Makes a prediction for the next number of days.
        """
        # Create a simple time feature
        last_day = 0 # Assuming the training data ended at day 0
        future_days = np.array(range(last_day + 1, last_day + days + 1)).reshape(-1, 1)

        # For this simple model, we'll just create a trend based on the current price
        # A real model would use the input data.
        predictions = self.model.predict(future_days)

        # A more realistic approach would be to use a real time series model.
        # For now, let's just return a list of 7 prices starting from the current price.
        return [current_price * (1 + 0.01 * i) for i in range(days)]
