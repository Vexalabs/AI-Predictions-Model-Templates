import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, "winner_model.joblib")
ENCODER_FILE = os.path.join(MODEL_DIR, "winner_encoder.joblib")
SCALER_FILE = os.path.join(MODEL_DIR, "winner_scaler.joblib")

class AIModel:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.scaler = None
        self.last_trained_at = None
        self._load_model()

    def _load_model(self):
        """Load trained model if it exists, otherwise keep None."""
        if os.path.exists(MODEL_FILE):
            self.model = joblib.load(MODEL_FILE)
            self.encoder = joblib.load(ENCODER_FILE)
            self.scaler = joblib.load(SCALER_FILE)
            logger.info("Model loaded from disk.")
        else:
            logger.warning("No trained model found. Please call /retrain first.")

    def predict(self, data: dict):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("No trained model loaded. Please retrain first.")

        home_team = data["home_team"]
        away_team = data["away_team"]
        home_odds = float(data["home_team_odds_avg"])
        away_odds = float(data["away_team_odds_avg"])

        X_scaled = self.scaler.transform([[home_odds, away_odds]])
        probs = self.model.predict_proba(X_scaled)[0]
        idx = probs.argmax()
        winner = self.encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx] * 100)

        return {
            "winner": winner,
            "winner_confidence_pct": round(confidence, 2),
            "winner_best_bet_odds": home_odds if winner == home_team else away_odds,
            "over_under": None,
            "over_under_confidence_pct": None,
            "over_under_best_bet_odds": None,
            "spread": None,
            "spread_confidence_pct": None,
            "spread_best_bet_odds": None,
        }

    def retrain(self, data_path="data/training_data.csv"):
        """Train a new model from the dataset."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        df = pd.read_csv(data_path)
        required_cols = ["home_team_odds_avg", "away_team_odds_avg", "winner"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Dataset missing required column: {col}")

        X = df[["home_team_odds_avg", "away_team_odds_avg"]].astype(float)
        y = df["winner"].astype(str)

        encoder = LabelEncoder()
        y_enc = encoder.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_scaled, y_enc)

        joblib.dump(clf, MODEL_FILE)
        joblib.dump(encoder, ENCODER_FILE)
        joblib.dump(scaler, SCALER_FILE)

        self.model = clf
        self.encoder = encoder
        self.scaler = scaler
        logger.info("Model retrained and saved successfully.")
        return {"status": "success", "message": "Model retrained successfully."}
