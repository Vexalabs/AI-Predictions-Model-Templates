import os
import datetime
import logging

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, "sports_model.joblib")
TEAM_ENCODER_FILE = os.path.join(MODEL_DIR, "team_encoder.joblib")
WINNER_ENCODER_FILE = os.path.join(MODEL_DIR, "winner_encoder.joblib")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.joblib")


class AIModel:
    def __init__(self):
        self.model = None
        self.team_encoder = None
        self.winner_encoder = None
        self.scaler = None
        self.last_trained_at = None
        self._load_model()

    def _load_model(self):
        """Load model and artifacts from disk if they exist."""
        if (
            os.path.exists(MODEL_FILE)
            and os.path.exists(TEAM_ENCODER_FILE)
            and os.path.exists(WINNER_ENCODER_FILE)
            and os.path.exists(SCALER_FILE)
        ):
            try:
                self.model = joblib.load(MODEL_FILE)
                self.team_encoder = joblib.load(TEAM_ENCODER_FILE)
                self.winner_encoder = joblib.load(WINNER_ENCODER_FILE)
                self.scaler = joblib.load(SCALER_FILE)
                logger.info("Model and artifacts loaded from disk.")
            except Exception as e:
                logger.exception(f"Failed to load model artifacts: {e}")
                self.model = None
        else:
            logger.warning("No trained model found. Please call /retrain first.")

    def retrain(self, data_path: str = "data/training_data.csv"):
        """Train or retrain the model using the CSV at data_path.
        Expects columns: home_team, away_team, home_team_odds_avg, away_team_odds_avg, winner
        """
        logger.info("Starting retrain process...")
        if not os.path.exists(data_path):
            msg = f"Dataset not found at {data_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        # Load data
        df = pd.read_csv(data_path)

        required_cols = [
            "home_team",
            "away_team",
            "home_team_odds_avg",
            "away_team_odds_avg",
            "winner",
        ]
        for col in required_cols:
            if col not in df.columns:
                msg = f"Dataset missing required column: {col}"
                logger.error(msg)
                raise ValueError(msg)

        # Ensure types and drop bad rows
        df["home_team"] = df["home_team"].astype(str)
        df["away_team"] = df["away_team"].astype(str)
        df["winner"] = df["winner"].astype(str)
        df["home_team_odds_avg"] = pd.to_numeric(df["home_team_odds_avg"], errors="coerce")
        df["away_team_odds_avg"] = pd.to_numeric(df["away_team_odds_avg"], errors="coerce")

        df = df.dropna(subset=required_cols)  # remove rows with missing required data
        if df.shape[0] < 10:
            msg = "Not enough valid rows in dataset after cleaning."
            logger.error(msg)
            raise ValueError(msg)

        # Encode team names (fit on both home + away values)
        combined = pd.concat([df["home_team"], df["away_team"]], ignore_index=True)
        team_encoder = LabelEncoder()
        team_encoder.fit(combined)

        df["home_team_enc"] = team_encoder.transform(df["home_team"])
        df["away_team_enc"] = team_encoder.transform(df["away_team"])

        # Encode winners
        winner_encoder = LabelEncoder()
        df["winner_enc"] = winner_encoder.fit_transform(df["winner"])

        # Prepare features and target
        X = df[["home_team_enc", "away_team_enc", "home_team_odds_avg", "away_team_odds_avg"]].astype(
            float
        )
        y = df["winner_enc"].astype(int)

        # Scale numeric features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split for quick validation (optional)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Optional validation score
        try:
            test_score = model.score(X_test, y_test)
            logger.info(f"Retrain complete — validation accuracy: {test_score:.4f}")
        except Exception:
            test_score = None

        # Save artifacts
        joblib.dump(model, MODEL_FILE)
        joblib.dump(team_encoder, TEAM_ENCODER_FILE)
        joblib.dump(winner_encoder, WINNER_ENCODER_FILE)
        joblib.dump(scaler, SCALER_FILE)

        # Update in-memory objects
        self.model = model
        self.team_encoder = team_encoder
        self.winner_encoder = winner_encoder
        self.scaler = scaler
        self.last_trained_at = datetime.datetime.utcnow().isoformat()

        logger.info(f"Model retrained and saved at {self.last_trained_at}")
        return {
            "status": "success",
            "message": "Model retrained successfully.",
            "validation_score": test_score,
            "last_trained_at": self.last_trained_at,
        }

    def predict(self, data: dict):
        """Make prediction using the trained model and return structured response."""
        if self.model is None or self.team_encoder is None or self.winner_encoder is None or self.scaler is None:
            raise ValueError("No trained model loaded. Please retrain first.")

        # Extract inputs
        home_team = str(data.get("home_team", ""))
        away_team = str(data.get("away_team", ""))
        try:
            home_odds = float(data.get("home_team_odds_avg"))
            away_odds = float(data.get("away_team_odds_avg"))
        except Exception:
            raise ValueError("Odds must be numeric values.")

        # Encode teams; handle unseen teams by falling back to 0 and logging
        try:
            home_enc = self.team_encoder.transform([home_team])[0]
        except ValueError:
            logger.warning(f"Unknown home_team '{home_team}' — falling back to 0.")
            home_enc = 0
        try:
            away_enc = self.team_encoder.transform([away_team])[0]
        except ValueError:
            logger.warning(f"Unknown away_team '{away_team}' — falling back to 0.")
            away_enc = 0

        # Prepare and scale input
        X = [[home_enc, away_enc, home_odds, away_odds]]
        X_scaled = self.scaler.transform(X)

        # Predict probabilities
        probs = self.model.predict_proba(X_scaled)[0]
        idx = int(probs.argmax())
        winner_label = self.winner_encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx] * 100.0)

        # Best bet odds logic (choose odds for predicted winner if string matches; otherwise choose higher confidence odds)
        best_bet_odds = None
        if winner_label == home_team:
            best_bet_odds = home_odds
        elif winner_label == away_team:
            best_bet_odds = away_odds
        else:
            # If labels don't match exact input names, fall back to smaller odds (higher implied prob)
            best_bet_odds = min(home_odds, away_odds)

        return {
            "winner": winner_label,
            "winner_confidence_pct": round(confidence, 2),
            "winner_best_bet_odds": float(best_bet_odds) if best_bet_odds is not None else None,
            "over_under": None,
            "over_under_confidence_pct": None,
            "over_under_best_bet_odds": None,
            "spread": None,
            "spread_confidence_pct": None,
            "spread_best_bet_odds": None,
        }
