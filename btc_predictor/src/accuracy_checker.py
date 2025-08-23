import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# CNN-LSTM Model
class CNN_LSTM(nn.Module):
    def __init__(self, input_size=7, cnn_channels=64, lstm_hidden=512, lstm_layers=3):
        super(CNN_LSTM, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=3)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(lstm_hidden * 2, 1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return self.sigmoid(x)

# Feature Engineering
def create_features(df):
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
    
    df = df.dropna()
    return df

# Download Data
print("Downloading BTC-USD data for last 5 years...")
df = yf.download("BTC-USD", period="5y", interval="1d", progress=False)
df.reset_index(inplace=True)
df = create_features(df)

# Load Features & Scaler
feature_cols = joblib.load("models/feature_cols.pkl")
scaler = joblib.load("models/scaler.pkl")
X = df[feature_cols].copy()
y = df['target'].values
X_scaled = scaler.transform(X)

# Spliting
X_train, X_test = X_scaled[:-250], X_scaled[-250:]
y_train, y_test = y[:-250], y[-250:]

# Load Models
models = {}
model_files = {
    'xgb': "models/xgb.pkl",
    'rf': "models/rf.pkl",
    'svm': "models/svm.pkl",
}
for name, path in model_files.items():
    if Path(path).exists():
        models[name] = joblib.load(path)

# Load CNN-LSTM
dl_path = Path("models/cnn_lstm.pt")
if dl_path.exists():
    try:
        dl_model = CNN_LSTM(
            input_size=len(feature_cols),  # match trained features
            cnn_channels=64,
            lstm_hidden=512,
            lstm_layers=3
        )
        # Allow partial load
        state_dict = torch.load(dl_path, map_location=torch.device('cpu'))
        dl_model.load_state_dict(state_dict, strict=False)
        dl_model.eval()
        models['dl'] = dl_model
    except Exception as e:
        print(f"Skipping CNN-LSTM due to load error: {e}")

# Load Ensemble Weights
ensemble_weights = {}
weights_path = Path("models/ensemble_weights.pkl")
if weights_path.exists():
    ensemble_weights = joblib.load(weights_path)

# Evaluation
results = {}
def evaluate_model(name, preds):
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)
    results[name] = acc
    print(f"\n{name.upper()} Metrics:")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

# Evaluate Each Model
for name, model in models.items():
    if name == 'dl':
        X_dl = torch.tensor(
            X_test.reshape(X_test.shape[0], len(feature_cols), 1),
            dtype=torch.float32
        )
        preds = (model(X_dl).detach().numpy().flatten() > 0.5).astype(int)
    else:
        preds = model.predict(X_test)
    evaluate_model(name, preds)

# Ensemble 
if ensemble_weights:
    preds_sum = np.zeros(len(y_test))
    total_weight = 0
    for name, weight in ensemble_weights.items():
        if name not in models:
            print(f"Skipping ensemble weight for missing model: {name}")
            continue
        if name == 'dl':
            X_dl = torch.tensor(
                X_test.reshape(X_test.shape[0], len(feature_cols), 1),
                dtype=torch.float32
            )
            pred_probs = models[name](X_dl).detach().numpy().flatten()
        else:
            if hasattr(models[name], 'predict_proba'):
                pred_probs = models[name].predict_proba(X_test)[:, 1]
            else:
                pred_probs = model.predict(X_test)
        preds_sum += weight * pred_probs
        total_weight += weight
    if total_weight > 0:
        ensemble_preds = (preds_sum / total_weight > 0.5).astype(int)
        evaluate_model("ensemble", ensemble_preds)
else:
    print("No ensemble weights found.")

# telling best model
best_model = max(results, key=results.get)
print(f"\nBest model based on Accuracy: {best_model.upper()} ({results[best_model]:.4f})")
