import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import optuna
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def download_btc():
    print("Downloading BTC-USD data...")
    df = yf.download('BTC-USD', period='max', interval='1d', progress=False)
    if 'Adj Close' in df.columns:
        df = df.drop(columns=['Adj Close'])
    df = df.reset_index()
    return df

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

def add_on_chain_features(df):
    try:
        on_chain_df = pd.read_csv('bitcoin.csv')
        on_chain_df['Date'] = pd.to_datetime(on_chain_df['Timestamp'], unit='s').dt.date
        on_chain_df = on_chain_df.groupby('Date').mean()
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df = pd.merge(df, on_chain_df[['transaction_count']], on='Date', how='left').fillna(0)
    except FileNotFoundError:
        print("bitcoin.csv not found. Skipping on-chain features.")
    return df

def tune_xgb(X_train, y_train):
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = dict(zip(np.unique(y_train), class_weights))
    sample_weights = np.array([weight_dict[yy] for yy in y_train])

    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
            'gamma': trial.suggest_float('gamma', 0, 5.0)
        }
        
        val_size = int(0.1 * len(X_train))
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
        sw_tr = sample_weights[:-val_size]
        
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=sw_tr)
        probas = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, probas)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=200, show_progress_bar=True)
    print(f"Best XGBoost params: {study.best_params}")
    return study.best_params

class CNNLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3):
        super(CNNLSTMClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        conv_out = F.relu(self.conv1(x)).transpose(1, 2)
        lstm_out, _ = self.lstm(conv_out)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out

def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)

def feature_selection(X, y):
    model = XGBRFClassifier(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X, y)
    importances = model.feature_importances_
    selected = importances > np.mean(importances)
    return selected

def main():
    df = download_btc()
    df = create_features(df)
    df = add_on_chain_features(df)
    
    feature_cols = ['open-close', 'low-high', 'is_quarter_end', 'vol_change', 'lag_return', 'rsi_14', 'macd', 'sma_fast', 'obv', 'bb_pos', 'atr_14']
    if 'transaction_count' in df.columns:
        feature_cols.append('transaction_count')
    X = df[feature_cols].values
    y = df['target'].values
    
    selected_features = feature_selection(X, y)
    X = X[:, selected_features]
    feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if selected_features[i]]
    print(f"Selected features: {feature_cols}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pickle.dump(scaler, open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb'))
    pickle.dump(feature_cols, open(os.path.join(MODELS_DIR, 'feature_cols.pkl'), 'wb'))
    
    val_size = int(0.1 * len(X_train))
    X_tr_scaled = X_train_scaled[:-val_size]
    y_tr = y_train[:-val_size]
    X_val_scaled = X_train_scaled[-val_size:]
    y_val = y_train[-val_size:]
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    weight_dict = dict(zip(np.unique(y_tr), class_weights))
    print(f"Class weights: {weight_dict}")
    
    sample_weights_train = np.array([weight_dict[yy] for yy in y_train])
    
    xgb_params = tune_xgb(X_train_scaled, y_train)
    xgb_model = XGBClassifier(**xgb_params, random_state=RANDOM_SEED, n_jobs=-1)
    xgb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
    pickle.dump(xgb_model, open(os.path.join(MODELS_DIR, 'xgb.pkl'), 'wb'))
    
    rf_model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, class_weight='balanced')
    rf_model.fit(X_train_scaled, y_train)
    pickle.dump(rf_model, open(os.path.join(MODELS_DIR, 'rf.pkl'), 'wb'))
    
    svm_model = SVC(probability=True, random_state=RANDOM_SEED, class_weight='balanced')
    svm_model.fit(X_train_scaled, y_train)
    pickle.dump(svm_model, open(os.path.join(MODELS_DIR, 'svm.pkl'), 'wb'))
    
    lgb_model = LGBMClassifier(n_estimators=200, random_state=RANDOM_SEED, class_weight='balanced')
    lgb_model.fit(X_train_scaled, y_train)
    pickle.dump(lgb_model, open(os.path.join(MODELS_DIR, 'lgb.pkl'), 'wb'))
    
    seq_length = 40
    X_tr_seq, y_tr_seq = create_sequences(X_tr_scaled, y_tr, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, seq_length)
    
    train_dataset = TensorDataset(torch.tensor(X_tr_seq, dtype=torch.float32), torch.tensor(y_tr_seq, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32), torch.tensor(y_val_seq, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32), torch.tensor(y_test_seq, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = CNNLSTMClassifier(input_size=len(feature_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    best_val_acc = 0.0
    best_state = None
    patience = 50
    counter = 0
    
    for epoch in range(300):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            weights = torch.tensor([weight_dict[int(l.item())] for l in labels], device=device)
            loss = (loss * weights).mean()
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            val_probas = []
            val_loss = 0.0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probas = torch.sigmoid(outputs.squeeze())
                val_probas.extend(probas.cpu().numpy())
                predicted = (probas > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs.squeeze(), labels).mean().item()
            val_acc = correct / total if total > 0 else 0
            val_auc = roc_auc_score(y_val_seq, val_probas) if len(set(y_val_seq)) > 1 else 0
            val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        print(f'Epoch {epoch+1}: Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, LR: {optimizer.param_groups[0]["lr"]}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            print("Early stopping")
            break
    
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'cnn_lstm.pt'))
    
    model.eval()
    with torch.no_grad():
        test_probas = []
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probas = torch.sigmoid(outputs.squeeze())
            test_probas.extend(probas.cpu().numpy())
    dl_probs_test = np.array(test_probas)
    
    with torch.no_grad():
        val_probas = []
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probas = torch.sigmoid(outputs.squeeze())
            val_probas.extend(probas.cpu().numpy())
    dl_probs_val = np.array(val_probas)
    
    xgb_probs_val = xgb_model.predict_proba(X_val_scaled)[:, 1][seq_length:]
    xgb_probs_test = xgb_model.predict_proba(X_test_scaled)[:, 1][seq_length:]
    
    rf_probs_val = rf_model.predict_proba(X_val_scaled)[:, 1][seq_length:]
    rf_probs_test = rf_model.predict_proba(X_test_scaled)[:, 1][seq_length:]
    
    svm_probs_val = svm_model.predict_proba(X_val_scaled)[:, 1][seq_length:]
    svm_probs_test = svm_model.predict_proba(X_test_scaled)[:, 1][seq_length:]
    
    lgb_probs_val = lgb_model.predict_proba(X_val_scaled)[:, 1][seq_length:]
    lgb_probs_test = lgb_model.predict_proba(X_test_scaled)[:, 1][seq_length:]
    
    ensemble_probs_val = (dl_probs_val + xgb_probs_val + rf_probs_val + svm_probs_val + lgb_probs_val) / 5
    ensemble_preds_val = (ensemble_probs_val > 0.5).astype(int)
    ensemble_val_acc = accuracy_score(y_val_seq, ensemble_preds_val)
    print(f'Ensemble Val Acc: {ensemble_val_acc:.4f}')
    
    ensemble_probs_test = (dl_probs_test + xgb_probs_test + rf_probs_test + svm_probs_test + lgb_probs_test) / 5
    ensemble_preds = (ensemble_probs_test > 0.5).astype(int)
    ensemble_acc = accuracy_score(y_test_seq, ensemble_preds)
    ensemble_auc = roc_auc_score(y_test_seq, ensemble_probs_test)
    print(f'Ensemble - Test Acc: {ensemble_acc:.4f}, Test AUC: {ensemble_auc:.4f}')
    
    with open(os.path.join(MODELS_DIR, 'ensemble_weights.pkl'), 'wb') as f:
        pickle.dump({'dl': 0.2, 'xgb': 0.2, 'rf': 0.2, 'svm': 0.2, 'lgb': 0.2}, f)

if __name__ == "__main__":
    main()
