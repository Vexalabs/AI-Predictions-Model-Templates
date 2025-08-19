import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from data_preprocessing import load_and_clean_data

MODEL_PATH = "D:\\cricket_prediction\\cricket_model.pkl"
DATA_PATH = "D:\\cricket_prediction\\dataset\\CWC2023.csv"
ACCURACY_PATH = "D:\\cricket_prediction\\model_accuracy.txt"

def train_model():
    df = load_and_clean_data(DATA_PATH)

    if df is None:
        return

    # Map dataset column names.
    # New features: 'Toss Decision'
    features = ["Toss Winner", "Stadium", "Team A", "Team B", "Toss Decision"]
    target = "Wining Team"

    # Encode categorical data
    for col in features + [target]:
        df[col] = df[col].astype("category").cat.codes

    X = df[features]
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate and save model accuracy
    accuracy = model.score(X_test, y_test)
    with open(ACCURACY_PATH, 'w') as f:
        f.write(str(accuracy))

    # Save trained model
    joblib.dump(model, MODEL_PATH)

    print("Model trained and saved at", MODEL_PATH)
    print("Model accuracy saved at", ACCURACY_PATH)

if __name__ == "__main__":
    train_model()