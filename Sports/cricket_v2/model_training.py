import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from data_preprocessing import process_and_merge_data, TEAM_MAP_PATH, STADIUM_MAP_PATH

MODEL_PATH = "cricket_model.pkl"

def train_model():
    """
    Loads processed data, trains a RandomForestClassifier model,
    and saves the trained model.
    """
    # Load and process the combined dataset
    df = process_and_merge_data()

    if df is None:
        print("Failed to load data. Exiting model training.")
        return

    # Dynamically load the saved mappings
    team_map = joblib.load(TEAM_MAP_PATH)
    stadium_map = joblib.load(STADIUM_MAP_PATH)

    # Encode categorical features
    df['Team A Encoded'] = df['Team A'].map(team_map)
    df['Team B Encoded'] = df['Team B'].map(team_map)
    df['Toss Winner Encoded'] = df['Toss Winner'].map(team_map)
    df['Stadium Encoded'] = df['Stadium'].map(stadium_map)
    
    # Target encoding: map 'Winning Team' to an integer
    df['Winning Team Encoded'] = df['Match Winner'].map(team_map)

    # Define features for the model
    features = [
        'Toss Winner Encoded', 'Toss Decision', 'Score A', 'Wickets A', 'Overs Played A',
        'Runrate A', 'Score B', 'Wickets B', 'Overs Played B', 'Runrate B',
        'Team A Encoded', 'Team B Encoded', 'Stadium Encoded'
    ]
    
    # Filter out rows with missing data for training
    df_model = df.dropna(subset=features + ['Winning Team Encoded'])
    
    # One-hot encode the 'Toss Decision' column
    df_model = pd.get_dummies(df_model, columns=['Toss Decision'], drop_first=True)

    # Update features list with the new one-hot encoded columns
    features.remove('Toss Decision')
    features.extend([col for col in df_model.columns if 'Toss Decision_' in col])
    
    X = df_model[features]
    y = df_model['Winning Team Encoded']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, MODEL_PATH)

    print("Model trained and saved at", MODEL_PATH)

if __name__ == "__main__":
    train_model()
