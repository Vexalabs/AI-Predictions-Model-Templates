import joblib
import pandas as pd
from data_preprocessing import process_and_merge_data, TEAM_MAP_PATH, STADIUM_MAP_PATH

# Paths to the model and mappings
MODEL_PATH = "cricket_model.pkl"

def predict_match(toss_winner, stadium, team_a, team_b, toss_decision):
    """
    Predicts the winner of a cricket match using a multi-tiered approach.
    1. Primary Model Prediction (if historical data exists for the match-up).
    2. Fallback 1: Head-to-Head record.
    3. Fallback 2: Overall win rate.
    
    Args:
        toss_winner (str): The name of the team that won the toss.
        stadium (str): The name of the stadium.
        team_a (str): The name of Team A.
        team_b (str): The name of Team B.
        toss_decision (str): The decision of the toss-winning team ('Bat' or 'Field').

    Returns:
        str: The predicted winner and the prediction method.
    """
    try:
        # Load the trained model and mappings
        model = joblib.load(MODEL_PATH)
        team_map = joblib.load(TEAM_MAP_PATH)
        stadium_map = joblib.load(STADIUM_MAP_PATH)
        
        # Invert the team map to get team names from codes
        team_map_inv = {v: k for k, v in team_map.items()}
        
        # Load the combined dataset for fallback logic. This is the only place we need to load data.
        df_merged = process_and_merge_data()

        if df_merged is None:
            return "Prediction error: Failed to load and process data for fallbacks."
        
        # --- PRIMARY PREDICTION (ML Model) ---
        print("Attempting primary prediction using ML model...")
        
        # This try-except block handles cases where the primary model fails,
        # such as when a team or stadium is not in the training data.
        try:
            # Map inputs to encoded values
            team_a_encoded = team_map.get(team_a)
            team_b_encoded = team_map.get(team_b)
            toss_winner_encoded = team_map.get(toss_winner)
            stadium_encoded = stadium_map.get(stadium)

            # Check if any of the key values are not found (unforeseen data)
            if any(val is None for val in [team_a_encoded, team_b_encoded, toss_winner_encoded, stadium_encoded]):
                raise ValueError("Data not found in model's training set. Proceeding to fallbacks.")
            
            # Create a dictionary for input features to prevent FutureWarning
            features = {
                'Toss Winner Encoded': toss_winner_encoded,
                'Team A Encoded': team_a_encoded,
                'Team B Encoded': team_b_encoded,
                'Stadium Encoded': stadium_encoded,
                'Toss Decision_Bat': 1 if toss_decision.lower() == 'bat' else 0,
                'Toss Decision_Field': 1 if toss_decision.lower() == 'field' else 0,
                # Placeholders for other features
                'Score A': 250,
                'Wickets A': 5,
                'Overs Played A': 50,
                'Runrate A': 5.0,
                'Score B': 240,
                'Wickets B': 5,
                'Overs Played B': 45,
                'Runrate B': 5.5
            }
            
            # Create the input DataFrame from the dictionary
            X_pred = pd.DataFrame([features])
            
            # Ensure the input DataFrame has the same columns as the training data
            trained_features = list(model.feature_names_in_)
            X_pred = X_pred.reindex(columns=trained_features, fill_value=0)

            # Make the prediction
            prediction = model.predict(X_pred)[0]
            predicted_winner = team_map_inv.get(prediction)

            # Generate the reason for the prediction
            reason = f"Based on historical data (foreseen data) and the trained model, the prediction is influenced by key factors. The toss winner, {toss_winner}, chose to {toss_decision.lower()}."
            
            # Check if the stadium is a home ground for one of the teams
            if toss_winner == team_a or toss_winner == team_b:
                home_team_stadiums = df_merged[df_merged['Team_Innings'] == toss_winner]['Venue'].unique()
                if stadium in home_team_stadiums:
                    reason += f" The match is being played at {stadium}, which is a home ground for {toss_winner}, which historically can provide a significant advantage."
            
            return f"Predicted winner: {predicted_winner} (Method: ML Model)\nReason: {reason}"

        except (KeyError, ValueError) as e:
            print(f"Primary prediction failed because of unforeseen data: {e}")
            print("Proceeding to fallback models.")
        
        # --- FALLBACK 1 (Head-to-Head) ---
        print("Attempting fallback 1: Head-to-Head analysis...")
        
        # Use the merged DataFrame directly, which already contains match data
        h2h_matches = df_merged[
            ((df_merged['Team A'] == team_a) & (df_merged['Team B'] == team_b)) |
            ((df_merged['Team A'] == team_b) & (df_merged['Team B'] == team_a))
        ]
        
        if not h2h_matches.empty:
            team_a_wins = h2h_matches[h2h_matches['Match Winner'] == team_a].shape[0]
            team_b_wins = h2h_matches[h2h_matches['Match Winner'] == team_b].shape[0]

            if team_a_wins > team_b_wins:
                return f"Predicted winner: {team_a} (Method: Head-to-Head Record)\nReason: The ML model could not be used due to unforeseen data. {team_a} is predicted to win as they have a better historical record against {team_b}."
            elif team_b_wins > team_a_wins:
                return f"Predicted winner: {team_b} (Method: Head-to-Head Record)\nReason: The ML model could not be used due to unforeseen data. {team_b} is predicted to win as they have a better historical record against {team_a}."
            else:
                print("Head-to-head record is a tie. Proceeding to fallback 2.")
        else:
            print("No historical head-to-head data found. Proceeding to fallback 2.")

        # --- FALLBACK 2 (Overall Win Rate) ---
        print("Attempting fallback 2: Overall Win Rate analysis...")
        
        # Calculate overall win rates for each team
        total_matches = df_merged['Match ID'].nunique()
        team_a_win_rate = df_merged[df_merged['Match Winner'] == team_a].shape[0] / total_matches
        team_b_win_rate = df_merged[df_merged['Match Winner'] == team_b].shape[0] / total_matches

        if team_a_win_rate > team_b_win_rate:
            return f"Predicted winner: {team_a} (Method: Overall Win Rate)\nReason: The ML model and Head-to-Head analysis could not be used. {team_a} is predicted to win based on a higher overall win rate from all available data."
        elif team_b_win_rate > team_a_win_rate:
            return f"Predicted winner: {team_b} (Method: Overall Win Rate)\nReason: The ML model and Head-to-Head analysis could not be used. {team_b} is predicted to win based on a higher overall win rate from all available data."
        else:
            return "Prediction inconclusive. Teams have an equal overall win rate and no other historical data is available for a clear prediction."

    except Exception as e:
        return f"Prediction error: {e}"

if __name__ == "__main__":
    toss_winner_input = "India"
    stadium_input = "Ahmedabad"
    team_a_input = "India"
    team_b_input = "Australia"
    toss_decision_input = "Bat"
    
    result = predict_match(toss_winner_input, stadium_input, team_a_input, team_b_input, toss_decision_input)
    print(result)
