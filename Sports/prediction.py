import joblib
import pandas as pd

MODEL_PATH = "D:\\cricket_prediction\\cricket_model.pkl"
DATA_PATH = "D:\\cricket_prediction\\dataset\\CWC2023.csv"
ACCURACY_PATH = "D:\\cricket_prediction\\model_accuracy.txt"

def get_team_stats(df, team, stadium):
    """Calculates team win percentage at a specific stadium."""
    stadium_matches = df[df['Stadium'] == stadium]
    team_matches_at_stadium = stadium_matches[(stadium_matches['Team A'] == team) | (stadium_matches['Team B'] == team)]
    if team_matches_at_stadium.empty:
        return 0, 0
    
    wins = (team_matches_at_stadium['Wining Team'] == team).sum()
    total_matches = len(team_matches_at_stadium)
    return wins, total_matches

def get_head_to_head_stats(df, team_a, team_b):
    """Calculates head-to-head win percentage between two teams."""
    head_to_head_matches = df[((df['Team A'] == team_a) & (df['Team B'] == team_b)) | ((df['Team A'] == team_b) & (df['Team B'] == team_a))]
    if head_to_head_matches.empty:
        return 0, 0, 0
        
    wins_a = (head_to_head_matches['Wining Team'] == team_a).sum()
    wins_b = (head_to_head_matches['Wining Team'] == team_b).sum()
    
    return wins_a, wins_b, len(head_to_head_matches)

def get_overall_stats(df, team):
    """Calculates a team's overall win percentage in the tournament."""
    team_matches = df[(df['Team A'] == team) | (df['Team B'] == team)]
    if team_matches.empty:
        return 0, 0

    wins = (team_matches['Wining Team'] == team).sum()
    total_matches = len(team_matches)
    return wins, total_matches

def get_accuracy():
    """Loads the model accuracy from the saved file."""
    try:
        with open(ACCURACY_PATH, 'r') as f:
            accuracy = float(f.read())
        return accuracy
    except FileNotFoundError:
        return None
    except Exception as e:
        return f"Error loading accuracy: {e}"

def predict_match(toss_winner, stadium, team_a, team_b, toss_decision):
    try:
        original_df = pd.read_csv(DATA_PATH)

        # Check if the exact match combination exists in the dataset
        exact_match = original_df[(
            ((original_df['Team A'] == team_a) & (original_df['Team B'] == team_b)) |
            ((original_df['Team A'] == team_b) & (original_df['Team B'] == team_a))
        ) & (original_df['Stadium'] == stadium)]

        if not exact_match.empty:
            model = joblib.load(MODEL_PATH)
            
            # Use the trained model for known scenarios
            categories = {
                'Toss Winner': original_df['Toss Winner'].astype('category').cat.categories.tolist(),
                'Stadium': original_df['Stadium'].astype('category').cat.categories.tolist(),
                'Team A': original_df['Team A'].astype('category').cat.categories.tolist(),
                'Team B': original_df['Team B'].astype('category').cat.categories.tolist(),
                'Toss Decision': original_df['Toss Decision'].astype('category').cat.categories.tolist()
            }
            
            for column, value in [('Toss Winner', toss_winner), ('Stadium', stadium), ('Team A', team_a), ('Team B', team_b), ('Toss Decision', toss_decision)]:
                if value not in categories[column]:
                    return f"Prediction error: The value '{value}' for '{column}' is not a valid category. Please use a value from the training data."

            input_df = pd.DataFrame([{
                "Toss Winner": toss_winner,
                "Stadium": stadium,
                "Team A": team_a,
                "Team B": team_b,
                "Toss Decision": toss_decision
            }])
            
            for col in input_df.columns:
                dtype = pd.CategoricalDtype(categories[col])
                input_df[col] = input_df[col].astype(dtype)

            input_df_encoded = input_df.apply(lambda x: x.cat.codes)
            prediction = model.predict(input_df_encoded)[0]
            
            wining_team_cats = original_df['Wining Team'].astype('category').cat.categories
            wining_team_mapping = dict(enumerate(wining_team_cats))
            predicted_winner = wining_team_mapping.get(prediction, "Unknown")
            
            if predicted_winner not in [team_a, team_b]:
                return f"Prediction for known match: The model's prediction is unreliable. Predicted winner is {predicted_winner}."
            else:
                return f"Predicted Match Winner (Model): {predicted_winner}"

        else:
            # Fallback prediction for unseen match-ups
            print("This match-up has not been seen before. Using fallback prediction method.")
            
            # Stadium performance analysis
            wins_a_stadium, total_a_stadium = get_team_stats(original_df, team_a, stadium)
            wins_b_stadium, total_b_stadium = get_team_stats(original_df, team_b, stadium)

            # Head-to-head performance analysis
            wins_a_h2h, wins_b_h2h, total_h2h = get_head_to_head_stats(original_df, team_a, team_b)

            # Calculate scores
            score_a = 0
            score_b = 0
            reasons = []

            # Factor 1: Stadium Performance
            if total_a_stadium > 0 or total_b_stadium > 0:
                win_rate_a_stadium = wins_a_stadium / total_a_stadium if total_a_stadium > 0 else 0
                win_rate_b_stadium = wins_b_stadium / total_b_stadium if total_b_stadium > 0 else 0
                
                if win_rate_a_stadium > win_rate_b_stadium:
                    score_a += 1
                    reasons.append(f"{team_a} has a better record at {stadium} ({wins_a_stadium}/{total_a_stadium} wins) than {team_b} ({wins_b_stadium}/{total_b_stadium} wins).")
                elif win_rate_b_stadium > win_rate_a_stadium:
                    score_b += 1
                    reasons.append(f"{team_b} has a better record at {stadium} ({wins_b_stadium}/{total_b_stadium} wins) than {team_a} ({wins_a_stadium}/{total_a_stadium} wins).")
                else:
                    reasons.append("Both teams have similar records or no data for this stadium.")
            else:
                reasons.append("No historical data for either team at this stadium.")

            # Factor 2: Head-to-head performance
            if total_h2h > 0:
                if wins_a_h2h > wins_b_h2h:
                    score_a += 1
                    reasons.append(f"{team_a} has a better head-to-head record against {team_b} ({wins_a_h2h}/{total_h2h} wins).")
                elif wins_b_h2h > wins_a_h2h:
                    score_b += 1
                    reasons.append(f"{team_b} has a better head-to-head record against {team_a} ({wins_b_h2h}/{total_h2h} wins).")
                else:
                    reasons.append("Head-to-head record is tied or no data available.")
            else:
                reasons.append("No historical head-to-head data for these teams.")
            
            # Tie-breaking logic: Overall tournament performance
            if score_a == score_b:
                wins_a_overall, total_a_overall = get_overall_stats(original_df, team_a)
                wins_b_overall, total_b_overall = get_overall_stats(original_df, team_b)
                
                win_rate_a_overall = wins_a_overall / total_a_overall if total_a_overall > 0 else 0
                win_rate_b_overall = wins_b_overall / total_b_overall if total_b_overall > 0 else 0
                
                if win_rate_a_overall > win_rate_b_overall:
                    reasons.append(f"Tie-breaker: {team_a} has a better overall tournament win rate ({win_rate_a_overall:.2f}) than {team_b} ({win_rate_b_overall:.2f}).")
                    return f"Predicted Match Winner (Fallback): {team_a}\nReasoning:\n- " + "\n- ".join(reasons)
                elif win_rate_b_overall > win_rate_a_overall:
                    reasons.append(f"Tie-breaker: {team_b} has a better overall tournament win rate ({win_rate_b_overall:.2f}) than {team_a} ({win_rate_a_overall:.2f}).")
                    return f"Predicted Match Winner (Fallback): {team_b}\nReasoning:\n- " + "\n- ".join(reasons)
                else:
                    return "Predicted Match Winner (Fallback): The match is too close to call based on available data."


            # Final result based on combined score
            if score_a > score_b:
                return f"Predicted Match Winner (Fallback): {team_a}\nReasoning:\n- " + "\n- ".join(reasons)
            elif score_b > score_a:
                return f"Predicted Match Winner (Fallback): {team_b}\nReasoning:\n- " + "\n- ".join(reasons)

    except FileNotFoundError:
        return f"Prediction error: The data file at '{DATA_PATH}' was not found. Please ensure the file exists at the correct path."
    except Exception as e:
        return f"Prediction error: {e}"