import pandas as pd
import joblib

# Paths for the new datasets
DATA_PATH = "D:/cricket_prediction/dataset/CWC2023.csv"
BATTING_SUMMARY_PATH = "D:/cricket_prediction/dataset/batting_summary.csv"
BOWLING_SUMMARY_PATH = "D:/cricket_prediction/dataset/bowling_summary.csv"
PLAYER_INFO_PATH = "D:/cricket_prediction/dataset/world_cup_players_info.csv"
MATCH_SCHEDULE_PATH = "D:/cricket_prediction/dataset/match_schedule_results.csv"

# Paths to save the mappings for the prediction script
TEAM_MAP_PATH = "cricket_team_map.pkl"
STADIUM_MAP_PATH = "cricket_stadium_map.pkl"

def load_all_data():
    """
    Loads all datasets and merges them into a single comprehensive DataFrame.
    """
    try:
        df_main = pd.read_csv(DATA_PATH)
        df_schedule = pd.read_csv(MATCH_SCHEDULE_PATH)
        df_batting = pd.read_csv(BATTING_SUMMARY_PATH)
        df_bowling = pd.read_csv(BOWLING_SUMMARY_PATH)
        df_players = pd.read_csv(PLAYER_INFO_PATH)

        # Drop duplicates and fill missing values
        for df in [df_main, df_schedule, df_batting, df_bowling, df_players]:
            df.drop_duplicates(inplace=True)
            # Fix for FutureWarning: use obj.ffill() instead of fillna(method="ffill")
            df.ffill(inplace=True)
        
        return df_main, df_schedule, df_batting, df_bowling, df_players
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None

def process_and_merge_data():
    """
    Processes and merges all loaded datasets.
    """
    df_main, df_schedule, df_batting, df_bowling, df_players = load_all_data()

    if df_main is None:
        return None

    # Merge main data with match schedule to get the winner column
    df_merged = pd.merge(df_main, df_schedule[['Match_no', 'Winner']], left_on='Match ID', right_on='Match_no', how='left')
    df_merged.rename(columns={'Winner': 'Match Winner'}, inplace=True)
    df_merged.drop('Match_no', axis=1, inplace=True)
    
    # Clean team names for consistent merging
    df_merged['Team A'] = df_merged['Team A'].str.strip()
    df_merged['Team B'] = df_merged['Team B'].str.strip()
    df_merged['Match Winner'] = df_merged['Match Winner'].str.strip()

    print("Data loaded and preprocessed successfully.")
    return df_merged

def create_and_save_mappings(df):
    """
    Dynamically creates team and stadium mappings and saves them.
    """
    # Create team map
    all_teams = pd.concat([df['Team A'], df['Team B']]).unique()
    team_map = {name: i for i, name in enumerate(all_teams)}
    joblib.dump(team_map, TEAM_MAP_PATH)
    print("Team map created and saved.")

    # Create stadium map
    stadium_map = {name: i for i, name in enumerate(df['Stadium'].unique())}
    joblib.dump(stadium_map, STADIUM_MAP_PATH)
    print("Stadium map created and saved.")

if __name__ == "__main__":
    df = process_and_merge_data()
    if df is not None:
        create_and_save_mappings(df)
        print("Final DataFrame head:")
        print(df.head())
