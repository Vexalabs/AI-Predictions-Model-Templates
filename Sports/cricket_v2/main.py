import pandas as pd
import sys
import os
from prediction import predict_match

# List of CSV files to check for
REQUIRED_FILES = [r'D:\cricket_prediction\dataset\world_cup_players_info.csv']

def get_player_list(team_name):
    """
    Reads the player info CSV and returns a list of players for the given team.
    
    Args:
        team_name (str): The name of the team.
        
    Returns:
        list: A list of player names.
    """
    try:
        # Use the explicit file path from REQUIRED_FILES
        df_players = pd.read_csv(REQUIRED_FILES[0])
        team_players = df_players[df_players['team_name'].str.strip() == team_name.strip()]['player_name'].tolist()
        return team_players
    except FileNotFoundError:
        print(f"Error: The file '{REQUIRED_FILES[0]}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the player data: {e}")
        sys.exit(1)

def select_players(team_name):
    """
    Prompts the user to select 11 players for a team from a given list by number.
    
    Args:
        team_name (str): The name of the team.
        
    Returns:
        list: The list of selected player names.
    """
    all_players = get_player_list(team_name)
    selected_players = []
    
    print(f"\nAvailable players for {team_name}:")
    for i, player in enumerate(all_players):
        print(f"  {i+1}. {player}")
    
    print(f"\nPlease select 11 players for {team_name} by entering the corresponding number.")
    
    while len(selected_players) < 11:
        try:
            selection_input = input(f"Enter player #{len(selected_players) + 1} (1- {len(all_players)}): ").strip()
            
            if not selection_input.isdigit():
                print("Invalid input. Please enter a number.")
                continue

            player_index = int(selection_input) - 1
            
            if player_index < 0 or player_index >= len(all_players):
                print("Invalid number. Please select a number from the list.")
                continue

            player_name = all_players[player_index]

            if player_name in selected_players:
                print(f"{player_name} is already selected. Please choose another player.")
            else:
                selected_players.append(player_name)
                print(f"Added {player_name}.")

        except ValueError:
            print("Invalid input. Please enter a valid number.")
        except IndexError:
            print("Invalid selection. Please choose a number from the list.")

    return selected_players

def main():
    print("Cricket Match Outcome Predictor")
    
    # Check for required data files
    for file in REQUIRED_FILES:
        if not os.path.exists(file):
            print(f"Error: Missing required data file '{file}'. Please ensure it is in the same directory.")
            sys.exit(1)

    team_a = input("Enter Team A: ").strip()
    team_b = input("Enter Team B: ").strip()

    # Player selection for each team
    print("\n--- Team Selection ---")
    squad_a = select_players(team_a)
    squad_b = select_players(team_b)
    
    print("\n--- Match Details ---")
    toss_winner = input("Enter Toss Winner Team: ").strip()
    toss_decision = input("Enter Toss Decision (Bat or Field): ").strip()
    stadium = input("Enter Stadium: ").strip()

    # The prediction function is called here. Note: The current model does not use
    # player squad data for prediction.
    result = predict_match(toss_winner, stadium, team_a, team_b, toss_decision)

    print(f"\nPrediction Result: {result}")

if __name__ == "__main__":
    main()
