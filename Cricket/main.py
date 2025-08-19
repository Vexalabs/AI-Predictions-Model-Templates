from prediction import predict_match, get_accuracy

def main():
    print("Cricket Match Outcome Predictor")

    accuracy = get_accuracy()
    if accuracy:
        print(f"Model Accuracy: {accuracy:.2f}")
    
    toss_winner = input("Enter Toss Winner Team: ")
    stadium = input("Enter Stadium: ")
    team_a = input("Enter Team A: ")
    team_b = input("Enter Team B: ")
    toss_decision = input("Enter Toss Decision (Bat/Field): ")

    result = predict_match(toss_winner, stadium, team_a, team_b, toss_decision)

    print(f"Predicted Match Winner: {result}")

if __name__ == "__main__":
    main()