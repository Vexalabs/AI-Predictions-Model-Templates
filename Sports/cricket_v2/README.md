

Cricket Match Outcome Predictor

This project is a machine learning-based application that predicts the winning team of a cricket match. The model is trained on the CWC2023.csv dataset and uses a robust prediction system with multiple fallback mechanisms to provide a result even for match-ups not seen in the training data.



Features (Version 2)

Trained Model Prediction: Uses a RandomForestClassifier model to predict the winner based on historical data.



Intelligent Fallback System: If a specific match-up (teams, stadium) is not in the training data, the model uses a fallback method.



Performance-Based Fallback: The fallback logic considers head-to-head records and overall win rates.



Enhanced Prediction Accuracy: The model now takes the user-provided "Toss Decision" (Bat or Field) as a direct input, making predictions more specific and accurate.



Player Selection: The program allows users to select 11 players for each team from a numbered list, preparing the application for future enhancements that could use this data for more detailed predictions.



AI-Powered Player Insights: The application uses a large language model (LLM) to generate a qualitative analysis of key players who are likely to perform well in the match, adding a new layer of insight.



Input Validation \& User-Friendly Interface: The application provides informative messages for invalid inputs and uses a simple, menu-driven interface for player selection.



Project Structure

data\_preprocessing.py: Handles the loading, cleaning, and preprocessing of the raw dataset.



model\_training.py: The script to train the machine learning model (RandomForestClassifier), calculate its accuracy, and save both to files (.pkl and .txt).



prediction.py: Contains the core prediction logic, including the primary model, the advanced fallback system, and the AI-powered player insight generation.



main.py: The main entry point for the application, which interacts with the user and orchestrates the prediction and insights generation.



CWC2023.csv, match\_schedule\_results.csv, batting\_summary.csv, bowling\_summary.csv, world\_cup\_players\_info.csv: The datasets used for training and insights.



cricket\_model.pkl: The trained machine learning model saved to disk.



requirements.txt: Lists all the necessary Python dependencies for the project.



A Comparative Analysis of Version 1 and Version 2

The evolution from our initial predictive model (Version 1) to the enhanced system (Version 2) demonstrates a significant increase in the application's robustness, transparency, and value to the user.



Prediction Accuracy \& Data Utilization:



Version 1: Had a significant flaw where it assumed the toss winner always chose to bat. This limited its practical accuracy as it didn't use real-world user input.



Version 2: Solved this by adding a user prompt for the toss decision, which is then fed directly to the model. This makes the predictions more relevant and accurate.



Robustness \& Fallback:



Version 1: Was fragile and would fail with an error if it encountered a team or stadium not in its training data.



Version 2: Is robust and reliable. It includes a multi-tiered fallback system that ensures a prediction is always provided, even for unforeseen matchups, by using head-to-head records or overall win rates.



User Experience \& Value Proposition:



Version 1: Was a simple, functional predictor. Its output was a win/loss result with a basic explanation.



Version 2: Is a more comprehensive pre-match analysis tool. It not only provides a reliable prediction but also generates qualitative insights on key players from both teams. This added layer of value makes the application more engaging and useful for the user.



Setup and Usage

Follow these steps to set up and run the project on your local machine.



Step 1: Install Dependencies

First, navigate to the project directory and install the required Python libraries using pip.



pip install -r requirements.txt



Step 2: Train the Model

Run the model\_training.py script to train the model and save it to a file. This step must be completed before you can make any predictions.



python model\_training.py



Step 3: Run the Prediction Application

Now you can run the main application and enter the details of the match you want to predict.



python main.py



