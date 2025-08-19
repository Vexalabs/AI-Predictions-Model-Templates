Cricket Match Outcome Predictor
This project is a machine learning-based application that predicts the winning team of a cricket match. The model is trained on the CWC2023.csv dataset and uses a robust prediction system with multiple fallback mechanisms to provide a result even for match-ups not seen in the training data.

Features
Trained Model Prediction: Uses a RandomForestClassifier model to predict the winner based on historical data.

Intelligent Fallback System: If a specific match-up (teams, stadium) is not in the training data, the model uses a fallback method.

Performance-Based Fallback: The fallback logic considers:

Team performance at the specific stadium.

Head-to-head records between the two teams.

Tie-Breaker Logic: In the case of a tie in the fallback analysis, the model uses the overall tournament win rate of each team to determine a winner.

Accuracy Display: The model's accuracy on the test data is calculated and displayed every time the prediction script is run.

Input Validation: The application provides informative error messages for invalid user inputs, such as misspellings of team or stadium names.

Project Structure
data_preprocessing.py: Handles the loading, cleaning, and preprocessing of the raw dataset.

model_training.py: The script to train the machine learning model (RandomForestClassifier), calculate its accuracy, and save both to files (.pkl and .txt).

prediction.py: Contains the core prediction logic, including the primary model and the advanced fallback system.

main.py: The main entry point for the application, which interacts with the user and displays the final prediction.

CWC2023.csv: The dataset used to train the model.

requirements.txt: Lists all the necessary Python dependencies for the project.

cricket_model.pkl: The trained machine learning model saved to disk.

model_accuracy.txt: A text file containing the model's accuracy score.

Setup and Usage
Follow these steps to set up and run the project on your local machine.

Step 1: Install Dependencies
First, navigate to the cricket_prediction directory and install the required Python libraries using pip.

pip install -r requirements.txt

Step 2: Train the Model
Run the model_training.py script to train the model and save it to a file. This step must be completed before you can make any predictions.

python model_training.py

Step 3: Run the Prediction Application
Now you can run the main application and enter the details of the match you want to predict.

python main.py

Example Output
Here is an example of a predicted output, demonstrating the fallback logic.

Made changes? Awesome! Here's how to see them in action:

1.  Go back to your *first* terminal (where the model is running) and press `Ctrl + c` to stop it.
2.  Simply repeat the "Fire Up" and "Is It Working" steps above. Easy peasy!

**Congratulations, you are now a bona fide model builder!**

---

### Peek Under the Hood: Project Files Explained

Curious about what makes the Sports model tick? Here's a quick tour:

* **`main.py`**: This is the heart of your project, the starting point for your API and where your model gets called.
* **`src/api.py`**: Defines how your model talks to the world (the API structure).
* **`src/models.py`**: **This is where YOUR custom prediction logic lives!** Get creative here.
* **`src/schemas.py`**: Lays out the data structures the API expects and provides.

And the rest:

* **`.dockerignore`**: Tells Docker what files to skip. You probably won't need to touch this often.
* **`.gitignore`**: Tells Git what files to ignore. No need to edit unless you add new files you don't want tracked.
* **`LICENSE`**: The project's license.
* **`README.md`**: What you're reading right now!
* **`requirements.txt`**: Lets Docker know which Python libraries your project needs. **Important! If you add new Python libraries to your custom logic, remember to add them here!**

