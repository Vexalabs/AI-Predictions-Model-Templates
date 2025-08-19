# Heart Disease Prediction Application

This project is an end-to-end machine learning application that predicts the likelihood of a patient having heart disease based on their medical attributes. The final model is deployed as an interactive web application using Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app) 
* https://heart-disease-predictor-o7wgyxdm3jx9irfi2kfd7k.streamlit.app/ *


## Table of Contents
- [Project Objective](#project-objective)
- [How it Works](#how-it-works)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Technologies Used](#technologies-used)
- [How to Run This Project Locally](#how-to-run-this-project-locally)
- [Project Structure](#project-structure)

## Project Objective
The primary goal is to provide a simple, accessible tool that leverages machine learning to assist healthcare professionals in making faster, data-driven decisions regarding a patient's heart disease risk. This serves as a clinical decision-support system, aiming for early diagnosis and intervention.

##  How it Works
The application functions as follows:
1.  A user (e.g., a doctor) enters a patient's medical information into the web interface.
2.  The input data is processed and scaled in the same way the model was trained.
3.  The trained Random Forest model predicts the probability an end-to-end machine learning application that predicts the likelihood of a patient having heart disease based on their medical attributes. The final model is deployed as a user-friendly, interactive web application using Streamlit.


---


##  Dataset
This project uses the well-known **Heart Disease Dataset** from the UCI Machine Learning Repository.
- **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Key Features:** `age`, `sex`, `chest pain type`, `resting blood pressure`, `cholesterol`, `max heart rate`, and other relevant medical indicators.

##  Workflow
The project was built following a standard machine learning workflow:
1.  **Data Preprocessing:** Handled missing values (imputed with the median) and ensured data was clean and numerical.
2.  **Exploratory Data Analysis (EDA):** Used visualizations like a correlation heatmap to understand feature relationships and identify key predictors.
3.  **Model Training & Comparison:** Trained and evaluated three different models: Logistic Regression, Decision Tree, and Random Forest.
4.  **Model Selection:** The **Random Forest Classifier** was selected as the best-performing model based on ROC-AUC score and overall accuracy.
5.  **Hyperparameter Tuning:** Used `GridSearchCV` to find the optimal settings for the Random Forest model, further enhancing its predictive power.
6.  **Deployment:** Built an interactive web application with Streamlit and deployed it on Streamlit Community Cloud.

##  Technologies Used
- **Programming Language:** Python 3
- **Data Science Libraries:** Pandas **"Heart Disease Dataset"** from the UCI Machine Learning Repository.

*   **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
*   **Source Institution:** Cleveland Clinic Foundation
*   **Key Features Used:** `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (cholesterol), `thalach` (max heart rate achieved), etc.

The full analysis and model development process can be viewed in the included Jupyter/Colab notebook: `Heart_Disease_Analysis.ipynb`.

---

## Technical Workflow

The project followed a standard end-to-end machine learning pipeline:

1.  **Data Cleaning & Preprocessing:** Handled missing values (marked as '?') by imputing them with the column median.
2.  **Exploratory Data Analysis (EDA):** Used visualizations like correlation heatmaps to understand feature relationships and identify key predictors.
3.  **Model Training & Comparison:** Trained and evaluated three different
