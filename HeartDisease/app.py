import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from ucimlrepo import fetch_ucirepo
import pickle

# --- THIS IS THE LINE THAT WAS MOVED ---
# It MUST be the first Streamlit command.
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# --- 1. Model Training (and Caching) ---
@st.cache_data
def train_model():
    # Fetch dataset 
    heart_disease = fetch_ucirepo(id=45) 
    
    # Use .copy() to avoid the SettingWithCopyWarning
    X = heart_disease.data.features.copy() 
    y = heart_disease.data.targets.copy()
    
    # --- Data Cleaning & Preprocessing ---
    X.replace('?', np.nan, inplace=True)
    X['ca'] = pd.to_numeric(X['ca'], errors='coerce')
    X['thal'] = pd.to_numeric(X['thal'], errors='coerce')
    X.fillna(X.median(), inplace=True)
    
    # Create binary target
    y_binary = (y['num'] > 0).astype(int)
    
    # --- Scaling and Training ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use the best parameters found during tuning
    best_params = {'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 200}
    model = RandomForestClassifier(random_state=42, **best_params)
    model.fit(X_scaled, y_binary)
    
    return model, scaler

# Load the model and scaler
model, scaler = train_model()


# --- 2. Application Interface ---

# --- Header ---
st.title('❤️ Heart Disease Prediction Tool')
st.markdown("""
This application predicts the likelihood of a patient having heart disease based on their medical information. 
Please enter the patient's details in the sidebar to get a prediction.
This is a **decision-support tool** and not a substitute for professional medical advice.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header('Patient Information')

def user_input_features():
    # ... (the rest of your code remains exactly the same) ...
    age = st.sidebar.number_input('Age', 1, 100, 50)
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    cp = st.sidebar.selectbox('Chest Pain Type', (0, 1, 2, 3), format_func=lambda x: {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}[x])
    trestbps = st.sidebar.number_input('Resting Blood Pressure (mm Hg)', 80, 200, 120)
    chol = st.sidebar.number_input('Serum Cholestoral (mg/dl)', 100, 600, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ('False', 'True'))
    restecg = st.sidebar.selectbox('Resting ECG Results', (0, 1, 2), format_func=lambda x: {0: 'Normal', 1: 'ST-T Wave Abnormality', 2: 'Left Ventricular Hypertrophy'}[x])
    thalach = st.sidebar.number_input('Maximum Heart Rate Achieved', 60, 220, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', ('No', 'Yes'))
    oldpeak = st.sidebar.slider('ST Depression (oldpeak)', 0.0, 6.2, 1.0, 0.1)
    slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment', (0, 1, 2), format_func=lambda x: {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}[x])
    ca = st.sidebar.selectbox('Number of Major Vessels Colored by Flouroscopy', (0, 1, 2, 3, 4))
    thal = st.sidebar.selectbox('Thalassemia', (0, 1, 2, 3), format_func=lambda x: {0: 'Unknown', 1: 'Normal', 2: 'Fixed Defect', 3: 'Reversable Defect'}[x])

    sex_num = 1 if sex == 'Male' else 0
    fbs_num = 1 if fbs == 'True' else 0
    exang_num = 1 if exang == 'Yes' else 0
    
    data = {
        'age': age,
        'sex': sex_num,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs_num,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang_num,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display the user's input
st.subheader('Patient Data Entered:')
st.write(input_df)

# --- 3. Prediction and Output ---
if st.sidebar.button('**Predict Heart Disease Risk**'):
    # Scale the user input
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction_proba = model.predict_proba(input_scaled)
    prediction = model.predict(input_scaled)
    
    risk_score = prediction_proba[0][1] * 100 
    
    st.subheader('Prediction Result')
    
    if prediction[0] == 1:
        st.error(f'**High Risk of Heart Disease**')
    else:
        st.success(f'**Low Risk of Heart Disease**')
        
    st.metric(label="Patient's Risk Score", value=f"{risk_score:.2f}%")

    st.write("""
    The risk score represents the model's confidence in the prediction. A higher score indicates a greater likelihood of heart disease. 
    This information should be used to supplement, not replace, a professional medical evaluation.
    """)