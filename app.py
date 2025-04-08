import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# App config
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("Titanic Survival Predictor")
st.markdown("Enter the details of a passenger to predict their survival.")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
age = st.slider("Age", min_value=0, max_value=100, value=30)
fare = st.slider("Fare", min_value=0.0, max_value=600.0, value=50.0)
sibsp = st.number_input("Number of Siblings / Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents / Children Aboard", min_value=0, max_value=10, value=0)
sex = st.radio("Sex", ['male', 'female'])
embarked = st.radio("Port of Embarkation", ['C', 'Q', 'S'])

# Encode categorical values
sex_male = 1 if sex == 'male' else 0
embarked_C = 1 if embarked == 'C' else 0
embarked_Q = 1 if embarked == 'Q' else 0

# Prepare feature vector in correct order
input_data = np.array([[pclass, age, fare, sibsp, parch, sex_male, embarked_C, embarked_Q]])
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"Survived (Probability: {probability:.2%})")
    else:
        st.error(f"Did Not Survive (Probability: {probability:.2%})")
