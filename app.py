import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Prediction")
st.markdown("Enter passenger details to predict survival chance.")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.slider("Age", 0, 100, 25)
fare = st.slider("Fare", 0.0, 500.0, 50.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
sex = st.selectbox("Sex", ['male', 'female'])
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# Encode categorical values
sex_male = 1 if sex == 'male' else 0
embarked_C = 1 if embarked == 'C' else 0
embarked_Q = 1 if embarked == 'Q' else 0

# Feature vector
features = np.array([[pclass, age, fare, sibsp, parch, sex_male, embarked_C, embarked_Q]])
features_scaled = scaler.transform(features)

# Predict
if st.button("Predict"):
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    st.subheader("Prediction Result")
    st.success("Survived " if prediction == 1 else "Did Not Survive")
    st.info(f"Survival Probability: {probability:.2%}")
