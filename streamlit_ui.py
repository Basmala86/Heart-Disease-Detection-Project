import streamlit as st
import pandas as pd
import joblib
 
# Load model
model = joblib.load("../ml_model/decision_tree_model.pkl")
 
# Streamlit app
st.title("Heart Disease Risk Prediction")
age = st.slider("Age", 0, 100, 50)
trestbps = st.slider("Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 300, 200)
 
if st.button("Predict Risk"):
    input_data = pd.DataFrame({
        "age": [age],
        "trestbps": [trestbps],
        "chol": [chol]
    })
    prediction = model.predict(input_data)
    st.write(f"Risk Prediction: {'High' if prediction[0] == 1 else 'Low'}")