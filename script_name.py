import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib  # Import joblib for loading the model

# Load the pre-trained logistic regression model
model = joblib.load("heart_model.joblib")  # Replace with the actual path to your saved model file

def main():
    st.title("Heart Disease Prediction App")

    # Get user inputs
    age = st.slider("Enter age:", min_value=1, max_value=100, value=25)

    sex = st.selectbox("Select sex:", ["Female", "Male"])
    sex = 1 if sex == "Male" else 0
    cp = st.selectbox("Enter chest pain type (cp):", [0, 1, 2, 3], index=0)
    trestbpd = st.selectbox("Enter resting blood pressure (trestbps):", list(range(80, 201)), index=60)
    chol = st.selectbox("Enter cholesterol level:", list(range(100, 401)), index=150)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs):", ["No", "Yes"], index=0)
    fbs = 1 if fbs == "Yes" else 0
    restecg = st.selectbox("Enter resting electrocardiographic results (restecg):", [0, 1, 2], index=0)
    thalach = st.selectbox("Enter maximum heart rate achieved (thalach):", list(range(60, 221)), index=80)
    exang = st.selectbox("Exercise induced angina (exang):", ["No", "Yes"], index=0)
    exang = 1 if exang == "Yes" else 0
    oldpeak = st.selectbox("Enter ST depression induced by exercise relative to rest (oldpeak):", np.arange(0.0, 6.3, 0.1), index=0)
    slope = st.selectbox("Enter the slope of the peak exercise ST segment (slope):", [0, 1, 2], index=0)
    ca = st.selectbox("Enter number of major vessels colored by fluoroscopy (ca):", list(range(0, 5)), index=0)
    thal = st.selectbox("Enter thalassemia type (thal):", [0, 1, 2, 3], index=2)

    input_data = np.array([age, sex, cp, trestbpd, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
    input_data_reshaped = input_data.reshape(1, -1)

    if st.button("Predict"):
        prediction = model.predict(input_data_reshaped)

        if prediction[0] == 0:
            st.success("The person does not have a heart disease.")
        else:
            st.error("The person has a heart disease.")

if __name__ == "__main__":
    main()
