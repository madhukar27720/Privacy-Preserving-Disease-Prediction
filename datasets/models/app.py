import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Privacy Preserving Disease Prediction")
st.write("Heart Disease Prediction using Machine Learning")

model = joblib.load("models/heart_model.pkl")

age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", 80,200)
chol = st.number_input("Cholesterol",100,600)
fbs = st.selectbox("Fasting Blood Sugar",[0,1])
restecg = st.selectbox("Rest ECG",[0,1,2])
thalach = st.number_input("Maximum Heart Rate",60,220)
exang = st.selectbox("Exercise Induced Angina",[0,1])
oldpeak = st.number_input("Old Peak",0.0,10.0)
slope = st.selectbox("Slope",[0,1,2])
ca = st.selectbox("CA",[0,1,2,3,4])
thal = st.selectbox("Thal",[0,1,2,3])

if st.button("Predict Disease"):

    data = [[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]]

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")
