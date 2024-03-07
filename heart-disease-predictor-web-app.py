import pickle as pk
import pandas as pd
import numpy as np
import streamlit as st

model_path = "Heart_Disease_Model.pkl"
data_path = "heart_disease.csv"

model = pk.load(open(model_path, "rb"))
data = pd.read_csv(data_path)
print(data)

st.header("Heart Disease Predictor")

gender = st.selectbox("Choose Gender", data["Gender"].unique())
if gender == "Male ":
    gen = 1
else:
    gen = 0

age = st.number_input("Enter Age")
currentSmoker = st.number_input("Is patient currentSmoker	")
cigsPerDay = st.number_input("Enter cigsPerDay	")
BPMeds = st.number_input("Is patient on BPMeds")
prevalentStroke = st.number_input("Is patient had Stroke")
prevalentHyp = st.number_input("Enter prevalentHyp status")
diabetes = st.number_input("Enter diabetes status")
totChol = st.number_input("Enter totCho")
sysBP = st.number_input("Enter sysBP")
diaBP = st.number_input("Enter diaBP")
BMI = st.number_input("Enter BMI")
heartRate = st.number_input("heartRate	")
glucose = st.number_input("Enter glucose")

input1 = np.array(
    [
        [
            gen,
            age,
            currentSmoker,
            cigsPerDay,
            BPMeds,
            prevalentStroke,
            prevalentHyp,
            diabetes,
            totChol,
            sysBP,
            diaBP,
            BMI,
            heartRate,
            glucose,
        ]
    ]
)
if st.button("Predict"):
    input1 = np.array(
        [
            [
                gen,
                age,
                currentSmoker,
                cigsPerDay,
                BPMeds,
                prevalentStroke,
                prevalentHyp,
                diabetes,
                totChol,
                sysBP,
                diaBP,
                BMI,
                heartRate,
                glucose,
            ]
        ]
    )
    output = model.predict(input1)
    if output[0] == 0:
        stn = "Patient is Healthy, No heart Disease"
    else:
        stn = "Patient May Have Heart Disease"
    st.markdown(stn)
