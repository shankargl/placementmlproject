import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Load trained model
# -----------------------------

model = joblib.load("placement_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("🎓 Student Placement Predictor")

st.write(
    "Predict whether a student is likely to be placed "
    "using Logistic Regression."
)


# -----------------------------
# User Inputs
# -----------------------------

IQ = st.number_input(
    "IQ",
    min_value=50,
    max_value=160,
    value=100
)

CGPA = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0
)

Academic_Performance = st.number_input(
    "Academic Performance",
    min_value=0,
    max_value=10,
    value=7
)

Internship_Experience = st.selectbox(
    "Internship Experience",
    ["No", "Yes"]
)

Extra_Curricular_Score = st.number_input(
    "Extra Curricular Score",
    min_value=0,
    max_value=10,
    value=5
)

Communication_Skills = st.number_input(
    "Communication Skills",
    min_value=0,
    max_value=10,
    value=5
)

Projects_Completed = st.number_input(
    "Projects Completed",
    min_value=0,
    max_value=20,
    value=2
)


# -----------------------------
# Convert Internship to 0/1
# -----------------------------

internship_no = 1 if Internship_Experience == "No" else 0
internship_yes = 1 if Internship_Experience == "Yes" else 0


# -----------------------------
# Create Input DataFrame
# -----------------------------

input_data = pd.DataFrame({

    "IQ": [IQ],

    "CGPA": [CGPA],

    "Academic_Performance": [Academic_Performance],

    "Extra_Curricular_Score": [Extra_Curricular_Score],

    "Communication_Skills": [Communication_Skills],

    "Projects_Completed": [Projects_Completed],

    "Internship_Experience_No": [internship_no],

    "Internship_Experience_Yes": [internship_yes]
})


# -----------------------------
# Make sure feature order
# is same as training
# -----------------------------

input_data = input_data[features]


# -----------------------------
# Scale input
# -----------------------------

input_scaled = scaler.transform(input_data)


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict Placement"):

    probability = model.predict_proba(input_scaled)[0][1]

    threshold = 0.70

    if probability >= threshold:
        st.success("🎉 Likely to be Placed")
    else:
        st.error("❌ Likely Not to be Placed")

    st.write(
        f"Placement Probability: {probability * 100:.2f}%"
    )