import streamlit as st
import pandas as pd
import joblib



model = joblib.load("placement_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")
threshold = joblib.load("threshold.pkl")


st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="centered"
)



st.title("🎓 Student Placement Predictor")

st.write(
    "Predict whether a student is likely to be placed "
    "using Logistic Regression."
)

st.divider()



st.subheader("Enter Student Details")


IQ = st.number_input(
    "IQ",
    min_value=50,
    max_value=160,
    value=100,
    step=1
)


Prev_Sem_Result = st.number_input(
    "Previous Semester Result",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.01
)


CGPA = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.01
)


Academic_Performance = st.number_input(
    "Academic Performance",
    min_value=0,
    max_value=10,
    value=7,
    step=1
)


Internship_Experience = st.selectbox(
    "Internship Experience",
    ["No", "Yes"]
)


Extra_Curricular_Score = st.number_input(
    "Extra Curricular Score",
    min_value=0,
    max_value=10,
    value=5,
    step=1
)


Communication_Skills = st.number_input(
    "Communication Skills",
    min_value=0,
    max_value=10,
    value=5,
    step=1
)


Projects_Completed = st.number_input(
    "Projects Completed",
    min_value=0,
    max_value=20,
    value=2,
    step=1
)



if Internship_Experience == "No":

    internship_no = 1
    internship_yes = 0

else:

    internship_no = 0
    internship_yes = 1


input_data = pd.DataFrame({

    "IQ": [IQ],

    "Prev_Sem_Result": [Prev_Sem_Result],

    "CGPA": [CGPA],

    "Academic_Performance": [Academic_Performance],

    "Extra_Curricular_Score": [Extra_Curricular_Score],

    "Communication_Skills": [Communication_Skills],

    "Projects_Completed": [Projects_Completed],

    "Internship_Experience_No": [internship_no],

    "Internship_Experience_Yes": [internship_yes]

})



input_data = input_data[features]




if st.button(
    "🔮 Predict Placement",
    use_container_width=True
):


    input_scaled = scaler.transform(input_data)



    probability = model.predict_proba(
        input_scaled
    )[0][1]



    prediction = int(
        probability >= threshold
    )



    st.divider()

    st.subheader("Prediction Result")


    if prediction == 1:

        st.success(
            "🎉 Likely to be Placed"
        )

    else:

        st.error(
            "❌ Likely Not to be Placed"
        )


    st.metric(
        "Placement Probability",
        f"{probability * 100:.2f}%"
    )



    st.progress(
        float(probability)
    )