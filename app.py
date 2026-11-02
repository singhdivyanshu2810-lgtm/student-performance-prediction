import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Sample training data
data = pd.DataFrame({
    "attendance": [90, 80, 60, 50, 95, 40, 75, 85, 65, 30,
                   88, 55, 92, 45, 78, 70, 35, 98, 82, 62],
    "study_hours": [5, 4, 2, 1, 6, 1, 3, 5, 2, 1,
                    5, 2, 7, 1, 4, 3, 1, 7, 4, 2],
    "previous_marks": [85, 75, 45, 35, 90, 25, 65, 80, 50, 20,
                       82, 40, 95, 30, 72, 60, 25, 98, 78, 48],
    "assignment_marks": [90, 80, 50, 40, 95, 30, 70, 85, 55, 25,
                         88, 45, 96, 35, 75, 65, 30, 99, 82, 52],
    "result": [1, 1, 0, 0, 1, 0, 1, 1, 0, 0,
               1, 0, 1, 0, 1, 1, 0, 1, 1, 0]
})

X = data.drop("result", axis=1)
y = data["result"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.title("🎓 Student Performance Prediction System")
st.write("Predict student performance using Machine Learning.")

st.divider()

st.subheader("Enter Student Details")

attendance = st.slider("Attendance (%)", 0, 100, 75)
study_hours = st.slider("Daily Study Hours", 0, 12, 4)
previous_marks = st.slider("Previous Marks", 0, 100, 70)
assignment_marks = st.slider("Assignment Marks", 0, 100, 75)

if st.button("Predict Performance"):
    input_data = pd.DataFrame([{
        "attendance": attendance,
        "study_hours": study_hours,
        "previous_marks": previous_marks,
        "assignment_marks": assignment_marks
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success("Prediction: PASS 🎉")
    else:
        st.error("Prediction: FAIL")

    st.metric("Prediction Confidence", f"{max(probability)*100:.2f}%")

st.divider()
st.caption("Developed using Python, Streamlit and Random Forest ML.")
