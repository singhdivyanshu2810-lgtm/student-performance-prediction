import json

import joblib
import pandas as pd
import streamlit as st

from utils import FEATURES, score, suggestions, validate

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


@st.cache_data
def load_metrics():
    with open("metrics.json") as f:
        return json.load(f)


try:
    model, metrics = load_model(), load_metrics()
except FileNotFoundError:
    st.error("model.joblib not found. Run `python train_model.py` first.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Pass probability threshold (%)", 30, 70, 50, 5, key="threshold",
                          help="Students with a pass probability below this value are predicted to fail. "
                               "Raise it to catch more at-risk students.") / 100
    min_att = st.slider("Minimum attendance rule (%)", 0, 100, 75, key="min_att",
                        help="Students below this attendance are flagged, whatever the model predicts.")

st.title("🎓 Student Performance Prediction System")
st.write("Predict whether a student will pass, using Machine Learning.")

tab_predict, tab_batch, tab_perf, tab_data = st.tabs(
    ["🔮 Predict", "📂 Batch Prediction", "📊 Model Performance", "🗂 Dataset"])

with tab_predict:
    c1, c2 = st.columns(2)
    with c1:
        attendance = st.slider("Attendance (%)", 0, 100, 75)
        study_hours = st.slider("Daily Study Hours", 0.0, 12.0, 4.0, 0.5)
    with c2:
        previous_marks = st.slider("Previous Marks", 0, 100, 70)
        assignment_marks = st.slider("Assignment Marks", 0, 100, 75)

    if st.button("Predict Performance", type="primary"):
        x = pd.DataFrame([[attendance, study_hours, previous_marks, assignment_marks]], columns=FEATURES)
        p_pass = float(model.predict_proba(x)[0][1])
        pred = int(p_pass >= threshold)

        m1, m2 = st.columns(2)
        with m1:
            (st.success if pred == 1 else st.error)("Prediction: PASS 🎉" if pred == 1 else "Prediction: FAIL ⚠️")
        with m2:
            st.metric("Probability of Passing", f"{p_pass * 100:.1f}%")
        st.progress(p_pass)

        if attendance < min_att:
            st.warning(f"Attendance rule: {attendance}% is below the minimum of {min_att}%. "
                       "The student may be ineligible for the exam, whatever the model predicts.")

        tips = suggestions(attendance, study_hours, previous_marks, assignment_marks)
        st.subheader("💡 Suggestions")
        if tips:
            for t in tips:
                st.write("- " + t)
        else:
            st.write("All indicators look healthy. Keep it up!")

with tab_batch:
    st.subheader("Predict for a whole class")
    st.write("Upload a CSV with the columns: " + ", ".join(f"`{c}`" for c in FEATURES) + ".")
    template = pd.read_csv("data/students.csv")[FEATURES].head(10)
    st.download_button("⬇️ Download sample CSV template", template.to_csv(index=False),
                       "template.csv", "text/csv")
    upload = st.file_uploader("Upload CSV file", type="csv")
    if upload is not None:
        try:
            raw = pd.read_csv(upload)
            clean, removed = validate(raw)
        except ValueError as e:
            st.error(str(e))
        except Exception:
            st.error("Could not read this file. Please upload a valid CSV.")
        else:
            if removed:
                st.warning(f"{removed} row(s) were skipped because of missing, non-numeric or out-of-range values.")
            if clean.empty:
                st.error("No valid rows found.")
            else:
                result = score(clean, model, threshold, min_att)
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Students", len(result))
                k2.metric("Predicted Pass", int((result["prediction"] == "Pass").sum()))
                k3.metric("Predicted Fail", int((result["prediction"] == "Fail").sum()))
                k4.metric("Attendance alerts", int(result["attendance_alert"].sum()))
                st.subheader("⚠️ At-risk students")
                risk = result[result["at_risk"]].sort_values("pass_probability")
                if risk.empty:
                    st.write("No at-risk students found.")
                else:
                    st.dataframe(risk, width="stretch")
                st.subheader("All results")
                st.dataframe(result, width="stretch")
                st.download_button("⬇️ Download results as CSV", result.to_csv(index=False),
                                   "predictions.csv", "text/csv")

with tab_perf:
    st.subheader("Model Comparison")
    st.caption(f"Best model: **{metrics['best_model']}** | Train rows: {metrics['n_train']} | Test rows: {metrics['n_test']}")
    table = pd.DataFrame(metrics["table"]).set_index("Model")
    st.dataframe(table.style.format("{:.3f}"), width="stretch")
    a, b = st.columns(2)
    with a:
        st.write("**Confusion matrix (test data)**")
        cm = pd.DataFrame(metrics["confusion_matrix"], index=["Actual Fail", "Actual Pass"],
                          columns=["Predicted Fail", "Predicted Pass"])
        st.dataframe(cm)
    with b:
        st.write("**Feature importance**")
        st.bar_chart(pd.Series(metrics["feature_importance"]))

with tab_data:
    st.subheader("Training Data")
    df = pd.read_csv("data/students.csv")
    st.write(f"{len(df)} students, {df.shape[1] - 1} features.")
    st.dataframe(df.head(50), width="stretch")
    st.bar_chart(df["result"].map({1: "Pass", 0: "Fail"}).value_counts())

st.divider()
st.caption("Built with Python, Streamlit and scikit-learn (Logistic Regression, Decision Tree, Random Forest).")
