# 🎓 Student Performance Prediction

A Machine Learning web app that predicts whether a student will **pass or fail** from attendance, study hours, previous marks and assignment marks.

**Live demo:** https://student--performance--predictor.streamlit.app/

<!-- Add screenshots here: ![Predict tab](images/screenshot_predict.png) -->

## Features
- **Predict:** pass probability, Pass/Fail result and personalised suggestions
- **Batch Prediction:** upload a CSV for a whole class, see at-risk students, download results
- **Attendance rule:** configurable minimum-attendance warning (default 75%), independent of the model
- **Adjustable threshold:** raise the pass-probability threshold to catch more at-risk students
- **Model Performance:** 3 models (Logistic Regression, Decision Tree, Random Forest) compared with train/test split, 5-fold cross-validation, accuracy, precision, recall, F1, confusion matrix and feature importance
- Trained model saved with `joblib` (no retraining on every run)
- 10 automated tests (`tests.py`)

## Project structure
```
app.py            # Streamlit app (4 tabs + sidebar settings)
utils.py          # validation, scoring and suggestion helpers
train_model.py    # data generation, training, evaluation
tests.py          # automated tests
data/students.csv # dataset (replace with a real one if available)
model.joblib      # saved best model
metrics.json      # evaluation results
images/           # charts
```

## Run locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Run the tests
```bash
python -m unittest tests -v
```

## Batch CSV format
Columns: `attendance, study_hours, previous_marks, assignment_marks`. Valid ranges: attendance, previous_marks, assignment_marks 0-100; study_hours 0-12. Invalid rows are skipped with a warning. A template can be downloaded from the Batch Prediction tab.

## Using your own dataset
Put a CSV at `data/students.csv` with columns `attendance, study_hours, previous_marks, assignment_marks, result` (1 = pass, 0 = fail), then run `python train_model.py` again.

## Note
The bundled dataset is **simulated** for demonstration. Replace it with real student data for real-world use. Predictions are an early-warning aid and should not be used as the only basis for decisions about a student.
