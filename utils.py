"""Helper functions shared by app.py and tests."""
import pandas as pd

FEATURES = ["attendance", "study_hours", "previous_marks", "assignment_marks"]
RANGES = {"attendance": (0, 100), "study_hours": (0, 12), "previous_marks": (0, 100), "assignment_marks": (0, 100)}


def suggestions(attendance, study_hours, previous_marks, assignment_marks):
    """Rule-based advice for one student."""
    tips = []
    if attendance < 75:
        tips.append("Attendance is below 75% - try to attend more classes.")
    if study_hours < 3:
        tips.append("Increase daily study time to at least 3-4 hours.")
    if previous_marks < 50:
        tips.append("Revise weak topics from previous exams.")
    if assignment_marks < 50:
        tips.append("Submit assignments on time and aim for better quality.")
    return tips


def validate(df):
    """Return (clean_df, n_removed). Raises ValueError if columns are missing."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    data = df.copy()
    for c in FEATURES:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    ok = data[FEATURES].notna().all(axis=1)
    for c, (lo, hi) in RANGES.items():
        ok &= data[c].between(lo, hi)
    return data[ok].reset_index(drop=True), int((~ok).sum())


def score(df, model, threshold=0.5, min_attendance=75):
    """Add prediction columns to a validated DataFrame."""
    out = df.copy()
    p = model.predict_proba(out[FEATURES])[:, 1]
    out["pass_probability"] = (p * 100).round(1)
    out["prediction"] = ["Pass" if v >= threshold else "Fail" for v in p]
    out["attendance_alert"] = out["attendance"] < min_attendance
    out["at_risk"] = (p < threshold) | out["attendance_alert"]
    return out
