"""Train, evaluate and save the Student Performance model.

Run:  python train_model.py
- Uses data/students.csv if present (replace it with a real dataset),
  otherwise generates a 500-row simulated dataset.
- Compares 3 models with a train/test split + cross-validation.
- Saves the best model, metrics and charts for the Streamlit app.
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["attendance", "study_hours", "previous_marks", "assignment_marks"]
TARGET = "result"
DATA_PATH = "data/students.csv"


def make_dataset(n=500, seed=42):
    """Simulated data: result depends on all 4 factors plus random noise."""
    rng = np.random.default_rng(seed)
    attendance = np.clip(rng.normal(75, 15, n), 20, 100)
    study = np.clip(rng.normal(4, 2, n), 0, 12)
    prev = np.clip(0.3 * attendance + 4 * study + rng.normal(25, 12, n), 0, 100)
    assign = np.clip(0.25 * attendance + 0.4 * prev + 2 * study + rng.normal(12, 8, n), 0, 100)
    score = 0.2 * attendance + 3 * study + 0.35 * prev + 0.25 * assign + rng.normal(0, 6, n)
    result = (score > np.percentile(score, 35)).astype(int)  # ~65% pass
    return pd.DataFrame({
        "attendance": attendance.round().astype(int),
        "study_hours": study.round(1),
        "previous_marks": prev.round().astype(int),
        "assignment_marks": assign.round().astype(int),
        TARGET: result,
    })


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded {len(df)} rows from {DATA_PATH}")
    else:
        df = make_dataset()
        df.to_csv(DATA_PATH, index=False)
        print(f"Generated simulated dataset ({len(df)} rows) -> {DATA_PATH}")

    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    rows, fitted = [], {}
    for name, m in models.items():
        cv = cross_val_score(m, X_tr, y_tr, cv=5, scoring="accuracy")
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, p),
            "Precision": precision_score(y_te, p),
            "Recall": recall_score(y_te, p),
            "F1": f1_score(y_te, p),
            "CV Accuracy (5-fold)": cv.mean(),
        })
        fitted[name] = m
        print(f"{name:20s} acc={rows[-1]['Accuracy']:.3f} f1={rows[-1]['F1']:.3f} cv={cv.mean():.3f}")

    best = max(rows, key=lambda r: r["F1"])["Model"]
    best_model = fitted[best]
    print("Best model:", best)

    cm = confusion_matrix(y_te, best_model.predict(X_te))
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Fail", "Pass"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {best}")
    fig.tight_layout(); fig.savefig("images/confusion_matrix.png", dpi=150); plt.close(fig)

    rf = fitted["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    imp.plot.barh(ax=ax, color="#4C72B0")
    ax.set_title("Feature Importance (Random Forest)"); ax.set_xlabel("Importance")
    fig.tight_layout(); fig.savefig("images/feature_importance.png", dpi=150); plt.close(fig)

    joblib.dump(best_model, "model.joblib")
    with open("metrics.json", "w") as f:
        json.dump({
            "best_model": best,
            "n_rows": int(len(df)),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "table": rows,
            "confusion_matrix": cm.tolist(),
            "feature_importance": imp.to_dict(),
        }, f, indent=2)
    print("Saved model.joblib, metrics.json, images/")


if __name__ == "__main__":
    main()
