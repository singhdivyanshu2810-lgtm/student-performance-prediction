"""Automated tests. Run:  python -m unittest tests -v"""
import json
import unittest

import joblib
import pandas as pd

from utils import FEATURES, score, suggestions, validate


class TestProject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load("model.joblib")
        cls.data = pd.read_csv("data/students.csv")

    def test_dataset_columns_and_no_nulls(self):
        self.assertTrue(set(FEATURES + ["result"]).issubset(self.data.columns))
        self.assertEqual(int(self.data.isna().sum().sum()), 0)

    def test_strong_student_passes(self):
        out = score(pd.DataFrame([[95, 8, 90, 92]], columns=FEATURES), self.model)
        self.assertEqual(out.loc[0, "prediction"], "Pass")

    def test_weak_student_fails(self):
        out = score(pd.DataFrame([[30, 1, 25, 30]], columns=FEATURES), self.model)
        self.assertEqual(out.loc[0, "prediction"], "Fail")

    def test_probability_range(self):
        out = score(self.data[FEATURES], self.model)
        self.assertTrue(out["pass_probability"].between(0, 100).all())

    def test_threshold_changes_prediction(self):
        row = pd.DataFrame([[60, 3, 50, 55]], columns=FEATURES)
        p = score(row, self.model).loc[0, "pass_probability"] / 100
        self.assertEqual(score(row, self.model, threshold=min(p + 0.05, 0.99)).loc[0, "prediction"], "Fail")

    def test_attendance_rule_flags_student(self):
        row = pd.DataFrame([[60, 6, 80, 85]], columns=FEATURES)
        out = score(row, self.model, min_attendance=75)
        self.assertTrue(out.loc[0, "attendance_alert"])
        self.assertTrue(out.loc[0, "at_risk"])

    def test_validate_removes_bad_rows(self):
        raw = pd.DataFrame({"attendance": [80, 150, "abc"], "study_hours": [4, 4, 4],
                            "previous_marks": [70, 70, 70], "assignment_marks": [70, 70, 70]})
        clean, removed = validate(raw)
        self.assertEqual((len(clean), removed), (1, 2))

    def test_validate_missing_column(self):
        with self.assertRaises(ValueError):
            validate(pd.DataFrame({"attendance": [80]}))

    def test_suggestions(self):
        self.assertEqual(len(suggestions(50, 1, 40, 40)), 4)
        self.assertEqual(suggestions(90, 6, 80, 80), [])

    def test_model_quality(self):
        with open("metrics.json") as f:
            m = json.load(f)
        best = [r for r in m["table"] if r["Model"] == m["best_model"]][0]
        self.assertGreater(best["Accuracy"], 0.7)


if __name__ == "__main__":
    unittest.main()
