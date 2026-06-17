"""
Student Study Performance
Step 2: 학습된 모델로 점수 예측
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_COLS = ["math_score", "reading_score", "writing_score"]


def load_artifacts():
    models = {}
    for t in TARGET_COLS:
        models[t] = joblib.load(os.path.join(MODEL_DIR, f"model_{t}.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    with open(os.path.join(MODEL_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(MODEL_DIR, "eda.json"), encoding="utf-8") as f:
        eda = json.load(f)
    return models, encoders, meta, eda


def predict_scores(input_dict):
    """
    input_dict 예시:
    {
        "gender": "female",
        "race_ethnicity": "group C",
        "parental_level_of_education": "bachelor's degree",
        "lunch": "standard",
        "test_preparation_course": "completed"
    }
    """
    models, encoders, meta, _ = load_artifacts()
    feat_cols = meta["feat_cols"]

    row = {}
    for col in feat_cols:
        val = input_dict.get(col, "")
        le = encoders[col]
        known = set(le.classes_)
        val = val if val in known else le.classes_[0]
        row[col] = le.transform([val])[0]

    X = pd.DataFrame([row])[feat_cols]
    result = {}
    for target, model in models.items():
        pred = float(model.predict(X)[0])
        result[target] = round(min(max(pred, 0), 100), 1)

    result["average"] = round(sum(result.values()) / len(result), 1)
    return result
