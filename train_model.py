"""
Student Study Performance
Step 1: 데이터 전처리 + 모델 학습 + 저장
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib
import json
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "study_performance.csv")
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

CAT_COLS = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
TARGET_COLS = ["math_score", "reading_score", "writing_score"]


def load_and_preprocess(path):
    print("▶ 데이터 로딩 중...")
    df = pd.read_csv(path)
    print(f"  행: {len(df):,}  |  컬럼: {len(df.columns)}")

    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feat_cols = CAT_COLS
    return df, feat_cols, encoders


def train_one(df, feat_cols, target):
    X = df[feat_cols]
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    r2  = r2_score(y_val, preds)
    mae = mean_absolute_error(y_val, preds)
    print(f"  [{target}]  R²={r2:.4f}  MAE={mae:.2f}점")
    return model, r2, mae


def save_artifacts(models, metrics, encoders, feat_cols):
    print("\n▶ 모델 저장 중...")
    for target, model in models.items():
        joblib.dump(model, os.path.join(MODEL_DIR, f"model_{target}.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    meta = {
        "feat_cols": feat_cols,
        "cat_cols": CAT_COLS,
        "targets": TARGET_COLS,
        "metrics": metrics,
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("  저장 완료: model_*.pkl / encoders.pkl / meta.json")


if __name__ == "__main__":
    df, feat_cols, encoders = load_and_preprocess(DATA_PATH)
    models, metrics = {}, {}
    print("\n▶ XGBoost 모델 학습 중...")
    for target in TARGET_COLS:
        model, r2, mae = train_one(df, feat_cols, target)
        models[target] = model
        metrics[target] = {"r2": round(r2, 4), "mae": round(mae, 2)}
    save_artifacts(models, metrics, encoders, feat_cols)

    # EDA 통계 저장
    df_raw = pd.read_csv(DATA_PATH)
    eda = {}
    for col in CAT_COLS:
        eda[col] = df_raw.groupby(col)[TARGET_COLS].mean().round(2).to_dict()
    with open(os.path.join(MODEL_DIR, "eda.json"), "w", encoding="utf-8") as f:
        json.dump(eda, f, ensure_ascii=False, indent=2)
    print("\n  EDA 통계 저장 완료: eda.json")
    print("\n✅ 학습 완료!")
