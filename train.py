"""
train.py — Diabetes Prediction System
======================================
Dataset  : mrsimple07/diabetes-prediction (Kaggle)
File     : Diabetes_prediction.csv  (1000 rows, 9 cols)
Target   : Diagnosis  (0 = Non-Diabetic, 1 = Diabetic)
Algorithm: Logistic Regression
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "Diabetes_prediction.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "model")
MODEL_PATH   = os.path.join(MODEL_DIR, "diabetes_model.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age",
]
TARGET_COL = "Diagnosis"


# ─── 1. Load ──────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    print(f"\n{'='*58}")
    print("  DIABETES PREDICTION SYSTEM — Model Training Pipeline")
    print(f"{'='*58}\n")
    print("[1/5] Loading dataset ...")
    df = pd.read_csv(path)
    print(f"      Shape   : {df.shape}")
    print(f"      Columns : {list(df.columns)}")
    counts = df[TARGET_COL].value_counts()
    print(f"\n      Class distribution (Diagnosis):")
    for label, cnt in counts.items():
        tag = "Diabetic" if label == 1 else "Non-Diabetic"
        print(f"         {tag} ({label}): {cnt}  ({cnt/len(df)*100:.1f}%)")
    return df


# ─── 2. Preprocess ────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    print("\n[2/5] Preprocessing data ...")

    # This dataset has continuous float values — no zero imputation needed.
    # Clip any negative ages or unrealistic values that may exist.
    before = len(df)
    df = df[df["Age"] >= 0].copy()
    clipped = before - len(df)
    if clipped:
        print(f"      → Removed {clipped} row(s) with negative Age")
    else:
        print("      → No invalid rows found — dataset is clean")

    print(f"      → Null values : {df.isnull().sum().sum()}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    print(f"      → Features : {FEATURE_COLS}")
    print(f"      → Target   : {TARGET_COL}  (0 = Non-Diabetic, 1 = Diabetic)")
    return X, y


# ─── 3. Split & Scale ─────────────────────────────────────────────────────────
def split_and_scale(X, y, test_size: float = 0.20, random_state: int = 42):
    print(f"\n[3/5] Splitting & scaling ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"      → Train samples : {len(X_train)}")
    print(f"      → Test  samples : {len(X_test)}")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print("      → StandardScaler applied (fit on train, transform on test)")
    return X_train, X_test, y_train, y_test, scaler


# ─── 4. Train ─────────────────────────────────────────────────────────────────
def train_model(X_train, y_train) -> LogisticRegression:
    print("\n[4/5] Training Logistic Regression ...")
    model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs", C=1.0)
    model.fit(X_train, y_train)
    print("      → Training complete.")
    return model


# ─── 5. Evaluate ──────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    print("\n[5/5] Evaluating on test set ...")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm      = confusion_matrix(y_test, y_pred)

    print(f"\n{'─'*58}")
    print("  MODEL PERFORMANCE")
    print(f"{'─'*58}")
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"               Pred: No   Pred: Yes")
    print(f"  Actual: No     {cm[0][0]:4d}       {cm[0][1]:4d}")
    print(f"  Actual: Yes    {cm[1][0]:4d}       {cm[1][1]:4d}")
    print(f"\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=["Non-Diabetic", "Diabetic"])
    for line in report.splitlines():
        print(f"  {line}")
    print(f"{'─'*58}")
    return acc


# ─── 6. Save ──────────────────────────────────────────────────────────────────
def save_artifacts(model, scaler):
    with open(MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    print(f"\n  Saved model  → {MODEL_PATH}")
    print(f"  Saved scaler → {SCALER_PATH}")
    print(f"\n{'='*58}")
    print("  Pipeline complete. Run:  streamlit run app.py")
    print(f"{'='*58}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df                                        = load_data(DATASET_PATH)
    X, y                                      = preprocess(df)
    X_train, X_test, y_train, y_test, scaler  = split_and_scale(X, y)
    model                                     = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)
    save_artifacts(model, scaler)
