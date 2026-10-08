"""
SHAPxAdopt-GH -- sample inference script
========================================
Predicts the AI-adoption readiness tier (Low / Medium / High) for new survey
responses and explains each prediction with SHAP.

It uses exactly the same features and XGBoost settings as the thesis
(shapxadopt_revision_experiments.py must be in the same folder).

STEP 1  Train the primary model once (70/15/15 split, seed 42) and save it:
        python predict.py train --data AI_Readiness_Cleaned_Workbook_300_1.xlsx

STEP 2  Score new responses (an .xlsx or .csv with the same columns as the
        Modeling_Data sheet: F1_score ... F7_score, Uses_*, A1, A2, A5-A8, C3, C4, D1-D8):
        python predict.py predict --new new_responses.xlsx
        If --new is omitted, the 45 held-out test cases are scored as a demonstration.

Output: predictions.csv with the predicted tier, the three class probabilities
and the three features that pushed the prediction most strongly towards the
predicted tier (largest positive SHAP values).
"""
import argparse, json
import joblib
import numpy as np
import pandas as pd
import shap

import shapxadopt_revision_experiments as S

MODEL_FILE = "shapxadopt_model.joblib"


def train(data_path):
    df = S.load(data_path)
    y = S.make_target(df[S.TARGET_SOURCE].values, S.THRESHOLDS_MAIN)
    X = S.encode(df)
    tr, va, te = S.split(y, 42)
    model = S.fit("xgb", X.iloc[tr], y[tr], 42, X.iloc[va], y[va])
    joblib.dump({"model": model, "columns": list(X.columns), "test_index": te.tolist()}, MODEL_FILE)
    pd.DataFrame(df.iloc[te]).to_csv("demo_test_cases.csv", index=False)
    print(f"Model saved to {MODEL_FILE} ({X.shape[1]} features, {model.best_iteration + 1} trees).")
    print("Held-out test cases written to demo_test_cases.csv for the demonstration.")


def predict(new_path):
    saved = joblib.load(MODEL_FILE)
    model, cols = saved["model"], saved["columns"]
    path = new_path or "demo_test_cases.csv"
    new = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path, sheet_name=S.SHEET)
    X = S.encode(new).reindex(columns=cols, fill_value=0.0)   # unseen answer options -> all zeros
    proba = model.predict_proba(X)
    pred = proba.argmax(axis=1)

    sv = shap.TreeExplainer(model).shap_values(X)
    sv = np.array(sv)
    if sv.shape[0] == len(S.CLASS_NAMES) and sv.shape[1] == len(X):   # (class, row, feature) layout
        sv = np.transpose(sv, (1, 2, 0))                                 # -> (row, feature, class)

    rows = []
    for i in range(len(X)):
        contrib = sv[i, :, pred[i]]
        top = np.argsort(contrib)[::-1][:3]
        rows.append({"row": i + 1,
                     "predicted_tier": S.CLASS_NAMES[pred[i]],
                     **{f"p_{c}": round(float(proba[i, k]), 3) for k, c in enumerate(S.CLASS_NAMES)},
                     **{f"driver_{j + 1}": f"{cols[t]} ({contrib[t]:+.2f})" for j, t in enumerate(top)}})
    out = pd.DataFrame(rows)
    out.to_csv("predictions.csv", index=False)
    print(out.head(10).to_string(index=False))
    print(f"\n{len(out)} predictions written to predictions.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["train", "predict"])
    ap.add_argument("--data", help="cleaned workbook (train mode)")
    ap.add_argument("--new", help="new responses to score (predict mode)")
    a = ap.parse_args()
    train(a.data) if a.mode == "train" else predict(a.new)
