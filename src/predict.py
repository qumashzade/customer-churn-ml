"""
Yadda saxlanılmış modeli yükləyir və yeni datada proqnoz verir.

İstifadə:
    python predict.py path/to/new_customers.csv
"""
import sys
import os
import json
import joblib
import pandas as pd

from preprocessing import clean_data, encode_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)
    return model, scaler, feature_names


def predict(csv_path: str) -> pd.DataFrame:
    model, scaler, feature_names = load_artifacts()

    raw = pd.read_csv(csv_path)
    ids = raw["customerID"] if "customerID" in raw.columns else raw.index

    df = raw.copy()
    if "Churn" not in df.columns:
        df["Churn"] = "No"  # clean_data() Churn sütununu gözləyir, olmasa əlavə edirik ki, xəta verməsin
    df = clean_data(df)
    df = encode_features(df)
    df = df.drop(columns=["Churn"])

    # train zamanı olan sütunlarla eyniləşdiririk (yeni datada olmayan dummy sütunlar üçün 0 qoyuruq)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    result = pd.DataFrame({
        "customerID": ids,
        "churn_prediction": ["Yes" if p == 1 else "No" for p in preds],
        "churn_probability": probs.round(4),
    })
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("İstifadə: python predict.py <csv_path>")
        sys.exit(1)

    output = predict(sys.argv[1])
    print(output.to_string(index=False))
    output.to_csv("predictions.csv", index=False)
    print("\nNəticələr 'predictions.csv' faylına yazıldı.")
