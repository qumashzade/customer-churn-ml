"""
Telco Customer Churn datası üzərində bir neçə klassifikasiya modeli təlim edir,
müqayisə edir və ən yaxşısını diskə yazır.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

from preprocessing import prepare_dataset

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Telco-Customer-Churn.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def evaluate(model, X_test, y_test, name: str) -> dict:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
    }
    return metrics


def main():
    print("Data yüklənir və hazırlanır...")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_dataset(DATA_PATH)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # StratifiedKFold seçdim ki, hər fold-da churn/no-churn nisbəti eyni qalsın,
    # data balanslı olmadığı üçün normal KFold bəzi fold-larda çox az churn
    # nümunəsi ilə qala bilərdi

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3 modeli seçmə səbəbim: Logistic Regression baseline kimi (sadə, izah olunandır),
    # Random Forest və XGBoost isə daha güclü, qeyri-xətti modellər kimi.
    # Hər birinin grid-i kiçikdir çünki kompüterimdə çox uzun çəkməsin istəmirdim
    models_and_grids = {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"C": [0.01, 0.1, 1, 10]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [200, 400], "max_depth": [6, 10, None]},
        ),
        "XGBoost": (
            XGBClassifier(random_state=42, eval_metric="logloss"),
            {"n_estimators": [200, 400], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        ),
    }

    results = []
    best_model = None
    best_score = -1
    best_name = None

    for name, (estimator, param_grid) in models_and_grids.items():
        print(f"\n=== {name} üçün GridSearchCV işə düşür ===")
        search = GridSearchCV(
            estimator, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        search.fit(X_train, y_train)
        print(f"Ən yaxşı parametrlər: {search.best_params_}")

        metrics = evaluate(search.best_estimator_, X_test, y_test, name)
        metrics["best_params"] = search.best_params_
        results.append(metrics)
        print(metrics)

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = search.best_estimator_
            best_name = name

    print(f"\n>>> Ən yaxşı model: {best_name} (ROC-AUC = {best_score}) <<<")

    # ən yaxşı modeli, scaler-i, feature sırasını və metrikaları yadda saxlayırıq
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # qalib model üçün ətraflı hesabat
    preds = best_model.predict(X_test)
    report = classification_report(y_test, preds, target_names=["No Churn", "Churn"])
    cm = confusion_matrix(y_test, preds)
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Best model: {best_name}\n\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{cm}\n")

    print(f"\nModel və artefaktlar '{MODEL_DIR}' qovluğuna yadda saxlanıldı.")


if __name__ == "__main__":
    main()
