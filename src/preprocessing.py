"""
Telco Customer Churn datası üçün preprocessing pipeline.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_raw_data(path: str) -> pd.DataFrame:
    """CSV faylını oxuyur."""
    df = pd.read_csv(path)
    return df


def clean_data(path_or_df) -> pd.DataFrame:
    """Datanı təmizləyir: tipləri düzəldir, boş dəyərləri doldurur, lazımsız sütunu atır."""
    df = path_or_df.copy()

    # TotalCharges əvvəlcə string kimi oxunurdu, düz rəqəm kimi işləmirdi.
    # 11 sətirdə boş dəyər var idi (yəqin tenure=0 olan yeni müştərilər),
    # onları median ilə doldurdum, silmək data itkisi olardı
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # customerID-nin proqnozla heç bir əlaqəsi yoxdur, ona görə atırıq
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Hədəf sütununu 0/1-ə çeviririk ki, model üçün rahat olsun
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def encode_features(df: pd.DataFrame):
    """Kateqorik sütunları rəqəmlərə çevirir (one-hot və label encoding qarışıq)."""
    df = df.copy()

    # binary olanlara (2 dəyərli) LabelEncoder kifayətdir, amma Contract kimi
    # 3+ dəyəri olanlara one-hot lazımdır, yoxsa model onlar arasında
    # süni sıralama olduğunu düşünə bilər (məs. one-year > month-to-month kimi)
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    multi_cat_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod",
    ]
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    return df


def prepare_dataset(path: str, test_size: float = 0.2, random_state: int = 42):
    """Tam pipeline: oxu -> təmizlə -> encode et -> böl -> scale et."""
    df = load_raw_data(path)
    df = clean_data(df)
    df = encode_features(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Yalnız train datasına fit edirik, test datasına transform - bu vacibdir,
    # yoxsa test datasının məlumatı model qurmağa "sızar" (data leakage)
    scaler = StandardScaler()
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test, scaler, list(X.columns)
