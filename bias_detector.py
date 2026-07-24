import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, target_column):
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str))
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def demographic_parity(y_pred, sensitive_col):
    groups = sensitive_col.unique()
    rates = {}
    for group in groups:
        mask = sensitive_col == group
        rates[group] = y_pred[mask].mean()
    return rates

def disparate_impact(y_pred, sensitive_col):
    rates = demographic_parity(y_pred, sensitive_col)
    values = list(rates.values())
    if max(values) == 0:
        return 0
    return min(values) / max(values)

def equalized_odds(y_pred, y_true, sensitive_col):
    groups = sensitive_col.unique()
    tpr = {}
    fpr = {}
    for group in groups:
        mask = sensitive_col == group
        y_p = y_pred[mask]
        y_t = y_true[mask]
        tp = ((y_p == 1) & (y_t == 1)).sum()
        fn = ((y_p == 0) & (y_t == 1)).sum()
        fp = ((y_p == 1) & (y_t == 0)).sum()
        tn = ((y_p == 0) & (y_t == 0)).sum()
        tpr[group] = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr[group] = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr, fpr

def detect_bias(df, target_column, sensitive_column):
    X, y = preprocess_data(df, target_column)
    model, X_test, y_test = train_model(X, y)

    sensitive_test = df.loc[X_test.index, sensitive_column]
    # Re-encode sensitive column to match preprocessed data
    le = LabelEncoder()
    sensitive_test_encoded = pd.Series(
        le.fit_transform(sensitive_test.astype(str)),
        index=sensitive_test.index
    )

    y_pred = model.predict(X_test)
    y_pred_series = pd.Series(y_pred, index=X_test.index)
    y_test_series = pd.Series(y_test.values, index=X_test.index)

    dp = demographic_parity(y_pred_series, sensitive_test_encoded)
    di = disparate_impact(y_pred_series, sensitive_test_encoded)
    tpr, fpr = equalized_odds(y_pred_series, y_test_series, sensitive_test_encoded)

    return {
        "model": model,
        "demographic_parity": dp,
        "disparate_impact": di,
        "equalized_odds": {"tpr": tpr, "fpr": fpr},
        "y_pred": y_pred_series,
        "y_test": y_test_series,
        "X_test": X_test,
        "sensitive_test": sensitive_test_encoded
    }