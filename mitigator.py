import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from bias_detector import preprocess_data, train_model, detect_bias


def reweighing(df, target_column, sensitive_column):
    df = df.copy()
    groups = df[sensitive_column].unique()
    weights = {}

    for group in groups:
        mask = df[sensitive_column] == group
        group_size = mask.sum()
        expected = len(df) / len(groups)
        weights[group] = expected / group_size

    sample_weights = df[sensitive_column].map(weights)
    return sample_weights


def train_model_with_weights(df, target_column, sample_weights):
    X, y = preprocess_data(df, target_column)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train, sample_weight=w_train)
    return model, X_test, y_test


def suppress_sensitive_feature(df, sensitive_column):
    return df.drop(columns=[sensitive_column])


def apply_mitigation(df, target_column, sensitive_column, strategy="reweighing"):
    if strategy == "reweighing":
        sample_weights = reweighing(df, target_column, sensitive_column)
        model, X_test, y_test = train_model_with_weights(df, target_column, sample_weights)

        le = LabelEncoder()
        sensitive_test = df.loc[X_test.index, sensitive_column]
        sensitive_test_encoded = pd.Series(
            le.fit_transform(sensitive_test.astype(str)),
            index=sensitive_test.index
        )

        y_pred = pd.Series(model.predict(X_test), index=X_test.index)
        y_test_series = pd.Series(y_test.values, index=X_test.index)

        from bias_detector import demographic_parity, disparate_impact, equalized_odds
        dp = demographic_parity(y_pred, sensitive_test_encoded)
        di = disparate_impact(y_pred, sensitive_test_encoded)
        tpr, fpr = equalized_odds(y_pred, y_test_series, sensitive_test_encoded)

        return {
            "model": model,
            "strategy": "Reweighing",
            "demographic_parity": dp,
            "disparate_impact": di,
            "equalized_odds": {"tpr": tpr, "fpr": fpr},
            "y_pred": y_pred,
            "y_test": y_test_series,
            "X_test": X_test,
            "sensitive_test": sensitive_test_encoded
        }

    elif strategy == "suppression":
        df_suppressed = suppress_sensitive_feature(df, sensitive_column)
        results = detect_bias(df_suppressed, target_column, sensitive_column)
        results["strategy"] = "Feature Suppression"
        return results


def compare_results(before, after):
    comparison = {
        "disparate_impact": {
            "before": before["disparate_impact"],
            "after": after["disparate_impact"]
        },
        "demographic_parity_diff": {
            "before": max(before["demographic_parity"].values()) - min(before["demographic_parity"].values()),
            "after": max(after["demographic_parity"].values()) - min(after["demographic_parity"].values())
        },
        "tpr_diff": {
            "before": max(before["equalized_odds"]["tpr"].values()) - min(before["equalized_odds"]["tpr"].values()),
            "after": max(after["equalized_odds"]["tpr"].values()) - min(after["equalized_odds"]["tpr"].values())
        },
        "fpr_diff": {
            "before": max(before["equalized_odds"]["fpr"].values()) - min(before["equalized_odds"]["fpr"].values()),
            "after": max(after["equalized_odds"]["fpr"].values()) - min(after["equalized_odds"]["fpr"].values())
        }
    }
    return comparison