import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


def preprocess_data(df, target_column, sensitive_columns=None):
    """
    Preprocess data: encode categoricals, separate features and targets
    """
    X = df.drop(columns=[target_column] + (sensitive_columns or []))
    y = df[target_column]
    
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
    
    return X, y


def train_model(X_train, y_train, random_state=42):
    """Train Random Forest classifier"""
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10)
    rf.fit(X_train, y_train)
    return rf


def demographic_parity(y_pred, sensitive_col):
    """
    Demographic Parity: Do all groups receive favorable outcome at equal rates?
    Returns: dict of rates per group
    """
    groups = sensitive_col.unique()
    rates = {}
    for group in groups:
        mask = sensitive_col == group
        if mask.sum() > 0:
            rates[group] = y_pred[mask].mean()
        else:
            rates[group] = np.nan
    return rates


def disparate_impact(y_pred, sensitive_col):
    """
    Disparate Impact: Ratio of worst-off to best-off group rate (EEOC 4/5 rule)
    Returns: ratio (≥0.8 is fair)
    """
    rates = demographic_parity(y_pred, sensitive_col)
    valid_rates = [r for r in rates.values() if not np.isnan(r)]
    
    if len(valid_rates) == 0 or max(valid_rates) == 0:
        return 0.0
    
    return min(valid_rates) / max(valid_rates)


def equalized_odds(y_pred, y_true, sensitive_col):
    """
    Equalized Odds: Equal TPR and FPR across groups
    Returns: dict with 'tpr' and 'fpr' per group
    """
    groups = sensitive_col.unique()
    tpr, fpr = {}, {}
    
    for group in groups:
        mask = sensitive_col == group
        y_p, y_t = y_pred[mask], y_true[mask]
        
        if len(y_t) == 0:
            tpr[group] = np.nan
            fpr[group] = np.nan
            continue
        
        tp = ((y_p == 1) & (y_t == 1)).sum()
        fn = ((y_p == 0) & (y_t == 1)).sum()
        fp = ((y_p == 1) & (y_t == 0)).sum()
        tn = ((y_p == 0) & (y_t == 0)).sum()
        
        tpr[group] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr[group] = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    
    return {'tpr': tpr, 'fpr': fpr}

def intersectional_demographic_parity(y_pred, df, attr1, attr2):
    """
    Compute Demographic Parity at intersection of two attributes
    E.g., Gender × Race → Female-Asian, Female-Black, Male-Asian, etc.
    Returns: dict of rates per intersection
    """
    intersections = df.groupby([attr1, attr2]).groups
    rates = {}
    
    for (a1, a2), group_idx in intersections.items():
        mask = df.index.isin(group_idx)
        if mask.sum() > 0:
            rates[(a1, a2)] = y_pred[mask].mean()
        else:
            rates[(a1, a2)] = np.nan
    
    return rates


def intersectional_disparate_impact(y_pred, df, attr1, attr2):
    """
    Disparate Impact for intersectional groups
    Computes min/max ratio across all intersections
    """
    rates = intersectional_demographic_parity(y_pred, df, attr1, attr2)
    valid_rates = [r for r in rates.values() if not np.isnan(r)]
    
    if len(valid_rates) == 0 or max(valid_rates) == 0:
        return 0.0
    
    return min(valid_rates) / max(valid_rates)


def intersectional_equalized_odds(y_pred, y_true, df, attr1, attr2):
    """
    Equalized Odds for intersectional groups
    Returns: dict with TPR/FPR per intersection
    """
    intersections = df.groupby([attr1, attr2]).groups
    tpr, fpr = {}, {}
    
    for (a1, a2), group_idx in intersections.items():
        mask = df.index.isin(group_idx)
        y_p, y_t = y_pred[mask], y_true[mask]
        
        if len(y_t) == 0:
            tpr[(a1, a2)] = np.nan
            fpr[(a1, a2)] = np.nan
            continue
        
        tp = ((y_p == 1) & (y_t == 1)).sum()
        fn = ((y_p == 0) & (y_t == 1)).sum()
        fp = ((y_p == 1) & (y_t == 0)).sum()
        tn = ((y_p == 0) & (y_t == 0)).sum()
        
        tpr[(a1, a2)] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr[(a1, a2)] = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    
    return {'tpr': tpr, 'fpr': fpr}

def worst_group_accuracy(y_pred, y_true, X_test, n_clusters=5):
    """
    Identify worst-performing cluster without knowing demographics
    Clusters the feature space unsupervised; finds group with lowest accuracy
    Returns: worst_cluster_id, worst_accuracy, all_accuracies
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_test)
    
    accuracies = {}
    group_sizes = {}
    for cluster_id in range(n_clusters):
        mask = clusters == cluster_id
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).mean()
            accuracies[cluster_id] = acc
            group_sizes[cluster_id] = mask.sum()
    
    worst_cluster = min(accuracies, key=accuracies.get)
    worst_accuracy = accuracies[worst_cluster]
    
    return {
        'worst_cluster': worst_cluster,
        'worst_accuracy': worst_accuracy,
        'accuracies': accuracies,
        'group_sizes': group_sizes,
        'accuracy_gap': max(accuracies.values()) - min(accuracies.values())
    }


def adversarial_bias_detection(y_pred, X_test, n_iterations=10):
    """
    Adversarial bias detection: Train an adversary to predict y_pred from features
    If adversary can't predict well, bias is low (prediction is independent of features)
    Returns: adversary accuracy (high = model predictions correlated with features)
    """
    from sklearn.linear_model import LogisticRegression
    
    adversary_accs = []
    n_samples = len(X_test)
    
    for _ in range(n_iterations):
        # Random train/test split for adversary
        train_idx = np.random.choice(n_samples, n_samples // 2, replace=False)
        test_idx = np.array([i for i in range(n_samples) if i not in train_idx])
        
        X_train_adv, X_test_adv = X_test.iloc[train_idx], X_test.iloc[test_idx]
        y_train_adv, y_test_adv = y_pred[train_idx], y_pred[test_idx]
        
        adv = LogisticRegression(max_iter=1000, random_state=42)
        adv.fit(X_train_adv, y_train_adv)
        adv_acc = adv.score(X_test_adv, y_test_adv)
        adversary_accs.append(adv_acc)
    
    mean_adversary_acc = np.mean(adversary_accs)

    return {
        'mean_adversary_accuracy': mean_adversary_acc,
        'bias_level': 'low' if mean_adversary_acc < 0.6 else 'medium' if mean_adversary_acc < 0.7 else 'high'
    }


def proxy_attribute_warnings(df, sensitive_columns):
    """
    Check if non-sensitive features are proxies for sensitive attributes
    E.g., ZIP code may proxy for race
    Returns: list of high-correlation pairs
    """
    warnings_list = []
    feature_cols = [col for col in df.columns if col not in sensitive_columns]
    
    for sens_col in sensitive_columns:
        # Encode sensitive column if categorical
        if df[sens_col].dtype == 'object':
            le = LabelEncoder()
            sens_encoded = le.fit_transform(df[sens_col])
        else:
            sens_encoded = df[sens_col]
        
        for feat_col in feature_cols[:10]:  # Check first 10 features for speed
            if df[feat_col].dtype in ['int64', 'float64']:
                corr = np.abs(np.corrcoef(sens_encoded, df[feat_col])[0, 1])
                if corr > 0.3:  # Moderate correlation threshold
                    warnings_list.append({
                        'sensitive_attr': sens_col,
                        'proxy_feature': feat_col,
                        'correlation': corr
                    })
    
    return warnings_list


def detect_bias(df, target_column, sensitive_columns, test_size=0.2, 
                compute_intersectional=True, compute_fairness_without_demographics=True): 
    from sklearn.model_selection import train_test_split
    
    # Preprocess
    X, y = preprocess_data(df, target_column, sensitive_columns)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Align sensitive columns with test set
    sensitive_test = {}
    for sens_col in sensitive_columns:
        sensitive_test[sens_col] = df.loc[X_test.index, sens_col].reset_index(drop=True)
    
    # Train model
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Standard metrics
    results = {
        'model': model,
        'y_pred': y_pred,
        'y_test': y_test.values,
        'X_test': X_test.reset_index(drop=True),
        'sensitive_test': sensitive_test,
        'metrics': {}
    }
    
    # Compute DP, DI, EO for each sensitive column
    for sens_col in sensitive_columns:
        sensitive_col_test = pd.Series(sensitive_test[sens_col])
        
        results['metrics'][sens_col] = {
            'demographic_parity': demographic_parity(y_pred, sensitive_col_test),
            'disparate_impact': disparate_impact(y_pred, sensitive_col_test),
            'equalized_odds': equalized_odds(y_pred, y_test.values, sensitive_col_test)
        }
    
    # NEW: Intersectional Fairness
    if compute_intersectional and len(sensitive_columns) >= 2:
        results['intersectional_metrics'] = {}
        df_test = pd.DataFrame(sensitive_test)
        
        for i, col1 in enumerate(sensitive_columns):
            for col2 in sensitive_columns[i+1:]:
                key = f"{col1} × {col2}"
                results['intersectional_metrics'][key] = {
                    'demographic_parity': intersectional_demographic_parity(y_pred, df_test, col1, col2),
                    'disparate_impact': intersectional_disparate_impact(y_pred, df_test, col1, col2),
                    'equalized_odds': intersectional_equalized_odds(y_pred, y_test.values, df_test, col1, col2)
                }
    
    # NEW: Fairness Without Protected Attributes
    if compute_fairness_without_demographics:
        results['fairness_without_demographics'] = {
            'worst_group_accuracy': worst_group_accuracy(y_pred, y_test.values, X_test),
            'adversarial_bias': adversarial_bias_detection(y_pred, X_test),
            'proxy_warnings': proxy_attribute_warnings(df, sensitive_columns)
        }
    
    return results
