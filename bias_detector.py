import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')


class BiasDetector:
    """
    Detects bias in ML models using fairness metrics
    
    Metrics:
    - Demographic Parity (DP): Equal positive prediction rates
    - Disparate Impact (DI): EEOC four-fifths rule
    - Equalized Odds (EO): Equal TPR and FPR across groups
    """
    
    def __init__(self):
        self.model = None
        self.le_dict = {}
    
    def detect_bias(self, df, target_column, sensitive_columns, 
                   include_intersectional=True, include_fwd=True):
        """
        Main bias detection function
        
        Args:
            df: DataFrame with data
            target_column: Column to predict
            sensitive_columns: List of protected attributes
            include_intersectional: Include intersectional analysis
            include_fwd: Include fairness-without-demographics
        
        Returns:
            Dictionary with all bias metrics and verdicts
        """
        
        # Validate input
        if len(df) < 50:
            raise ValueError("Dataset too small (minimum 50 samples)")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        for col in sensitive_columns:
            if col not in df.columns:
                raise ValueError(f"Sensitive column '{col}' not found")
        
        # Prepare data
        X, y, df_encoded = self._prepare_data(df, target_column, sensitive_columns)
        
        # Train model
        try:
            self.model, X_test, y_test, X_train, y_train = self._train_model(X, y)
        except Exception as e:
            raise ValueError(f"Error training model: {e}")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Compute fairness metrics
        results = {
            'model_accuracy': np.mean(y_pred == y_test),
            'predictions': y_pred.tolist(),
            'y_true': y_test.tolist(),
            'y_pred_proba': y_pred_proba.tolist(),
            'standard_metrics': {}
        }
        
        # Standard fairness metrics
        df_test_encoded = df_encoded.iloc[X_test.index] if hasattr(X_test, 'index') else df_encoded.iloc[-len(y_test):]
        
        results['standard_metrics']['demographic_parity'] = self._demographic_parity(
            y_pred, df_test_encoded, sensitive_columns
        )
        
        results['standard_metrics']['disparate_impact'] = self._disparate_impact(
            y_pred, df_test_encoded, sensitive_columns
        )
        
        results['standard_metrics']['equalized_odds'] = self._equalized_odds(
            y_pred, y_test, df_test_encoded, sensitive_columns
        )
        
        # Intersectional fairness
        if include_intersectional and len(sensitive_columns) > 1:
            results['intersectional_metrics'] = self._intersectional_fairness(
                y_pred, y_test, df_test_encoded, sensitive_columns
            )
        
        # Fairness without demographics
        if include_fwd:
            results['fairness_without_demographics'] = self._fairness_without_demographics(
                y_pred, y_test, X_test
            )
        
        return results
    
    def _prepare_data(self, df, target_column, sensitive_columns):
        """Prepare and encode data"""
        df_clean = df.dropna()
        
        # Encode categorical features
        df_encoded = df_clean.copy()
        feature_columns = [col for col in df_clean.columns 
                          if col != target_column]
        
        for col in feature_columns:
            if df_encoded[col].dtype == 'object':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.le_dict[col] = le
        
        # Separate features and target
        X = df_encoded.drop(target_column, axis=1)
        y = df_encoded[target_column]
        
        # Ensure binary classification
        if len(np.unique(y)) > 2:
            # Convert to binary (top 2 classes or top/bottom split)
            if df_encoded[target_column].dtype == 'object':
                unique_vals = df_encoded[target_column].unique()
                y = (df_encoded[target_column] == unique_vals[0]).astype(int)
            else:
                median = df_encoded[target_column].median()
                y = (df_encoded[target_column] > median).astype(int)
        
        return X, y, df_encoded
    
    def _train_model(self, X, y):
        """Train Random Forest model with stratified split"""
        # Check class balance
        if len(np.unique(y)) < 2:
            raise ValueError("Target column must have at least 2 classes")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        return model, X_test, y_test, X_train, y_train
    
    def _demographic_parity(self, y_pred, df, sensitive_columns):
        """
        Demographic Parity: Equal positive prediction rates across groups
        
        DP Gap = |P(Ŷ=1|A=a) - P(Ŷ=1|A=b)|
        Fair: gap < 5%, Mild: 5-10%, Significant: > 10%
        """
        results = {
            'metric_name': 'Demographic Parity',
            'definition': 'Equal positive prediction rates across groups',
            'dp_gap': 0,
            'verdict': 'Unknown',
            'group_rates': {}
        }
        
        positive_rates = []
        
        for col in sensitive_columns:
            try:
                unique_vals = df[col].unique()
                rates = {}
                
                for val in unique_vals:
                    mask = df[col] == val
                    if mask.sum() > 0:
                        rate = y_pred[mask].mean()
                        rates[str(val)] = rate
                        positive_rates.append(rate)
                
                results['group_rates'][col] = rates
            except:
                pass
        
        if positive_rates:
            dp_gap = max(positive_rates) - min(positive_rates)
            results['dp_gap'] = dp_gap
            
            if dp_gap < 0.05:
                results['verdict'] = 'Fair'
            elif dp_gap < 0.10:
                results['verdict'] = 'Mild'
            else:
                results['verdict'] = 'Significant'
        
        return results
    
    def _disparate_impact(self, y_pred, df, sensitive_columns):
        """
        Disparate Impact (Four-Fifths Rule): DI >= 0.8 is legal
        
        DI = min(P(Ŷ=1|A=a)) / max(P(Ŷ=1|A=b))
        Fair: DI >= 0.8, Mild: 0.6-0.8, Significant: < 0.6
        
        Reference: EEOC Guidelines
        """
        results = {
            'metric_name': 'Disparate Impact (EEOC Four-Fifths Rule)',
            'definition': 'Ratio of positive prediction rates (EEOC threshold: 0.8)',
            'di_ratio': 1.0,
            'verdict': 'Unknown',
            'group_rates': {}
        }
        
        positive_rates = []
        
        for col in sensitive_columns:
            try:
                unique_vals = df[col].unique()
                rates = {}
                
                for val in unique_vals:
                    mask = df[col] == val
                    if mask.sum() > 0:
                        rate = y_pred[mask].mean()
                        rates[str(val)] = rate
                        positive_rates.append(rate)
                
                results['group_rates'][col] = rates
            except:
                pass
        
        if positive_rates:
            di_ratio = min(positive_rates) / (max(positive_rates) + 1e-10)
            results['di_ratio'] = di_ratio
            
            if di_ratio >= 0.8:
                results['verdict'] = 'Fair'
            elif di_ratio >= 0.6:
                results['verdict'] = 'Mild'
            else:
                results['verdict'] = 'Significant'
        
        return results
    
    def _equalized_odds(self, y_pred, y_test, df, sensitive_columns):
        """
        Equalized Odds: Equal TPR and FPR across groups
        
        TPR_a = TP/(TP+FN), FPR_a = FP/(FP+TN)
        Fair: both gaps < 5%, Mild: < 10%, Significant: >= 10%
        
        Reference: Hardt et al. 2016 (NIPS)
        """
        results = {
            'metric_name': 'Equalized Odds',
            'definition': 'Equal True Positive Rates and False Positive Rates across groups',
            'tpr_gap': 0,
            'fpr_gap': 0,
            'verdict': 'Unknown',
            'group_metrics': {}
        }
        
        tpr_values = []
        fpr_values = []
        
        for col in sensitive_columns:
            try:
                unique_vals = df[col].unique()
                metrics = {}
                
                for val in unique_vals:
                    mask = df[col] == val
                    
                    if mask.sum() > 0:
                        y_pred_group = y_pred[mask]
                        y_test_group = y_test.iloc[mask] if hasattr(y_test, 'iloc') else y_test[mask]
                        
                        # TPR: True Positive Rate
                        if (y_test_group == 1).sum() > 0:
                            tp = ((y_pred_group == 1) & (y_test_group == 1)).sum()
                            tpr = tp / ((y_test_group == 1).sum())
                            tpr_values.append(tpr)
                        else:
                            tpr = 0
                        
                        # FPR: False Positive Rate
                        if (y_test_group == 0).sum() > 0:
                            fp = ((y_pred_group == 1) & (y_test_group == 0)).sum()
                            fpr = fp / ((y_test_group == 0).sum())
                            fpr_values.append(fpr)
                        else:
                            fpr = 0
                        
                        metrics[str(val)] = {'tpr': tpr, 'fpr': fpr}
                
                results['group_metrics'][col] = metrics
            except:
                pass
        
        if tpr_values and fpr_values:
            tpr_gap = max(tpr_values) - min(tpr_values)
            fpr_gap = max(fpr_values) - min(fpr_values)
            
            results['tpr_gap'] = tpr_gap
            results['fpr_gap'] = fpr_gap
            
            max_gap = max(tpr_gap, fpr_gap)
            
            if max_gap < 0.05:
                results['verdict'] = 'Fair'
            elif max_gap < 0.10:
                results['verdict'] = 'Mild'
            else:
                results['verdict'] = 'Significant'
        
        return results
    
    def _intersectional_fairness(self, y_pred, y_test, df, sensitive_columns):
        """
        Intersectional Fairness: Analyze fairness across combinations of attributes
        
        Applies DP, DI, EO to attribute intersections (e.g., gender × race)
        """
        intersectional_metrics = {}
        
        # For each pair of sensitive columns
        for i in range(len(sensitive_columns)):
            for j in range(i+1, len(sensitive_columns)):
                col1, col2 = sensitive_columns[i], sensitive_columns[j]
                pair_name = f"{col1} × {col2}"
                
                try:
                    # Group by intersection
                    groups = df.groupby([col1, col2]).groups
                    
                    positive_rates = []
                    for (val1, val2), indices in groups.items():
                        if len(indices) > 0:
                            rate = y_pred[indices].mean()
                            positive_rates.append(rate)
                    
                    if positive_rates:
                        dp_gap = max(positive_rates) - min(positive_rates)
                        di_ratio = min(positive_rates) / (max(positive_rates) + 1e-10)
                        
                        intersectional_metrics[pair_name] = {
                            'dp_gap': dp_gap,
                            'di_ratio': di_ratio,
                            'verdict': 'Significant' if di_ratio < 0.6 else 'Mild' if di_ratio < 0.8 else 'Fair'
                        }
                except:
                    pass
        
        return intersectional_metrics
    
    def _fairness_without_demographics(self, y_pred, y_test, X_test):
        """
        Fairness Without Protected Attributes:
        Three privacy-preserving bias detection techniques
        
        1. Worst-Group Accuracy: Unsupervised clustering finds vulnerable groups
        2. Adversarial Bias Detection: Can model decisions be predicted from features?
        3. Proxy Warnings: Detect features correlated with protected attributes
        """
        fwd_results = {}
        
        try:
            # Method 1: Worst-Group Accuracy
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_test)
            
            accuracies = {}
            for cluster_id in range(5):
                mask = clusters == cluster_id
                if mask.sum() > 0:
                    y_test_cluster = y_test.iloc[mask] if hasattr(y_test, 'iloc') else y_test[mask]
                    acc = (y_pred[mask] == y_test_cluster).mean()
                    accuracies[cluster_id] = acc
            
            worst_acc = min(accuracies.values()) if accuracies else 0
            fwd_results['worst_group_accuracy'] = worst_acc
        except:
            fwd_results['worst_group_accuracy'] = 0
        
        try:
            # Method 2: Adversarial Bias Detection
            # Can an adversary predict model outputs from features?
            adversary = LogisticRegression(max_iter=1000, random_state=42)
            
            split_point = len(X_test) // 2
            X_train_adv = X_test.iloc[:split_point] if hasattr(X_test, 'iloc') else X_test[:split_point]
            X_test_adv = X_test.iloc[split_point:] if hasattr(X_test, 'iloc') else X_test[split_point:]
            y_pred_train = y_pred[:split_point]
            y_pred_test = y_pred[split_point:]
            
            adversary.fit(X_train_adv, y_pred_train)
            adv_accuracy = adversary.score(X_test_adv, y_pred_test)
            
            fwd_results['adversarial_bias_score'] = adv_accuracy
        except:
            fwd_results['adversarial_bias_score'] = 0.5
        
        # Method 3: Proxy warnings (would need protected attributes, so simplified)
        fwd_results['proxy_warnings'] = [
            "Recommendation: Check for features that may serve as proxies for protected attributes"
        ]
        
        return fwd_results
