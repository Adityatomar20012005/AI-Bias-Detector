"""
Fairness Mitigation Module
Implements reweighting and feature suppression strategies
to reduce bias and improve fairness
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class FairnessMitigator:
    """Implements fairness mitigation strategies"""
    
    def apply_reweighting_mitigation(self, df, sensitive_column, target_column):
        """
        Reweighting Mitigation: Upweight underrepresented groups
        
        Gives higher weight to samples from disadvantaged groups during training.
        This encourages the model to pay more attention to minority groups.
        
        Args:
            df: DataFrame
            sensitive_column: Sensitive attribute to mitigate
            target_column: Target variable
        
        Returns:
            Dictionary with mitigation results
        """
        
        try:
            df_clean = df.dropna()
            
            # Prepare features
            X = df_clean.drop(target_column, axis=1)
            y = df_clean[target_column]
            
            # Encode categorical features
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            # Encode target
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y.astype(str))
            
            # Ensure binary
            if len(np.unique(y)) > 2:
                median = y.median() if hasattr(y, 'median') else np.median(y)
                y = (y > median).astype(int)
            
            # Calculate sample weights based on sensitive attribute
            sensitive_attr = df_clean[sensitive_column]
            
            # Encode sensitive attribute if needed
            if sensitive_attr.dtype == 'object':
                le_sens = LabelEncoder()
                sensitive_attr_encoded = le_sens.fit_transform(sensitive_attr.astype(str))
            else:
                sensitive_attr_encoded = sensitive_attr.values
            
            # Compute group sizes
            unique_groups = np.unique(sensitive_attr_encoded)
            group_sizes = {}
            
            for group in unique_groups:
                group_sizes[group] = (sensitive_attr_encoded == group).sum()
            
            # Create sample weights (inverse of group size)
            sample_weights = np.ones(len(X))
            
            for group, size in group_sizes.items():
                mask = sensitive_attr_encoded == group
                # Weight inversely proportional to group size
                weight = 1.0 / (size + 1e-10)
                sample_weights[mask] = weight
            
            # Normalize weights
            sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
            
            # Train/test split
            X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
                X, y, sample_weights,
                test_size=0.2,
                random_state=42
            )
            
            # Train model with sample weights
            model_mitigated = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model_mitigated.fit(X_train, y_train, sample_weight=weights_train)
            
            # Evaluate
            y_pred_mitigated = model_mitigated.predict(X_test)
            accuracy_mitigated = np.mean(y_pred_mitigated == y_test)
            
            # Compute fairness metrics after mitigation
            df_test = df_clean.iloc[X_test.index] if hasattr(X_test, 'index') else df_clean.iloc[-len(y_test):]
            
            # DP Gap after mitigation
            positive_rates = []
            for group in np.unique(sensitive_attr_encoded):
                mask = (df_test[sensitive_column] == group) if sensitive_column in df_test.columns else np.zeros(len(df_test), dtype=bool)
                if mask.sum() > 0:
                    rate = y_pred_mitigated[mask].mean()
                    positive_rates.append(rate)
            
            dp_gap_mitigated = (max(positive_rates) - min(positive_rates)) if positive_rates else 0
            
            # DI Ratio after mitigation
            di_mitigated = (min(positive_rates) / (max(positive_rates) + 1e-10)) if positive_rates else 1.0
            
            return {
                'accuracy': accuracy_mitigated,
                'dp_gap': dp_gap_mitigated,
                'di_ratio': di_mitigated,
                'strategy': 'Reweighting',
                'sensitive_column': sensitive_column,
                'success': True
            }
        
        except Exception as e:
            print(f"Error in reweighting mitigation: {e}")
            return {
                'accuracy': 0,
                'dp_gap': 0,
                'di_ratio': 1.0,
                'strategy': 'Reweighting',
                'sensitive_column': sensitive_column,
                'success': False,
                'error': str(e)
            }
    
    def apply_suppression_mitigation(self, df, sensitive_column, target_column):
        """
        Feature Suppression Mitigation: Remove sensitive attribute from training
        
        This is the "fairness through unawareness" approach.
        Removes the sensitive attribute from features so model can't directly 
        discriminate based on it.
        
        Note: Proxy features may still leak protected attribute information.
        
        Args:
            df: DataFrame
            sensitive_column: Sensitive attribute to suppress
            target_column: Target variable
        
        Returns:
            Dictionary with mitigation results
        """
        
        try:
            df_clean = df.dropna()
            
            # Remove sensitive column from features
            X = df_clean.drop([target_column, sensitive_column], axis=1, errors='ignore')
            y = df_clean[target_column]
            
            # Encode categorical features
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            # Encode target
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y.astype(str))
            
            # Ensure binary
            if len(np.unique(y)) > 2:
                median = y.median() if hasattr(y, 'median') else np.median(y)
                y = (y > median).astype(int)
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
            
            # Train model WITHOUT sensitive attribute
            model_suppressed = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model_suppressed.fit(X_train, y_train)
            
            # Evaluate
            y_pred_suppressed = model_suppressed.predict(X_test)
            accuracy_suppressed = np.mean(y_pred_suppressed == y_test)
            
            # Compute fairness metrics after suppression
            df_test = df_clean.iloc[X_test.index] if hasattr(X_test, 'index') else df_clean.iloc[-len(y_test):]
            
            # DP Gap after suppression
            positive_rates = []
            if sensitive_column in df_test.columns:
                unique_vals = df_test[sensitive_column].unique()
                for val in unique_vals:
                    mask = df_test[sensitive_column] == val
                    if mask.sum() > 0:
                        rate = y_pred_suppressed[mask].mean()
                        positive_rates.append(rate)
            
            dp_gap_suppressed = (max(positive_rates) - min(positive_rates)) if positive_rates else 0
            di_suppressed = (min(positive_rates) / (max(positive_rates) + 1e-10)) if positive_rates else 1.0
            
            return {
                'accuracy': accuracy_suppressed,
                'dp_gap': dp_gap_suppressed,
                'di_ratio': di_suppressed,
                'strategy': 'Feature Suppression',
                'sensitive_column': sensitive_column,
                'success': True
            }
        
        except Exception as e:
            print(f"Error in suppression mitigation: {e}")
            return {
                'accuracy': 0,
                'dp_gap': 0,
                'di_ratio': 1.0,
                'strategy': 'Feature Suppression',
                'sensitive_column': sensitive_column,
                'success': False,
                'error': str(e)
            }
    
    def compare_strategies(self, original_results, reweighting_results, suppression_results):
        """
        Compare different mitigation strategies
        
        Returns:
            Dictionary comparing results
        """
        
        comparison = {
            'original': {
                'accuracy': original_results.get('model_accuracy', 0),
                'dp_gap': original_results.get('standard_metrics', {}).get('demographic_parity', {}).get('dp_gap', 0),
                'di_ratio': original_results.get('standard_metrics', {}).get('disparate_impact', {}).get('di_ratio', 1.0)
            },
            'reweighting': {
                'accuracy': reweighting_results.get('accuracy', 0),
                'dp_gap': reweighting_results.get('dp_gap', 0),
                'di_ratio': reweighting_results.get('di_ratio', 1.0)
            },
            'suppression': {
                'accuracy': suppression_results.get('accuracy', 0),
                'dp_gap': suppression_results.get('dp_gap', 0),
                'di_ratio': suppression_results.get('di_ratio', 1.0)
            }
        }
        
        # Calculate improvements
        comparison['improvements'] = {
            'reweighting': {
                'accuracy_change': reweighting_results.get('accuracy', 0) - comparison['original']['accuracy'],
                'dp_improvement': comparison['original']['dp_gap'] - reweighting_results.get('dp_gap', 0),
                'di_improvement': reweighting_results.get('di_ratio', 1.0) - comparison['original']['di_ratio']
            },
            'suppression': {
                'accuracy_change': suppression_results.get('accuracy', 0) - comparison['original']['accuracy'],
                'dp_improvement': comparison['original']['dp_gap'] - suppression_results.get('dp_gap', 0),
                'di_improvement': suppression_results.get('di_ratio', 1.0) - comparison['original']['di_ratio']
            }
        }
        
        return comparison
