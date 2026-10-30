"""
Fairness Mitigation Module
Reweighing (Kamiran & Calders 2012), feature suppression, and proxy removal
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

MIN_GROUP = 20


class FairnessMitigator:
    """Implements fairness mitigation strategies"""

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _encode(df_clean, target_column, drop_cols):
        """Return numeric X and binary 0/1 y (positional index)"""
        X = df_clean.drop(columns=[target_column] + list(drop_cols), errors='ignore').copy()
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        y = df_clean[target_column]
        if not pd.api.types.is_numeric_dtype(y):
            if y.nunique() == 2:
                y = pd.Series(LabelEncoder().fit_transform(y.astype(str)))
            else:
                y = (y == y.value_counts().index[0]).astype(int)
        elif y.nunique() > 2:
            y = (y > y.median()).astype(int)
        return X, pd.Series(np.asarray(y)).astype(int)

    @staticmethod
    def _fairness_after(y_pred, y_true, sens_test):
        rates = [float(y_pred[sens_test == v].mean())
                 for v in np.unique(sens_test) if (sens_test == v).sum() >= MIN_GROUP]
        if len(rates) < 2:
            return 0.0, 1.0
        return max(rates) - min(rates), min(rates) / (max(rates) + 1e-10)

    def _train_eval(self, X, y, df_clean, sensitive_column, sample_weight=None):
        idx = np.arange(len(df_clean))
        tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                       random_state=42, n_jobs=-1)
        model.fit(X.iloc[tr], y.iloc[tr],
                  sample_weight=None if sample_weight is None else sample_weight[tr])
        y_pred = model.predict(X.iloc[te])
        sens_test = df_clean[sensitive_column].astype(str).values[te]
        dp_gap, di = self._fairness_after(y_pred, y.iloc[te].values, sens_test)
        return {'accuracy': float(np.mean(y_pred == y.iloc[te].values)),
                'dp_gap': float(dp_gap), 'di_ratio': float(di), 'success': True}

    @staticmethod
    def reweighing_weights(group_keys, y):
        """Kamiran & Calders: weight(g, y) = P(g) * P(y) / P(g, y)"""
        group_keys = np.asarray(group_keys).astype(str)
        y = np.asarray(y)
        w = np.ones(len(y))
        for g in np.unique(group_keys):
            for yv in np.unique(y):
                mask = (group_keys == g) & (y == yv)
                if mask.any():
                    w[mask] = (group_keys == g).mean() * (y == yv).mean() / mask.mean()
        return w

    def build_mitigated_dataset(self, df, sensitive_columns, target_column,
                                strategy, proxy_features=None):
        """
        Returns the dataset as it would be fed to the mitigated model.
          reweighting   -> cleaned data + 'sample_weight' column (all sensitive cols combined)
          suppression   -> sensitive columns removed
          proxy_removal -> sensitive columns AND detected proxy features removed
        """
        df_clean = df.dropna().reset_index(drop=True)
        if strategy == 'reweighting':
            _, y = self._encode(df_clean, target_column, drop_cols=[])
            keys = df_clean[list(sensitive_columns)].astype(str).agg('|'.join, axis=1)
            out = df_clean.copy()
            out['sample_weight'] = self.reweighing_weights(keys.values, y.values).round(4)
            return out
        drop = list(sensitive_columns)
        if strategy == 'proxy_removal':
            drop += [c for c in (proxy_features or []) if c not in drop]
        return df_clean.drop(columns=[c for c in drop if c in df_clean.columns])

    # ------------------------------------------------------------ strategies
    def apply_reweighting_mitigation(self, df, sensitive_column, target_column):
        """
        Reweighing (Kamiran & Calders 2012):
        weight(s, y) = P(s) * P(y) / P(s, y)
        Combinations of group and outcome that are rarer than independence
        would predict are up-weighted, so the model cannot learn the correlation.
        """
        try:
            df_clean = df.dropna().reset_index(drop=True)
            X, y = self._encode(df_clean, target_column, drop_cols=[])
            s = df_clean[sensitive_column].astype(str).values

            weights = self.reweighing_weights(s, y.values)

            out = self._train_eval(X, y, df_clean, sensitive_column, sample_weight=weights)
            out.update({'strategy': 'Reweighting', 'sensitive_column': sensitive_column})
            return out
        except Exception as e:
            return {'accuracy': 0, 'dp_gap': 0, 'di_ratio': 1.0, 'strategy': 'Reweighting',
                    'sensitive_column': sensitive_column, 'success': False, 'error': str(e)}

    def apply_suppression_mitigation(self, df, sensitive_column, target_column):
        """Fairness through unawareness: drop the sensitive column (proxies may remain)"""
        out = self.apply_proxy_removal(df, sensitive_column, target_column, proxy_features=[])
        out.update({'strategy': 'Feature Suppression', 'sensitive_column': sensitive_column})
        return out

    def apply_proxy_removal(self, df, sensitive_column, target_column, proxy_features):
        """Drop the sensitive column AND its proxy features, retrain, measure fairness"""
        try:
            df_clean = df.dropna().reset_index(drop=True)
            drop_cols = [sensitive_column] + [c for c in proxy_features if c in df_clean.columns]
            X, y = self._encode(df_clean, target_column, drop_cols)
            out = self._train_eval(X, y, df_clean, sensitive_column)
            out['dropped'] = drop_cols
            return out
        except Exception as e:
            return {'accuracy': 0, 'dp_gap': 0, 'di_ratio': 1.0, 'success': False, 'error': str(e)}

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
