import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import warnings
from mitigator import FairnessMitigator

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
        self.target_classes = None
    
    def detect_bias(self, df, target_column, sensitive_columns, 
                   include_intersectional=True, include_fwd=True,
                   mitigation='none'):
        """
        Main bias detection function
        
        Args:
            df: DataFrame with data
            target_column: Column to predict
            sensitive_columns: List of protected attributes
            include_intersectional: Include intersectional analysis
            include_fwd: Include fairness-without-demographics
            mitigation: 'none' | 'reweighting' | 'suppression' | 'proxy_removal'
                        (applied to the model that is audited)
        
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
        
        # Proxy features (found on the full encoded data)
        proxies = self._find_proxies(X, sensitive_columns)
        proxy_names = sorted({p['feature'] for p in proxies})

        # Choose what the audited model is trained on
        drop_cols, groups = [], None
        if mitigation == 'suppression':
            drop_cols = list(sensitive_columns)
        elif mitigation == 'proxy_removal':
            drop_cols = list(sensitive_columns) + proxy_names
        elif mitigation == 'reweighting':
            groups = df_encoded[list(sensitive_columns)].astype(str).agg('|'.join, axis=1)
        elif mitigation != 'none':
            raise ValueError(f"Unknown mitigation '{mitigation}'")

        X_model = X.drop(columns=drop_cols)

        # Train model
        try:
            self.model, X_test_m, y_test, X_train, y_train = self._train_model(X_model, y, groups)
        except Exception as e:
            raise ValueError(f"Error training model: {e}")

        X_test = X.loc[X_test_m.index]          # full feature set for FWD / proxy checks

        # Get predictions
        y_pred = self.model.predict(X_test_m)
        y_pred_proba = self.model.predict_proba(X_test_m)[:, 1]

        # Compute fairness metrics
        results = {
            'model_accuracy': np.mean(y_pred == y_test),
            'predictions': y_pred.tolist(),
            'y_true': y_test.tolist(),
            'y_pred_proba': y_pred_proba.tolist(),
            'test_index': X_test.index.tolist(),
            'test_sensitive_data': {c: df.loc[X_test.index, c].tolist() for c in sensitive_columns},
            'feature_importance': dict(zip(X_test_m.columns, self.model.feature_importances_.tolist())),
            'mitigation': {'strategy': mitigation, 'dropped_columns': drop_cols,
                           'proxy_features': proxies},
            'standard_metrics': {}
        }
        
        # Standard fairness metrics
        df_test_encoded = df_encoded.loc[X_test.index] if hasattr(X_test, 'index') else df_encoded.iloc[-len(y_test):]
        
        results['standard_metrics'] = self._metrics_bundle(
            y_pred, y_test, df_test_encoded, sensitive_columns
        )

        # Before-mitigation reference (same split, all features, no weights)
        if mitigation != 'none':
            base_model, _, _, _, _ = self._train_model(X, y)
            base_pred = base_model.predict(X_test)
            results['baseline'] = {
                'model_accuracy': float(np.mean(base_pred == y_test)),
                'standard_metrics': self._metrics_bundle(
                    base_pred, y_test, df_test_encoded, sensitive_columns)
            }
        
        # Intersectional fairness
        if include_intersectional and len(sensitive_columns) > 1:
            results['intersectional_metrics'] = self._intersectional_fairness(
                y_pred, y_test, df_test_encoded, sensitive_columns
            )

        # Fairness without demographics
        if include_fwd:
            results['fairness_without_demographics'] = self._fairness_without_demographics(
                y_pred, y_test, X_test, sensitive_columns, proxies
            )

        return results

    def _metrics_bundle(self, y_pred, y_test, df_test_encoded, sensitive_columns):
        return {
            'demographic_parity': self._demographic_parity(y_pred, df_test_encoded, sensitive_columns),
            'disparate_impact': self._disparate_impact(y_pred, df_test_encoded, sensitive_columns),
            'equalized_odds': self._equalized_odds(y_pred, y_test, df_test_encoded, sensitive_columns),
        }

    def _find_proxies(self, X, sensitive_columns, threshold=0.3):
        """Features whose |Spearman correlation| with a sensitive column >= threshold"""
        proxies = []
        try:
            X_num = pd.DataFrame(X).reset_index(drop=True)
            for s in sensitive_columns:
                if s not in X_num.columns:
                    continue
                for feat in X_num.columns:
                    if feat in sensitive_columns:
                        continue
                    c = X_num[feat].corr(X_num[s], method='spearman')
                    if pd.notna(c) and abs(c) >= threshold:
                        proxies.append({'feature': feat, 'sensitive': s,
                                        'correlation': round(float(c), 3)})
            proxies.sort(key=lambda d: -abs(d['correlation']))
        except Exception:
            pass
        return proxies

    def _decode(self, col, val):
        """Turn encoded value back into its original label"""
        if col in self.le_dict:
            try:
                return self.le_dict[col].inverse_transform([int(val)])[0]
            except Exception:
                pass
        return val

    def _intersectional_fairness(self, y_pred, y_test, df, sensitive_columns):
        metrics = {}
        df = df.reset_index(drop=True)      # positions now match y_pred
        y_pred = np.asarray(y_pred)

        for i in range(len(sensitive_columns)):
            for j in range(i + 1, len(sensitive_columns)):
                col1, col2 = sensitive_columns[i], sensitive_columns[j]
                pair_name = f"{col1} × {col2}"
                try:
                    rates = {}
                    for (v1, v2), g in df.groupby([col1, col2]):
                        if len(g) >= 20:    # skip tiny, unreliable groups
                            label = f"{self._decode(col1, v1)} + {self._decode(col2, v2)}"
                            rates[label] = float(y_pred[g.index].mean())

                    if len(rates) < 2:
                        metrics[pair_name] = {'error': 'Not enough groups with 20+ samples'}
                        continue

                    best = max(rates, key=rates.get)
                    worst = min(rates, key=rates.get)
                    dp_gap = rates[best] - rates[worst]
                    di_ratio = rates[worst] / (rates[best] + 1e-10)

                    metrics[pair_name] = {
                        'dp_gap': dp_gap,
                        'di_ratio': di_ratio,
                        'verdict': 'Significant' if di_ratio < 0.6 else 'Mild' if di_ratio < 0.8 else 'Fair',
                        'most_favored': best,
                        'least_favored': worst,
                        'rates': rates
                    }
                except Exception as e:
                    metrics[pair_name] = {'error': str(e)}
        return metrics

    def _fairness_without_demographics(self, y_pred, y_test, X_test, sensitive_columns=None,
                                       proxy_features=None):
        fwd_results = {}
        y_pred = np.asarray(y_pred)

        # Method 1: Worst-group accuracy (unchanged)
        try:
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_test)
            accuracies = {}
            for cid in range(5):
                mask = clusters == cid
                if mask.sum() > 0:
                    y_c = y_test.iloc[mask] if hasattr(y_test, 'iloc') else y_test[mask]
                    accuracies[cid] = (y_pred[mask] == y_c).mean()
            fwd_results['worst_group_accuracy'] = min(accuracies.values()) if accuracies else 0
        except Exception:
            fwd_results['worst_group_accuracy'] = 0

        # Method 2: Adversarial score + baseline to compare against
        try:
            adversary = LogisticRegression(max_iter=1000, random_state=42)
            sp = len(X_test) // 2
            adversary.fit(X_test.iloc[:sp], y_pred[:sp])
            fwd_results['adversarial_bias_score'] = adversary.score(X_test.iloc[sp:], y_pred[sp:])
            p = float(np.mean(y_pred[sp:]))
            fwd_results['adversarial_baseline'] = max(p, 1 - p)   # accuracy of "always guess majority"
        except Exception:
            fwd_results['adversarial_bias_score'] = 0.5
            fwd_results['adversarial_baseline'] = 0.5

        # Method 3: Real proxy detection (computed once on the full data)
        if proxy_features is None:
            proxy_features = self._find_proxies(X_test, sensitive_columns or [])

        fwd_results['proxy_features'] = proxy_features
        fwd_results['proxy_warnings'] = (
            [f"{p['feature']} is correlated with {p['sensitive']} (r = {p['correlation']})"
             for p in proxy_features]
            or ["No strong proxy features found (|r| < 0.3)"]
        )
        return fwd_results

    def _prepare_data(self, df, target_column, sensitive_columns):
        """Prepare and encode data (features AND target)"""
        df_clean = df.dropna()
        df_encoded = df_clean.copy()

        # Encode every non-numeric feature column
        for col in df_clean.columns:
            if col == target_column:
                continue
            if not pd.api.types.is_numeric_dtype(df_encoded[col]):
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.le_dict[col] = le

        # Encode target -> 0/1
        y = df_encoded[target_column]
        if not pd.api.types.is_numeric_dtype(y):
            if y.nunique() == 2:
                le_t = LabelEncoder()           # sorted: '<=50K' -> 0, '>50K' -> 1
                y = pd.Series(le_t.fit_transform(y.astype(str)), index=y.index)
                self.target_classes = list(le_t.classes_)
            else:
                top = y.value_counts().index[0]
                y = (y == top).astype(int)
        elif y.nunique() > 2:
            y = (y > y.median()).astype(int)
        df_encoded[target_column] = y

        X = df_encoded.drop(target_column, axis=1)
        return X, y, df_encoded

    def _train_model(self, X, y, groups=None):
        """Train Random Forest with a stratified split (optionally with reweighing)"""
        if len(np.unique(y)) < 2:
            raise ValueError("Target column must have at least 2 classes")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        sample_weight = None
        if groups is not None:      # weights estimated on TRAIN rows only
            sample_weight = FairnessMitigator.reweighing_weights(
                groups.loc[X_train.index].values, y_train.values)

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)

        return model, X_test, y_test, X_train, y_train
    
    MIN_GROUP = 20   # ignore tiny groups (unreliable rates)

    @staticmethod
    def _worst(verdicts):
        order = {'Fair': 0, 'Mild': 1, 'Significant': 2}
        known = [v for v in verdicts if v in order]
        return max(known, key=order.get) if known else 'Unknown'

    def _group_rates(self, y_pred, df, col):
        """Positive-prediction rate per group of ONE sensitive column"""
        rates = {}
        for val, g in df.groupby(col):
            if len(g) >= self.MIN_GROUP:
                rates[str(self._decode(col, val))] = float(y_pred[g.index].mean())
        return rates

    def _demographic_parity(self, y_pred, df, sensitive_columns):
        """DP gap per attribute. Fair < 5%, Mild < 10%, Significant >= 10%"""
        y_pred = np.asarray(y_pred)
        df = df.reset_index(drop=True)
        res = {'metric_name': 'Demographic Parity',
               'definition': 'Equal positive prediction rates across groups',
               'dp_gap': 0, 'verdict': 'Unknown',
               'group_rates': {}, 'per_attribute': {}}
        for col in sensitive_columns:
            rates = self._group_rates(y_pred, df, col)
            res['group_rates'][col] = rates
            if len(rates) >= 2:
                gap = max(rates.values()) - min(rates.values())
                res['per_attribute'][col] = {
                    'dp_gap': gap,
                    'verdict': 'Fair' if gap < 0.05 else 'Mild' if gap < 0.10 else 'Significant'}
        if res['per_attribute']:
            res['dp_gap'] = max(v['dp_gap'] for v in res['per_attribute'].values())
            res['verdict'] = self._worst([v['verdict'] for v in res['per_attribute'].values()])
        return res

    def _disparate_impact(self, y_pred, df, sensitive_columns):
        """DI per attribute (four-fifths rule). Fair >= 0.8, Mild >= 0.6"""
        y_pred = np.asarray(y_pred)
        df = df.reset_index(drop=True)
        res = {'metric_name': 'Disparate Impact (EEOC Four-Fifths Rule)',
               'definition': 'Ratio of positive prediction rates (EEOC threshold: 0.8)',
               'di_ratio': 1.0, 'verdict': 'Unknown',
               'group_rates': {}, 'per_attribute': {}}
        for col in sensitive_columns:
            rates = self._group_rates(y_pred, df, col)
            res['group_rates'][col] = rates
            if len(rates) >= 2:
                di = min(rates.values()) / (max(rates.values()) + 1e-10)
                res['per_attribute'][col] = {
                    'di_ratio': di,
                    'verdict': 'Fair' if di >= 0.8 else 'Mild' if di >= 0.6 else 'Significant'}
        if res['per_attribute']:
            res['di_ratio'] = min(v['di_ratio'] for v in res['per_attribute'].values())
            res['verdict'] = self._worst([v['verdict'] for v in res['per_attribute'].values()])
        return res

    def _equalized_odds(self, y_pred, y_test, df, sensitive_columns):
        """TPR/FPR gaps per attribute (Hardt et al. 2016)"""
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_test)
        df = df.reset_index(drop=True)
        res = {'metric_name': 'Equalized Odds',
               'definition': 'Equal True Positive Rates and False Positive Rates across groups',
               'tpr_gap': 0, 'fpr_gap': 0, 'verdict': 'Unknown',
               'group_metrics': {}, 'per_attribute': {}}
        for col in sensitive_columns:
            groups, tprs, fprs = {}, [], []
            for val, g in df.groupby(col):
                if len(g) < self.MIN_GROUP:
                    continue
                yp, yt = y_pred[g.index], y_true[g.index]
                pos, neg = (yt == 1).sum(), (yt == 0).sum()
                tpr = float(((yp == 1) & (yt == 1)).sum() / pos) if pos else None
                fpr = float(((yp == 1) & (yt == 0)).sum() / neg) if neg else None
                groups[str(self._decode(col, val))] = {'tpr': tpr, 'fpr': fpr}
                if tpr is not None: tprs.append(tpr)
                if fpr is not None: fprs.append(fpr)
            res['group_metrics'][col] = groups
            if len(tprs) >= 2 and len(fprs) >= 2:
                tg, fg = max(tprs) - min(tprs), max(fprs) - min(fprs)
                m = max(tg, fg)
                res['per_attribute'][col] = {
                    'tpr_gap': tg, 'fpr_gap': fg,
                    'verdict': 'Fair' if m < 0.05 else 'Mild' if m < 0.10 else 'Significant'}
        if res['per_attribute']:
            res['tpr_gap'] = max(v['tpr_gap'] for v in res['per_attribute'].values())
            res['fpr_gap'] = max(v['fpr_gap'] for v in res['per_attribute'].values())
            res['verdict'] = self._worst([v['verdict'] for v in res['per_attribute'].values()])
        return res
