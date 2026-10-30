import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

sns.set_style("whitegrid")

MIN_GROUP = 20      # ignore tiny groups (unreliable rates)
FIGSIZE = (6, 3.5)


class CompactVisualizer:
    """Creates compact fairness metric visualizations.

    Uses matplotlib's Figure directly (not pyplot), so figures are never
    leaked in memory and Streamlit reruns stay thread-safe.
    """

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _new_fig():
        fig = Figure(figsize=FIGSIZE)
        return fig, fig.subplots()

    @staticmethod
    def _grouping(df, col):
        """Series used for grouping; bins continuous columns (e.g. age) into quartiles"""
        s = df[col].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 8:
            s = pd.qcut(s, 4, duplicates='drop').astype(str)
        return s.astype(str)

    def _groups(self, df, col):
        """Yield (label, positional_index_array) for each group with enough samples"""
        s = self._grouping(df, col)
        for val, idx in s.groupby(s).indices.items():
            if len(idx) >= MIN_GROUP:
                yield val, idx

    # ------------------------------------------------------------------ public
    def show_all_plots_compact(self, y_pred, y_true, df, sensitive_columns,
                               y_proba=None, feature_importance=None):
        """
        y_pred / y_true / y_proba must be for the TEST rows only, and `df`
        must be the matching test rows (reset_index(drop=True)).
        Returns dict of matplotlib Figures.
        """
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        df = df.reset_index(drop=True)

        candidates = {
            'demographic_parity': lambda: self._plot_demographic_parity(y_pred, df, sensitive_columns),
            'disparate_impact': lambda: self._plot_disparate_impact(y_pred, df, sensitive_columns),
            'equalized_odds': lambda: self._plot_equalized_odds(y_pred, y_true, df, sensitive_columns),
            'confusion_matrix': lambda: self._plot_confusion_matrix(y_pred, y_true),
            'feature_importance': lambda: self._plot_feature_importance(feature_importance, sensitive_columns),
            'prediction_distribution': lambda: self._plot_prediction_distribution(y_proba, y_true),
        }
        plots = {}
        for name, fn in candidates.items():
            fig = fn()
            if fig is not None:
                plots[name] = fig
        return plots

    # ------------------------------------------------------------------ plots
    def _plot_demographic_parity(self, y_pred, df, sensitive_columns):
        try:
            fig, ax = self._new_fig()
            overall = float(y_pred.mean())
            labels, rates, colors = [], [], []

            for col in sensitive_columns:
                for val, idx in self._groups(df, col):
                    rate = float(y_pred[idx].mean())
                    labels.append(f"{col}: {val}")
                    rates.append(rate)
                    dev = abs(rate - overall)          # colour = distance from overall rate
                    colors.append('#2ca02c' if dev < 0.05 else '#ff7f0e' if dev < 0.10 else '#d62728')

            if not rates:
                return None
            ax.barh(labels, rates, color=colors, alpha=0.8, edgecolor='black')
            ax.axvline(overall, color='blue', linestyle='--', linewidth=2,
                       label=f'Overall rate ({overall:.2f})')
            ax.set_xlabel('Positive Prediction Rate', fontsize=10, fontweight='bold')
            ax.set_title('Demographic Parity\n(green <5 pts, orange <10 pts, red ≥10 pts from overall)',
                         fontsize=10, fontweight='bold')
            ax.set_xlim(0, max(0.05, max(rates) * 1.2))
            ax.legend(fontsize=8)
            fig.tight_layout()
            return fig
        except Exception as e:
            print(f"Error plotting DP: {e}")
            return None

    def _plot_disparate_impact(self, y_pred, df, sensitive_columns):
        try:
            fig, ax = self._new_fig()
            labels, ratios = [], []
            for col in sensitive_columns:
                rates = [float(y_pred[idx].mean()) for _, idx in self._groups(df, col)]
                if len(rates) >= 2:
                    labels.append(col)
                    ratios.append(min(rates) / (max(rates) + 1e-10))
            if not ratios:
                return None

            colors = ['#2ca02c' if r >= 0.8 else '#ff7f0e' if r >= 0.6 else '#d62728' for r in ratios]
            bars = ax.bar(labels, ratios, color=colors, alpha=0.8, edgecolor='black')
            for b, r in zip(bars, ratios):
                ax.text(b.get_x() + b.get_width() / 2, min(r, 1) + 0.02, f"{r:.2f}",
                        ha='center', fontsize=9, fontweight='bold')
            ax.axhline(0.8, color='green', linestyle='--', linewidth=2, label='Legal threshold (0.8)')
            ax.axhline(0.6, color='orange', linestyle='--', linewidth=1.5, label='Significant below (0.6)')
            ax.set_ylabel('Disparate Impact Ratio', fontsize=10, fontweight='bold')
            ax.set_title('Disparate Impact (EEOC four-fifths rule)', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.15)
            ax.legend(fontsize=8)
            fig.tight_layout()
            return fig
        except Exception as e:
            print(f"Error plotting DI: {e}")
            return None

    def _plot_equalized_odds(self, y_pred, y_true, df, sensitive_columns):
        try:
            fig, ax = self._new_fig()
            labels, tprs, fprs = [], [], []
            for col in sensitive_columns:
                for val, idx in self._groups(df, col):
                    yp, yt = y_pred[idx], y_true[idx]
                    pos, neg = (yt == 1).sum(), (yt == 0).sum()
                    tprs.append(((yp == 1) & (yt == 1)).sum() / pos if pos else 0)
                    fprs.append(((yp == 1) & (yt == 0)).sum() / neg if neg else 0)
                    labels.append(f"{col}: {val}")
            if not labels:
                return None

            x = np.arange(len(labels))
            w = 0.35
            ax.bar(x - w / 2, tprs, w, label='TPR', alpha=0.8, color='#1f77b4')
            ax.bar(x + w / 2, fprs, w, label='FPR', alpha=0.8, color='#ff7f0e')
            ax.set_ylabel('Rate', fontsize=10, fontweight='bold')
            ax.set_title('Equalized Odds (TPR vs FPR)\nbars should match across groups',
                         fontsize=10, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            fig.tight_layout()
            return fig
        except Exception as e:
            print(f"Error plotting EO: {e}")
            return None

    def _plot_confusion_matrix(self, y_pred, y_true):
        try:
            fig, ax = self._new_fig()
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])   # always 2x2
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                        xticklabels=['Negative', 'Positive'],
                        yticklabels=['Negative', 'Positive'])
            ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
            ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
            fig.tight_layout()
            return fig
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")
            return None

    def _plot_feature_importance(self, feature_importance, sensitive_columns):
        """Real importances from the trained model (dict: feature -> importance)"""
        try:
            if not feature_importance:
                return None
            top = sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)[:10][::-1]
            names = [k for k, _ in top]
            vals = [v for _, v in top]
            colors = ['#d62728' if n in sensitive_columns else '#1f77b4' for n in names]

            fig, ax = self._new_fig()
            ax.barh(names, vals, color=colors, alpha=0.8, edgecolor='black')
            ax.set_xlabel('Feature Importance', fontsize=10, fontweight='bold')
            ax.set_title('Top Feature Importances\n(Red = sensitive attribute used by the model)',
                         fontsize=10, fontweight='bold')
            fig.tight_layout()
            return fig
        except Exception as e:
            print(f"Error plotting feature importance: {e}")
            return None

    def _plot_prediction_distribution(self, y_proba, y_true):
        """Predicted probability of the positive class, split by true label"""
        try:
            if y_proba is None or len(y_proba) == 0:
                return None
            y_proba = np.asarray(y_proba)
            fig, ax = self._new_fig()
            bins = np.linspace(0, 1, 21)
            ax.hist(y_proba[y_true == 0], bins=bins, alpha=0.7, color='#1f77b4', label='True negative class')
            ax.hist(y_proba[y_true == 1], bins=bins, alpha=0.7, color='#ff7f0e', label='True positive class')
            ax.axvline(0.5, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('Predicted probability of positive class', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_title('Prediction Distribution by True Label', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            fig.tight_layout()
            return fig
        except Exception as e:
            print(f"Error plotting prediction distribution: {e}")
            return None
