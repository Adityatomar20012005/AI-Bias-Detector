import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (6, 3.5)


class CompactVisualizer:
    """Creates compact fairness metric visualizations"""
    
    def __init__(self):
        self.fig_count = 0
    
    def show_all_plots_compact(self, y_pred, y_true, df, sensitive_columns):
        """
        Generate all visualization plots in compact format
        
        Returns:
            Dictionary of matplotlib figures
        """
        plots = {}
        
        # Demographic Parity visualization
        fig_dp = self._plot_demographic_parity(y_pred, df, sensitive_columns)
        if fig_dp:
            plots['demographic_parity'] = fig_dp
        
        # Disparate Impact visualization
        fig_di = self._plot_disparate_impact(y_pred, df, sensitive_columns)
        if fig_di:
            plots['disparate_impact'] = fig_di
        
        # Equalized Odds visualization
        fig_eo = self._plot_equalized_odds(y_pred, y_true, df, sensitive_columns)
        if fig_eo:
            plots['equalized_odds'] = fig_eo
        
        # Confusion matrix
        fig_cm = self._plot_confusion_matrix(y_pred, y_true)
        if fig_cm:
            plots['confusion_matrix'] = fig_cm
        
        # Feature importance
        fig_fi = self._plot_feature_importance(df, sensitive_columns)
        if fig_fi:
            plots['feature_importance'] = fig_fi
        
        # Prediction distribution
        fig_dist = self._plot_prediction_distribution(y_pred, y_true)
        if fig_dist:
            plots['prediction_distribution'] = fig_dist
        
        return plots
    
    def _plot_demographic_parity(self, y_pred, df, sensitive_columns):
        """Plot positive prediction rates by sensitive attribute"""
        try:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            
            colors = []
            labels = []
            rates = []
            
            for col in sensitive_columns:
                unique_vals = df[col].unique()
                
                for val in unique_vals:
                    mask = df[col] == val
                    if mask.sum() > 0:
                        rate = y_pred[mask].mean()
                        rates.append(rate)
                        labels.append(f"{col}:\n{str(val)}")
                        # Color based on rate
                        if rate > 0.3:
                            colors.append('#d62728')
                        elif rate > 0.15:
                            colors.append('#ff7f0e')
                        else:
                            colors.append('#2ca02c')
            
            if rates:
                bars = ax.barh(labels, rates, color=colors, alpha=0.7, edgecolor='black')
                ax.axvline(np.mean(rates), color='blue', linestyle='--', linewidth=2, label='Mean')
                ax.set_xlabel('Positive Prediction Rate', fontsize=10, fontweight='bold')
                ax.set_title('Demographic Parity\n(DP)', fontsize=12, fontweight='bold')
                ax.set_xlim(0, 1)
                ax.legend(fontsize=8)
                
                plt.tight_layout()
                return fig
        except Exception as e:
            print(f"Error plotting DP: {e}")
            return None
    
    def _plot_disparate_impact(self, y_pred, df, sensitive_columns):
        """Plot disparate impact (DI) by sensitive attribute"""
        try:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            
            di_ratios = []
            labels = []
            
            for col in sensitive_columns:
                unique_vals = df[col].unique()
                rates = []
                
                for val in unique_vals:
                    mask = df[col] == val
                    if mask.sum() > 0:
                        rate = y_pred[mask].mean()
                        rates.append(rate)
                
                if rates:
                    di = min(rates) / (max(rates) + 1e-10)
                    di_ratios.append(di)
                    labels.append(col)
            
            if di_ratios:
                colors = ['#2ca02c' if di >= 0.8 else '#ff7f0e' if di >= 0.6 else '#d62728' 
                         for di in di_ratios]
                bars = ax.bar(labels, di_ratios, color=colors, alpha=0.7, edgecolor='black')
                ax.axhline(0.8, color='green', linestyle='--', linewidth=2, label='Legal Threshold (0.8)')
                ax.axhline(0.6, color='orange', linestyle='--', linewidth=1.5, label='Mild Bias (0.6)')
                ax.set_ylabel('Disparate Impact Ratio', fontsize=10, fontweight='bold')
                ax.set_title('Disparate Impact (EEOC)\n(DI)', fontsize=12, fontweight='bold')
                ax.set_ylim(0, 1)
                ax.legend(fontsize=8)
                
                plt.tight_layout()
                return fig
        except Exception as e:
            print(f"Error plotting DI: {e}")
            return None
    
    def _plot_equalized_odds(self, y_pred, y_true, df, sensitive_columns):
        """Plot equalized odds (TPR and FPR) by sensitive attribute"""
        try:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            
            tpr_values = []
            fpr_values = []
            labels = []
            
            for col in sensitive_columns:
                unique_vals = df[col].unique()
                
                for val in unique_vals:
                    mask = df[col] == val
                    if mask.sum() > 0:
                        y_pred_group = y_pred[mask]
                        y_true_group = y_true.iloc[mask] if hasattr(y_true, 'iloc') else y_true[mask]
                        
                        # TPR
                        if (y_true_group == 1).sum() > 0:
                            tp = ((y_pred_group == 1) & (y_true_group == 1)).sum()
                            tpr = tp / ((y_true_group == 1).sum())
                        else:
                            tpr = 0
                        
                        # FPR
                        if (y_true_group == 0).sum() > 0:
                            fp = ((y_pred_group == 1) & (y_true_group == 0)).sum()
                            fpr = fp / ((y_true_group == 0).sum())
                        else:
                            fpr = 0
                        
                        tpr_values.append(tpr)
                        fpr_values.append(fpr)
                        labels.append(f"{col}:{val}")
            
            if tpr_values and fpr_values:
                x = np.arange(len(labels))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, tpr_values, width, label='TPR', alpha=0.7, color='#1f77b4')
                bars2 = ax.bar(x + width/2, fpr_values, width, label='FPR', alpha=0.7, color='#ff7f0e')
                
                ax.set_ylabel('Rate', fontsize=10, fontweight='bold')
                ax.set_title('Equalized Odds\n(TPR vs FPR)', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                ax.set_ylim(0, 1)
                ax.legend(fontsize=8)
                
                plt.tight_layout()
                return fig
        except Exception as e:
            print(f"Error plotting EO: {e}")
            return None
    
    def _plot_confusion_matrix(self, y_pred, y_true):
        """Plot confusion matrix"""
        try:
            from sklearn.metrics import confusion_matrix
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            
            ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
            ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")
            return None
    
    def _plot_feature_importance(self, df, sensitive_columns):
        """Plot top feature importance"""
        try:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            
            # Create dummy feature importance (in real scenario, from trained model)
            feature_names = [col for col in df.columns if col not in sensitive_columns][:10]
            importances = np.random.rand(len(feature_names))
            importances = importances / importances.sum()
            
            # Highlight sensitive columns
            colors = ['#d62728' if name in sensitive_columns else '#1f77b4' 
                     for name in feature_names]
            
            bars = ax.barh(feature_names, importances, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Feature Importance', fontsize=10, fontweight='bold')
            ax.set_title('Feature Importance\n(Red = Sensitive)', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error plotting feature importance: {e}")
            return None
    
    def _plot_prediction_distribution(self, y_pred, y_true):
        """Plot prediction distribution by true label"""
        try:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            
            y_true_array = y_true.values if hasattr(y_true, 'values') else y_true
            
            ax.hist([y_pred[y_true_array == 0], y_pred[y_true_array == 1]],
                   label=['True Negative', 'True Positive'],
                   bins=20, alpha=0.7, color=['#1f77b4', '#ff7f0e'])
            
            ax.set_xlabel('Prediction Value', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_title('Prediction Distribution\nby True Label', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error plotting prediction distribution: {e}")
            return None
