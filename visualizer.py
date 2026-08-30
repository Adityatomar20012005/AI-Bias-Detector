import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'sans-serif'


def plot_demographic_parity(rates_dict, title="Demographic Parity by Group"):

    fig, ax = plt.subplots(figsize=(10, 5))
    
    groups = list(rates_dict.keys())
    values = list(rates_dict.values())
    
    # Filter out NaN
    valid_data = [(g, v) for g, v in zip(groups, values) if not np.isnan(v)]
    if not valid_data:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
        return fig
    
    groups, values = zip(*valid_data)
    mean_val = np.mean(values)
    
    colors = ['#d62728' if abs(v - mean_val) > 0.1 else '#2ca02c' if abs(v - mean_val) < 0.05 else '#ff7f0e' 
              for v in values]
    
    bars = ax.bar(groups, values, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Average: {mean_val:.2%}')
    
    ax.set_ylabel('Positive Prediction Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend()
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_disparate_impact(di_ratio, title="Disparate Impact Ratio"):
    """
    Gauge plot showing DI ratio vs. 0.8 threshold
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color based on threshold
    if di_ratio >= 0.8:
        color = '#2ca02c'  # Green - Fair
        verdict = 'FAIR (≥0.8)'
    elif di_ratio >= 0.6:
        color = '#ff7f0e'  # Orange - Mild bias
        verdict = 'MILD BIAS (0.6–0.8)'
    else:
        color = '#d62728'  # Red - Significant bias
        verdict = 'SIGNIFICANT BIAS (<0.6)'
    
    # Horizontal bar
    ax.barh(['Disparate Impact'], [di_ratio], height=0.3, color=color, edgecolor='black', linewidth=2)
    ax.axvline(0.8, color='black', linestyle='--', linewidth=2.5, label='EEOC 4/5 Threshold')
    
    ax.set_xlim(0, 1)
    ax.set_xlabel('Ratio (Min Rate / Max Rate)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Add annotation
    ax.text(di_ratio + 0.02, 0, f'{di_ratio:.3f}\n{verdict}', 
            va='center', fontweight='bold', fontsize=11,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    return fig


def plot_equalized_odds(tpr_dict, fpr_dict, title="Equalized Odds (TPR & FPR by Group)"):
    """
    Grouped bar chart showing TPR and FPR per group
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    groups = list(tpr_dict.keys())
    tpr_vals = list(tpr_dict.values())
    fpr_vals = list(fpr_dict.values())
    
    # Filter out NaN
    valid_idx = [i for i, (t, f) in enumerate(zip(tpr_vals, fpr_vals)) 
                 if not np.isnan(t) and not np.isnan(f)]
    if not valid_idx:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
        return fig
    
    groups = [groups[i] for i in valid_idx]
    tpr_vals = [tpr_vals[i] for i in valid_idx]
    fpr_vals = [fpr_vals[i] for i in valid_idx]
    
    x = np.arange(len(groups))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, tpr_vals, width, label='True Positive Rate (TPR)', 
                   color='#1f77b4', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, fpr_vals, width, label='False Positive Rate (FPR)', 
                   color='#ff7f0e', edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                        f'{height:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_intersectional_demographic_parity_heatmap(rates_dict, attr1, attr2, 
                                                   title="Intersectional Demographic Parity Heatmap"):
    """
    Heatmap of DP rates across intersectional groups
    E.g., rows=Gender, cols=Race
    """
    # Convert dict to DataFrame for heatmap
    data = {}
    for (a1, a2), rate in rates_dict.items():
        if a1 not in data:
            data[a1] = {}
        data[a1][a2] = rate
    
    df_heatmap = pd.DataFrame(data).T
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(df_heatmap, annot=True, fmt='.2%', cmap='RdYlGn', center=0.5,
                cbar_kws={'label': 'Positive Prediction Rate'}, ax=ax,
                linewidths=1, linecolor='gray')
    
    ax.set_xlabel(attr2, fontsize=12, fontweight='bold')
    ax.set_ylabel(attr1, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_intersectional_disparities(di_ratio, attr1, attr2, 
                                    title="Intersectional Disparate Impact"):
    """
    Show worst vs. best intersectional group disparity
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    color = '#2ca02c' if di_ratio >= 0.8 else '#ff7f0e' if di_ratio >= 0.6 else '#d62728'
    
    ax.barh(['Intersectional DI'], [di_ratio], height=0.3, color=color, edgecolor='black', linewidth=2)
    ax.axvline(0.8, color='black', linestyle='--', linewidth=2.5, label='EEOC 4/5 Threshold')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Min Rate / Max Rate (Across All Intersections)', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\n{attr1} × {attr2}', fontsize=14, fontweight='bold')
    ax.legend()
    
    ax.text(di_ratio + 0.02, 0, f'{di_ratio:.3f}', va='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_worst_group_accuracy(accuracy_dict, group_sizes, title="Accuracy by Cluster (Fairness Without Demographics)"):
    """
    Bar chart showing accuracy per cluster + highlight worst performer
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    clusters = sorted(accuracy_dict.keys())
    accuracies = [accuracy_dict[c] for c in clusters]
    sizes = [group_sizes[c] for c in clusters]
    
    worst_idx = np.argmin(accuracies)
    colors = ['#d62728' if i == worst_idx else '#2ca02c' for i in range(len(clusters))]
    
    bars = ax.bar(range(len(clusters)), accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add group size as text
    for i, (bar, size) in enumerate(zip(bars, sizes)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{accuracies[i]:.1%}\n(n={size})', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_xlabel('Cluster (Unsupervised Group)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([f'Cluster {c}' for c in clusters])
    ax.set_ylim(0, 1)
    ax.axhline(np.mean(accuracies), color='blue', linestyle='--', linewidth=2, 
               label=f'Mean Accuracy: {np.mean(accuracies):.1%}')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_adversarial_bias(adversary_acc, bias_level, title="Adversarial Bias Detection"):
    """
    Show adversary's ability to predict model predictions from features
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    color_map = {'low': '#2ca02c', 'medium': '#ff7f0e', 'high': '#d62728'}
    color = color_map.get(bias_level, '#1f77b4')
    
    ax.barh(['Adversary Accuracy'], [adversary_acc], height=0.3, color=color, edgecolor='black', linewidth=2)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=2, label='Random (0.5)')
    
    ax.set_xlim(0.4, 1)
    ax.set_xlabel('Adversary Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    
    interpretation = "Predictions independent of features" if adversary_acc < 0.6 else \
                     "Moderate correlation with features" if adversary_acc < 0.7 else \
                     "Strong correlation with features"
    
    ax.text(adversary_acc + 0.01, 0, f'{adversary_acc:.3f}\nBias: {bias_level.upper()}\n{interpretation}',
            va='center', fontweight='bold', fontsize=10,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    return fig


def plot_proxy_warnings(proxy_warnings, title="Proxy Feature Warnings"):
    """
    Show features that correlate with sensitive attributes (potential proxies)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if not proxy_warnings:
        ax.text(0.5, 0.5, 'No significant proxy features detected', ha='center', va='center', fontsize=14)
        plt.tight_layout()
        return fig
    
    warnings_df = pd.DataFrame(proxy_warnings).sort_values('correlation', ascending=False).head(10)
    
    y_pos = np.arange(len(warnings_df))
    colors = ['#d62728' if corr > 0.5 else '#ff7f0e' for corr in warnings_df['correlation']]
    
    bars = ax.barh(y_pos, warnings_df['correlation'], color=colors, edgecolor='black', linewidth=1.2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['proxy_feature']} → {row['sensitive_attr']}" 
                         for _, row in warnings_df.iterrows()])
    ax.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    
    # Add value labels
    for bar, corr in zip(bars, warnings_df['correlation']):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{corr:.3f}', va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    return fig


def show_all_plots(bias_results, show_intersectional=True, show_fairness_without_demographics=True):
    """
    Generate all visualizations
    """
    figs = []
    
    # Standard metrics
    for sens_col in bias_results['sensitive_test'].keys():
        metrics = bias_results['metrics'][sens_col]
        
        figs.append(('demographic_parity', 
                    plot_demographic_parity(metrics['demographic_parity'], 
                                           f"Demographic Parity - {sens_col}")))
        figs.append(('disparate_impact',
                    plot_disparate_impact(metrics['disparate_impact'],
                                         f"Disparate Impact - {sens_col}")))
        figs.append(('equalized_odds',
                    plot_equalized_odds(metrics['equalized_odds']['tpr'],
                                       metrics['equalized_odds']['fpr'],
                                       f"Equalized Odds - {sens_col}")))
    
    # Intersectional metrics
    if show_intersectional and 'intersectional_metrics' in bias_results:
        for intersection_key, intersection_metrics in bias_results['intersectional_metrics'].items():
            attr1, attr2 = intersection_key.split(' × ')
            
            figs.append(('intersectional_dp_heatmap',
                        plot_intersectional_demographic_parity_heatmap(
                            intersection_metrics['demographic_parity'], attr1, attr2,
                            f"Intersectional DP: {intersection_key}")))
            
            figs.append(('intersectional_di',
                        plot_intersectional_disparities(
                            intersection_metrics['disparate_impact'], attr1, attr2,
                            f"Intersectional DI: {intersection_key}")))
    
    # Fairness without demographics
    if show_fairness_without_demographics and 'fairness_without_demographics' in bias_results:
        fwd = bias_results['fairness_without_demographics']
        
        wga = fwd['worst_group_accuracy']
        figs.append(('worst_group_accuracy',
                    plot_worst_group_accuracy(wga['accuracies'], wga['group_sizes'])))
        
        adv = fwd['adversarial_bias']
        figs.append(('adversarial_bias',
                    plot_adversarial_bias(adv['mean_adversary_accuracy'], adv['bias_level'])))
        
        if fwd['proxy_warnings']:
            figs.append(('proxy_warnings',
                        plot_proxy_warnings(fwd['proxy_warnings'])))
    
    return figs
