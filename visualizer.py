import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

def plot_demographic_parity(dp_rates):
    groups = [str(g) for g in dp_rates.keys()]
    rates = [r * 100 for r in dp_rates.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(groups, rates, color=sns.color_palette("Set2", len(groups)))
    ax.set_title("Demographic Parity — Positive Prediction Rate by Group")
    ax.set_xlabel("Group")
    ax.set_ylabel("Positive Prediction Rate (%)")
    ax.axhline(y=sum(rates)/len(rates), color='red', linestyle='--', label='Average')
    ax.legend()
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{rate:.1f}%", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    return fig


def plot_disparate_impact(di_score):
    fig, ax = plt.subplots(figsize=(6, 4))
    color = "green" if di_score >= 0.8 else "orange" if di_score >= 0.6 else "red"
    ax.barh(["Disparate Impact"], [di_score], color=color)
    ax.axvline(x=0.8, color='black', linestyle='--', label='Fair threshold (0.8)')
    ax.set_xlim(0, 1.2)
    ax.set_title("Disparate Impact Score")
    ax.set_xlabel("Score")
    ax.legend()
    ax.text(di_score + 0.02, 0, f"{di_score:.2f}", va='center', fontsize=12)
    plt.tight_layout()
    return fig


def plot_equalized_odds(tpr, fpr):
    groups = [str(g) for g in tpr.keys()]
    tpr_vals = [v * 100 for v in tpr.values()]
    fpr_vals = [v * 100 for v in fpr.values()]

    x = range(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width/2 for i in x], tpr_vals, width, label='TPR', color='steelblue')
    ax.bar([i + width/2 for i in x], fpr_vals, width, label='FPR', color='salmon')
    ax.set_title("Equalized Odds — TPR and FPR by Group")
    ax.set_xlabel("Group")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.legend()
    plt.tight_layout()
    return fig


def show_all_plots(bias_results, df=None, sensitive_col=None):
    st.subheader("Demographic Parity")
    fig1 = plot_demographic_parity(bias_results["demographic_parity"])
    st.pyplot(fig1)

    st.subheader("Disparate Impact")
    fig2 = plot_disparate_impact(bias_results["disparate_impact"])
    st.pyplot(fig2)

    st.subheader("Equalized Odds")
    tpr = bias_results["equalized_odds"]["tpr"]
    fpr = bias_results["equalized_odds"]["fpr"]
    fig3 = plot_equalized_odds(tpr, fpr)
    st.pyplot(fig3)