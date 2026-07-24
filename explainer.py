import pandas as pd

def explain_demographic_parity(dp_rates):
    lines = ["**Demographic Parity**"]
    lines.append("This measures whether different groups receive positive outcomes at equal rates.\n")
    for group, rate in dp_rates.items():
        lines.append(f"- Group {group}: {rate:.2%} positive prediction rate")
    values = list(dp_rates.values())
    diff = max(values) - min(values)
    if diff < 0.05:
        lines.append("\n✅ **Fair** — prediction rates are close across groups.")
    elif diff < 0.10:
        lines.append("\n⚠️ **Mild bias detected** — there is a moderate gap between groups.")
    else:
        lines.append("\n❌ **Significant bias detected** — large gap in prediction rates across groups.")
    return "\n".join(lines)


def explain_disparate_impact(di_score):
    lines = ["**Disparate Impact**"]
    lines.append("This compares the positive prediction rate of the least favoured group to the most favoured group.")
    lines.append(f"\nScore: **{di_score:.2f}**")
    if di_score >= 0.8:
        lines.append("✅ **Fair** — score is above 0.8 (the standard legal threshold).")
    elif di_score >= 0.6:
        lines.append("⚠️ **Mild bias** — score is below 0.8, indicating some disparity.")
    else:
        lines.append("❌ **Significant bias** — score is well below 0.8, indicating serious disparity.")
    return "\n".join(lines)


def explain_equalized_odds(tpr, fpr):
    lines = ["**Equalized Odds**"]
    lines.append("This checks whether the model makes equally accurate predictions across groups.\n")
    lines.append("*True Positive Rate (TPR)* — how often the model correctly predicts a positive outcome:")
    for group, rate in tpr.items():
        lines.append(f"- Group {group}: {rate:.2%}")
    lines.append("\n*False Positive Rate (FPR)* — how often the model incorrectly predicts a positive outcome:")
    for group, rate in fpr.items():
        lines.append(f"- Group {group}: {rate:.2%}")
    tpr_diff = max(tpr.values()) - min(tpr.values())
    fpr_diff = max(fpr.values()) - min(fpr.values())
    if tpr_diff < 0.05 and fpr_diff < 0.05:
        lines.append("\n✅ **Fair** — TPR and FPR are consistent across groups.")
    elif tpr_diff < 0.10 or fpr_diff < 0.10:
        lines.append("\n⚠️ **Mild bias** — some difference in error rates across groups.")
    else:
        lines.append("\n❌ **Significant bias** — error rates differ substantially across groups.")
    return "\n".join(lines)


def generate_full_explanation(bias_results):
    dp = explain_demographic_parity(bias_results["demographic_parity"])
    di = explain_disparate_impact(bias_results["disparate_impact"])
    tpr = bias_results["equalized_odds"]["tpr"]
    fpr = bias_results["equalized_odds"]["fpr"]
    eo = explain_equalized_odds(tpr, fpr)
    return f"{dp}\n\n---\n\n{di}\n\n---\n\n{eo}"