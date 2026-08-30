def get_verdict(value, metric_type='dp'):
    if metric_type == 'dp':
        # Demographic Parity gap
        if value < 0.05:
            return '✅ FAIR', 'green'
        elif value < 0.10:
            return '⚠️ MILD BIAS', 'orange'
        else:
            return '❌ SIGNIFICANT BIAS', 'red'
    
    elif metric_type == 'di':
        # Disparate Impact ratio
        if value >= 0.8:
            return '✅ FAIR (≥0.8)', 'green'
        elif value >= 0.6:
            return '⚠️ MILD BIAS (0.6–0.8)', 'orange'
        else:
            return '❌ SIGNIFICANT BIAS (<0.6)', 'red'
    
    elif metric_type == 'eo':
        # Equalized Odds gap
        if value < 0.05:
            return '✅ FAIR', 'green'
        elif value < 0.10:
            return '⚠️ MILD BIAS', 'orange'
        else:
            return '❌ SIGNIFICANT BIAS', 'red'


def explain_demographic_parity(rates_dict, sensitive_col):

    rates = {k: v for k, v in rates_dict.items() if not (isinstance(v, float) and v != v)}  # Remove NaN
    
    if not rates:
        return "No valid data to explain."
    
    max_group = max(rates, key=rates.get)
    min_group = min(rates, key=rates.get)
    max_rate = rates[max_group]
    min_rate = rates[min_group]
    gap = max_rate - min_rate
    
    explanation = f"""
### Demographic Parity: "{sensitive_col}"

**What it means:** Are all groups receiving favorable outcomes at similar rates?

**Your data:**
- **Best-performing group:** {max_group} gets favorable outcome at {max_rate:.1%}
- **Worst-performing group:** {min_group} gets favorable outcome at {min_rate:.1%}
- **Gap:** {gap:.1%}

**Plain English:** 
If the model is perfectly fair on this metric, both groups should get positive predictions at roughly the same rate.
The gap of {gap:.1%} means the model favors {max_group} candidates significantly more than {min_group} candidates.

**Verdict:** {get_verdict(gap, 'dp')[0]}
"""
    return explanation


def explain_disparate_impact(di_ratio, sensitive_col):
    """
    Plain-English explanation of Disparate Impact
    """
    verdict, color = get_verdict(di_ratio, 'di')
    
    explanation = f"""
### Disparate Impact: "{sensitive_col}"

**What it means:** Does the model comply with the EEOC "four-fifths rule"?

**Your ratio:** {di_ratio:.3f}

**Plain English:**
The U.S. Equal Employment Opportunity Commission uses a 0.8 ratio as a legal threshold in employment cases.
If the worst-off group gets favorable outcomes at 80% or higher the rate of the best-off group, the disparity is legally considered acceptable.

Your ratio of {di_ratio:.3f} means:
- If the best group gets 100 positive predictions, the worst group gets only {int(di_ratio * 100)} predictions
- This disparate treatment is **legally risky** in hiring/credit/lending contexts

**Verdict:** {verdict}
"""
    return explanation


def explain_equalized_odds(tpr_dict, fpr_dict, sensitive_col):
    """
    Plain-English explanation of Equalized Odds
    """
    tpr = {k: v for k, v in tpr_dict.items() if not (isinstance(v, float) and v != v)}
    fpr = {k: v for k, v in fpr_dict.items() if not (isinstance(v, float) and v != v)}
    
    if not tpr or not fpr:
        return "No valid data to explain."
    
    tpr_gap = max(tpr.values()) - min(tpr.values())
    fpr_gap = max(fpr.values()) - min(fpr.values())
    max_gap = max(tpr_gap, fpr_gap)
    
    explanation = f"""
### Equalized Odds: "{sensitive_col}"

**What it means:** Does the model make equally accurate predictions for all groups?

**Your data:**
- **TPR Gap (True Positive Rate):** {tpr_gap:.1%} — Difference in correct positive predictions across groups
- **FPR Gap (False Positive Rate):** {fpr_gap:.1%} — Difference in incorrect positive predictions across groups

**Details by group:**
- **True Positive Rates (correctly identifying eligible candidates):**
{chr(10).join([f"  - {g}: {v:.1%}" for g, v in tpr.items()])}

- **False Positive Rates (incorrectly giving positive to ineligible candidates):**
{chr(10).join([f"  - {g}: {v:.1%}" for g, v in fpr.items()])}

**Plain English:**
Equalized Odds asks: "Does the model make equally good *and equally bad* mistakes across groups?"

A gap of {tpr_gap:.1%} in TPR means one group is less likely to be correctly identified as eligible.
A gap of {fpr_gap:.1%} in FPR means one group is more likely to receive a false positive.

**Verdict:** {get_verdict(max_gap, 'eo')[0]}
"""
    return explanation

def explain_intersectional_demographic_parity(rates_dict, attr1, attr2):
    """
    Explain Demographic Parity at intersection of two attributes
    """
    rates = {k: v for k, v in rates_dict.items() if not (isinstance(v, float) and v != v)}
    
    if not rates:
        return "No valid intersectional data."
    
    # Find best and worst
    best_group = max(rates, key=rates.get)
    worst_group = min(rates, key=rates.get)
    gap = rates[best_group] - rates[worst_group]
    
    explanation = f"""
### Intersectional Demographic Parity: "{attr1}" × "{attr2}"

**What it means:** Are all combinations of {attr1} and {attr2} receiving favorable outcomes equally?

**Key findings:**
- **Best-performing intersection:** {best_group} → {rates[best_group]:.1%} positive predictions
- **Worst-performing intersection:** {worst_group} → {rates[worst_group]:.1%} positive predictions
- **Intersectional gap:** {gap:.1%}

**Why this matters:**
Checking {attr1} and {attr2} separately might show fairness, but their *intersection* could reveal compound discrimination.
For example: Women of color might face worse bias than women alone or minorities alone.

**All intersections:**
{chr(10).join([f"- {group}: {rate:.1%}" for group, rate in sorted(rates.items(), key=lambda x: x[1], reverse=True)])}

**Verdict:** {get_verdict(gap, 'dp')[0]}
"""
    return explanation


def explain_intersectional_disparate_impact(di_ratio, attr1, attr2):
    """
    Explain Disparate Impact for intersectional groups
    """
    verdict, color = get_verdict(di_ratio, 'di')
    
    explanation = f"""
### Intersectional Disparate Impact: "{attr1}" × "{attr2}"

**What it means:** Do intersectional groups comply with the EEOC four-fifths rule?

**Your intersectional DI ratio:** {di_ratio:.3f}

**Plain English:**
This ratio compares the *worst-off intersectional group* to the *best-off intersectional group*.
If {attr1} and {attr2} compound discrimination (they make each other worse), this ratio will be lower than single-attribute DI.

**Verdict:** {verdict}

**Implication:** 
Intersectional bias is often invisible in single-attribute audits. If your single-attribute DI looks fair but 
this intersectional DI is low, it means one intersection is being systematically disadvantaged.
"""
    return explanation


def explain_worst_group_accuracy(wga_dict):
    """
    Explain worst-group accuracy findings
    """
    worst_acc = wga_dict['worst_accuracy']
    accuracy_gap = wga_dict['accuracy_gap']
    worst_cluster = wga_dict['worst_cluster']
    group_sizes = wga_dict['group_sizes']
    
    explanation = f"""
### Fairness Without Protected Attributes: Worst-Group Accuracy

**What it means:** The model's accuracy varies dramatically across *unsupervised* subgroups—even without knowing demographics.

**Your findings:**
- **Worst-performing cluster:** Cluster {worst_cluster} with accuracy {worst_acc:.1%} (n={group_sizes[worst_cluster]} samples)
- **Accuracy gap:** {accuracy_gap:.1%} between best and worst clusters
- **Overall pattern:** Model is significantly less accurate for some unseen subgroup

**Why this matters:**
You don't need explicit demographic labels to detect unfairness. If the model performs poorly on an unsupervised cluster,
that cluster likely contains individuals from a disadvantaged group—even if you don't know *which* group.

**Practical takeaway:**
Fix the worst-performing cluster's accuracy. This invisible subgroup deserves equal model performance.

**Verdict:** ❌ SIGNIFICANT ACCURACY DISPARITY
"""
    return explanation


def explain_adversarial_bias(adv_dict):
    """
    Explain adversarial bias detection
    """
    adv_acc = adv_dict['mean_adversary_accuracy']
    bias_level = adv_dict['bias_level']
    
    interpretation_map = {
        'low': "Predictions are **independent** of feature patterns—minimal bias",
        'medium': "Predictions have **moderate correlation** with feature patterns",
        'high': "Predictions are **strongly driven** by feature patterns—high risk of bias"
    }
    
    explanation = f"""
### Fairness Without Protected Attributes: Adversarial Bias Detection

**What it means:** Can we predict the model's output just from its input features?

**Your adversarial accuracy:** {adv_acc:.1%}

**Plain English:**
We trained an adversary that tries to predict your model's decisions from the input features alone.
If it succeeds (>60% accuracy), your model's predictions are correlated with feature patterns in a way that suggests bias.

**Your result:**
- Adversary accuracy: {adv_acc:.1%}
- Bias level: **{bias_level.upper()}**
- Interpretation: {interpretation_map[bias_level]}

**Random baseline:** An adversary guessing randomly achieves 50% accuracy (binary classification).
Your adversary achieved {adv_acc:.1%}, meaning predictions are {'' if adv_acc < 0.6 else 'significantly '} feature-dependent.

**Verdict:** {"✅ LOW BIAS RISK" if adv_acc < 0.6 else "⚠️ MODERATE BIAS RISK" if adv_acc < 0.7 else "❌ HIGH BIAS RISK"}
"""
    return explanation


def explain_proxy_warnings(proxy_warnings):
    """
    Explain proxy feature warnings
    """
    if not proxy_warnings:
        return "✅ No significant proxy features detected. Non-sensitive features do not strongly correlate with protected attributes."
    
    top_warnings = sorted(proxy_warnings, key=lambda x: x['correlation'], reverse=True)[:5]
    
    explanation = f"""
### Proxy Feature Warnings: Hidden Discrimination

**What it means:** Some non-protected features correlate strongly with protected attributes. 
Using these features allows discrimination "under the radar."

**Examples from your data:**
{chr(10).join([f"- **{w['proxy_feature']}** → **{w['sensitive_attr']}** (correlation: {w['correlation']:.3f})" 
               for w in top_warnings])}

**Plain English:**
Even if you remove the sensitive attribute (e.g., "gender"), related features (e.g., "name gender", "clothing style") 
can indirectly re-introduce bias. These are called **proxy features**.

**Why it matters:**
- **ZIP code** often proxies for race (segregated neighborhoods)
- **First name** proxies for ethnicity
- **School type** proxies for socioeconomic status and race
- **Age-related features** can proxy for protected classes

**Recommendation:**
Review features with high correlations. Either:
1. Remove or anonymize them if they're not essential
2. Audit the model's performance on subgroups defined by these proxies
3. Apply fairness constraints to reduce their influence

**Verdict:** ⚠️ AUDIT REQUIRED
"""
    return explanation


def generate_full_explanation(bias_results, sensitive_col='all', include_intersectional=True, 
                             include_fairness_without_demographics=True):
    """
    Generate comprehensive explanation of all findings
    """
    all_explanations = []
    
    # Standard metrics
    if sensitive_col == 'all':
        columns_to_explain = list(bias_results['sensitive_test'].keys())
    else:
        columns_to_explain = [sensitive_col]
    
    for col in columns_to_explain:
        metrics = bias_results['metrics'][col]
        
        all_explanations.append(explain_demographic_parity(metrics['demographic_parity'], col))
        all_explanations.append(explain_disparate_impact(metrics['disparate_impact'], col))
        all_explanations.append(explain_equalized_odds(metrics['equalized_odds']['tpr'],
                                                       metrics['equalized_odds']['fpr'], col))
    
    # Intersectional explanations
    if include_intersectional and 'intersectional_metrics' in bias_results:
        for intersection_key, intersection_metrics in bias_results['intersectional_metrics'].items():
            attr1, attr2 = intersection_key.split(' × ')
            all_explanations.append(explain_intersectional_demographic_parity(
                intersection_metrics['demographic_parity'], attr1, attr2))
            all_explanations.append(explain_intersectional_disparate_impact(
                intersection_metrics['disparate_impact'], attr1, attr2))
    
    # Fairness without demographics explanations
    if include_fairness_without_demographics and 'fairness_without_demographics' in bias_results:
        fwd = bias_results['fairness_without_demographics']
        all_explanations.append(explain_worst_group_accuracy(fwd['worst_group_accuracy']))
        all_explanations.append(explain_adversarial_bias(fwd['adversarial_bias']))
        if fwd['proxy_warnings']:
            all_explanations.append(explain_proxy_warnings(fwd['proxy_warnings']))
    
    return all_explanations
