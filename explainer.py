"""
Fairness Metric Explainer
Provides plain-English interpretations of fairness metrics
with legal and regulatory context
"""


class FairnessExplainer:
    """Explains fairness metrics in non-technical language"""
    
    def explain_demographic_parity(self, dp_data):
        """Explain Demographic Parity"""
        gap = dp_data.get('dp_gap', 0)
        verdict = dp_data.get('verdict', 'Unknown')
        
        explanation = f"""
### What is Demographic Parity?

Demographic Parity means that a model should make positive predictions 
at roughly the same rate for all groups. In other words, if 30% of 
applicants are approved for a loan, it should be 30% for all demographic groups.

### What does your result mean?

**Gap: {gap*100:.2f}%**

This means there's a {gap*100:.2f}% difference in positive prediction rates 
between the most-favored and least-favored groups.

**Verdict: {verdict}**

"""
        
        if verdict == "Fair":
            explanation += """
✅ **GOOD NEWS:** Your model appears to satisfy demographic parity. 
Groups are receiving positive predictions at similar rates.

**Implication:** Your model is unlikely to face discrimination lawsuits 
based on demographic parity alone.
"""
        
        elif verdict == "Mild":
            explanation += """
⚠️ **CAUTION:** Your model shows moderate demographic parity concerns.

**Implication:** While not severe, this suggests some group is getting 
positive predictions significantly less often. Consider:
- Reviewing your training data for historical bias
- Testing mitigation strategies (reweighting)
- Consulting with legal and ethics teams
"""
        
        else:  # Significant
            explanation += """
❌ **SERIOUS ISSUE:** Your model violates demographic parity significantly.

**Implication:** One or more groups is receiving positive predictions 
at substantially lower rates. This could:
- Expose your organization to discrimination lawsuits
- Violate fair lending laws (FCRA)
- Violate employment discrimination laws (Title VII)
- Violate EU AI Act requirements (if deployed in EU)

**Recommended Actions:**
1. Investigate the root cause (data bias, feature engineering, model bias)
2. Test mitigation strategies immediately
3. Consult legal counsel before deployment
4. Consider alternative fairness metrics (Disparate Impact, Equalized Odds)
"""
        
        return explanation
    
    def explain_disparate_impact(self, di_data):
        """Explain Disparate Impact (EEOC Four-Fifths Rule)"""
        ratio = di_data.get('di_ratio', 0)
        verdict = di_data.get('verdict', 'Unknown')
        
        explanation = f"""
### What is Disparate Impact?

Disparate Impact (also called the "Four-Fifths Rule") is a legal standard 
used by the U.S. Equal Employment Opportunity Commission (EEOC).

The rule states: The selection rate for a protected group should be at 
least 80% (4/5) of the selection rate for the most-favored group.

**Formula:** DI = (Lowest group rate) / (Highest group rate)

- **Legal:** DI ≥ 0.80
- **Gray Zone:** DI 0.60-0.80
- **Illegal:** DI < 0.60

### What does your result mean?

**DI Ratio: {ratio:.3f}**

This means the least-favored group is receiving positive predictions at 
{ratio*100:.1f}% the rate of the most-favored group.

**Verdict: {verdict}**

"""
        
        if verdict == "Fair":
            explanation += """
✅ **LEGALLY SAFE:** Your model passes the EEOC four-fifths rule.

**Implication:** 
- Your organization can defend this model in hiring/lending lawsuits
- The model is likely compliant with U.S. employment discrimination law
- Still consider other fairness metrics for comprehensive assessment

**Next Step:** Check Demographic Parity and Equalized Odds metrics as well.
"""
        
        elif verdict == "Mild":
            explanation += """
⚠️ **LEGAL GRAY ZONE:** Your model is approaching the EEOC threshold.

**Implication:**
- This could be challenged in court but has some defense
- EEOC may investigate if formal complaint is filed
- Demonstrates some discriminatory impact

**Recommended Actions:**
1. Gather documentation showing business necessity
2. Show you've tested alternative hiring/lending standards
3. Test mitigation strategies to improve DI
4. Consult legal counsel
"""
        
        else:  # Significant
            explanation += """
❌ **LEGALLY RISKY:** Your model violates the EEOC four-fifths rule.

**Implication:**
- This is illegal under Title VII of the Civil Rights Act (employment)
- This violates the Fair Credit Reporting Act (FCRA) for lending
- EEOC can sue your organization
- Individual plaintiffs can file discrimination lawsuits
- Damages can include back pay, lost benefits, and punitive damages

**Mandatory Actions:**
1. DO NOT DEPLOY this model without mitigation
2. Consult legal counsel IMMEDIATELY
3. Test reweighting/suppression mitigation strategies
4. Consider alternative fairness metrics
5. Investigate the source of discrimination in your data/model
"""
        
        return explanation
    
    def explain_equalized_odds(self, eo_data):
        """Explain Equalized Odds"""
        tpr_gap = eo_data.get('tpr_gap', 0)
        fpr_gap = eo_data.get('fpr_gap', 0)
        verdict = eo_data.get('verdict', 'Unknown')
        
        explanation = f"""
### What is Equalized Odds?

Equalized Odds means that a model should have:
- **Equal True Positive Rates (TPR)** across groups: If someone truly 
  deserves a positive outcome, the model should recognize it at the same rate
- **Equal False Positive Rates (FPR)** across groups: If someone truly 
  doesn't deserve a positive outcome, the model should correctly reject them 
  at the same rate

This ensures the model makes errors equally across groups.

**Reference:** Hardt et al. 2016 (NeurIPS) - "Equality of Opportunity in Supervised Learning"

### What does your result mean?

**TPR Gap: {tpr_gap*100:.2f}%**
**FPR Gap: {fpr_gap*100:.2f}%**

- TPR Gap measures if some groups are wrongly rejected more often
- FPR Gap measures if some groups are wrongly approved more often

**Verdict: {verdict}**

"""
        
        if verdict == "Fair":
            explanation += """
✅ **EXCELLENT:** Your model satisfies equalized odds.

**Implication:**
- Your model treats all groups equally in terms of true positives and false positives
- Errors are distributed fairly across groups
- This is one of the strongest fairness guarantees

**Advantage:** Unlike Demographic Parity, this metric considers prediction accuracy
and doesn't require equal approval rates (just equal error rates).
"""
        
        elif verdict == "Mild":
            explanation += """
⚠️ **CAUTION:** Your model shows moderate equalized odds issues.

**Implication:**
- Errors are not distributed equally across groups
- Some group is being wrongly rejected OR wrongly approved more often
- This could violate equal protection principles

**Investigation needed:**
- Is TPR gap high? Some groups are being "false negatives" - rejected unfairly
- Is FPR gap high? Some groups are being "false positives" - approved unfairly
- What's causing these differences? Data bias? Feature engineering? Model choice?
"""
        
        else:  # Significant
            explanation += """
❌ **SERIOUS ISSUE:** Your model violates equalized odds significantly.

**Implication:**
- The model makes errors very differently across groups
- This is unfair and could face legal challenges
- May violate equal protection under the law
- Could expose organization to discrimination claims

**What's happening:**
- High TPR gap: Some groups are being rejected unfairly (higher false negatives)
- High FPR gap: Some groups are being approved unfairly (higher false positives)

**Recommended Actions:**
1. Analyze which group is disadvantaged
2. Investigate data collection and model training for bias
3. Test fairness-aware learning algorithms
4. Consider constraint-based optimization for equalized odds
"""
        
        return explanation
    
    def explain_intersectional_fairness(self, intersectional_data):
        """Explain intersectional fairness findings"""
        explanation = """
### What is Intersectional Fairness?

Intersectional fairness recognizes that people belong to multiple groups 
at once. For example, someone can be both a woman AND a racial minority. 
Bias can compound at these intersections.

**Example:** A model might be fair to women overall, fair to minorities 
overall, but severely biased against Black women specifically.

### Why does this matter?

Research shows that single-attribute fairness audits miss 40-70% of 
discrimination. Only by examining intersections can you find hidden bias.

### Your intersectional findings:

"""
        
        for pair, metrics in intersectional_data.items():
            di_ratio = metrics.get('di_ratio', 0)
            dp_gap = metrics.get('dp_gap', 0)
            verdict = metrics.get('verdict', 'Unknown')
            
            explanation += f"""
**{pair}:**
- DI Ratio: {di_ratio:.3f}
- DP Gap: {dp_gap*100:.2f}%
- Verdict: {verdict}

"""
        
        explanation += """
### What this means:

If you see "Significant" verdicts for intersectional groups, it means:
- Your model might be overall fair to single attributes
- But severely discriminates against people at the intersection
- This requires targeted mitigation

### Recommended Action:

If any intersection shows bias, consider:
1. Stratified analysis by intersection (not just overall)
2. Targeted mitigation for specific intersections
3. Broader data collection or balancing for disadvantaged intersections
"""
        
        return explanation
    
    def explain_fairness_without_demographics(self, fwd_data):
        """Explain fairness-without-demographics findings"""
        explanation = """
### What is Fairness Without Protected Attributes?

In some domains (healthcare, government), storing demographic data is:
- Illegal under GDPR/HIPAA
- Ethically problematic
- Risky for data breaches

Fairness-without-demographics uses unsupervised techniques to detect bias 
WITHOUT needing demographic labels.

### Three techniques we use:

**1. Worst-Group Accuracy**
- Clusters data into groups WITHOUT demographic labels
- Finds the group where model accuracy is lowest
- If overall accuracy is high but worst-group accuracy is low → hidden bias

**2. Adversarial Bias Detection**
- Trains an "adversary" model to predict model decisions from features alone
- If adversary can predict decisions well → decisions depend on feature patterns
- If adversary can't predict → decisions are more independent

**3. Proxy Attribute Warnings**
- Identifies features that correlate with protected attributes
- These "proxy" features can cause indirect discrimination
- Example: ZIP code correlates with race, causing racial discrimination

### Your results:

"""
        
        if 'worst_group_accuracy' in fwd_data:
            wga = fwd_data['worst_group_accuracy']
            explanation += f"**Worst-Group Accuracy:** {wga*100:.2f}%\n"
            if wga < 0.7:
                explanation += "⚠️ This is significantly lower than overall accuracy - hidden bias likely\n\n"
            else:
                explanation += "✅ Reasonable worst-group accuracy\n\n"
        
        if 'adversarial_bias_score' in fwd_data:
            abs_score = fwd_data['adversarial_bias_score']
            explanation += f"**Adversarial Bias Score:** {abs_score:.3f}\n"
            if abs_score > 0.7:
                explanation += "⚠️ HIGH RISK - Model decisions are driven by feature patterns\n\n"
            elif abs_score > 0.6:
                explanation += "⚠️ MEDIUM RISK - Some decision correlation with features\n\n"
            else:
                explanation += "✅ LOW RISK - Decisions are relatively independent of features\n\n"
        
        if 'proxy_warnings' in fwd_data:
            explanation += f"**Proxy Warnings:** {len(fwd_data['proxy_warnings'])} potential proxy features identified\n"
            explanation += "Review these features for potential indirect discrimination\n"
        
        explanation += """
### What to do:

1. Use these results even without demographic data
2. If worst-group accuracy is low → investigate feature engineering
3. If adversarial score is high → consider feature suppression
4. Check proxy features in your feature engineering pipeline
"""
        
        return explanation
