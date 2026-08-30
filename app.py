import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

from bias_detector import detect_bias, demographic_parity, disparate_impact, equalized_odds
from visualizer import show_all_plots
from explainer import generate_full_explanation
from data_loader import load_sample_data

st.set_page_config(
    page_title="AI Bias & Fairness Detector",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY_COLOR = "#1E40AF"    
SUCCESS_COLOR = "#10B981"
WARNING_COLOR = "#F59E0B"    
DANGER_COLOR = "#EF4444" 
LIGHT_BG = "#F9FAFB"         

# Custom CSS
st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1E40AF;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        font-weight: 500;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .success-box {
        background-color: #ECFDF5;
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .danger-box {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Data source selection
    data_source = st.radio(
        "📊 Data Source",
        ["Upload CSV", "Use Sample Data (UCI Adult)"],
        index=0
    )
    
    df = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    else:
        if st.button("📥 Load UCI Adult Income Dataset"):
            with st.spinner("Loading UCI Adult dataset..."):
                df = load_sample_data()
                st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    st.markdown("---")
    
    if df is not None:
        st.markdown("### 🎯 Target & Sensitive Columns")
        
        target_col = st.selectbox(
            "Target Column (what to predict)",
            df.columns
        )
        
        sensitive_cols = st.multiselect(
            "Sensitive Attributes (protected characteristics)",
            [col for col in df.columns if col != target_col],
            default=[col for col in df.columns if col.lower() in ['sex', 'gender', 'race']][:2]
        )
        
        st.markdown("---")
        st.markdown("### 🔧 Analysis Options")
        
        compute_intersectional = st.checkbox(
            "🔗 Compute Intersectional Fairness",
            value=True,
            help="Check fairness across combinations of multiple sensitive attributes"
        )
        
        compute_fwd = st.checkbox(
            "🔎 Compute Fairness Without Demographics",
            value=True,
            help="Detect fairness issues without explicit demographic labels"
        )
        
        st.markdown("---")
        
        if st.button("🚀 Run Full Bias Detection", key="main_detect", use_container_width=True):
            st.session_state.run_detection = True
            st.session_state.bias_results = None

col1, col2 = st.columns([0.8, 0.2])

with col1:
    st.markdown('<div class="main-header">⚖️ AI Bias & Fairness Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detect, explain, and mitigate machine learning bias in seconds</div>', 
                unsafe_allow_html=True)

with col2:
    st.markdown("")
    st.markdown("")
    if st.button("📖 About", help="Learn more about this tool"):
        st.info("""
        This tool detects and explains bias in machine learning models using:
        - **Demographic Parity**: Equal outcome rates across groups
        - **Disparate Impact**: EEOC four-fifths rule compliance
        - **Equalized Odds**: Equal accuracy across groups
        - **Intersectional Fairness**: Compound discrimination detection
        - **Fairness Without Demographics**: Bias detection without explicit labels
        """)

st.markdown("---")

# Check if detection has been run
if df is None:
    st.warning("👈 Please configure and load your data in the sidebar to begin")

elif 'run_detection' not in st.session_state or not st.session_state.run_detection:
    st.info("👈 Configure your analysis in the sidebar, then click 'Run Full Bias Detection'")

else:
    # Run detection
    with st.spinner("🔄 Running comprehensive bias analysis..."):
        try:
            bias_results = detect_bias(
                df,
                target_col,
                sensitive_cols,
                compute_intersectional=compute_intersectional,
                compute_fairness_without_demographics=compute_fwd
            )
            st.session_state.bias_results = bias_results
        except Exception as e:
            st.error(f"❌ Error during detection: {str(e)}")
            st.stop()
    
    st.success("✅ Analysis complete!")
    st.markdown("---")
 
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Visualizations",
        "🔍 Detailed Explanations",
        "🔗 Intersectional Analysis",
        "🔎 Fairness Without Demographics"
    ])

    with tab1:
        st.markdown("### Summary of Findings")
        
        # Create metric cards for each sensitive column
        for sens_col in sensitive_cols:
            st.markdown(f"#### {sens_col}")
            
            metrics = bias_results['metrics'][sens_col]
            dp = metrics['demographic_parity']
            di = metrics['disparate_impact']
            eo = metrics['equalized_odds']
            
            # Calculate gaps
            dp_vals = [v for v in dp.values() if not np.isnan(v)]
            if dp_vals:
                dp_gap = max(dp_vals) - min(dp_vals)
            else:
                dp_gap = 0
            
            eo_vals = list(eo['tpr'].values()) + list(eo['fpr'].values())
            eo_vals = [v for v in eo_vals if not np.isnan(v)]
            if eo_vals:
                eo_gap = max(eo_vals) - min(eo_vals)
            else:
                eo_gap = 0
            
            # Metric columns
            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.metric(
                    "Demographic Parity Gap",
                    f"{dp_gap:.1%}",
                    delta="✅ Fair" if dp_gap < 0.05 else "⚠️ Mild" if dp_gap < 0.1 else "❌ Significant",
                    delta_color="off"
                )
            
            with m2:
                di_verdict = "✅" if di >= 0.8 else "⚠️" if di >= 0.6 else "❌"
                st.metric(
                    "Disparate Impact Ratio",
                    f"{di:.3f}",
                    delta=di_verdict,
                    delta_color="off"
                )
            
            with m3:
                st.metric(
                    "Equalized Odds Gap",
                    f"{eo_gap:.1%}",
                    delta="✅ Fair" if eo_gap < 0.05 else "⚠️ Mild" if eo_gap < 0.1 else "❌ Significant",
                    delta_color="off"
                )
            
            # Show detailed rates
            with st.expander(f"📋 Detailed Rates for {sens_col}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Demographic Parity Rates:**")
                    for group, rate in dp.items():
                        if not np.isnan(rate):
                            st.write(f"- {group}: {rate:.1%}")
                
                with col2:
                    st.markdown("**Equalized Odds:**")
                    st.write("**True Positive Rates (TPR):**")
                    for group, rate in eo['tpr'].items():
                        if not np.isnan(rate):
                            st.write(f"- {group}: {rate:.1%}")
                    st.write("**False Positive Rates (FPR):**")
                    for group, rate in eo['fpr'].items():
                        if not np.isnan(rate):
                            st.write(f"- {group}: {rate:.1%}")
        
        st.markdown("---")
        
        # Model Performance
        st.markdown("### 🎯 Model Performance")
        accuracy = (bias_results['y_pred'] == bias_results['y_test']).mean()
        st.metric("Overall Accuracy", f"{accuracy:.1%}")
  
    with tab2:
        st.markdown("### 📊 Bias Visualizations")
        
        figs = show_all_plots(
            bias_results,
            show_intersectional=compute_intersectional,
            show_fairness_without_demographics=compute_fwd
        )
        
        for fig_name, fig in figs:
            st.pyplot(fig, use_container_width=True)
            st.markdown("---")
       
    with tab3:
        st.markdown("### 📝 Detailed Explanations")
        
        explanations = generate_full_explanation(
            bias_results,
            sensitive_col='all',
            include_intersectional=compute_intersectional,
            include_fairness_without_demographics=compute_fwd
        )
        
        for exp in explanations:
            st.markdown(exp)
            st.markdown("---")
    
    with tab4:
        if compute_intersectional and 'intersectional_metrics' in bias_results:
            st.markdown("### 🔗 Intersectional Fairness Analysis")
            st.info("""
            Intersectional fairness checks bias across *combinations* of sensitive attributes.
            A model might appear fair for gender alone, but show bias at the intersection of gender × race.
            """)
            
            for intersection_key, metrics in bias_results['intersectional_metrics'].items():
                st.markdown(f"#### {intersection_key}")
                
                attr1, attr2 = intersection_key.split(' × ')
                
                # Show DI ratio for intersection
                di_ratio = metrics['disparate_impact']
                col1, col2 = st.columns(2)
                
                with col1:
                    di_verdict = "✅" if di_ratio >= 0.8 else "⚠️" if di_ratio >= 0.6 else "❌"
                    st.metric(
                        f"DI Ratio ({intersection_key})",
                        f"{di_ratio:.3f}",
                        delta=di_verdict,
                        delta_color="off"
                    )
                
                with col2:
                    st.markdown(f"**Interpretation:**")
                    if di_ratio >= 0.8:
                        st.success("✅ Fair across both attributes")
                    elif di_ratio >= 0.6:
                        st.warning("⚠️ Mild bias detected at intersection")
                    else:
                        st.error("❌ Significant compound discrimination")
                
                # Show rates matrix
                with st.expander(f"📊 Rates for {intersection_key}", expanded=False):
                    rates = metrics['demographic_parity']
                    df_rates = pd.DataFrame(list(rates.items()), columns=['Group', 'Positive Rate'])
                    df_rates['Positive Rate'] = df_rates['Positive Rate'].apply(lambda x: f"{x:.1%}")
                    st.dataframe(df_rates, use_container_width=True)
                
                st.markdown("---")
        else:
            st.info("💡 Enable 'Compute Intersectional Fairness' in the sidebar to see intersectional analysis")

    with tab5:
        if compute_fwd and 'fairness_without_demographics' in bias_results:
            st.markdown("### 🔎 Fairness Without Protected Attributes")
            st.info("""
            These methods detect fairness issues **without** explicit demographic labels.
            Useful when demographics are unavailable or privacy-sensitive.
            """)
            
            fwd = bias_results['fairness_without_demographics']
            
            # Worst-Group Accuracy
            st.markdown("#### 📉 Worst-Group Accuracy")
            wga = fwd['worst_group_accuracy']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Worst Cluster Accuracy", f"{wga['worst_accuracy']:.1%}")
            with col2:
                st.metric("Accuracy Gap", f"{wga['accuracy_gap']:.1%}")
            with col3:
                if wga['accuracy_gap'] > 0.1:
                    st.error("❌ Large disparity")
                elif wga['accuracy_gap'] > 0.05:
                    st.warning("⚠️ Mild disparity")
                else:
                    st.success("✅ Small disparity")
            
            # Adversarial Bias
            st.markdown("#### 🎯 Adversarial Bias Detection")
            adv = fwd['adversarial_bias']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Adversary Accuracy", f"{adv['mean_adversary_accuracy']:.1%}")
            with col2:
                bias_color = "green" if adv['bias_level'] == 'low' else "orange" if adv['bias_level'] == 'medium' else "red"
                st.markdown(f"**Bias Level:** <span style='color:{bias_color}'>{adv['bias_level'].upper()}</span>", 
                           unsafe_allow_html=True)
            
            st.info("""
            **Interpretation:** If the adversary can predict model outputs from features (>60% accuracy),
            your model's decisions are correlated with input patterns in a potentially biased way.
            """)
            
            # Proxy Warnings
            if fwd['proxy_warnings']:
                st.markdown("#### ⚠️ Proxy Feature Warnings")
                st.warning("""
                Some features correlate with sensitive attributes and may be acting as proxies for discrimination.
                Consider removing or carefully auditing these features.
                """)
                
                proxy_df = pd.DataFrame(fwd['proxy_warnings']).head(10)
                proxy_df['correlation'] = proxy_df['correlation'].apply(lambda x: f"{x:.3f}")
                st.dataframe(proxy_df[['proxy_feature', 'sensitive_attr', 'correlation']], 
                           use_container_width=True)
            else:
                st.success("✅ No significant proxy features detected")
        
        else:
            st.info("💡 Enable 'Compute Fairness Without Demographics' in the sidebar to see these metrics")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>
    <p>AI Bias & Fairness Detector v2.0 | Enhanced with Intersectional & Fairness-Without-Demographics Metrics</p>
    <p>Built with ❤️ for responsible AI | MIET Meerut CSE-AI Capstone Project</p>
</div>
""", unsafe_allow_html=True)
