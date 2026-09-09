import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import sys

# Custom modules
from bias_detector import BiasDetector
from visualizer import CompactVisualizer
from explainer import FairnessExplainer
from mitigator import FairnessMitigator
from data_loader import DataLoader
from data_storage import BiasAnalysisStorage

# Page config
st.set_page_config(
    page_title="AI Bias & Fairness Detector",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .fair-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .warning-badge {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .danger-badge {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'sensitive_columns' not in st.session_state:
        st.session_state.sensitive_columns = []
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None  # 'upload' or 'uci'
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'analysis_id' not in st.session_state:
        st.session_state.analysis_id = None
    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = None

init_session_state()

# Initialize storage
storage = BiasAnalysisStorage()
storage.init_db()

# Sidebar
st.sidebar.title("⚖️ AI Bias & Fairness Detector")
st.sidebar.markdown("---")

# Sidebar: Data Source Selection
st.sidebar.subheader("📊 Data Source")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Upload CSV", "UCI Adult Dataset", "Load Previous Analysis"],
    key="data_source_radio"
)

if data_source == "Upload CSV":
    st.sidebar.subheader("📁 Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.session_state.data_source = 'upload'
            st.session_state.dataset_name = uploaded_file.name
            st.sidebar.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            st.session_state.data_loaded = False

elif data_source == "UCI Adult Dataset":
    
    st.sidebar.subheader("📊 UCI Adult Dataset")
    st.sidebar.info("Loading census income dataset (48,842 rows)")
    
    try:
        data_loader = DataLoader()
        df = data_loader.load_uci_adult()
        
        if df is not None:
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.session_state.data_source = 'uci'
            st.session_state.dataset_name = "UCI Adult Income"
            st.sidebar.success(f"✅ Loaded UCI Adult Dataset ({len(df)} rows)")
        else:
            st.sidebar.error("Failed to load UCI Adult dataset")
            st.session_state.data_loaded = False
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        st.session_state.data_loaded = False

elif data_source == "Load Previous Analysis":
    """Load previous analyses - BUG FIX: No longer requires file upload first"""
    st.sidebar.subheader("📚 Previous Analyses")
    
    try:
        analyses_summary = storage.load_analyses_summary(limit=20)
        
        if analyses_summary:
            analysis_options = [
                f"{a['dataset_name']} - {a['timestamp']}"
                for a in analyses_summary
            ]
            selected_analysis = st.sidebar.selectbox(
                "Select analysis to load:",
                range(len(analysis_options)),
                format_func=lambda i: analysis_options[i]
            )
            
            if st.sidebar.button("Load Analysis", key="load_analysis_btn"):
                selected = analyses_summary[selected_analysis]
                st.session_state.analysis_id = selected['id']
                st.session_state.dataset_name = selected['dataset_name']
                st.session_state.target_column = selected['target_column']
                st.session_state.sensitive_columns = json.loads(selected['sensitive_columns'])
                
                # Reload analysis results from storage
                results_json = json.loads(selected['results_json'])
                st.session_state.analysis_results = results_json
                
                st.sidebar.success(f"✅ Loaded: {selected['dataset_name']}")
                st.rerun()
        else:
            st.sidebar.info("No previous analyses found")
    except Exception as e:
        st.sidebar.error(f"Error loading analyses: {e}")

if st.session_state.data_loaded and st.session_state.df is not None:
    df = st.session_state.df
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configuration")
    
    # Target column selection
    columns = df.columns.tolist()
    target_col = st.sidebar.selectbox(
        "Select target column (what to predict):",
        columns,
        key="target_col_select"
    )
    st.session_state.target_column = target_col
    
    default_sensitive = st.session_state.sensitive_columns if st.session_state.sensitive_columns else []
    
    sensitive_cols = st.sidebar.multiselect(
        "Select sensitive attributes (protected columns):",
        columns,
        default=default_sensitive,
        key="sensitive_cols_select"
    )
    st.session_state.sensitive_columns = sensitive_cols
    
    # Remove target from sensitive columns if included
    sensitive_cols = [col for col in sensitive_cols if col != target_col]
    st.session_state.sensitive_columns = sensitive_cols
    
    if not sensitive_cols:
        st.sidebar.warning("⚠️ Please select at least one sensitive attribute")
    
    # Analysis options
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Analysis Options")
    
    include_intersectional = st.sidebar.checkbox(
        "Include intersectional fairness analysis",
        value=True,
        key="intersectional_check"
    )
    
    include_fwd = st.sidebar.checkbox(
        "Include fairness-without-demographics",
        value=True,
        key="fwd_check"
    )
    
    # Run analysis button
    if st.sidebar.button("🚀 Run Fairness Analysis", key="run_analysis_btn"):
        if target_col and sensitive_cols:
            with st.spinner("🔄 Analyzing fairness metrics..."):
                try:
                    detector = BiasDetector()
                    results = detector.detect_bias(
                        df=df,
                        target_column=target_col,
                        sensitive_columns=sensitive_cols,
                        include_intersectional=include_intersectional,
                        include_fwd=include_fwd
                    )
                    
                    st.session_state.analysis_results = results
                    st.sidebar.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.sidebar.error(f"Error during analysis: {e}")
        else:
            st.sidebar.error("Please select both target and sensitive columns")
    
    # Save analysis button
    if st.session_state.analysis_results:
        if st.sidebar.button("💾 Save This Analysis", key="save_analysis_btn"):
            try:
                analysis_id = storage.save_analysis(
                    dataset_name=st.session_state.dataset_name or "Custom Dataset",
                    target_column=target_col,
                    sensitive_columns=sensitive_cols,
                    analysis_results=st.session_state.analysis_results,
                    accuracy=st.session_state.analysis_results.get('model_accuracy', 0),
                    dp_gap=st.session_state.analysis_results.get('standard_metrics', {}).get('demographic_parity', {}).get('dp_gap', 0),
                    di_ratio=st.session_state.analysis_results.get('standard_metrics', {}).get('disparate_impact', {}).get('di_ratio', 0)
                )
                st.session_state.analysis_id = analysis_id
                st.sidebar.success(f"✅ Saved! Analysis ID: {analysis_id}")
            except Exception as e:
                st.sidebar.error(f"Error saving: {e}")

if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Overview", "📈 Visualizations", "🔍 Explanations", "⚙️ Mitigation", "🔗 Advanced"]
    )
    
    # TAB 1: Overview
    with tab1:
        st.title("📊 Fairness Analysis Overview")
        
        col1, col2, col3 = st.columns(3)
        
        # Model accuracy
        accuracy = results.get('model_accuracy', 0)
        with col1:
            st.metric("Model Accuracy", f"{accuracy*100:.2f}%")
        
        # Overall fairness verdict
        standard_metrics = results.get('standard_metrics', {})
        verdicts = []
        for metric_name, metric_data in standard_metrics.items():
            verdicts.append(metric_data.get('verdict', 'Unknown'))
        
        overall_bias_level = "Fair" if all(v == "Fair" for v in verdicts) else "Significant"
        with col2:
            st.metric("Overall Bias Level", overall_bias_level)
        
        # Number of attributes analyzed
        with col3:
            st.metric("Protected Attributes", len(st.session_state.sensitive_columns))
        
        st.markdown("---")
        
        # Metric cards for each sensitive attribute
        st.subheader("📋 Fairness Metrics by Attribute")
        
        for sensitive_col in st.session_state.sensitive_columns:
            st.write(f"**{sensitive_col}**")
            
            col1, col2, col3 = st.columns(3)
            
            # DP
            if 'demographic_parity' in standard_metrics:
                dp = standard_metrics['demographic_parity']
                dp_gap = dp.get('dp_gap', 0)
                dp_verdict = dp.get('verdict', 'Unknown')
                with col1:
                    if dp_verdict == "Fair":
                        st.markdown(f'<div class="fair-badge">✅ DP Gap: {dp_gap*100:.2f}%</div>', unsafe_allow_html=True)
                    elif dp_verdict == "Mild":
                        st.markdown(f'<div class="warning-badge">⚠️ DP Gap: {dp_gap*100:.2f}%</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="danger-badge">❌ DP Gap: {dp_gap*100:.2f}%</div>', unsafe_allow_html=True)
            
            # DI
            if 'disparate_impact' in standard_metrics:
                di = standard_metrics['disparate_impact']
                di_ratio = di.get('di_ratio', 0)
                di_verdict = di.get('verdict', 'Unknown')
                with col2:
                    if di_verdict == "Fair":
                        st.markdown(f'<div class="fair-badge">✅ DI: {di_ratio:.3f}</div>', unsafe_allow_html=True)
                    elif di_verdict == "Mild":
                        st.markdown(f'<div class="warning-badge">⚠️ DI: {di_ratio:.3f}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="danger-badge">❌ DI: {di_ratio:.3f}</div>', unsafe_allow_html=True)
            
            # EO
            if 'equalized_odds' in standard_metrics:
                eo = standard_metrics['equalized_odds']
                eo_verdict = eo.get('verdict', 'Unknown')
                with col3:
                    if eo_verdict == "Fair":
                        st.markdown(f'<div class="fair-badge">✅ EO: Fair</div>', unsafe_allow_html=True)
                    elif eo_verdict == "Mild":
                        st.markdown(f'<div class="warning-badge">⚠️ EO: Mild</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="danger-badge">❌ EO: Significant</div>', unsafe_allow_html=True)
            
            st.markdown("---")
    
    # TAB 2: Visualizations
    with tab2:
        st.title("📈 Fairness Visualizations")
        
        visualizer = CompactVisualizer()
        plots = visualizer.show_all_plots_compact(
            y_pred=np.array(results.get('predictions', [])),
            y_true=np.array(results.get('y_true', [])),
            df=st.session_state.df,
            sensitive_columns=st.session_state.sensitive_columns
        )
        
        # Display plots
        if plots:
            for plot_name, plot_fig in plots.items():
                st.pyplot(plot_fig)
    
    # TAB 3: Explanations
    with tab3:
        st.title("🔍 Fairness Metric Explanations")
        
        explainer = FairnessExplainer()
        
        standard_metrics = results.get('standard_metrics', {})
        
        # DP Explanation
        if 'demographic_parity' in standard_metrics:
            st.subheader("📊 Demographic Parity (DP)")
            dp_data = standard_metrics['demographic_parity']
            explanation = explainer.explain_demographic_parity(dp_data)
            st.markdown(explanation)
            st.info(f"**Verdict:** {dp_data.get('verdict', 'Unknown')}")
        
        st.markdown("---")
        
        # DI Explanation
        if 'disparate_impact' in standard_metrics:
            st.subheader("⚖️ Disparate Impact (DI)")
            di_data = standard_metrics['disparate_impact']
            explanation = explainer.explain_disparate_impact(di_data)
            st.markdown(explanation)
            st.info(f"**Verdict:** {di_data.get('verdict', 'Unknown')}")
        
        st.markdown("---")
        
        # EO Explanation
        if 'equalized_odds' in standard_metrics:
            st.subheader("🎯 Equalized Odds (EO)")
            eo_data = standard_metrics['equalized_odds']
            explanation = explainer.explain_equalized_odds(eo_data)
            st.markdown(explanation)
            st.info(f"**Verdict:** {eo_data.get('verdict', 'Unknown')}")
    
    # TAB 4: Mitigation
    with tab4:
        st.title("⚙️ Fairness Mitigation")
        
        mitigation_strategy = st.selectbox(
            "Select mitigation strategy:",
            ["Reweighting", "Feature Suppression"]
        )
        
        mitigation_sensitive_col = st.selectbox(
            "Apply mitigation to:",
            st.session_state.sensitive_columns
        )
        
        if st.button("🔧 Test Mitigation Strategy"):
            try:
                mitigator = FairnessMitigator()
                
                if mitigation_strategy == "Reweighting":
                    mitigated_results = mitigator.apply_reweighting_mitigation(
                        df=st.session_state.df,
                        sensitive_column=mitigation_sensitive_col,
                        target_column=st.session_state.target_column
                    )
                else:
                    mitigated_results = mitigator.apply_suppression_mitigation(
                        df=st.session_state.df,
                        sensitive_column=mitigation_sensitive_col,
                        target_column=st.session_state.target_column
                    )
                
                # Display before/after comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Before Mitigation")
                    st.metric("Accuracy", f"{results['model_accuracy']*100:.2f}%")
                    if 'demographic_parity' in standard_metrics:
                        st.metric("DP Gap", f"{standard_metrics['demographic_parity']['dp_gap']*100:.2f}%")
                
                with col2:
                    st.subheader("After Mitigation")
                    st.metric("Accuracy", f"{mitigated_results['accuracy']*100:.2f}%")
                    st.metric("DP Gap", f"{mitigated_results['dp_gap']*100:.2f}%")
                
                # Save mitigation test
                if st.session_state.analysis_id:
                    try:
                        storage.save_mitigation_test(
                            analysis_id=st.session_state.analysis_id,
                            sensitive_column=mitigation_sensitive_col,
                            strategy=mitigation_strategy,
                            accuracy_before=results['model_accuracy'],
                            accuracy_after=mitigated_results['accuracy'],
                            di_before=standard_metrics.get('disparate_impact', {}).get('di_ratio', 0),
                            di_after=mitigated_results.get('di_ratio', 0),
                            dp_gap_before=standard_metrics.get('demographic_parity', {}).get('dp_gap', 0),
                            dp_gap_after=mitigated_results['dp_gap']
                        )
                        st.success("✅ Mitigation test saved!")
                    except Exception as e:
                        st.warning(f"Could not save mitigation test: {e}")
                
            except Exception as e:
                st.error(f"Error applying mitigation: {e}")
    
    # TAB 5: Advanced
    with tab5:
        st.title("🔗 Advanced Analysis")
        
        # Intersectional analysis
        if 'intersectional_metrics' in results:
            st.subheader("🔀 Intersectional Fairness Analysis")
            st.info("Analyzes fairness across combinations of protected attributes")
            
            intersectional_data = results['intersectional_metrics']
            
            # Display intersectional metrics
            for attr_pair, metrics in intersectional_data.items():
                st.write(f"**{attr_pair}**")
                st.write(f"- DP Gap: {metrics.get('dp_gap', 0)*100:.2f}%")
                st.write(f"- DI Ratio: {metrics.get('di_ratio', 0):.3f}")
            
            st.markdown("---")
        
        # Fairness-without-demographics
        if 'fairness_without_demographics' in results:
            st.subheader("🔐 Fairness Without Protected Attributes")
            st.info("Privacy-preserving bias detection techniques")
            
            fwd_data = results['fairness_without_demographics']
            
            if 'worst_group_accuracy' in fwd_data:
                st.write(f"**Worst Group Accuracy:** {fwd_data['worst_group_accuracy']*100:.2f}%")
            
            if 'adversarial_bias_score' in fwd_data:
                st.write(f"**Adversarial Bias Score:** {fwd_data['adversarial_bias_score']:.3f}")
            
            if 'proxy_warnings' in fwd_data:
                st.write("**Proxy Attribute Warnings:**")
                for warning in fwd_data['proxy_warnings']:
                    st.write(f"- {warning}")

else:
    # Welcome screen
    st.title("⚖️ AI Bias & Fairness Detector")
    
    st.markdown("""
    Welcome to the AI Bias & Fairness Detector! This tool helps you identify, 
    understand, and mitigate bias in machine learning models.
    
    ### 🚀 Getting Started
    
    1. **Choose a data source** from the sidebar:
       - Upload your own CSV file
       - Use the UCI Adult dataset (standard fairness benchmark)
       - Load a previous analysis
    
    2. **Configure your analysis:**
       - Select your target column (what you're predicting)
       - Select sensitive attributes (protected columns to analyze)
    
    3. **Run the analysis** to get:
       - Fairness metrics (Demographic Parity, Disparate Impact, Equalized Odds)
       - Intersectional fairness analysis
       - Privacy-preserving bias detection
       - Plain-English explanations
       - Mitigation strategy testing
    
    ### 📊 Features
    
    - **Standard Fairness Metrics:** Detect single-attribute bias
    - **Intersectional Analysis:** Find compound discrimination
    - **Fairness-Without-Demographics:** Privacy-preserving bias detection
    - **Mitigation Testing:** Test reweighting and feature suppression
    - **Data Persistence:** Save and load previous analyses
    
    ### 📚 About
    
    This tool is built on research-grounded fairness metrics from:
    - Hardt et al. 2016 (Equalized Odds)
    - Feldman et al. 2015 (Disparate Impact)
    - Mehrabi et al. 2021 (Fairness Survey)
    - Rambhatla et al. 2023 (Intersectional Fairness)
    - Lahoti et al. 2020 (Fairness Without Demographics)
    
    ---
    
    **Created by:** Aditya Tomar, Akshay Pal, Adnan Ameer, Ansh Parashar  
    **Supervisor:** Mr. Vijay Kumar Sharma  
    **Institution:** MIET Meerut, AKTU Lucknow
    """)


st.sidebar.markdown("---")
st.sidebar.caption("AI Bias & Fairness Detector v2.1")
st.sidebar.caption("© 2026 MIET Meerut | AKTU Lucknow")
