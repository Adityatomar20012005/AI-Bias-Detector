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

@st.cache_data
def load_adult_cached():
    return DataLoader().load_uci_adult()

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

MITIGATION_LABELS = {
    'none': 'None (original model)',
    'reweighting': 'Reweighting',
    'suppression': 'Remove sensitive columns',
    'proxy_removal': 'Remove sensitive columns + proxies',
}
VERDICT_ORDER = {'Fair': 0, 'Mild': 1, 'Significant': 2}


def worst_verdict(standard_metrics):
    """Overall level = the worst verdict across all metrics"""
    known = [m.get('verdict') for m in standard_metrics.values() if m.get('verdict') in VERDICT_ORDER]
    return max(known, key=VERDICT_ORDER.get) if known else 'Unknown'

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
        df = load_adult_cached()
        
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

# ============================================================================
# SECTION 2: CONFIGURATION & ANALYSIS
# ============================================================================

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

    # Update session state
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
        "Include intersectional analysis",
        value=True,
        key="intersectional_check"
    )

    include_fwd = st.sidebar.checkbox(
        "Include fairness-without-demographics",
        value=True,
        key="fwd_check"
    )
    
    mitigation_mode = st.sidebar.selectbox(
        "Bias mitigation applied to the model:",
        list(MITIGATION_LABELS.keys()),
        format_func=lambda k: MITIGATION_LABELS[k],
        key="mitigation_mode_select",
        help="Trains the audited model with this fix and shows before vs after on the Overview tab."
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
                        include_fwd=include_fwd,
                        mitigation=mitigation_mode
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
        overall_bias_level = worst_verdict(standard_metrics)
        baseline = results.get('baseline')
        was = worst_verdict(baseline['standard_metrics']) if baseline else None
        with col2:
            st.metric("Overall Bias Level", overall_bias_level,
                      delta=f"was {was} before mitigation" if was else None,
                      delta_color="off")

        # Number of attributes analyzed
        with col3:
            st.metric("Protected Attributes", len(st.session_state.sensitive_columns))
        
        st.markdown("---")

        mit = results.get('mitigation', {})
        strategy = mit.get('strategy', 'none')
        if strategy != 'none' and baseline:
            dropped = mit.get('dropped_columns', [])
            detail = (f"Columns removed from training: `{', '.join(dropped)}`." if dropped
                      else "Training rows were reweighted so group and outcome are independent.")
            st.info(f"🛡️ Model trained with **{MITIGATION_LABELS.get(strategy, strategy)}**. {detail}")

            rows = []
            for c in st.session_state.sensitive_columns:
                b_di = baseline['standard_metrics']['disparate_impact']['per_attribute'].get(c, {})
                a_di = standard_metrics['disparate_impact']['per_attribute'].get(c, {})
                b_dp = baseline['standard_metrics']['demographic_parity']['per_attribute'].get(c, {})
                a_dp = standard_metrics['demographic_parity']['per_attribute'].get(c, {})
                rows.append({
                    'Attribute': c,
                    'DI before': round(b_di.get('di_ratio', 0), 3),
                    'DI after': round(a_di.get('di_ratio', 0), 3),
                    'DP gap before (%)': round(b_dp.get('dp_gap', 0) * 100, 2),
                    'DP gap after (%)': round(a_dp.get('dp_gap', 0) * 100, 2),
                    'Verdict': f"{b_di.get('verdict', '?')} → {a_di.get('verdict', '?')}",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption(f"Accuracy: {baseline['model_accuracy']*100:.2f}% before → "
                       f"{accuracy*100:.2f}% after (the fairness-accuracy trade-off).")
            st.markdown("---")

        # Metric cards for each sensitive attribute
        st.subheader("📋 Fairness Metrics by Attribute")
        
        for sensitive_col in st.session_state.sensitive_columns:
            st.write(f"**{sensitive_col}**")
            
            col1, col2, col3 = st.columns(3)
            
            # DP
            if 'demographic_parity' in standard_metrics:
                dp = standard_metrics['demographic_parity'].get('per_attribute', {}).get(sensitive_col, {})
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
                di = standard_metrics['disparate_impact'].get('per_attribute', {}).get(sensitive_col, {})
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
                eo = standard_metrics['equalized_odds'].get('per_attribute', {}).get(sensitive_col, {})
                eo_verdict = eo.get('verdict', 'Unknown')
                with col3:
                    if eo_verdict == "Fair":
                        st.markdown(f'<div class="fair-badge">✅ EO: Fair (TPR gap {eo.get("tpr_gap", 0)*100:.1f}%, FPR gap {eo.get("fpr_gap", 0)*100:.1f}%)</div>', unsafe_allow_html=True)
                    elif eo_verdict == "Mild":
                        st.markdown(f'<div class="warning-badge">⚠️ EO: Mild (TPR gap {eo.get("tpr_gap", 0)*100:.1f}%, FPR gap {eo.get("fpr_gap", 0)*100:.1f}%)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="danger-badge">❌ EO: {eo_verdict} (TPR gap {eo.get("tpr_gap", 0)*100:.1f}%, FPR gap {eo.get("fpr_gap", 0)*100:.1f}%)</div>', unsafe_allow_html=True)
            
            st.markdown("---")
    
    # TAB 2: Visualizations
    with tab2:
        st.title("📈 Fairness Visualizations")
        
        # Prefer data stored inside the results (works for loaded analyses too)
        df_test_viz = None
        if 'test_sensitive_data' in results:
            df_test_viz = pd.DataFrame(results['test_sensitive_data'])
        elif 'test_index' in results and st.session_state.df is not None:
            df_test_viz = st.session_state.df.loc[results['test_index']].reset_index(drop=True)

        if df_test_viz is None:
            st.warning("This analysis was saved before plot data was stored. "
                       "Load the dataset, re-run the analysis and save it again.")
        else:
            viz_cols = [c for c in st.session_state.sensitive_columns if c in df_test_viz.columns]
            visualizer = CompactVisualizer()
            plots = visualizer.show_all_plots_compact(
                y_pred=np.array(results['predictions']),
                y_true=np.array(results['y_true']),
                df=df_test_viz,
                sensitive_columns=viz_cols,
                y_proba=np.array(results.get('y_pred_proba', [])),
                feature_importance=results.get('feature_importance')
            )
            if plots:
                for plot_name, plot_fig in plots.items():
                    st.pyplot(plot_fig)
            else:
                st.info("No plots could be generated for the selected attributes.")

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
        
        if st.session_state.df is None:
            st.info("Mitigation needs the original dataset. Load the dataset from the sidebar first.")
        elif st.button("🔧 Test Mitigation Strategy"):
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
                
                if not mitigated_results.get('success'):
                    st.error(f"Mitigation failed: {mitigated_results.get('error')}")
                    raise RuntimeError(mitigated_results.get("error"))

                dp_before = standard_metrics.get('demographic_parity', {}).get('per_attribute', {}).get(mitigation_sensitive_col, {}).get('dp_gap', 0)
                di_before = standard_metrics.get('disparate_impact', {}).get('per_attribute', {}).get(mitigation_sensitive_col, {}).get('di_ratio', 1.0)

                # Display before/after comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Before Mitigation")
                    st.metric("Accuracy", f"{results['model_accuracy']*100:.2f}%")
                    st.metric("DP Gap", f"{dp_before*100:.2f}%")
                    st.metric("DI Ratio", f"{di_before:.3f}")
                
                with col2:
                    st.subheader("After Mitigation")
                    st.metric("Accuracy", f"{mitigated_results['accuracy']*100:.2f}%")
                    st.metric("DP Gap", f"{mitigated_results['dp_gap']*100:.2f}%")
                    st.metric("DI Ratio", f"{mitigated_results['di_ratio']:.3f}")
                
                # Save mitigation test
                if st.session_state.analysis_id:
                    try:
                        storage.save_mitigation_test(
                            analysis_id=st.session_state.analysis_id,
                            sensitive_column=mitigation_sensitive_col,
                            strategy=mitigation_strategy,
                            accuracy_before=results['model_accuracy'],
                            accuracy_after=mitigated_results['accuracy'],
                            di_before=di_before,
                            di_after=mitigated_results.get('di_ratio', 0),
                            dp_gap_before=dp_before,
                            dp_gap_after=mitigated_results['dp_gap']
                        )
                        st.success("✅ Mitigation test saved!")
                    except Exception as e:
                        st.warning(f"Could not save mitigation test: {e}")
                
            except Exception as e:
                st.error(f"Error applying mitigation: {e}")
    
        # ---------- Export mitigated dataset ----------
        st.markdown("---")
        st.subheader("📥 Export mitigated dataset")
        if st.session_state.df is None:
            st.info("Load the original dataset (sidebar) to export a mitigated copy.")
        else:
            export_labels = {
                'reweighting': "Reweighting (adds a sample_weight column)",
                'suppression': "Sensitive columns removed",
                'proxy_removal': "Sensitive columns + proxy features removed",
            }
            keys = list(export_labels.keys())
            active = results.get('mitigation', {}).get('strategy', 'none')
            export_strategy = st.selectbox(
                "Dataset to export:", keys,
                index=keys.index(active) if active in keys else 0,
                format_func=lambda k: export_labels[k],
                key="export_strategy_select"
            )
            proxy_names = sorted({p['feature'] for p in results.get('mitigation', {}).get('proxy_features', [])})
            mitigated_df = FairnessMitigator().build_mitigated_dataset(
                st.session_state.df, st.session_state.sensitive_columns,
                st.session_state.target_column, export_strategy, proxy_names
            )
            st.caption(f"{len(mitigated_df):,} rows × {len(mitigated_df.columns)} columns "
                       "(rows with missing values are dropped, as in the analysis)")
            st.dataframe(mitigated_df.head(10))
            st.download_button(
                "⬇️ Download mitigated dataset (CSV)",
                data=mitigated_df.to_csv(index=False).encode('utf-8'),
                file_name=f"mitigated_{export_strategy}.csv",
                mime="text/csv",
                key="download_mitigated_btn"
            )

    # TAB 5: Advanced
    with tab5:
        st.title("🔗 Advanced Analysis")

        # ---------- Intersectional ----------
        if 'intersectional_metrics' in results:
            st.subheader("🔀 Intersectional Fairness Analysis")
            st.info("Analyzes fairness across combinations of protected attributes")

            for pair, m in results['intersectional_metrics'].items():
                st.write(f"**{pair}**")
                if 'error' in m:
                    st.warning(m['error'])
                    continue

                best, worst = m['most_favored'], m['least_favored']
                st.write(f"- DP Gap: {m['dp_gap']*100:.2f}%")
                st.write(f"- DI Ratio: {m['di_ratio']:.3f} ({m['verdict']})")
                st.write(f"- Most favored: **{best}** ({m['rates'][best]*100:.1f}% positive)")
                st.write(f"- Least favored: **{worst}** ({m['rates'][worst]*100:.1f}% positive)")

                if m['verdict'] == 'Fair':
                    st.success("✅ No action needed for this intersection.")
                else:
                    st.warning(
                        f"⚠️ **{worst}** gets positive outcomes far less often than **{best}**. "
                        f"**Action:** go to the ⚙️ Mitigation tab and test Reweighting on "
                        f"one of these attributes, then re-run the analysis."
                    )
            st.markdown("---")

        # ---------- Fairness without demographics ----------
        if 'fairness_without_demographics' in results:
            st.subheader("🔐 Fairness Without Protected Attributes")
            st.info("Privacy-preserving bias detection techniques")
            fwd = results['fairness_without_demographics']

            # Worst-group accuracy vs overall
            wga = fwd.get('worst_group_accuracy', 0)
            overall = results['model_accuracy']
            st.write(f"**Worst Group Accuracy:** {wga*100:.2f}%  (overall: {overall*100:.2f}%)")
            if overall - wga > 0.10:
                st.warning("⚠️ Some cluster of people gets much worse accuracy than average. "
                           "**Action:** collect more data for that segment or add features that describe it.")
            else:
                st.success("✅ Accuracy is fairly even across clusters.")

            # Adversarial score vs baseline
            adv = fwd.get('adversarial_bias_score', 0.5)
            base = fwd.get('adversarial_baseline', 0.5)
            st.write(f"**Adversarial Bias Score:** {adv:.3f}  (baseline: {base:.3f})")
            if adv - base > 0.10:
                st.warning("⚠️ Decisions are easy to predict from features alone, beyond the majority-class baseline. "
                           "Check the proxy features below.")
            else:
                st.success("✅ Decisions are not strongly predictable beyond the baseline.")

            # Proxy features + action
            st.write("**Proxy Attribute Warnings:**")
            for w in fwd.get('proxy_warnings', []):
                st.write(f"- {w}")

            proxies = fwd.get('proxy_features', [])
            if proxies:
                st.markdown("##### 🛠️ Apply: remove proxies and retest")
                target_sens = st.selectbox("Evaluate fairness for:",
                                           st.session_state.sensitive_columns,
                                           key="proxy_sens_select")
                if st.session_state.df is None:
                    st.info("Load the original dataset to run proxy removal.")
                elif st.button("🧹 Remove proxy features & retest", key="proxy_btn"):
                    names = sorted({p['feature'] for p in proxies if p['sensitive'] == target_sens})
                    if not names:
                        st.info(f"No proxy features found for '{target_sens}'.")
                    else:
                        mit = FairnessMitigator()
                        kwargs = dict(df=st.session_state.df,
                                      sensitive_column=target_sens,
                                      target_column=st.session_state.target_column)
                        before = mit.apply_proxy_removal(proxy_features=[], **kwargs)
                        after = mit.apply_proxy_removal(proxy_features=names, **kwargs)

                        if before['success'] and after['success']:
                            st.write(f"Removed: `{', '.join(names)}`")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.subheader("Without proxy removal")
                                st.metric("Accuracy", f"{before['accuracy']*100:.2f}%")
                                st.metric("DI Ratio", f"{before['di_ratio']:.3f}")
                                st.metric("DP Gap", f"{before['dp_gap']*100:.2f}%")
                            with c2:
                                st.subheader("With proxies removed")
                                st.metric("Accuracy", f"{after['accuracy']*100:.2f}%")
                                st.metric("DI Ratio", f"{after['di_ratio']:.3f}")
                                st.metric("DP Gap", f"{after['dp_gap']*100:.2f}%")
                        else:
                            st.error(after.get('error') or before.get('error'))

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
