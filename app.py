import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
from data_loader import load_data, get_columns_names, load_sample_data
from bias_detector import detect_bias
from explainer import generate_full_explanation
from visualizer import show_all_plots
from mitigator import apply_mitigation, compare_results

st.set_page_config(page_title="AI Bias & Fairness Detector", layout="wide")
st.title("🔍 AI Bias & Fairness Detector")
st.markdown("Detect and mitigate bias in AI/ML models with ease.")

# --- Sidebar: Configuration ---
st.sidebar.header("Configuration")
data_option = st.sidebar.radio("Choose Data Source", ["Upload CSV", "Use Sample Dataset"])

df = None

if data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.success("✅ File uploaded successfully!")
else:
    if st.sidebar.button("Load Sample Dataset"):
        df = load_sample_data()
        st.session_state["df"] = df
        st.success("✅ Sample dataset loaded!")

if "df" in st.session_state and df is None:
    df = st.session_state["df"]

# --- Column Selection ---
if df is not None:
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    columns = get_columns_names(df)

    target_column = st.selectbox("Select Target Column (what the model predicts)", columns)
    sensitive_columns = st.multiselect("Select Sensitive Column(s) (e.g. gender, race)",
                                        [c for c in columns if c != target_column])

    # --- Bias Detection ---
    if st.button("🔎 Detect Bias"):
        if not sensitive_columns:
            st.error("Please select at least one sensitive column.")
        else:
            all_results = {}
            summary_rows = []

            with st.spinner("Analysing bias..."):
                for col in sensitive_columns:
                    result = detect_bias(df, target_column, col)
                    all_results[col] = result

                    dp_vals = list(result["demographic_parity"].values())
                    summary_rows.append({
                        "Sensitive Column": col,
                        "Disparate Impact": round(result["disparate_impact"], 3),
                        "DP Diff": round(max(dp_vals) - min(dp_vals), 3),
                        "TPR Diff": round(max(result["equalized_odds"]["tpr"].values()) - min(result["equalized_odds"]["tpr"].values()), 3),
                        "FPR Diff": round(max(result["equalized_odds"]["fpr"].values()) - min(result["equalized_odds"]["fpr"].values()), 3),
                        "DI Fair?": "✅" if result["disparate_impact"] >= 0.8 else ("⚠️" if result["disparate_impact"] >= 0.6 else "❌")
                    })

                st.session_state["all_results"] = all_results
                st.session_state["summary_rows"] = summary_rows
                st.session_state["df"] = df
                st.session_state["target_column"] = target_column
                st.session_state["sensitive_columns"] = sensitive_columns
                st.session_state["chat_history"] = []

            st.success("✅ Bias detection complete!")

            st.subheader("📊 Combined Bias Summary")
            summary_df = pd.DataFrame(summary_rows)
            st.table(summary_df)

            for col in sensitive_columns:
                st.markdown(f"---\n### 🔍 Results for: `{col}`")
                # Pass original df and col so visualizer can use real labels
                show_all_plots(all_results[col], df, col)
                st.markdown(generate_full_explanation(all_results[col]))

# --- Mitigation ---
if "all_results" in st.session_state:
    st.markdown("---")
    st.subheader("🛠️ Bias Mitigation")

    sensitive_columns = st.session_state["sensitive_columns"]
    strategy = st.selectbox("Select Mitigation Strategy", ["reweighing", "suppression"])
    selected_col = st.selectbox("Select Sensitive Column to Mitigate", sensitive_columns)

    if st.button("⚙️ Apply Mitigation"):
        with st.spinner("Applying mitigation..."):
            mitigated = apply_mitigation(
                st.session_state["df"],
                st.session_state["target_column"],
                selected_col,
                strategy=strategy
            )
            before = st.session_state["all_results"][selected_col]
            comparison = compare_results(before, mitigated)

        st.success(f"✅ Mitigation applied using: **{mitigated['strategy']}** on `{selected_col}`")

        st.subheader("📊 After Mitigation — Visualisations")
        show_all_plots(mitigated, st.session_state["df"], selected_col)

        st.subheader("📝 After Mitigation — Explanation")
        st.markdown(generate_full_explanation(mitigated))

        st.subheader("📈 Before vs After Comparison")
        comp_df = pd.DataFrame({
            "Metric": ["Disparate Impact", "Demographic Parity Diff", "TPR Diff", "FPR Diff"],
            "Before": [
                round(comparison["disparate_impact"]["before"], 3),
                round(comparison["demographic_parity_diff"]["before"], 3),
                round(comparison["tpr_diff"]["before"], 3),
                round(comparison["fpr_diff"]["before"], 3),
            ],
            "After": [
                round(comparison["disparate_impact"]["after"], 3),
                round(comparison["demographic_parity_diff"]["after"], 3),
                round(comparison["tpr_diff"]["after"], 3),
                round(comparison["fpr_diff"]["after"], 3),
            ]
        })
        st.table(comp_df)

# --- Sidebar: Chatbot ---
if "all_results" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.header("💬 Bias Assistant")
    st.sidebar.markdown("Ask questions about your bias results.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.sidebar.markdown(f"**You:** {msg['content']}")
        else:
            st.sidebar.markdown(f"**Assistant:** {msg['content']}")

    user_input = st.sidebar.text_input("Ask a question...", key="chat_input")

    if st.sidebar.button("Send") and user_input.strip():
        all_results = st.session_state["all_results"]

        results_summary = ""
        for col, result in all_results.items():
            results_summary += f"""
Sensitive Column: {col}
- Disparate Impact: {round(result['disparate_impact'], 3)}
- Demographic Parity Rates: {dict(result['demographic_parity'])}
- TPR: {dict(result['equalized_odds']['tpr'])}
- FPR: {dict(result['equalized_odds']['fpr'])}
"""

        context = f"""
You are a bias and fairness assistant. Answer questions only based on the following bias detection results:

{results_summary}
Target Column: {st.session_state['target_column']}

Only answer questions related to these results. If asked something unrelated, politely say you can only help with the current bias results.
        """

        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        try:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                system_instruction=context
            )
            response = model.generate_content(user_input)
            reply = response.text
        except Exception as e:
            reply = f"Error contacting assistant: {str(e)}"

        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        st.rerun()