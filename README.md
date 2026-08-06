# AI Bias & Fairness Detector

**A no-code, accessible tool for detecting and mitigating bias in machine learning models.**

## 📋 Overview

This project addresses a critical gap in the AI fairness landscape: while bias detection and mitigation techniques are well-studied, they remain largely inaccessible to non-experts. Our tool bridges this gap by providing an intuitive, web-based interface that lets HR analysts, policy teams, and researchers audit their ML models for bias without writing code.

### The Problem

Machine learning models trained on historical data often automate and amplify existing inequalities:
- **Amazon's hiring bot (2018)** — learned to penalize resumes containing the word "women's"
- **COMPAS recidivism algorithm** — was nearly twice as likely to falsely flag Black defendants as high-risk
- **Credit scoring systems** — perpetuate racial and gender disparities in loan approval

Yet tools like Fairlearn and AIF360 remain **code-first**, requiring Python expertise — a barrier for most practitioners.

### Our Solution

A **detect → explain → mitigate → compare** workflow in a single Streamlit app:
1. **Upload data** or use the UCI Adult Income benchmark
2. **Detect bias** using three complementary metrics
3. **Read plain-English explanations** of what the numbers mean
4. **Apply a fix** (Reweighing or Feature Suppression)
5. **Compare before/after** fairness and accuracy side-by-side

---

## ✨ Features

### Core Detection
- **Demographic Parity** — Do all groups receive favorable outcomes at equal rates?
- **Disparate Impact** — Is the 0.8 four-fifths EEOC threshold violated?
- **Equalized Odds** — Do groups receive equally accurate predictions (TPR/FPR)?

### Accessibility
- No code required — upload CSV, click buttons, get results
- Plain-English explanations for every metric (not just numbers)
- Natural-language chatbot sidebar for follow-up questions

### Multi-Attribute Support
- Check multiple sensitive columns (gender, race, age) in one run
- Motivated by Chen et al. (2024): fixing one attribute can break another

### Mitigation Strategies
- **Reweighing** — Upweight underrepresented groups during training
- **Feature Suppression** — Remove sensitive attribute ("fairness through unawareness")
- Before/after fairness-accuracy comparison via Fairea benchmarking logic

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3 |
| **Framework** | Streamlit (v1.59+) |
| **ML Library** | scikit-learn 1.9.0 |
| **Data Handling** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Chatbot** | Groq API (LPU-based inference) |
| **Deployment** | Streamlit Community Cloud / local server |
| **Version Control** | Git & GitHub |

### Why This Stack?

- **Streamlit** — Web-first, no frontend expertise needed; security-hardened (2026)
- **scikit-learn 1.9.0** — Supports free-threaded CPython; production-ready
- **Groq LPU** — Fastest open-source inference available (750–1000+ tokens/sec)
- **UCI Adult Income** — Standard fairness benchmark used in 45+ research studies

---

## 📥 Installation

### Prerequisites
- Python 3.10 or later
- pip or conda
- API key for Groq (or Gemini as fallback)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-bias-fairness-detector.git
cd ai-bias-fairness-detector

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY="your_groq_api_key_here"
export GEMINI_API_KEY="your_gemini_api_key_here"  # Optional fallback

# Run the app
streamlit run app.py
```

The app will launch at `http://localhost:8501`.

---

## 🚀 Usage

### Step 1: Load Data
- **Option A:** Upload a CSV file (with a target column and sensitive attributes)
- **Option B:** Use the UCI Adult Income dataset (click "Load Sample Data")

### Step 2: Configure Detection
- Select your **target column** (what you're predicting: e.g., income, loan approval)
- Select **sensitive attributes** (e.g., sex, race, age)
- Choose ML model parameters (Random Forest is the default)

### Step 3: Detect Bias
Click **"Detect Bias"** to:
- Train a Random Forest classifier on 80% of data
- Compute Demographic Parity, Disparate Impact, and Equalized Odds on the held-out 20%
- Render interactive charts and plain-English verdicts

### Step 4: Mitigate (Optional)
- Choose a strategy: **Reweighing** or **Feature Suppression**
- Click **"Apply Mitigation"**
- View before/after comparison of fairness metrics and accuracy

### Step 5: Ask Questions (Sidebar)
Use the **Chatbot** sidebar to ask natural-language questions about the results.

---

## 📁 Project Structure

```
ai-bias-fairness-detector/
├── app.py                  # Main Streamlit application
├── bias_detector.py        # Core fairness metrics (DP, DI, EO)
├── explainer.py           # Plain-English verdict generation
├── visualizer.py          # Matplotlib chart rendering
├── mitigator.py           # Reweighing & suppression strategies
├── data_loader.py         # CSV upload & UCI dataset fetching
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── presentations/         # Slide decks & scripts
│   ├── AI_Bias_Fairness_Detector.pptx       # Group presentation (10 slides)
│   ├── Aditya_Individual_Contribution.pptx  # Individual contribution (11 slides)
│   └── Problem_Statement_Script.md          # 5-min faculty presentation script
└── docs/                  # Additional documentation
    └── Literature_Review.md
```

---

## 🔬 Fairness Metrics Explained

### Demographic Parity
**Question:** Do all groups receive the favorable outcome at the same rate?

**Formula:** P(Ŷ = 1 | A = a) is equal for all values of a

**Verdict thresholds:**
- Gap < 5% → Fair ✅
- 5–10% → Mild bias ⚠️
- > 10% → Significant bias ❌

**Trade-off:** Can conflict mathematically with Equalized Odds

### Disparate Impact
**Question:** What is the ratio of worst-off group rate to best-off group rate?

**Formula:** min_a P(Ŷ=1|A=a) ÷ max_a P(Ŷ=1|A=a)

**Verdict thresholds:**
- ≥ 0.8 → Fair ✅ (EEOC four-fifths rule)
- 0.6–0.8 → Mild bias ⚠️
- < 0.6 → Significant bias ❌

**Legal grounding:** U.S. Equal Employment Opportunity Commission (EEOC) uses this in employment discrimination cases

### Equalized Odds
**Question:** Do groups receive equally accurate predictions?

**Formula:** TPR_a and FPR_a are equal for all values of a

**Components:**
- **TPR (True Positive Rate):** Correctly predicted positives ÷ all actual positives
- **FPR (False Positive Rate):** Incorrectly predicted positives ÷ all actual negatives

**Verdict thresholds:**
- Both gaps < 5% → Fair ✅
- Either gap < 10% → Mild bias ⚠️
- Else → Significant bias ❌

**Advantage over DP:** Accounts for whether predictions are actually correct

---

## 📊 Dataset

**UCI Adult Income Dataset** (Lichman, 2013)
- **Size:** 48,842 records
- **Task:** Predict whether income exceeds $50K
- **Sensitive attributes:** sex, race, age
- **Features:** 14 (education, occupation, hours-per-week, etc.)
- **Reference:** https://archive.ics.uci.edu/dataset/2/adult

This dataset is a fairness research standard cited in 45+ peer-reviewed studies.

---

## 📚 Literature Review

This project is grounded in **11 peer-reviewed sources from 2020–2024:**

| Author(s) | Year | Contribution |
|-----------|------|--------------|
| Mehrabi et al. | 2021 | Taxonomy of fairness definitions (basis for DP/DI/EO) |
| Caton & Haas | 2024 | Pre/in/post-processing survey (basis for mitigation strategies) |
| Pessach & Shmueli | 2022 | Metric-to-mitigation mapping |
| Le Quy et al. | 2022 | Fairness-aware dataset survey (justifies UCI Adult) |
| Pagano et al. | 2023 | Systematic review confirming DP/DI/EO as top 3 metrics |
| Siddique et al. | 2024 | Bias mitigation techniques survey |
| Ferrara | 2024 | Real-world bias impact & mitigation strategies |
| Weerts et al. | 2023 | Fairlearn toolkit (code-first precedent) |
| Hort, Zhang, Sarro, Harman | 2021 | Fairea benchmarking approach (basis for compare_results()) |
| Chen, Zhang, Sarro, Harman | 2024 | Multi-attribute fairness (motivates our multi-column support) |
| Biswas & Rajan | 2020 | Empirical bias in real-world ML models |

For full citations with links, see [Literature_Review.md](./docs/Literature_Review.md).

---

## 👥 Team

**Capstone Project — B.Tech. Final Year, MIET Meerut / AKTU Lucknow**

| Role | Name | Roll Number | Responsibility |
|------|------|-------------|-----------------|
| **Aditya Tomar** | 2300681520008 | Literature study, metric research, core implementation (DP/DI/EO) |
| **Akshay Pal** | 2300681520013 | Integration & testing |
| **Adnan Ameer** | 2300681520009 | UI/UX & frontend |
| **Ansh Parashar** | 2300681520020 | Deployment & documentation |

**Supervisor:** Mr. Vijay Kumar Sharma, Dept. of CSE (AI)

---

## 🎓 Regulatory Context

This project aligns with:
- **EU AI Act (2024)** — High-risk AI systems (hiring, credit, criminal justice) must undergo documented bias testing
- **NIST AI Risk Management Framework (2023)** — Guidance on fairness as a core risk dimension
- **EEOC Four-Fifths Rule** — 0.8 threshold for employment discrimination cases

---

## 🔮 Future Scope

- **Causal/counterfactual fairness** — Go beyond group fairness to individual fairness
- **LLM bias auditing** — Extend metrics to generative AI (text, images)
- **Production monitoring** — Continuous fairness tracking as data drift occurs
- **Intersectional fairness** — Simultaneous fairness across combinations of attributes
- **Fairness-accuracy Pareto frontier** — Interactive visualization of trade-offs

---

## 📖 How to Cite

If you use this project in research or industry applications:

```bibtex
@software{tomar2026aibias,
  title={AI Bias and Fairness Detector},
  author={Aditya Tomar, Adnan Ameer, Akshay Pal, Ansh Parashar},
  year={2026},
  institution={Meerut Institute of Engineering & Technology},
  url={https://github.com/Adityatomar20012005/ai-bias-fairness-detector},
  note={B.Tech. Capstone Project, supervised by Vijay Kumar Sharma}
}
```

---

## 📝 License

This project is licensed under the **MIT License** — see [LICENSE](./LICENSE) for details.

Free for academic, research, and commercial use. Attribution appreciated but not required.

---

## 🤝 Contributing

We welcome contributions! Please:

1. **Fork** the repository
2. **Create a feature branch:** `git checkout -b feature/your-feature-name`
3. **Commit your changes:** `git commit -m "Add your message"`
4. **Push to the branch:** `git push origin feature/your-feature-name`
5. **Open a pull request** with a clear description of your changes

### Contribution Guidelines

- Follow PEP 8 style for Python code
- Add docstrings to all functions
- Include unit tests for new metrics or mitigation strategies
- Update README if you add new features
- Reference relevant research papers in your commit messages

---

## 📧 Contact & Support

**For questions, feedback, or collaboration:**

- **Aditya Tomar** (Project Lead) — aditya.tomar.cseai.2023@miet.ac.in
- **Supervisor:** Mr. Vijay Kumar Sharma — vjay.sharma@miet.ac.in
- **Department:** Computer Science & Engineering (AI), MIET Meerut

**Issues & Bug Reports:** Open an issue on GitHub with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Python version & OS

---

## 🙏 Acknowledgments

- **Mehrabi et al., Caton & Haas, Pessach & Shmueli** for foundational fairness surveys
- **The Fairlearn team** for demonstrating the value of accessible fairness tooling
- **UCI Machine Learning Repository** for the Adult Income dataset
- **Anthropic's Claude** for research and code assistance

---

## 📄 Changelog

### v1.0 (July 2026)
- Initial release
- Core metrics: Demographic Parity, Disparate Impact, Equalized Odds
- Mitigation: Reweighing, Feature Suppression
- Streamlit UI with chatbot sidebar
- Multi-attribute support
- Support for CSV upload and UCI Adult Income dataset

---

## ⚖️ Disclaimer

This tool is provided **as-is** for research and educational purposes. While it implements established fairness metrics from peer-reviewed literature, fairness is inherently contextual and contested. **Always:**
- Involve domain experts and affected communities in fairness decisions
- Understand the trade-offs between metrics and objectives
- Consult legal and policy teams before deploying fairness interventions
- Monitor deployed models for fairness drift over time

This tool is a **starting point**, not a substitute for critical thinking about what fairness means in your specific context.

---

**Last Updated:** July 24, 2026  
**Current Version:** 1.0  
**Status:** Stable (Production-Ready)
