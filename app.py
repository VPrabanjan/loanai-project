"""
AI Loan Risk Assessor with Explainable AI
==========================================
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys

# Make sure local modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from nlp.extractor     import extract_from_text, format_extracted
from explainer.shap_explain import generate_explanation

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title  = "LoanAI — Smart Risk Assessor",
    page_icon   = "🏦",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ── CUSTOM CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #080810;
    color: #f0f0ff;
}
.stApp { background: linear-gradient(135deg, #080810 0%, #0d0d1a 100%); }

/* Header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(90deg, #00f5ff, #8b5cf6, #ff2d78);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1; margin-bottom: 0.5rem;
}
.hero-sub {
    color: rgba(240,240,255,0.55);
    font-size: 1.05rem; margin-bottom: 2rem;
}
.mono { font-family: 'JetBrains Mono', monospace; color: #00f5ff; font-size: 0.8rem; }

/* Cards */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-cyan  { border-color: rgba(0,245,255,0.25); background: rgba(0,245,255,0.04); }
.card-pink  { border-color: rgba(255,45,120,0.25); background: rgba(255,45,120,0.04); }
.card-green { border-color: rgba(0,255,136,0.25); background: rgba(0,255,136,0.04); }

/* Result boxes */
.result-approved {
    background: linear-gradient(135deg, rgba(0,255,136,0.15), rgba(0,245,255,0.08));
    border: 2px solid #00ff88;
    border-radius: 20px; padding: 2.5rem;
    text-align: center;
}
.result-rejected {
    background: linear-gradient(135deg, rgba(255,45,120,0.15), rgba(139,92,246,0.08));
    border: 2px solid #ff2d78;
    border-radius: 20px; padding: 2.5rem;
    text-align: center;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem; font-weight: 800;
}
.approved-text { color: #00ff88; }
.rejected-text { color: #ff2d78; }

/* Confidence bar */
.conf-bar-wrap {
    background: rgba(255,255,255,0.08);
    border-radius: 50px; height: 10px;
    margin: 1rem 0; overflow: hidden;
}
.conf-bar-fill-green {
    height: 100%; border-radius: 50px;
    background: linear-gradient(90deg, #00ff88, #00f5ff);
    transition: width 1s ease;
}
.conf-bar-fill-red {
    height: 100%; border-radius: 50px;
    background: linear-gradient(90deg, #ff2d78, #8b5cf6);
    transition: width 1s ease;
}

/* Metric boxes */
.metric-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem; font-weight: 800; color: #00f5ff;
}
.metric-label {
    font-size: 0.75rem; color: rgba(240,240,255,0.45);
    text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;
}

/* Extracted info table */
.info-row {
    display: flex; justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 0.88rem;
}
.info-key   { color: rgba(240,240,255,0.45); }
.info-val   { color: #f0f0ff; font-weight: 600; }

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px; color: rgba(240,240,255,0.6);
    font-weight: 600; padding: 0.5rem 1.5rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,245,255,0.1) !important;
    border-color: rgba(0,245,255,0.4) !important;
    color: #00f5ff !important;
}

/* Inputs */
.stTextArea textarea, .stNumberInput input, .stSelectbox select {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #f0f0ff !important;
}
.stTextArea textarea:focus, .stNumberInput input:focus {
    border-color: rgba(0,245,255,0.5) !important;
    box-shadow: 0 0 0 2px rgba(0,245,255,0.1) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #00f5ff, #8b5cf6) !important;
    color: #080810 !important; font-weight: 700 !important;
    border: none !important; border-radius: 50px !important;
    padding: 0.65rem 2.5rem !important; font-size: 0.95rem !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,245,255,0.35) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d0d1a !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* Success/Error messages */
.stSuccess { background: rgba(0,255,136,0.1) !important; border: 1px solid rgba(0,255,136,0.3) !important; color: #00ff88 !important; }
.stError   { background: rgba(255,45,120,0.1)  !important; border: 1px solid rgba(255,45,120,0.3)  !important; color: #ff2d78  !important; }
</style>
""", unsafe_allow_html=True)


# ── LOAD MODEL ────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path    = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')
    features_path = os.path.join(os.path.dirname(__file__), 'model', 'features.pkl')

    if not os.path.exists(model_path):
        st.error("❌ Model not found! Run `python model/train_model.py` first.")
        st.stop()

    model    = pickle.load(open(model_path, 'rb'))
    features = pickle.load(open(features_path, 'rb'))
    return model, features

model, FEATURES = load_model()


# ── HELPERS ───────────────────────────────────────────────
def predict(data: dict):
    df  = pd.DataFrame([data])[FEATURES]
    pred  = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    return int(pred), proba, df


def render_result(pred, proba, input_df):
    approved = pred == 1
    conf     = proba[1] if approved else proba[0]
    color    = "approved" if approved else "rejected"
    icon     = "✅" if approved else "❌"
    label    = "LOAN APPROVED" if approved else "LOAN REJECTED"
    bar_cls  = "conf-bar-fill-green" if approved else "conf-bar-fill-red"
    pct      = f"{conf * 100:.1f}%"

    st.markdown(f"""
    <div class='result-{color}'>
        <div class='result-title {color}-text'>{icon} {label}</div>
        <p style='color:rgba(240,240,255,0.6); margin-top:0.5rem; font-size:0.95rem;'>
            AI Confidence: <strong style='color:#f0f0ff'>{pct}</strong>
        </p>
        <div class='conf-bar-wrap'>
            <div class='{bar_cls}' style='width:{pct}'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # SHAP Explanation
    st.markdown("### 🧠 Why did AI make this decision?")
    pos, neg, sv, fig = generate_explanation(model, input_df, pred)

    col_p, col_n = st.columns(2)
    with col_p:
        if pos:
            st.markdown("**Factors in your favour:**")
            for r in pos:
                st.markdown(f"<div style='color:#00ff88; font-size:0.9rem; margin:4px 0'>{r}</div>",
                            unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:rgba(240,240,255,0.4)'>No strong positive factors detected.</div>",
                        unsafe_allow_html=True)
    with col_n:
        if neg:
            st.markdown("**Factors working against you:**")
            for r in neg:
                st.markdown(f"<div style='color:#ff2d78; font-size:0.9rem; margin:4px 0'>{r}</div>",
                            unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:rgba(240,240,255,0.4)'>No major risk factors detected.</div>",
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**📊 SHAP Feature Impact Chart:**")
    st.pyplot(fig)

    # Improvement tips if rejected
    if not approved:
        st.markdown("---")
        st.markdown("### 💡 How to Improve Your Chances")
        tips = []
        if sv[FEATURES.index('Credit_History')] < 0:
            tips.append("🔧 **Improve your credit score** — pay existing debts on time for 6+ months")
        if sv[FEATURES.index('ApplicantIncome')] < 0:
            tips.append("💰 **Increase income** — consider a co-applicant or additional income source")
        if sv[FEATURES.index('LoanAmount')] < 0:
            tips.append("📉 **Reduce loan amount** — apply for a smaller amount initially")
        if sv[FEATURES.index('Dependents')] < 0:
            tips.append("👨‍👩‍👧 **Reduce financial commitments** — fewer dependents means lower risk")
        if not tips:
            tips.append("📋 Strengthen your overall financial profile and reapply after 3–6 months.")
        for t in tips:
            st.markdown(t)


# ══════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════

# ── HERO ─────────────────────────────────────────────────
st.markdown("""
<div class='hero-title'>🏦 LoanAI</div>
<div class='hero-sub'>AI-Powered Loan Risk Assessor with Explainable Intelligence</div>
<div class='mono'>// Built with XGBoost · spaCy NLP · SHAP Explainability · Streamlit</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── STATS ROW ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("<div class='metric-box'><div class='metric-val'>~80%</div><div class='metric-label'>Model Accuracy</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='metric-box'><div class='metric-val'>XGBoost</div><div class='metric-label'>ML Algorithm</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='metric-box'><div class='metric-val'>SHAP</div><div class='metric-label'>Explainability</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='metric-box'><div class='metric-val'>spaCy</div><div class='metric-label'>NLP Engine</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── TABS ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "💬 Natural Language Input",
    "📋 Manual Form",
    "📖 How It Works"
])

# ══════════════════════════════════════════════════════════
# TAB 1 — NLP INPUT
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Describe your loan application in plain English")

    st.markdown("""
    <div class='card card-cyan'>
    <div class='mono' style='margin-bottom:0.5rem'>// Example inputs you can try:</div>
    <ul style='color:rgba(240,240,255,0.65); font-size:0.88rem; line-height:2; padding-left:1.2rem;'>
        <li>I am a married male graduate earning 60000 monthly. I need a loan of 200000 for 360 months. I have good credit history and 1 dependent. I live in an urban area.</li>
        <li>Female, self employed, not graduate, income 30000, loan 500000 for 180 months, bad credit, rural area, 3 dependents.</li>
        <li>I earn 80000 per month and want to borrow 150000 for 12 months. Good credit, semiurban, no dependents, married graduate.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    user_text = st.text_area(
        "Your Application Text",
        height=130,
        placeholder="Type your loan application details in plain English...",
        label_visibility="collapsed"
    )

    analyze_btn = st.button("🔍 Analyze with AI", key="nlp_btn")

    if analyze_btn:
        if not user_text.strip():
            st.error("Please enter some text first!")
        else:
            with st.spinner("🤖 Extracting information with NLP..."):
                extracted = extract_from_text(user_text)
                formatted = format_extracted(extracted)

            st.success("✅ Information successfully extracted from your text!")

            # Show extracted info
            st.markdown("**📋 What AI understood from your text:**")
            cols = st.columns(3)
            items = list(formatted.items())
            per_col = (len(items) + 2) // 3

            for i, col in enumerate(cols):
                with col:
                    for key, val in items[i*per_col:(i+1)*per_col]:
                        st.markdown(f"""
                        <div class='info-row'>
                            <span class='info-key'>{key}</span>
                            <span class='info-val'>{val}</span>
                        </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Predict
            with st.spinner("⚡ Running risk assessment..."):
                pred, proba, input_df = predict(extracted)

            st.divider()
            render_result(pred, proba, input_df)


# ══════════════════════════════════════════════════════════
# TAB 2 — MANUAL FORM
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Fill in your application details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Personal Details**")
        gender    = st.selectbox("Gender",       ["Male", "Female"])
        married   = st.selectbox("Married",      ["Yes", "No"])
        dependents = st.selectbox("Dependents",  [0, 1, 2, 3])
        education = st.selectbox("Education",    ["Graduate", "Not Graduate"])
        self_emp  = st.selectbox("Self Employed",["No", "Yes"])

    with col2:
        st.markdown("**Financial Details**")
        income    = st.number_input("Monthly Income (₹)",        1000,  1000000, 50000, 1000)
        co_income = st.number_input("Co-applicant Income (₹)",      0,   500000,     0, 1000)
        loan_amt  = st.number_input("Loan Amount (₹ thousands)",   10,      700,   150,    5)
        loan_term = st.selectbox("Loan Term (months)", [360, 180, 120, 84, 60, 36, 12])
        credit    = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])

    with col3:
        st.markdown("**Location**")
        area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        st.markdown("<br><br>", unsafe_allow_html=True)

        # Risk estimate preview
        dti = (loan_amt * 1000) / max(income, 1) * 100
        dti_color = "#00ff88" if dti < 40 else "#ffd60a" if dti < 70 else "#ff2d78"
        st.markdown(f"""
        <div class='card' style='margin-top:1rem;'>
            <div class='mono' style='margin-bottom:0.5rem'>// Quick estimates</div>
            <div style='font-size:0.85rem; color:rgba(240,240,255,0.6); margin-bottom:0.35rem;'>
                Debt-to-Income Ratio:
                <strong style='color:{dti_color}'>{dti:.1f}%</strong>
            </div>
            <div style='font-size:0.85rem; color:rgba(240,240,255,0.6);'>
                Monthly Repayment (est.):
                <strong style='color:#00f5ff'>₹{loan_amt*1000/max(loan_term,1):,.0f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    form_btn = st.button("🔍 Assess My Loan Risk", key="form_btn")

    if form_btn:
        area_map   = {"Urban": 2, "Semiurban": 1, "Rural": 0}
        data = {
            'Gender':            1 if gender == "Male" else 0,
            'Married':           1 if married == "Yes" else 0,
            'Dependents':        dependents,
            'Education':         0 if education == "Graduate" else 1,
            'Self_Employed':     1 if self_emp == "Yes" else 0,
            'ApplicantIncome':   income,
            'CoapplicantIncome': co_income,
            'LoanAmount':        loan_amt,
            'Loan_Amount_Term':  loan_term,
            'Credit_History':    1 if credit.startswith("Good") else 0,
            'Property_Area':     area_map[area],
        }

        with st.spinner("⚡ Running risk assessment..."):
            pred, proba, input_df = predict(data)

        st.divider()
        render_result(pred, proba, input_df)


# ══════════════════════════════════════════════════════════
# TAB 3 — HOW IT WORKS
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("### How LoanAI Works")

    steps = [
        ("1️⃣", "Input", "Enter your loan application either in plain English or via the structured form."),
        ("2️⃣", "NLP Extraction", "spaCy NLP model parses your text and extracts 11 key financial features automatically."),
        ("3️⃣", "Risk Prediction", "XGBoost ML model (trained on real loan data) predicts approval probability with ~80% accuracy."),
        ("4️⃣", "SHAP Explanation", "SHAP (SHapley Additive exPlanations) breaks down exactly which features drove the AI decision."),
        ("5️⃣", "Actionable Advice", "If rejected, the system provides specific, personalized tips to improve your application."),
    ]

    for icon, title, desc in steps:
        st.markdown(f"""
        <div class='card' style='display:flex; gap:1rem; align-items:flex-start; margin-bottom:0.75rem;'>
            <div style='font-size:1.8rem; flex-shrink:0'>{icon}</div>
            <div>
                <div style='font-family:Syne,sans-serif; font-weight:700; font-size:1rem; margin-bottom:4px'>{title}</div>
                <div style='color:rgba(240,240,255,0.55); font-size:0.9rem; line-height:1.6'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    tech = {
        "Machine Learning": "XGBoost — gradient boosted decision trees for high accuracy",
        "NLP Engine":        "spaCy — industrial-strength natural language processing",
        "Explainability":    "SHAP — game-theory based feature attribution",
        "Frontend":          "Streamlit — rapid Python web app framework",
        "Data Processing":   "Pandas + NumPy — data wrangling and numerical ops",
        "Visualization":     "Matplotlib + Plotly — interactive charts",
    }
    for k, v in tech.items():
        st.markdown(f"""
        <div class='info-row'>
            <span class='info-key' style='font-weight:600; color:#00f5ff'>{k}</span>
            <span class='info-val' style='color:rgba(240,240,255,0.65); font-size:0.85rem'>{v}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:rgba(240,240,255,0.3); font-size:0.8rem; padding:1rem'>
        Built by <strong style='color:#00f5ff'>Prabanjan V</strong> · 
        Data Scientist · Chennai, Tamil Nadu
    </div>
    """, unsafe_allow_html=True)
