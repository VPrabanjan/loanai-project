"""
SHAP Explainability module — explains model predictions in plain English.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


FEATURE_LABELS = {
    'Gender':             'Gender',
    'Married':            'Marital Status',
    'Dependents':         'Number of Dependents',
    'Education':          'Education Level',
    'Self_Employed':      'Self Employment',
    'ApplicantIncome':    'Monthly Income',
    'CoapplicantIncome':  'Co-applicant Income',
    'LoanAmount':         'Loan Amount',
    'Loan_Amount_Term':   'Loan Term (months)',
    'Credit_History':     'Credit History',
    'Property_Area':      'Property Area',
}

POSITIVE_REASONS = {
    'Credit_History':    "Your clean credit history strongly supports approval.",
    'ApplicantIncome':   "Your income level is favorable for this loan.",
    'LoanAmount':        "The loan amount is within an acceptable range.",
    'Married':           "Being married positively influences the assessment.",
    'Education':         "Your graduate education adds credibility.",
    'Property_Area':     "Your property location is in a favorable area.",
    'Dependents':        "Low number of dependents reduces financial burden.",
    'CoapplicantIncome': "Co-applicant income strengthens your application.",
}

NEGATIVE_REASONS = {
    'Credit_History':    "Poor credit history is the biggest risk factor.",
    'ApplicantIncome':   "Your income may be insufficient for this loan amount.",
    'LoanAmount':        "The requested loan amount is too high relative to income.",
    'Married':           "Marital status is flagged in the risk model.",
    'Education':         "Education level is affecting the risk score.",
    'Property_Area':     "Your property area carries higher lending risk.",
    'Dependents':        "High number of dependents increases financial risk.",
    'CoapplicantIncome': "No co-applicant income increases individual risk.",
}


def generate_explanation(model, input_df: pd.DataFrame, prediction: int):
    """
    Generate SHAP-based plain English explanation.

    Returns:
        positive_reasons (list of str)
        negative_reasons (list of str)
        shap_values (np.array)
        fig (matplotlib figure)
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # For binary classification, shap_values shape: (1, n_features)
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values[0]

    # Pair features with their SHAP impact
    impacts = list(zip(input_df.columns.tolist(), sv))
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)

    positive_reasons = []
    negative_reasons = []

    for feat, impact in impacts[:6]:
        label  = FEATURE_LABELS.get(feat, feat)
        if impact > 0.02:
            reason = POSITIVE_REASONS.get(feat, f"{label} is positively impacting your application.")
            positive_reasons.append(f"✅ {reason}")
        elif impact < -0.02:
            reason = NEGATIVE_REASONS.get(feat, f"{label} is negatively impacting your application.")
            negative_reasons.append(f"⚠️ {reason}")

    # ── Build a custom bar chart ──────────────────────────
    feat_names  = [FEATURE_LABELS.get(f, f) for f, _ in impacts[:8]]
    feat_values = [v for _, v in impacts[:8]]
    colors      = ['#00ff88' if v > 0 else '#ff2d78' for v in feat_values]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')

    bars = ax.barh(feat_names[::-1], feat_values[::-1], color=colors[::-1],
                   edgecolor='none', height=0.6)

    ax.axvline(0, color='rgba(255,255,255,0.2)', linewidth=0.8, linestyle='--', alpha=0.4)
    ax.set_xlabel('SHAP Impact on Prediction', color='#aaaacc', fontsize=9)
    ax.set_title('Feature Impact on Your Loan Decision', color='white', fontsize=11, pad=12)
    ax.tick_params(colors='#aaaacc', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333355')
    ax.spines['left'].set_color('#333355')

    pos_patch = mpatches.Patch(color='#00ff88', label='Positive Impact')
    neg_patch = mpatches.Patch(color='#ff2d78', label='Negative Impact')
    ax.legend(handles=[pos_patch, neg_patch], facecolor='#1a1a2e',
              edgecolor='#333355', labelcolor='white', fontsize=8)

    plt.tight_layout()

    return positive_reasons, negative_reasons, sv, fig
