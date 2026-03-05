"""
NLP module — extracts loan features from plain English text using spaCy + regex.
"""

import re
import spacy

# Load spaCy model (run: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def extract_from_text(text: str) -> dict:
    """
    Parse plain English loan application text and return a feature dict.

    Example input:
        "I am a married male graduate earning 60000 monthly.
         I need a loan of 200000 for 360 months. I have good credit history
         and 2 dependents. I live in an urban area."

    Returns:
        dict with keys matching model FEATURES
    """
    raw   = text
    text  = text.lower()
    doc   = nlp(text)

    result = {
        'Gender':             1,   # default Male
        'Married':            0,
        'Dependents':         0,
        'Education':          0,   # Graduate
        'Self_Employed':      0,
        'ApplicantIncome':    5000,
        'CoapplicantIncome':  0,
        'LoanAmount':         150, # in thousands
        'Loan_Amount_Term':   360,
        'Credit_History':     1,
        'Property_Area':      1,   # Semiurban
    }

    # ── Gender ────────────────────────────────────────────
    if 'female' in text or 'woman' in text or 'she ' in text:
        result['Gender'] = 0
    elif 'male' in text or 'man' in text or 'he ' in text:
        result['Gender'] = 1

    # ── Married ───────────────────────────────────────────
    if any(w in text for w in ['married', 'wife', 'husband', 'spouse', 'wed']):
        result['Married'] = 1
    if any(w in text for w in ['unmarried', 'single', 'bachelor', 'not married']):
        result['Married'] = 0

    # ── Education ─────────────────────────────────────────
    if any(w in text for w in ['not graduate', 'non graduate', 'no degree', 'diploma', 'high school']):
        result['Education'] = 1  # Not Graduate
    elif 'graduate' in text or 'degree' in text or 'university' in text or 'college' in text:
        result['Education'] = 0  # Graduate

    # ── Self Employed ─────────────────────────────────────
    if any(w in text for w in ['self employed', 'self-employed', 'own business',
                                'business owner', 'entrepreneur', 'freelance']):
        result['Self_Employed'] = 1

    # ── Credit History ────────────────────────────────────
    if any(w in text for w in ['bad credit', 'poor credit', 'no credit',
                                'defaulted', 'low credit', 'credit issues']):
        result['Credit_History'] = 0
    elif any(w in text for w in ['good credit', 'excellent credit',
                                  'clean credit', 'credit history']):
        result['Credit_History'] = 1

    # ── Property Area ─────────────────────────────────────
    if 'urban' in text and 'semi' not in text:
        result['Property_Area'] = 2
    elif any(w in text for w in ['semiurban', 'semi urban', 'suburban']):
        result['Property_Area'] = 1
    elif 'rural' in text or 'village' in text or 'countryside' in text:
        result['Property_Area'] = 0

    # ── Dependents ────────────────────────────────────────
    dep_map = {
        'no dependent': 0, 'zero dependent': 0,
        'one dependent': 1, '1 dependent': 1,
        'two dependent': 2, '2 dependent': 2,
        'three dependent': 3, '3 dependent': 3,
        '3+ dependent': 3,
    }
    for phrase, val in dep_map.items():
        if phrase in text:
            result['Dependents'] = val
            break

    # ── Extract all numbers from text ────────────────────
    # Remove commas so "1,50,000" → "150000"
    clean_text = re.sub(r'(\d),(\d)', r'\1\2', text)
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', clean_text)
    numbers = [float(n) for n in numbers]

    # ── Income detection ──────────────────────────────────
    income_patterns = [
        r'earn(?:ing|s)?\s+(?:rs\.?\s*)?(\d[\d,]*)',
        r'salary\s+(?:of\s+)?(?:rs\.?\s*)?(\d[\d,]*)',
        r'income\s+(?:of\s+)?(?:rs\.?\s*)?(\d[\d,]*)',
        r'(?:rs\.?\s*)?(\d[\d,]+)\s+(?:monthly|per month|a month)',
    ]
    for pat in income_patterns:
        match = re.search(pat, clean_text)
        if match:
            val = float(re.sub(r',', '', match.group(1)))
            result['ApplicantIncome'] = val
            break

    # ── Loan Amount detection ─────────────────────────────
    loan_patterns = [
        r'loan\s+(?:of\s+)?(?:rs\.?\s*)?(\d[\d,]*)',
        r'borrow\s+(?:rs\.?\s*)?(\d[\d,]*)',
        r'need\s+(?:rs\.?\s*)?(\d[\d,]*)',
        r'(?:rs\.?\s*)?(\d[\d,]+)\s+(?:loan|amount)',
    ]
    for pat in loan_patterns:
        match = re.search(pat, clean_text)
        if match:
            val = float(re.sub(r',', '', match.group(1)))
            # Convert to thousands if value > 10000
            result['LoanAmount'] = val / 1000 if val > 1000 else val
            break

    # ── Loan Term detection ───────────────────────────────
    term_patterns = [
        r'(\d+)\s+months?',
        r'(\d+)\s+years?\s+term',
        r'term\s+(?:of\s+)?(\d+)',
    ]
    for pat in term_patterns:
        match = re.search(pat, clean_text)
        if match:
            val = float(match.group(1))
            # If years given, convert to months
            if val <= 30:
                val = val * 12
            result['Loan_Amount_Term'] = val
            break

    # ── Co-applicant income ───────────────────────────────
    co_patterns = [
        r'co.?applicant\s+(?:income|earn)\s+(?:rs\.?\s*)?(\d[\d,]*)',
        r'spouse\s+(?:earn|income)\s+(?:rs\.?\s*)?(\d[\d,]*)',
        r'joint\s+(?:rs\.?\s*)?(\d[\d,]*)',
    ]
    for pat in co_patterns:
        match = re.search(pat, clean_text)
        if match:
            val = float(re.sub(r',', '', match.group(1)))
            result['CoapplicantIncome'] = val
            break

    return result


def format_extracted(data: dict) -> dict:
    """Return human-readable version for display."""
    return {
        'Gender':            'Male' if data['Gender'] == 1 else 'Female',
        'Married':           'Yes' if data['Married'] == 1 else 'No',
        'Dependents':        str(data['Dependents']),
        'Education':         'Graduate' if data['Education'] == 0 else 'Not Graduate',
        'Self Employed':     'Yes' if data['Self_Employed'] == 1 else 'No',
        'Monthly Income':    f"₹{data['ApplicantIncome']:,.0f}",
        'Co-app Income':     f"₹{data['CoapplicantIncome']:,.0f}",
        'Loan Amount':       f"₹{data['LoanAmount'] * 1000:,.0f}",
        'Loan Term':         f"{int(data['Loan_Amount_Term'])} months",
        'Credit History':    'Good ✅' if data['Credit_History'] == 1 else 'Poor ⚠️',
        'Property Area':     ['Rural', 'Semiurban', 'Urban'][data['Property_Area']],
    }
