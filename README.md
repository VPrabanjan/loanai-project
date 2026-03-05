# 🏦 LoanAI — AI Loan Risk Assessor with Explainable AI

> An intelligent loan risk assessment system that predicts approval probability using XGBoost ML, extracts features from plain English using spaCy NLP, and explains every decision using SHAP.

---

## 🚀 Features

- 💬 **Natural Language Input** — Describe your application in plain English
- 🧠 **NLP Extraction** — spaCy automatically extracts 11 financial features
- ⚡ **XGBoost Prediction** — ~80% accurate loan approval prediction
- 🔍 **SHAP Explainability** — Know exactly WHY the AI made its decision
- 💡 **Improvement Tips** — Personalized advice if your loan is rejected
- 📊 **Interactive Dashboard** — Beautiful Streamlit UI with dark theme

---

## 📁 Project Structure

```
loan-risk-assessor/
├── app.py                    ← Main Streamlit application
├── requirements.txt          ← Python dependencies
├── README.md
├── model/
│   ├── train_model.py        ← Train & save XGBoost model
│   ├── model.pkl             ← Saved model (generated after training)
│   └── features.pkl          ← Feature list (generated after training)
├── nlp/
│   ├── __init__.py
│   └── extractor.py          ← spaCy NLP feature extractor
├── explainer/
│   ├── __init__.py
│   └── shap_explain.py       ← SHAP explanation generator
└── data/
    └── loan_data.csv         ← Dataset (download from Kaggle)
```

---

## ⚙️ Setup & Run

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 2 — Get the Dataset
Download from Kaggle:
👉 https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

Save as `data/loan_data.csv`

### Step 3 — Train the Model
```bash
python model/train_model.py
```
Expected output: `✅ Model Accuracy: ~80%`

### Step 4 — Run the App
```bash
streamlit run app.py
```

Open browser at: `http://localhost:8501`

---

## 🧪 Example NLP Input

```
I am a married male graduate earning 60000 monthly.
I need a loan of 200000 for 360 months.
I have good credit history and 1 dependent.
I live in an urban area.
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| ML Model | XGBoost |
| NLP | spaCy |
| Explainability | SHAP |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Plotly |

---

## 👤 Author

**Prabanjan V** — Data Scientist  
📧 prabanjanvicram2005@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/VPrabanjan)  
⚙️ [GitHub](https://github.com/VPrabanjan)
