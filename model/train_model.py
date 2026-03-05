"""
Run this file ONCE to train and save the model.
Usage: python model/train_model.py
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# ── Load Dataset ──────────────────────────────────────────
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'loan_data.csv'))

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

# ── Clean Data ────────────────────────────────────────────
# Fill missing numeric with median
for col in df.select_dtypes(include='number').columns:
    df[col].fillna(df[col].median(), inplace=True)

# Fill missing categorical with mode
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Fix Dependents column (has '3+' as string)
df['Dependents'] = df['Dependents'].replace('3+', '3').astype(int)

# ── Encode Categorical ────────────────────────────────────
encode_map = {
    'Gender':        {'Male': 1, 'Female': 0},
    'Married':       {'Yes': 1, 'No': 0},
    'Education':     {'Graduate': 0, 'Not Graduate': 1},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0},
    'Loan_Status':   {'Y': 1, 'N': 0},
}

for col, mapping in encode_map.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# ── Features & Target ─────────────────────────────────────
FEATURES = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

X = df[FEATURES]
y = df['Loan_Status']

# ── Train / Test Split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train XGBoost ─────────────────────────────────────────
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

# ── Evaluate ──────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Save Model & Features ─────────────────────────────────
save_dir = os.path.dirname(__file__)
pickle.dump(model,    open(os.path.join(save_dir, 'model.pkl'),    'wb'))
pickle.dump(FEATURES, open(os.path.join(save_dir, 'features.pkl'), 'wb'))

print("\n✅ Model saved to model/model.pkl")
print("✅ Features saved to model/features.pkl")
