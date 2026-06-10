"""
Model Training Script
AI-Based Loan Default Predictor — XGBoost + SHAP Explainability
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try importing xgboost, fallback to RandomForest
try:
    from xgboost import XGBClassifier
    USE_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    USE_XGB = False

try:
    import shap
    USE_SHAP = True
except ImportError:
    USE_SHAP = False


FEATURE_COLS = [
    'business_age_years', 'monthly_avg_revenue', 'monthly_expense_ratio',
    'cash_flow_consistency_score', 'inventory_turnover_monthly',
    'seasonal_dip_percent', 'utility_bill_delays_per_year',
    'mobile_recharge_freq_monthly', 'num_employees', 'has_gst',
    'loan_amount_requested', 'previous_loan_taken', 'previous_loan_defaulted',
    'monthly_net_income', 'loan_to_income_ratio', 'locality_encoded',
    'business_type_encoded'
]


def load_and_preprocess():
    df = pd.read_csv('data/loan_data.csv')

    le_locality = LabelEncoder()
    le_btype = LabelEncoder()
    df['locality_encoded'] = le_locality.fit_transform(df['locality'])
    df['business_type_encoded'] = le_btype.fit_transform(df['business_type'])

    X = df[FEATURE_COLS]
    y = df['defaulted']

    os.makedirs('models', exist_ok=True)
    pickle.dump(le_locality, open('models/le_locality.pkl', 'wb'))
    pickle.dump(le_btype, open('models/le_btype.pkl', 'wb'))

    return X, y, df


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if USE_XGB:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        print("🚀 Training XGBoost model...")
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        print("🚀 Training Random Forest model (XGBoost not installed)...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n📊 Model Performance:")
    print(f"   Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"   ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=['No Default', 'Default']))

    pickle.dump(model, open('models/loan_model.pkl', 'wb'))
    print("✅ Model saved to models/loan_model.pkl")

    return model, X_test, y_test


def generate_shap_plots(model, X_train):
    if not USE_SHAP:
        print("⚠️  SHAP not installed, skipping explainability plots.")
        return None

    print("\n🔍 Generating SHAP explainability plots...")
    os.makedirs('data', exist_ok=True)

    sample = X_train.sample(min(300, len(X_train)), random_state=42)

    if USE_XGB:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(sample)

    # Handle both 1D and 2D shap values
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    # Summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, sample, feature_names=FEATURE_COLS,
                      show=False, plot_size=None)
    plt.title("Feature Impact on Loan Default Prediction", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig('data/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, sample, feature_names=FEATURE_COLS,
                      plot_type='bar', show=False)
    plt.title("Feature Importance (SHAP)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig('data/shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   SHAP plots saved: data/shap_summary.png, data/shap_bar.png")
    return explainer


def get_explanation_text(applicant_data: dict, model, feature_cols) -> str:
    """Generate human-readable Hindi-English explanation for a prediction."""
    row = pd.DataFrame([applicant_data])[feature_cols]
    prob = model.predict_proba(row)[0][1]
    pred = int(prob > 0.5)

    risk_level = "Low Risk 🟢" if prob < 0.35 else ("Medium Risk 🟡" if prob < 0.65 else "High Risk 🔴")

    positives, negatives = [], []

    if applicant_data['business_age_years'] >= 3:
        positives.append(f"Business {applicant_data['business_age_years']} saal purana hai (stable)")
    else:
        negatives.append("Business naya hai (less than 3 years)")

    if applicant_data['cash_flow_consistency_score'] >= 70:
        positives.append("Cash flow consistent hai")
    else:
        negatives.append("Cash flow inconsistent hai")

    if applicant_data['utility_bill_delays_per_year'] <= 2:
        positives.append("Utility bills time pe bhare hain")
    else:
        negatives.append(f"Utility bill delays: {applicant_data['utility_bill_delays_per_year']} baar late")

    if applicant_data['has_gst'] == 1:
        positives.append("GST registered hai (formal business)")

    if applicant_data['loan_to_income_ratio'] > 2:
        negatives.append(f"Loan-to-income ratio high hai ({applicant_data['loan_to_income_ratio']:.1f}x)")

    if applicant_data['previous_loan_defaulted'] == 1:
        negatives.append("Pehle loan default kar chuka hai ⚠️")

    explanation = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 LOAN ASSESSMENT RESULT
━━━━━━━━━━━━━━━━━━━━━━━━━━
Decision  : {'❌ REJECT (Default Risk)' if pred == 1 else '✅ APPROVE (Low Default Risk)'}
Risk Level: {risk_level}
Confidence: {abs(prob - 0.5) * 200:.1f}%

✅ Positive Factors:
{chr(10).join(f'  • {p}' for p in positives) if positives else '  • None significant'}

⚠️  Risk Factors:
{chr(10).join(f'  • {n}' for n in negatives) if negatives else '  • None significant'}
━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return explanation


if __name__ == '__main__':
    print("=" * 50)
    print("  LOAN DEFAULT PREDICTOR — MODEL TRAINING")
    print("=" * 50)

    X, y, df = load_and_preprocess()
    model, X_test, y_test = train_model(X, y)

    X_train_full, _ = train_test_split(X, test_size=0.2, random_state=42)
    generate_shap_plots(model, X_train_full)

    pickle.dump(FEATURE_COLS, open('models/feature_cols.pkl', 'wb'))
    print("\n🎉 Training complete! Run: streamlit run app.py")
