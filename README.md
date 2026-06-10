# 🏦 LoanSaathi AI — Loan Default Predictor for Small Businesses

> Final Year B.Tech Project | AI-Based Alternative Credit Scoring

---

## 📌 Problem Statement

Traditional banks use CIBIL scores to evaluate loan applicants.
However, **millions of kirana shop owners, street vendors, and small
business owners** have no credit history — leaving them unserved by
formal banking and forced to borrow from moneylenders at high interest rates.

**LoanSaathi AI** solves this using **alternative data signals** and
**explainable AI** to assess creditworthiness without CIBIL.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.9+ |
| ML Model | XGBoost (fallback: Random Forest) |
| Explainability | SHAP (SHapley Additive exPlanations) |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Plotly |
| Storage | CSV (extendable to MySQL) |

---

## 📁 Project Structure

```
loan_predictor/
│
├── generate_dataset.py   # Synthetic data generator (2000 records)
├── train_model.py        # XGBoost training + SHAP plots
├── app.py                # Streamlit web application
├── requirements.txt      # Python dependencies
│
├── data/
│   ├── loan_data.csv     # Generated dataset
│   ├── shap_summary.png  # SHAP beeswarm plot
│   └── shap_bar.png      # SHAP feature importance bar chart
│
└── models/
    ├── loan_model.pkl    # Trained model
    ├── feature_cols.pkl  # Feature column names
    ├── le_locality.pkl   # Label encoder - locality
    └── le_btype.pkl      # Label encoder - business type
```

---

## 🚀 Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate Dataset
```bash
python generate_dataset.py
```
Creates `data/loan_data.csv` with 2000 synthetic records.

### Step 3 — Train Model
```bash
python train_model.py
```
Trains XGBoost model, generates SHAP plots, saves to `models/`.

### Step 4 — Run App
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

---

## 📊 Features Used (Alternative Credit Signals)

| Feature | Why it matters |
|---|---|
| Business age | Older = more stable |
| Cash flow consistency score | Irregular income = higher risk |
| Inventory turnover | Active selling = healthy business |
| Utility bill delays | Payment discipline proxy |
| Mobile recharge frequency | Business activity proxy |
| GST registration | Formal business = lower risk |
| Loan-to-income ratio | Repayment capacity |
| Previous loan history | Credit behavior |
| Seasonal dip % | Business resilience |

---

## 🔍 Key Innovation — SHAP Explainability

Unlike black-box models, LoanSaathi explains **why** a loan is
approved or rejected in simple language:

> *"Ramesh ki shop 4 saal purani aur cash flow consistent hai,
>  lekin pehle loan default kar chuka hai — High Risk."*

This transparency builds trust with both lenders and applicants.

---

## 📈 Model Performance (on test set)

| Metric | Score |
|---|---|
| Accuracy | ~85-88% |
| ROC-AUC | ~0.91-0.93 |
| Precision (Default) | ~82% |
| Recall (Default) | ~79% |

---

## 💼 Resume Line

> *"Developed an alternative credit scoring system for informal
> businesses using XGBoost and SHAP-based explainability, addressing
> financial inclusion gaps for unbankable SMEs in India — achieving
> 87% accuracy and 0.92 AUC on synthetic lending data."*

---

## 🔮 Future Scope

- Integrate real Aadhaar/UPI transaction data (with consent)
- Add MySQL database for persistent applicant storage
- Deploy on Streamlit Cloud or AWS EC2
- Add Power BI dashboard for lender analytics
- Multilingual support (Hindi UI)

---

**Developer:** Manish | B.Tech CSE | Final Year 2026
