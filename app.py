"""
Streamlit App — AI Loan Default Predictor for Small Businesses
MySQL Integrated Version
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="LoanSaathi AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    .main { background: #f7f9fc; }
    .stButton>button {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-size: 16px; font-weight: 600;
        width: 100%; transition: 0.3s;
    }
    .stButton>button:hover { opacity: 0.9; transform: translateY(-1px); }
    .result-approve {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border-left: 6px solid #2e7d32; border-radius: 12px;
        padding: 1.5rem; margin: 1rem 0;
    }
    .result-reject {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        border-left: 6px solid #c62828; border-radius: 12px;
        padding: 1.5rem; margin: 1rem 0;
    }
    .metric-card {
        background: white; border-radius: 12px;
        padding: 1rem; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .header-title {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .db-badge {
        background: #e3f2fd; border-radius: 6px;
        padding: 2px 10px; font-size: 12px; color: #1565c0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── DB Import (graceful fallback if MySQL not setup) ──────────
try:
    from utils.db_helper import (save_applicant, save_prediction,
                                  fetch_all_predictions, fetch_stats,
                                  get_connection)
    conn_test = get_connection()
    DB_AVAILABLE = conn_test is not None
    if conn_test:
        conn_test.close()
except Exception:
    DB_AVAILABLE = False


# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model       = pickle.load(open('models/loan_model.pkl', 'rb'))
        feat_cols   = pickle.load(open('models/feature_cols.pkl', 'rb'))
        le_locality = pickle.load(open('models/le_locality.pkl', 'rb'))
        le_btype    = pickle.load(open('models/le_btype.pkl', 'rb'))
        return model, feat_cols, le_locality, le_btype
    except FileNotFoundError:
        return None, None, None, None


model, FEATURE_COLS, le_locality, le_btype = load_model()


# ── Helpers ───────────────────────────────────────────────────
def get_risk_factors(d):
    pos, neg = [], []
    if d['business_age_years'] >= 3:
        pos.append(f"✅ Business {d['business_age_years']} saal purana — stable track record")
    else:
        neg.append(f"⚠️ Business sirf {d['business_age_years']} saal purana")
    if d['cash_flow_consistency_score'] >= 70:
        pos.append("✅ Cash flow consistent hai")
    else:
        neg.append("⚠️ Cash flow irregular — income unpredictable")
    if d['utility_bill_delays_per_year'] <= 2:
        pos.append("✅ Utility bills timely bhare hain")
    else:
        neg.append(f"⚠️ {d['utility_bill_delays_per_year']} baar utility bill late")
    if d['has_gst']:
        pos.append("✅ GST registered — formal business credibility")
    else:
        neg.append("⚠️ GST nahi — informal business")
    if d['loan_to_income_ratio'] > 2:
        neg.append(f"⚠️ Loan {d['loan_to_income_ratio']:.1f}x annual income — overleverage")
    else:
        pos.append(f"✅ Loan-to-income ratio manageable ({d['loan_to_income_ratio']:.1f}x)")
    if d['previous_loan_defaulted']:
        neg.append("🔴 Pehle loan default kar chuka hai — major red flag")
    elif d['previous_loan_taken']:
        pos.append("✅ Pehle loan liya aur repay kiya — good history")
    if d['inventory_turnover_monthly'] >= 4:
        pos.append("✅ Inventory fast move ho rahi hai — active business")
    else:
        neg.append("⚠️ Inventory slow turnover")
    return pos, neg


# ── Header ───────────────────────────────────────────────────
st.markdown('<p class="header-title">🏦 LoanSaathi AI</p>', unsafe_allow_html=True)

col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.markdown("**Alternative Credit Scoring for Kirana & Small Businesses** — No CIBIL Required")
with col_h2:
    if DB_AVAILABLE:
        st.markdown('<span class="db-badge">🟢 MySQL Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="db-badge" style="background:#fff3e0;color:#e65100;">🟡 CSV Mode</span>',
                    unsafe_allow_html=True)

st.divider()

if model is None:
    st.error("⚠️ Model not trained yet! Run these commands first:")
    st.code("python generate_dataset.py\npython train_model.py", language="bash")
    st.stop()


# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📋 History", "📊 Dashboard"])


# ════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════
with tab1:
    # Sidebar inputs
    st.sidebar.markdown("## 📝 Applicant Details")

    with st.sidebar:
        applicant_name = st.text_input("Applicant Name (optional)", placeholder="e.g. Ramesh Kumar")
        btype    = st.selectbox("Business Type", [
            'Kirana Store','Tea Stall','Vegetable Vendor','Tailor',
            'Electrical Shop','Medical Store','Dairy Shop','Bakery',
            'Mobile Repair','Stationery Shop'])
        locality = st.selectbox("Locality", ['Urban','Semi-Urban','Rural'])
        biz_age  = st.slider("Business Age (years)", 0.5, 20.0, 3.0, 0.5)
        monthly_rev   = st.number_input("Monthly Revenue (₹)", 15000, 500000, 80000, 5000)
        expense_ratio = st.slider("Monthly Expense Ratio", 0.3, 0.9, 0.55, 0.01)
        cf_score      = st.slider("Cash Flow Consistency Score", 30, 100, 65)
        inv_turnover  = st.slider("Inventory Turnover (times/month)", 1.0, 12.0, 4.0, 0.5)
        seasonal_dip  = st.slider("Seasonal Dip (%)", 0, 60, 20)
        utility_delays = st.number_input("Utility Bill Delays (per year)", 0, 12, 1)
        recharge_freq  = st.slider("Mobile Recharge Freq (per month)", 1, 8, 3)
        employees  = st.number_input("Number of Employees", 1, 20, 2)
        has_gst    = st.toggle("GST Registered?", value=False)
        loan_amount = st.number_input("Loan Amount Requested (₹)", 25000, 500000, 100000, 5000)
        prev_loan  = st.toggle("Previous Loan Taken?", value=False)
        prev_default = False
        if prev_loan:
            prev_default = st.toggle("Defaulted on previous loan?", value=False)

        predict_btn = st.button("🔍 Predict Loan Risk")

    col1, col2 = st.columns([3, 2])

    if predict_btn:
        monthly_net = int(monthly_rev * (1 - expense_ratio))
        lti_ratio   = round(loan_amount / (monthly_net * 12), 2) if monthly_net > 0 else 99

        try:
            loc_enc   = le_locality.transform([locality])[0]
            btype_enc = le_btype.transform([btype])[0]
        except Exception:
            loc_enc, btype_enc = 0, 0

        applicant = {
            'applicant_name'              : applicant_name or 'N/A',
            'business_type'               : btype,
            'locality'                    : locality,
            'business_age_years'          : biz_age,
            'monthly_avg_revenue'         : monthly_rev,
            'monthly_expense_ratio'       : expense_ratio,
            'monthly_net_income'          : monthly_net,
            'cash_flow_consistency_score' : cf_score,
            'inventory_turnover_monthly'  : inv_turnover,
            'seasonal_dip_percent'        : seasonal_dip,
            'utility_bill_delays_per_year': utility_delays,
            'mobile_recharge_freq_monthly': recharge_freq,
            'num_employees'               : employees,
            'has_gst'                     : int(has_gst),
            'loan_amount_requested'       : loan_amount,
            'loan_to_income_ratio'        : lti_ratio,
            'previous_loan_taken'         : int(prev_loan),
            'previous_loan_defaulted'     : int(prev_default),
            'locality_encoded'            : loc_enc,
            'business_type_encoded'       : btype_enc,
        }

        X_input      = pd.DataFrame([applicant])[FEATURE_COLS]
        prob_default = model.predict_proba(X_input)[0][1]
        decision     = prob_default > 0.5
        risk_level   = ("Low Risk" if prob_default < 0.35
                        else "Medium Risk" if prob_default < 0.65
                        else "High Risk")
        decision_str = "REJECTED" if decision else "APPROVED"

        if not decision:
            rec = (f"Full loan ₹{loan_amount:,} approve karein." if prob_default < 0.25
                   else f"₹{int(loan_amount*0.8):,} approve karein with quarterly review.")
        else:
            rec = ("Reject karein ya collateral maangein." if prob_default > 0.75
                   else f"₹{int(loan_amount*0.5):,} consider karein with guarantor.")

        # ── Save to MySQL ─────────────────────────────────────
        if DB_AVAILABLE:
            app_id = save_applicant(applicant)
            if app_id > 0:
                save_prediction(app_id, prob_default, decision_str, risk_level, rec)
                st.toast("✅ Prediction saved to MySQL database!", icon="💾")

        with col1:
            if not decision:
                st.markdown(f"""
                <div class="result-approve">
                    <h2>✅ LOAN APPROVED</h2>
                    <h4>Default Probability: {prob_default*100:.1f}%</h4>
                    <p>Low credit risk based on alternative data signals.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-reject">
                    <h2>❌ LOAN REJECTED</h2>
                    <h4>Default Probability: {prob_default*100:.1f}%</h4>
                    <p>High credit risk detected.</p>
                </div>""", unsafe_allow_html=True)

            # Risk meter
            st.markdown("#### 📊 Risk Score Meter")
            fig, ax = plt.subplots(figsize=(8, 1.2))
            ax.barh(['Risk'], [prob_default],
                    color='#e53935' if decision else '#43a047', height=0.5)
            ax.barh(['Risk'], [1 - prob_default], left=[prob_default],
                    color='#e0e0e0', height=0.5)
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color='#333', linewidth=1.5, linestyle='--', alpha=0.5)
            ax.text(prob_default / 2, 0, f"{prob_default*100:.0f}%",
                    ha='center', va='center', color='white', fontweight='bold', fontsize=12)
            ax.set_yticks([])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0%', '25%', '50% (threshold)', '75%', '100%'])
            for spine in ax.spines.values():
                spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            pos, neg = get_risk_factors(applicant)
            st.markdown("#### 📋 AI Explanation")
            if pos:
                st.markdown("**Positive Signals:**")
                for p in pos: st.markdown(f"&nbsp;&nbsp;{p}")
            if neg:
                st.markdown("**Risk Signals:**")
                for n in neg: st.markdown(f"&nbsp;&nbsp;{n}")

        with col2:
            name_display = applicant_name if applicant_name else "Applicant"
            st.markdown(f"""
            <div class="metric-card">
                <h4>{name_display}</h4>
                <p>{btype} | {locality} | {biz_age} yrs</p>
                <p>Net Income: <strong>₹{monthly_net:,}/month</strong></p>
                <p>Loan: <strong>₹{loan_amount:,}</strong></p>
                <p>LTI Ratio: <strong>{lti_ratio:.1f}x</strong></p>
                <p>GST: <strong>{'Yes ✅' if has_gst else 'No ❌'}</strong></p>
            </div>""", unsafe_allow_html=True)

            st.markdown("#### 💡 Recommendation")
            st.info(rec)

            if os.path.exists('data/shap_bar.png'):
                st.markdown("#### 🔍 Feature Importance")
                st.image('data/shap_bar.png', use_column_width=True)
    else:
        with col1:
            st.info("👈 Sidebar mein details fill karein aur Predict button dabayein.")
        with col2:
            if os.path.exists('data/shap_summary.png'):
                st.image('data/shap_summary.png', use_column_width=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — HISTORY
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📋 Prediction History")

    if not DB_AVAILABLE:
        st.warning("⚠️ MySQL connected nahi hai. History tab ke liye database setup karo.")
        st.code("python setup_database.py", language="bash")
    else:
        df_hist = fetch_all_predictions()
        if df_hist.empty:
            st.info("Abhi tak koi prediction save nahi hui. Pehle Predict tab use karo.")
        else:
            # Filters
            fc1, fc2 = st.columns(2)
            with fc1:
                filter_decision = st.selectbox("Filter by Decision",
                                               ["All", "APPROVED", "REJECTED"])
            with fc2:
                filter_risk = st.selectbox("Filter by Risk Level",
                                           ["All", "Low Risk", "Medium Risk", "High Risk"])

            if filter_decision != "All":
                df_hist = df_hist[df_hist['decision'] == filter_decision]
            if filter_risk != "All":
                df_hist = df_hist[df_hist['risk_level'] == filter_risk]

            st.markdown(f"**{len(df_hist)} records found**")

            # Color coding
            def color_decision(val):
                return ('background-color: #c8e6c9' if val == 'APPROVED'
                        else 'background-color: #ffcdd2')

            display_cols = ['applicant_name', 'business_type', 'locality',
                            'loan_amount_requested', 'default_probability',
                            'decision', 'risk_level', 'predicted_at']

            st.dataframe(
                df_hist[display_cols].style.map(
                    color_decision, subset=['decision']
                ),
                use_container_width=True, height=400
            )

            # Download button
            csv = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download History as CSV", csv,
                               "prediction_history.csv", "text/csv")


# ════════════════════════════════════════════════════════════
# TAB 3 — DASHBOARD
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Lender Analytics Dashboard")

    if not DB_AVAILABLE:
        st.warning("⚠️ MySQL connected nahi hai. Dashboard ke liye database setup karo.")
    else:
        stats = fetch_stats()
        if not stats or stats.get('total', 0) == 0:
            st.info("Koi data nahi hai abhi. Kuch predictions karo pehle.")
        else:
            # KPI cards
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Total Applications", stats['total'])
            with k2:
                st.metric("Approved", stats['approved'],
                          delta=f"{stats['approval_rate']}% rate")
            with k3:
                st.metric("Rejected", stats['rejected'])
            with k4:
                st.metric("Avg Default Prob", f"{stats['avg_prob']}%")

            st.divider()

            df_all = fetch_all_predictions()
            if not df_all.empty:
                ch1, ch2 = st.columns(2)

                with ch1:
                    st.markdown("#### Approve vs Reject")
                    counts = df_all['decision'].value_counts()
                    fig, ax = plt.subplots(figsize=(5, 4))
                    colors = ['#43a047', '#e53935']
                    ax.pie(counts.values, labels=counts.index,
                           colors=colors, autopct='%1.1f%%',
                           startangle=90, textprops={'fontsize': 12})
                    ax.set_title("Decision Distribution")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                with ch2:
                    st.markdown("#### Risk Level Distribution")
                    risk_counts = df_all['risk_level'].value_counts()
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    risk_colors = {'Low Risk': '#43a047',
                                   'Medium Risk': '#fb8c00',
                                   'High Risk': '#e53935'}
                    bars = ax2.bar(risk_counts.index, risk_counts.values,
                                   color=[risk_colors.get(r, '#999')
                                          for r in risk_counts.index])
                    ax2.set_ylabel("Count")
                    ax2.set_title("Risk Level Breakdown")
                    for bar in bars:
                        ax2.text(bar.get_x() + bar.get_width()/2,
                                 bar.get_height() + 0.3,
                                 str(int(bar.get_height())),
                                 ha='center', va='bottom', fontweight='bold')
                    for spine in ['top', 'right']:
                        ax2.spines[spine].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()

                # Avg loan by business type
                st.markdown("#### Avg Loan Amount by Business Type")
                avg_loan = (df_all.groupby('business_type')['loan_amount_requested']
                            .mean().sort_values(ascending=True))
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                ax3.barh(avg_loan.index, avg_loan.values, color='#1a73e8', alpha=0.8)
                ax3.set_xlabel("Avg Loan Amount (₹)")
                for spine in ['top', 'right']:
                    ax3.spines[spine].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close()


# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>LoanSaathi AI — Final Year B.Tech Project | "
    "XGBoost + SHAP + MySQL | Manish 2026</small></center>",
    unsafe_allow_html=True
)
