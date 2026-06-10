"""
Database Helper — LoanSaathi AI
Handles all MySQL operations
"""

import mysql.connector
from mysql.connector import Error
import pandas as pd
import streamlit as st

# ── CONFIG ────────────────────────────────────────────────────
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',   # <-- APNA PASSWORD DAALO
    'database': 'loansaathi_db'
}


def get_connection():
    """MySQL connection return karta hai."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        st.error(f"❌ Database connection failed: {e}")
        return None


def save_applicant(data: dict) -> int:
    """
    Applicant data save karo, applicant_id return karo.
    data dict mein 'applicant_name' optional hai.
    """
    conn = get_connection()
    if not conn:
        return -1

    query = """
        INSERT INTO applicants (
            applicant_name, business_type, locality,
            business_age_years, monthly_avg_revenue, monthly_expense_ratio,
            monthly_net_income, cash_flow_consistency_score,
            inventory_turnover_monthly, seasonal_dip_percent,
            utility_bill_delays_per_year, mobile_recharge_freq_monthly,
            num_employees, has_gst, loan_amount_requested,
            loan_to_income_ratio, previous_loan_taken, previous_loan_defaulted
        ) VALUES (
            %(applicant_name)s, %(business_type)s, %(locality)s,
            %(business_age_years)s, %(monthly_avg_revenue)s, %(monthly_expense_ratio)s,
            %(monthly_net_income)s, %(cash_flow_consistency_score)s,
            %(inventory_turnover_monthly)s, %(seasonal_dip_percent)s,
            %(utility_bill_delays_per_year)s, %(mobile_recharge_freq_monthly)s,
            %(num_employees)s, %(has_gst)s, %(loan_amount_requested)s,
            %(loan_to_income_ratio)s, %(previous_loan_taken)s, %(previous_loan_defaulted)s
        )
    """
    try:
        cursor = conn.cursor()
        data = {k: int(v) if hasattr(v, 'item') else v for k, v in data.items()}
        cursor.execute(query, data)
        conn.commit()
        applicant_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return applicant_id
    except Error as e:
        st.error(f"❌ Save applicant failed: {e}")
        conn.close()
        return -1


def save_prediction(applicant_id: int, prob: float,
                    decision: str, risk_level: str, recommendation: str):
    """Prediction result save karo."""
    conn = get_connection()
    if not conn:
        return

    query = """
        INSERT INTO predictions
            (applicant_id, default_probability, decision, risk_level, recommendation)
        VALUES (%s, %s, %s, %s, %s)
    """
    try:
        cursor = conn.cursor()
        prob = float(prob)
        cursor.execute(query, (applicant_id, round(prob, 4),
                               decision, risk_level, recommendation))
        conn.commit()
        cursor.close()
        conn.close()
    except Error as e:
        st.error(f"❌ Save prediction failed: {e}")
        conn.close()


def fetch_all_predictions() -> pd.DataFrame:
    """Saari predictions history fetch karo."""
    conn = get_connection()
    if not conn:
        return pd.DataFrame()

    query = """
        SELECT
            p.id            AS pred_id,
            a.applicant_name,
            a.business_type,
            a.locality,
            a.loan_amount_requested,
            p.default_probability,
            p.decision,
            p.risk_level,
            p.recommendation,
            p.predicted_at
        FROM predictions p
        JOIN applicants a ON p.applicant_id = a.id
        ORDER BY p.predicted_at DESC
    """
    try:
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Error as e:
        st.error(f"❌ Fetch failed: {e}")
        conn.close()
        return pd.DataFrame()


def fetch_stats() -> dict:
    """Dashboard ke liye summary stats."""
    conn = get_connection()
    if not conn:
        return {}

    try:
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT COUNT(*) AS total FROM predictions")
        total = cursor.fetchone()['total']

        cursor.execute("SELECT COUNT(*) AS approved FROM predictions WHERE decision='APPROVED'")
        approved = cursor.fetchone()['approved']

        cursor.execute("SELECT AVG(default_probability) AS avg_prob FROM predictions")
        avg_prob = cursor.fetchone()['avg_prob'] or 0

        cursor.execute("SELECT AVG(loan_amount_requested) AS avg_loan FROM applicants")
        avg_loan = cursor.fetchone()['avg_loan'] or 0

        cursor.close()
        conn.close()

        return {
            'total': total,
            'approved': approved,
            'rejected': total - approved,
            'approval_rate': round(approved / total * 100, 1) if total > 0 else 0,
            'avg_prob': round(avg_prob * 100, 1),
            'avg_loan': int(avg_loan)
        }
    except Error as e:
        st.error(f"❌ Stats fetch failed: {e}")
        conn.close()
        return {}
