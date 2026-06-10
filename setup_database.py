"""
Database Setup Script — LoanSaathi AI
Run this ONCE to create MySQL database and tables.

Requirements:
- MySQL Workbench installed
- MySQL Server running
- Update DB_CONFIG below with your password
"""

import mysql.connector
from mysql.connector import Error

# ── CONFIG — Apna password yahan daalo ──────────────────────
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',   # <-- YAHAN APNA PASSWORD DAALO
}

DB_NAME = 'loansaathi_db'

# ── SQL Queries ───────────────────────────────────────────────
CREATE_DB = f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"

USE_DB = f"USE {DB_NAME}"

CREATE_APPLICANTS = """
CREATE TABLE IF NOT EXISTS applicants (
    id                          INT AUTO_INCREMENT PRIMARY KEY,
    applicant_name              VARCHAR(100) DEFAULT 'N/A',
    business_type               VARCHAR(100),
    locality                    VARCHAR(50),
    business_age_years          FLOAT,
    monthly_avg_revenue         INT,
    monthly_expense_ratio       FLOAT,
    monthly_net_income          INT,
    cash_flow_consistency_score FLOAT,
    inventory_turnover_monthly  FLOAT,
    seasonal_dip_percent        FLOAT,
    utility_bill_delays_per_year INT,
    mobile_recharge_freq_monthly INT,
    num_employees               INT,
    has_gst                     TINYINT(1),
    loan_amount_requested       INT,
    loan_to_income_ratio        FLOAT,
    previous_loan_taken         TINYINT(1),
    previous_loan_defaulted     TINYINT(1),
    created_at                  DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    applicant_id        INT,
    default_probability FLOAT,
    decision            VARCHAR(20),
    risk_level          VARCHAR(20),
    recommendation      TEXT,
    predicted_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (applicant_id) REFERENCES applicants(id)
)
"""

def setup_database():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute(CREATE_DB)
        print(f"✅ Database '{DB_NAME}' created/verified")

        cursor.execute(USE_DB)

        cursor.execute(CREATE_APPLICANTS)
        print("✅ Table 'applicants' created/verified")

        cursor.execute(CREATE_PREDICTIONS)
        print("✅ Table 'predictions' created/verified")

        conn.commit()
        cursor.close()
        conn.close()

        print("\n🎉 Database setup complete!")
        print(f"   MySQL Workbench mein '{DB_NAME}' database dikhega.")

    except Error as e:
        print(f"\n❌ Error: {e}")
        print("\n👉 Fix karo:")
        print("   1. MySQL Server chal raha hai? (Workbench mein check karo)")
        print("   2. DB_CONFIG mein sahi password daala?")
        print("   3. MySQL service start karo: services.msc → MySQL → Start")


if __name__ == '__main__':
    setup_database()
