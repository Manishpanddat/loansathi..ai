"""
Synthetic Dataset Generator
AI-Based Loan Default Predictor for Kirana/Small Businesses
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 2000  # number of records

def generate_dataset():
    business_types = ['Kirana Store', 'Tea Stall', 'Vegetable Vendor', 'Tailor',
                      'Electrical Shop', 'Medical Store', 'Dairy Shop', 'Bakery',
                      'Mobile Repair', 'Stationery Shop']

    localities = ['Urban', 'Semi-Urban', 'Rural']

    data = {
        'business_id': [f'BIZ{str(i).zfill(4)}' for i in range(1, N+1)],
        'business_type': np.random.choice(business_types, N),
        'locality': np.random.choice(localities, N, p=[0.5, 0.3, 0.2]),
        'business_age_years': np.round(np.random.exponential(scale=4, size=N).clip(0.5, 20), 1),
        'monthly_avg_revenue': np.random.randint(15000, 300000, N),
        'monthly_expense_ratio': np.round(np.random.uniform(0.4, 0.85, N), 2),
        'cash_flow_consistency_score': np.round(np.random.uniform(30, 100, N), 1),
        'inventory_turnover_monthly': np.round(np.random.uniform(1, 12, N), 1),
        'seasonal_dip_percent': np.round(np.random.uniform(0, 60, N), 1),
        'utility_bill_delays_per_year': np.random.randint(0, 12, N),
        'mobile_recharge_freq_monthly': np.random.randint(1, 8, N),
        'num_employees': np.random.randint(1, 15, N),
        'has_gst': np.random.choice([0, 1], N, p=[0.6, 0.4]),
        'loan_amount_requested': np.random.randint(25000, 500000, N),
        'previous_loan_taken': np.random.choice([0, 1], N, p=[0.45, 0.55]),
        'previous_loan_defaulted': np.zeros(N, dtype=int),
    }

    df = pd.DataFrame(data)

    # Previous default only possible if previous loan taken
    for i in df[df['previous_loan_taken'] == 1].index:
        df.at[i, 'previous_loan_defaulted'] = np.random.choice([0, 1], p=[0.8, 0.2])

    # Monthly net income derived
    df['monthly_net_income'] = (df['monthly_avg_revenue'] * (1 - df['monthly_expense_ratio'])).astype(int)

    # Loan to income ratio
    df['loan_to_income_ratio'] = np.round(df['loan_amount_requested'] / (df['monthly_net_income'] * 12), 2)

    # ---- Generate Target Variable (default = 1) ----
    risk_score = (
        - 0.03 * df['business_age_years']
        - 0.008 * df['cash_flow_consistency_score']
        - 0.02 * df['monthly_net_income'] / 10000
        + 0.015 * df['utility_bill_delays_per_year']
        + 0.012 * df['seasonal_dip_percent']
        + 0.08 * df['loan_to_income_ratio']
        + 0.15 * df['previous_loan_defaulted']
        - 0.05 * df['has_gst']
        - 0.01 * df['inventory_turnover_monthly']
        + np.random.normal(0, 0.1, N)
    )

    prob_default = 1 / (1 + np.exp(-risk_score))
    df['defaulted'] = (prob_default > 0.5).astype(int)

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/loan_data.csv', index=False)
    print(f"✅ Dataset generated: {N} records")
    print(f"   Default rate: {df['defaulted'].mean()*100:.1f}%")
    print(f"   Saved to: data/loan_data.csv")
    return df

if __name__ == '__main__':
    generate_dataset()
