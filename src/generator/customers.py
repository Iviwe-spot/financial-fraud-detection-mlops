"""Customer master-data generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import OCCUPATIONS, OCCUPATION_WEIGHTS, PROVINCES, PROVINCE_WEIGHTS


def generate_customers(n_customers: int, rng: np.random.Generator, as_of_date: str) -> pd.DataFrame:
    """Generate customers and latent-but-legitimate behavioural profiles."""
    if n_customers <= 0:
        raise ValueError("n_customers must be positive")

    age = np.clip(np.rint(rng.normal(41, 14, n_customers)), 18, 85).astype(int)
    province = rng.choice(PROVINCES, n_customers, p=PROVINCE_WEIGHTS)
    occupation = rng.choice(OCCUPATIONS, n_customers, p=OCCUPATION_WEIGHTS)
    income = np.clip(rng.lognormal(np.log(28_000), 0.72, n_customers), 3_500, 350_000).round(2)
    account_age_days = np.clip(rng.gamma(3.0, 650, n_customers).astype(int) + 30, 30, 8_000)
    risk_segment = pd.cut(income, bins=[0, 18_000, 65_000, np.inf], labels=["MASS", "MIDDLE", "AFFLUENT"]).astype(str)

    avg_ratio = np.clip(rng.lognormal(np.log(0.025), 0.48, n_customers), 0.004, 0.12)
    avg_transaction = np.clip(income * avg_ratio, 40, 15_000).round(2)
    std_transaction = np.maximum(avg_transaction * rng.uniform(0.35, 1.10, n_customers), 20).round(2)
    preferred_channel = rng.choice(["CARD", "ONLINE", "EFT", "ATM", "MOBILE"], n_customers, p=[0.46, 0.17, 0.13, 0.08, 0.16])
    international_rate = np.clip(rng.beta(0.7, 18, n_customers), 0, 0.35).round(4)
    usual_start_hour = np.clip(np.rint(rng.normal(6.5, 1.2, n_customers)), 4, 10).astype(int)
    usual_end_hour = np.clip(np.rint(rng.normal(21.0, 1.4, n_customers)), 17, 23).astype(int)

    return pd.DataFrame({
        "customer_id": [f"C{i:07d}" for i in range(1, n_customers + 1)],
        "age": age,
        "province": province,
        "monthly_income": income,
        "occupation": occupation,
        "account_age_days": account_age_days,
        "risk_segment": risk_segment,
        "preferred_channel": preferred_channel,
        "avg_transaction": avg_transaction,
        "std_transaction": std_transaction,
        "international_rate": international_rate,
        "usual_start_hour": usual_start_hour,
        "usual_end_hour": usual_end_hour,
        "profile_as_of_date": pd.Timestamp(as_of_date),
    })
