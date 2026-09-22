"""Account master-data generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_accounts(customers: pd.DataFrame, n_accounts: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate at least one account per customer, then allocate additional accounts."""
    n_customers = len(customers)
    if n_accounts < n_customers:
        raise ValueError("n_accounts must be at least the number of customers")

    customer_indices = np.concatenate([
        np.arange(n_customers),
        rng.choice(n_customers, n_accounts - n_customers, replace=True),
    ])
    rng.shuffle(customer_indices)
    owner = customers.iloc[customer_indices].reset_index(drop=True)
    account_type = rng.choice(["CHEQUE", "SAVINGS", "CREDIT"], n_accounts, p=[0.57, 0.29, 0.14])
    multiplier = np.where(account_type == "CREDIT", 0.35, np.where(account_type == "SAVINGS", 1.8, 0.8))
    opening_balance = np.maximum(owner["monthly_income"].to_numpy() * multiplier * rng.lognormal(0, 0.65, n_accounts), 500).round(2)
    status = rng.choice(["ACTIVE", "DORMANT"], n_accounts, p=[0.985, 0.015])

    return pd.DataFrame({
        "account_id": [f"A{i:08d}" for i in range(1, n_accounts + 1)],
        "customer_id": owner["customer_id"].to_numpy(),
        "account_type": account_type,
        "opening_balance": opening_balance,
        "account_status": status,
        "currency": "ZAR",
    })
