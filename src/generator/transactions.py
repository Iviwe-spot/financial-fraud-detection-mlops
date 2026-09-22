"""Normal, customer-conditioned transaction generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import CATEGORY_WEIGHTS, CHANNELS, MERCHANTS


def _sample_channels(preferred: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    keep_preference = rng.random(len(preferred)) < 0.72
    alternatives = rng.choice(CHANNELS, len(preferred), p=[0.38, 0.19, 0.14, 0.09, 0.20])
    return np.where(keep_preference, preferred, alternatives)


def generate_transactions(
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    n_transactions: int,
    start_date: str,
    end_date: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate normal transactions while maintaining account balance continuity."""
    if n_transactions <= 0:
        raise ValueError("n_transactions must be positive")

    active = accounts.loc[accounts["account_status"] == "ACTIVE"].copy()
    account_idx = rng.choice(len(active), n_transactions, replace=True)
    tx = active.iloc[account_idx][["account_id", "customer_id"]].reset_index(drop=True)
    profiles = customers.set_index("customer_id").loc[tx["customer_id"]].reset_index()

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    day_span = (end.normalize() - start.normalize()).days + 1
    dates = start.normalize() + pd.to_timedelta(rng.integers(0, day_span, n_transactions), unit="D")
    hours = np.array([
        rng.integers(s, e + 1) if rng.random() < 0.94 else rng.integers(0, 24)
        for s, e in zip(profiles["usual_start_hour"], profiles["usual_end_hour"])
    ])
    timestamps = dates + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(rng.integers(0, 3600, n_transactions), unit="s")

    raw_amount = rng.lognormal(
        np.log(np.maximum(profiles["avg_transaction"].to_numpy(), 1)),
        np.clip(np.log1p(profiles["std_transaction"].to_numpy() / profiles["avg_transaction"].to_numpy()), 0.18, 0.9),
    )
    amount = np.clip(raw_amount, 5, 80_000).round(2)
    channel = _sample_channels(profiles["preferred_channel"].to_numpy(), rng)
    category_names = np.array(list(MERCHANTS))
    merchant_category = rng.choice(category_names, n_transactions, p=CATEGORY_WEIGHTS)
    merchant = np.array([rng.choice(MERCHANTS[c]) for c in merchant_category])
    is_international = rng.random(n_transactions) < profiles["international_rate"].to_numpy()
    country = np.where(is_international, rng.choice(["United Kingdom", "United States", "Namibia", "Botswana", "Mauritius"], n_transactions), "South Africa")
    province = np.where(is_international, "INTERNATIONAL", profiles["province"].to_numpy())
    transaction_type = np.select(
        [channel == "ATM", channel == "EFT", merchant_category == "Utilities"],
        ["CASH_WITHDRAWAL", "TRANSFER", "DEBIT_ORDER"],
        default="PURCHASE",
    )

    device_pool = np.array([f"DEV-{i:08d}" for i in range(1, len(customers) + 1)])
    customer_number = profiles["customer_id"].str[1:].astype(int).to_numpy() - 1
    device_id = device_pool[customer_number]
    secondary = rng.random(n_transactions) < 0.12
    device_id = np.where(secondary, np.char.add(device_id.astype(str), "-B"), device_id)

    tx = tx.assign(
        timestamp=timestamps,
        amount=amount,
        merchant=merchant,
        merchant_category=merchant_category,
        transaction_type=transaction_type,
        channel=channel,
        device_id=device_id,
        province=province,
        country=country,
        is_international=is_international,
    ).sort_values(["account_id", "timestamp"], kind="stable").reset_index(drop=True)

    opening = active.set_index("account_id")["opening_balance"]
    signed_amount = np.where(tx["transaction_type"].eq("TRANSFER") & (rng.random(n_transactions) < 0.18), -tx["amount"], tx["amount"])
    tx["amount"] = np.abs(signed_amount).round(2)
    tx["direction"] = np.where(signed_amount < 0, "CREDIT", "DEBIT")
    balance_before = np.empty(n_transactions, dtype=np.int64)
    balance_after = np.empty(n_transactions, dtype=np.int64)
    adjusted_amount = np.maximum(np.rint(tx["amount"].to_numpy() * 100), 1).astype(np.int64)
    directions = tx["direction"].to_numpy(copy=True)
    for account_id, indices in tx.groupby("account_id", sort=False).indices.items():
        balance = int(round(float(opening.loc[account_id]) * 100))
        for idx in indices:
            balance_before[idx] = balance
            if directions[idx] == "DEBIT" and balance <= 1:
                directions[idx] = "CREDIT"
            if directions[idx] == "DEBIT":
                # Work in cents to preserve exact continuity and avoid rounding overdrafts.
                adjusted_amount[idx] = min(adjusted_amount[idx], max(int(balance * 0.85), 1))
                balance -= adjusted_amount[idx]
            else:
                balance += adjusted_amount[idx]
            balance_after[idx] = balance
    tx["amount"] = adjusted_amount / 100
    tx["direction"] = directions
    tx["balance_before"] = balance_before / 100
    tx["balance_after"] = balance_after / 100
    tx.insert(0, "transaction_id", [f"T{i:010d}" for i in range(1, n_transactions + 1)])
    return tx.sort_values("timestamp", kind="stable").reset_index(drop=True)
