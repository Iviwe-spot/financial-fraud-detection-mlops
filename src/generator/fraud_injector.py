from __future__ import annotations

import numpy as np
import pandas as pd


FRAUD_TYPE_ACCOUNT_TAKEOVER = "ACCOUNT_TAKEOVER"


def inject_account_takeover(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    customers: pd.DataFrame,
    n_scenarios: int = 100,
    min_transactions: int = 2,
    max_transactions: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Inject account-takeover fraud into legitimate transactions."""

    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be greater than zero")

    if min_transactions <= 0:
        raise ValueError("min_transactions must be greater than zero")

    if max_transactions < min_transactions:
        raise ValueError(
            "max_transactions must be greater than or equal to min_transactions"
        )

    required_columns = {
        "transaction_id",
        "account_id",
        "customer_id",
        "timestamp",
        "amount",
        "merchant",
        "merchant_category",
        "transaction_type",
        "channel",
        "device_id",
        "province",
        "country",
        "is_international",
        "direction",
        "balance_before",
        "balance_after",
    }

    missing_columns = required_columns - set(transactions.columns)

    if missing_columns:
        raise ValueError(
            f"Transactions DataFrame is missing columns: "
            f"{sorted(missing_columns)}"
        )

    rng = np.random.default_rng(seed)

    baseline = transactions.copy(deep=True)
    account_master = accounts.copy(deep=True)
    customer_master = customers.copy(deep=True)

    # Label all original transactions as legitimate.
    baseline["is_fraud"] = 0
    baseline["fraud_type"] = None
    baseline["fraud_scenario_id"] = None

    active_accounts = account_master[
        account_master["account_status"] == "ACTIVE"
    ].copy()

    eligible_accounts = active_accounts[
        active_accounts["account_id"].isin(baseline["account_id"])
    ].copy()

    if n_scenarios > len(eligible_accounts):
        raise ValueError(
            "n_scenarios cannot exceed the number of eligible active accounts"
        )

    victim_indices = rng.choice(
        eligible_accounts.index.to_numpy(),
        size=n_scenarios,
        replace=False,
    )

    victims = eligible_accounts.loc[victim_indices].reset_index(drop=True)

    fraudulent_rows = []
    global_max_timestamp = pd.to_datetime(baseline["timestamp"]).max()

    for scenario_number, victim in victims.iterrows():
        account_id = victim["account_id"]
        customer_id = victim["customer_id"]

        customer_rows = customer_master[
            customer_master["customer_id"] == customer_id
        ]

        if customer_rows.empty:
            continue

        customer = customer_rows.iloc[0]

        customer_history = baseline[
            baseline["customer_id"] == customer_id
        ].sort_values("timestamp")

        account_history = baseline[
            baseline["account_id"] == account_id
        ].sort_values("timestamp")

        if customer_history.empty or account_history.empty:
            continue

        scenario_id = f"ATO_{scenario_number + 1:06d}"
        attacker_device = f"ATO_DEVICE_{scenario_number + 1:06d}"

        number_of_transactions = int(
            rng.integers(min_transactions, max_transactions + 1)
        )

        last_timestamp = pd.Timestamp(customer_history["timestamp"].max())

        start_timestamp = last_timestamp + pd.Timedelta(
            days=int(rng.integers(1, 8))
        )

        if start_timestamp > global_max_timestamp:
            start_timestamp = global_max_timestamp - pd.Timedelta(
                days=int(rng.integers(0, 3))
            )

        usual_start = int(customer["usual_start_hour"])
        usual_end = int(customer["usual_end_hour"])

        unusual_hours = [
            hour
            for hour in range(24)
            if not _hour_is_usual(hour, usual_start, usual_end)
        ]

        if unusual_hours:
            fraud_hour = int(rng.choice(unusual_hours))
        else:
            fraud_hour = int(rng.integers(0, 24))

        start_timestamp = start_timestamp.replace(
            hour=fraud_hour,
            minute=int(rng.integers(0, 60)),
            second=int(rng.integers(0, 60)),
        )

        customer_average = max(float(customer["avg_transaction"]), 1.0)
        customer_std = max(float(customer["std_transaction"]), 1.0)

        current_balance = float(
            account_history.iloc[-1]["balance_after"]
        )

        previous_timestamp = start_timestamp

        for sequence_number in range(number_of_transactions):
            if current_balance <= 0.01:
                break

            if sequence_number == 0:
                fraud_timestamp = start_timestamp
            else:
                fraud_timestamp = previous_timestamp + pd.Timedelta(
                    minutes=int(rng.integers(1, 6))
                )

            previous_timestamp = fraud_timestamp

            abnormal_multiplier = float(rng.uniform(4.0, 9.0))

            proposed_amount = (
                customer_average * abnormal_multiplier
                + abs(rng.normal(0.0, customer_std))
            )

            amount = min(proposed_amount, current_balance)
            amount = round(max(amount, 0.01), 2)

            balance_before = round(current_balance, 2)
            balance_after = round(
                max(balance_before - amount, 0.0),
                2,
            )

            fraudulent_rows.append(
                {
                    "transaction_id": (
                        f"FRAUD_ATO_{scenario_number + 1:06d}_"
                        f"{sequence_number + 1:02d}"
                    ),
                    "account_id": account_id,
                    "customer_id": customer_id,
                    "timestamp": fraud_timestamp,
                    "amount": amount,
                    "merchant": "UNKNOWN_ONLINE_MERCHANT",
                    "merchant_category": "ELECTRONICS",
                    "transaction_type": "PURCHASE",
                    "channel": "ONLINE",
                    "device_id": attacker_device,
                    "province": customer["province"],
                    "country": "South Africa",
                    "is_international": False,
                    "direction": "DEBIT",
                    "balance_before": balance_before,
                    "balance_after": balance_after,
                    "is_fraud": 1,
                    "fraud_type": FRAUD_TYPE_ACCOUNT_TAKEOVER,
                    "fraud_scenario_id": scenario_id,
                }
            )

            current_balance = balance_after

    fraud = pd.DataFrame(fraudulent_rows)

    if fraud.empty:
        return baseline

    result = pd.concat(
        [baseline, fraud],
        ignore_index=True,
    )

    return result.sort_values(
        ["timestamp", "transaction_id"]
    ).reset_index(drop=True)


def _hour_is_usual(
    hour: int,
    usual_start: int,
    usual_end: int,
) -> bool:
    """Return whether an hour falls within the normal activity window."""

    if usual_start <= usual_end:
        return usual_start <= hour <= usual_end

    return hour >= usual_start or hour <= usual_end