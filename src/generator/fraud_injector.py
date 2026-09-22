from __future__ import annotations

import numpy as np
import pandas as pd


FRAUD_TYPE_ACCOUNT_TAKEOVER = "ACCOUNT_TAKEOVER"
FRAUD_TYPE_CARD_TESTING = "CARD_TESTING"

CARD_TESTING_MIN_AMOUNT = 1.00
CARD_TESTING_MAX_AMOUNT = 20.00


def inject_account_takeover(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    customers: pd.DataFrame,
    n_scenarios: int = 100,
    min_transactions: int = 2,
    max_transactions: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Inject synthetic Account Takeover fraud.

    Each scenario:

    - Selects an existing active account.
    - Uses a new attacker-controlled device.
    - Creates multiple transactions within a short period.
    - Uses unusually large customer-relative amounts.
    - Attempts to operate outside normal customer hours.
    - Adds fraud ground-truth labels.

    The source DataFrames are not modified.
    """

    _validate_parameters(
        n_scenarios=n_scenarios,
        min_transactions=min_transactions,
        max_transactions=max_transactions,
    )

    _validate_transaction_columns(transactions)

    rng = np.random.default_rng(seed)

    baseline = transactions.copy(deep=True)
    account_master = accounts.copy(deep=True)
    customer_master = customers.copy(deep=True)

    # Account Takeover is the first fraud injector.
    # Therefore, all original transactions are legitimate.
    baseline["is_fraud"] = 0
    baseline["fraud_type"] = None
    baseline["fraud_scenario_id"] = None

    eligible_accounts = _get_eligible_accounts(
        baseline=baseline,
        accounts=account_master,
    )

    if n_scenarios > len(eligible_accounts):
        raise ValueError(
            "n_scenarios cannot exceed the number of "
            "eligible active accounts"
        )

    victim_indices = rng.choice(
        eligible_accounts.index.to_numpy(),
        size=n_scenarios,
        replace=False,
    )

    victims = (
        eligible_accounts
        .loc[victim_indices]
        .reset_index(drop=True)
    )

    fraudulent_rows: list[dict] = []

    global_max_timestamp = pd.to_datetime(
        baseline["timestamp"]
    ).max()

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

        account_history = customer_history[
            customer_history["account_id"] == account_id
        ].sort_values("timestamp")

        if customer_history.empty or account_history.empty:
            continue

        scenario_id = (
            f"ATO_{scenario_number + 1:06d}"
        )

        number_of_transactions = int(
            rng.integers(
                min_transactions,
                max_transactions + 1,
            )
        )

        last_timestamp = pd.Timestamp(
            customer_history["timestamp"].max()
        )

        start_timestamp = (
            last_timestamp
            + pd.Timedelta(
                days=int(rng.integers(1, 8))
            )
        )

        if start_timestamp > global_max_timestamp:
            start_timestamp = (
                global_max_timestamp
                - pd.Timedelta(
                    days=int(rng.integers(0, 3))
                )
            )

        usual_start = int(
            customer["usual_start_hour"]
        )

        usual_end = int(
            customer["usual_end_hour"]
        )

        unusual_hours = [
            hour
            for hour in range(24)
            if not _hour_is_usual(
                hour=hour,
                usual_start=usual_start,
                usual_end=usual_end,
            )
        ]

        if unusual_hours:
            fraud_hour = int(
                rng.choice(unusual_hours)
            )
        else:
            fraud_hour = int(
                rng.integers(0, 24)
            )

        start_timestamp = start_timestamp.replace(
            hour=fraud_hour,
            minute=int(rng.integers(0, 60)),
            second=int(rng.integers(0, 60)),
        )

        attacker_device = (
            f"ATO_DEVICE_{scenario_number + 1:06d}"
        )

        customer_average = max(
            float(customer["avg_transaction"]),
            1.0,
        )

        customer_std = max(
            float(customer["std_transaction"]),
            1.0,
        )

        current_balance = float(
            account_history.iloc[-1][
                "balance_after"
            ]
        )

        previous_timestamp = start_timestamp

        for sequence_number in range(
            number_of_transactions
        ):
            if current_balance <= 0.01:
                break

            if sequence_number == 0:
                fraud_timestamp = start_timestamp
            else:
                fraud_timestamp = (
                    previous_timestamp
                    + pd.Timedelta(
                        minutes=int(
                            rng.integers(1, 6)
                        )
                    )
                )

            previous_timestamp = fraud_timestamp

            abnormal_multiplier = float(
                rng.uniform(4.0, 9.0)
            )

            proposed_amount = (
                customer_average
                * abnormal_multiplier
                + abs(
                    rng.normal(
                        loc=0.0,
                        scale=customer_std,
                    )
                )
            )

            amount = min(
                proposed_amount,
                current_balance,
            )

            amount = round(
                max(amount, 0.01),
                2,
            )

            balance_before = round(
                current_balance,
                2,
            )

            balance_after = round(
                max(
                    balance_before - amount,
                    0.0,
                ),
                2,
            )

            fraudulent_rows.append(
                {
                    "transaction_id": (
                        f"FRAUD_ATO_"
                        f"{scenario_number + 1:06d}_"
                        f"{sequence_number + 1:02d}"
                    ),
                    "account_id": account_id,
                    "customer_id": customer_id,
                    "timestamp": fraud_timestamp,
                    "amount": amount,
                    "merchant": (
                        "UNKNOWN_ONLINE_MERCHANT"
                    ),
                    "merchant_category": (
                        "ELECTRONICS"
                    ),
                    "transaction_type": (
                        "PURCHASE"
                    ),
                    "channel": "ONLINE",
                    "device_id": attacker_device,
                    "province": customer[
                        "province"
                    ],
                    "country": "South Africa",
                    "is_international": False,
                    "direction": "DEBIT",
                    "balance_before": (
                        balance_before
                    ),
                    "balance_after": (
                        balance_after
                    ),
                    "is_fraud": 1,
                    "fraud_type": (
                        FRAUD_TYPE_ACCOUNT_TAKEOVER
                    ),
                    "fraud_scenario_id": (
                        scenario_id
                    ),
                }
            )

            current_balance = balance_after

    return _combine_baseline_and_fraud(
        baseline=baseline,
        fraudulent_rows=fraudulent_rows,
    )


def inject_card_testing(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    customers: pd.DataFrame,
    n_scenarios: int = 100,
    min_transactions: int = 5,
    max_transactions: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Inject synthetic Card Testing fraud.

    Card Testing is represented as a rapid sequence of small
    transactions against an existing active account.

    This injector can accept a dataset that already contains
    Account Takeover fraud. Existing fraud labels are preserved.

    The source DataFrames are not modified.
    """

    _validate_parameters(
        n_scenarios=n_scenarios,
        min_transactions=min_transactions,
        max_transactions=max_transactions,
    )

    _validate_transaction_columns(transactions)

    rng = np.random.default_rng(seed)

    baseline = transactions.copy(deep=True)
    account_master = accounts.copy(deep=True)
    customer_master = customers.copy(deep=True)

    # Add fraud labels only when they do not already exist.
    if "is_fraud" not in baseline.columns:
        baseline["is_fraud"] = 0

    if "fraud_type" not in baseline.columns:
        baseline["fraud_type"] = None

    if "fraud_scenario_id" not in baseline.columns:
        baseline["fraud_scenario_id"] = None

    legitimate = baseline[
        baseline["is_fraud"] == 0
    ].copy()

    eligible_accounts = _get_eligible_accounts(
        baseline=legitimate,
        accounts=account_master,
    )

    # Obtain each account's latest legitimate balance.
    latest_balances = (
        legitimate
        .sort_values("timestamp")
        .groupby(
            "account_id",
            as_index=False,
        )
        .tail(1)[
            [
                "account_id",
                "balance_after",
            ]
        ]
        .rename(
            columns={
                "balance_after": "latest_balance"
            }
        )
    )

    eligible_accounts = eligible_accounts.merge(
        latest_balances,
        on="account_id",
        how="inner",
    )

    # Requiring sufficient funds for the maximum possible
    # burst ensures that every selected scenario can generate
    # all its requested transactions.
    minimum_required_balance = (
        CARD_TESTING_MAX_AMOUNT
        * max_transactions
    )

    eligible_accounts = eligible_accounts[
        eligible_accounts["latest_balance"]
        >= minimum_required_balance
    ].copy()

    if n_scenarios > len(eligible_accounts):
        raise ValueError(
            "n_scenarios cannot exceed the number of "
            "eligible active accounts with sufficient balance"
        )

    victim_indices = rng.choice(
        eligible_accounts.index.to_numpy(),
        size=n_scenarios,
        replace=False,
    )

    victims = (
        eligible_accounts
        .loc[victim_indices]
        .reset_index(drop=True)
    )

    fraudulent_rows: list[dict] = []

    global_max_timestamp = pd.to_datetime(
        legitimate["timestamp"]
    ).max()

    for scenario_number, victim in victims.iterrows():
        account_id = victim["account_id"]
        customer_id = victim["customer_id"]

        customer_rows = customer_master[
            customer_master["customer_id"] == customer_id
        ]

        if customer_rows.empty:
            continue

        customer = customer_rows.iloc[0]

        customer_history = legitimate[
            legitimate["customer_id"] == customer_id
        ].sort_values("timestamp")

        account_history = customer_history[
            customer_history["account_id"] == account_id
        ].sort_values("timestamp")

        if customer_history.empty or account_history.empty:
            continue

        scenario_id = (
            f"CARD_TEST_{scenario_number + 1:06d}"
        )

        number_of_transactions = int(
            rng.integers(
                min_transactions,
                max_transactions + 1,
            )
        )

        last_timestamp = pd.Timestamp(
            account_history["timestamp"].max()
        )

        start_timestamp = (
            last_timestamp
            + pd.Timedelta(
                hours=int(rng.integers(1, 25))
            )
        )

        if start_timestamp > global_max_timestamp:
            start_timestamp = (
                global_max_timestamp
                - pd.Timedelta(
                    hours=int(
                        rng.integers(0, 12)
                    )
                )
            )

        attacker_device = (
            f"CARD_TEST_DEVICE_"
            f"{scenario_number + 1:06d}"
        )

        current_balance = float(
            account_history.iloc[-1][
                "balance_after"
            ]
        )

        previous_timestamp = start_timestamp

        for sequence_number in range(
            number_of_transactions
        ):
            if sequence_number == 0:
                fraud_timestamp = start_timestamp
            else:
                fraud_timestamp = (
                    previous_timestamp
                    + pd.Timedelta(
                        seconds=int(
                            rng.integers(10, 91)
                        )
                    )
                )

            previous_timestamp = fraud_timestamp

            amount = round(
                float(
                    rng.uniform(
                        CARD_TESTING_MIN_AMOUNT,
                        CARD_TESTING_MAX_AMOUNT,
                    )
                ),
                2,
            )

            amount = min(
                amount,
                current_balance,
            )

            amount = round(amount, 2)

            balance_before = round(
                current_balance,
                2,
            )

            balance_after = round(
                max(
                    balance_before - amount,
                    0.0,
                ),
                2,
            )

            fraudulent_rows.append(
                {
                    "transaction_id": (
                        f"FRAUD_CARD_TEST_"
                        f"{scenario_number + 1:06d}_"
                        f"{sequence_number + 1:02d}"
                    ),
                    "account_id": account_id,
                    "customer_id": customer_id,
                    "timestamp": fraud_timestamp,
                    "amount": amount,
                    "merchant": (
                        "UNKNOWN_TEST_MERCHANT"
                    ),
                    "merchant_category": (
                        "ONLINE_SERVICES"
                    ),
                    "transaction_type": (
                        "PURCHASE"
                    ),
                    "channel": "ONLINE",
                    "device_id": attacker_device,
                    "province": customer[
                        "province"
                    ],
                    "country": "South Africa",
                    "is_international": False,
                    "direction": "DEBIT",
                    "balance_before": (
                        balance_before
                    ),
                    "balance_after": (
                        balance_after
                    ),
                    "is_fraud": 1,
                    "fraud_type": (
                        FRAUD_TYPE_CARD_TESTING
                    ),
                    "fraud_scenario_id": (
                        scenario_id
                    ),
                }
            )

            current_balance = balance_after

    return _combine_baseline_and_fraud(
        baseline=baseline,
        fraudulent_rows=fraudulent_rows,
    )


def _validate_parameters(
    n_scenarios: int,
    min_transactions: int,
    max_transactions: int,
) -> None:
    """Validate common fraud-injection parameters."""

    if n_scenarios <= 0:
        raise ValueError(
            "n_scenarios must be greater than zero"
        )

    if min_transactions <= 0:
        raise ValueError(
            "min_transactions must be greater than zero"
        )

    if max_transactions < min_transactions:
        raise ValueError(
            "max_transactions must be greater than or "
            "equal to min_transactions"
        )


def _validate_transaction_columns(
    transactions: pd.DataFrame,
) -> None:
    """Validate the required transaction columns."""

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

    missing_columns = (
        required_columns
        - set(transactions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Transactions DataFrame is missing columns: "
            f"{sorted(missing_columns)}"
        )


def _get_eligible_accounts(
    baseline: pd.DataFrame,
    accounts: pd.DataFrame,
) -> pd.DataFrame:
    """Return active accounts with transaction history."""

    active_accounts = accounts[
        accounts["account_status"] == "ACTIVE"
    ].copy()

    return active_accounts[
        active_accounts["account_id"].isin(
            baseline["account_id"]
        )
    ].copy()


def _combine_baseline_and_fraud(
    baseline: pd.DataFrame,
    fraudulent_rows: list[dict],
) -> pd.DataFrame:
    """Combine baseline and fraud transactions chronologically."""

    if not fraudulent_rows:
        return baseline

    fraud = pd.DataFrame(fraudulent_rows)

    result = pd.concat(
        [baseline, fraud],
        ignore_index=True,
    )

    return result.sort_values(
        [
            "timestamp",
            "transaction_id",
        ]
    ).reset_index(drop=True)


def _hour_is_usual(
    hour: int,
    usual_start: int,
    usual_end: int,
) -> bool:
    """Return whether an hour is within the usual window."""

    if usual_start <= usual_end:
        return (
            usual_start
            <= hour
            <= usual_end
        )

    # Handles activity windows that cross midnight.
    return (
        hour >= usual_start
        or hour <= usual_end
    )