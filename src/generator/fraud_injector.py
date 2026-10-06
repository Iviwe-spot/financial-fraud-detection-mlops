from __future__ import annotations

import numpy as np
import pandas as pd


FRAUD_TYPE_ACCOUNT_TAKEOVER = "ACCOUNT_TAKEOVER"
FRAUD_TYPE_CARD_TESTING = "CARD_TESTING"
FRAUD_TYPE_VELOCITY_ATTACK = "VELOCITY_ATTACK"
FRAUD_TYPE_ABNORMAL_AMOUNT = "ABNORMAL_AMOUNT"
FRAUD_TYPE_GEOGRAPHIC_ANOMALY = "GEOGRAPHIC_ANOMALY"

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

def inject_velocity_attack(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    customers: pd.DataFrame,
    n_scenarios: int = 50,
    min_transactions: int = 5,
    max_transactions: int = 10,
    seed: int = 44,
) -> pd.DataFrame:
    """
    Inject synthetic Velocity Attack fraud.

    A Velocity Attack consists of multiple customer-plausible
    transactions occurring within an abnormally short period.

    Unlike Card Testing:

    - amounts remain plausible for the customer;
    - a known customer device is reused where possible;
    - the primary anomaly is transaction frequency;
    - transactions occur 30-120 seconds apart.

    Existing fraud labels are preserved. Behavioural profiles are
    derived only from legitimate transaction history.

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

    # Behavioural profiles must only use legitimate history.
    legitimate = baseline[
        baseline["is_fraud"] == 0
    ].copy()

    eligible_accounts = _get_eligible_accounts(
        baseline=legitimate,
        accounts=account_master,
    )

    # Use the latest balance from all existing transactions.
    # This includes any previously injected fraud and therefore
    # preserves composability between fraud typologies.
    latest_account_rows = (
        baseline
        .sort_values("timestamp")
        .groupby(
            "account_id",
            as_index=False,
        )
        .tail(1)
    )

    latest_account_state = (
        latest_account_rows[
            [
                "account_id",
                "timestamp",
                "balance_after",
            ]
        ]
        .rename(
            columns={
                "timestamp": "latest_timestamp",
                "balance_after": "latest_balance",
            }
        )
    )

    eligible_accounts = eligible_accounts.merge(
        latest_account_state,
        on="account_id",
        how="inner",
    )

    # Add customer behavioural profiles so that accounts with
    # insufficient balances can be excluded before selection.
    customer_profiles = customer_master[
        [
            "customer_id",
            "avg_transaction",
            "std_transaction",
        ]
    ].copy()

    customer_profiles["avg_transaction"] = (
        customer_profiles["avg_transaction"]
        .astype(float)
        .clip(lower=1.0)
    )

    customer_profiles["std_transaction"] = (
        customer_profiles["std_transaction"]
        .astype(float)
        .clip(lower=1.0)
    )

    eligible_accounts = eligible_accounts.merge(
        customer_profiles,
        on="customer_id",
        how="inner",
    )

    # A Velocity Attack transaction is capped at 1.5 times the
    # customer's normal average. This condition guarantees enough
    # balance for the maximum requested number of transactions.
    eligible_accounts[
        "maximum_velocity_amount"
    ] = (
        eligible_accounts["avg_transaction"]
        * 1.50
    )

    eligible_accounts[
        "minimum_required_balance"
    ] = (
        eligible_accounts[
            "maximum_velocity_amount"
        ]
        * max_transactions
    )

    eligible_accounts = eligible_accounts[
        eligible_accounts["latest_balance"]
        >= eligible_accounts[
            "minimum_required_balance"
        ]
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

    for scenario_number, victim in victims.iterrows():
        account_id = victim["account_id"]
        customer_id = victim["customer_id"]

        customer_rows = customer_master[
            customer_master["customer_id"]
            == customer_id
        ]

        if customer_rows.empty:
            continue

        customer = customer_rows.iloc[0]

        # Legitimate history supplies behavioural evidence.
        customer_history = legitimate[
            legitimate["customer_id"]
            == customer_id
        ].sort_values("timestamp")

        legitimate_account_history = (
            customer_history[
                customer_history["account_id"]
                == account_id
            ]
            .sort_values("timestamp")
        )

        # All account history supplies the latest operational state.
        complete_account_history = (
            baseline[
                baseline["account_id"]
                == account_id
            ]
            .sort_values("timestamp")
        )

        if (
            customer_history.empty
            or legitimate_account_history.empty
            or complete_account_history.empty
        ):
            continue

        scenario_id = (
            f"VELOCITY_{scenario_number + 1:06d}"
        )

        number_of_transactions = int(
            rng.integers(
                min_transactions,
                max_transactions + 1,
            )
        )

        last_timestamp = pd.Timestamp(
            complete_account_history.iloc[-1][
                "timestamp"
            ]
        )

        # Start after the latest existing account transaction.
        # Do not force the event backwards into the baseline period.
        start_timestamp = (
            last_timestamp
            + pd.Timedelta(
                hours=int(rng.integers(1, 25))
            )
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
            complete_account_history.iloc[-1][
                "balance_after"
            ]
        )

        # Reuse an existing legitimate device where possible.
        known_devices = (
            customer_history["device_id"]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(known_devices) > 0:
            device_id = str(
                rng.choice(known_devices)
            )
        else:
            device_id = (
                f"KNOWN_DEVICE_{customer_id}"
            )

        # Use legitimate transaction contexts to avoid introducing
        # an artificial merchant or channel shortcut.
        context_candidates = (
            customer_history[
                customer_history[
                    "transaction_type"
                ] == "PURCHASE"
            ]
        )

        if context_candidates.empty:
            context_candidates = (
                customer_history.copy()
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
                            rng.integers(30, 121)
                        )
                    )
                )

            previous_timestamp = fraud_timestamp

            # Reduce the variance so that individual amounts remain
            # plausible. The main signal should be transaction speed.
            proposed_amount = float(
                rng.normal(
                    loc=customer_average,
                    scale=max(
                        customer_std * 0.35,
                        customer_average * 0.10,
                    ),
                )
            )

            minimum_amount = max(
                customer_average * 0.50,
                1.00,
            )

            maximum_amount = max(
                customer_average * 1.50,
                minimum_amount,
            )

            amount = float(
                np.clip(
                    proposed_amount,
                    minimum_amount,
                    maximum_amount,
                )
            )

            amount = round(
                min(
                    amount,
                    current_balance,
                ),
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

            # Reuse a legitimate merchant/category/channel context.
            context_index = int(
                rng.integers(
                    0,
                    len(context_candidates),
                )
            )

            context = (
                context_candidates.iloc[
                    context_index
                ]
            )

            fraudulent_rows.append(
                {
                    "transaction_id": (
                        f"FRAUD_VELOCITY_"
                        f"{scenario_number + 1:06d}_"
                        f"{sequence_number + 1:02d}"
                    ),
                    "account_id": account_id,
                    "customer_id": customer_id,
                    "timestamp": fraud_timestamp,
                    "amount": amount,
                    "merchant": context[
                        "merchant"
                    ],
                    "merchant_category": context[
                        "merchant_category"
                    ],
                    "transaction_type": (
                        "PURCHASE"
                    ),
                    "channel": context[
                        "channel"
                    ],
                    "device_id": device_id,
                    "province": context[
                        "province"
                    ],
                    "country": context[
                        "country"
                    ],
                    "is_international": bool(
                        context[
                            "is_international"
                        ]
                    ),
                    "direction": "DEBIT",
                    "balance_before": (
                        balance_before
                    ),
                    "balance_after": (
                        balance_after
                    ),
                    "is_fraud": 1,
                    "fraud_type": (
                        FRAUD_TYPE_VELOCITY_ATTACK
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
def inject_abnormal_amount(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    customers: pd.DataFrame,
    n_scenarios: int = 45,
    min_zscore: float = 4.0,
    max_zscore: float = 8.0,
    seed: int = 45,
) -> pd.DataFrame:
    """
    Inject synthetic Abnormal Amount fraud.

    Each scenario contains one transaction whose amount is extreme
    relative to the customer's normal transaction behaviour.

    The scenario deliberately avoids relying on new devices,
    unusual locations, or transaction bursts. Its primary signal
    is customer-relative transaction amount.

    Accounts already involved in an existing fraud scenario are
    excluded to prevent overlapping fraud typologies.

    Existing fraud labels are preserved.
    """

    if n_scenarios <= 0:
        raise ValueError(
            "n_scenarios must be greater than zero"
        )

    if min_zscore <= 0:
        raise ValueError(
            "min_zscore must be greater than zero"
        )

    if max_zscore < min_zscore:
        raise ValueError(
            "max_zscore must be greater than or "
            "equal to min_zscore"
        )

    _validate_transaction_columns(transactions)

    rng = np.random.default_rng(seed)

    baseline = transactions.copy(deep=True)
    account_master = accounts.copy(deep=True)
    customer_master = customers.copy(deep=True)

    if "is_fraud" not in baseline.columns:
        baseline["is_fraud"] = 0

    if "fraud_type" not in baseline.columns:
        baseline["fraud_type"] = None

    if "fraud_scenario_id" not in baseline.columns:
        baseline["fraud_scenario_id"] = None

    legitimate = baseline[
        baseline["is_fraud"] == 0
    ].copy()

    # Prevent different fraud typologies from being injected
    # independently into the same account.
    compromised_accounts = set(
        baseline.loc[
            baseline["is_fraud"] == 1,
            "account_id",
        ]
    )

    eligible_accounts = _get_eligible_accounts(
        baseline=legitimate,
        accounts=account_master,
    )

    eligible_accounts = eligible_accounts[
        ~eligible_accounts["account_id"].isin(
            compromised_accounts
        )
    ].copy()

    latest_balances = (
        legitimate
        .sort_values("timestamp")
        .groupby("account_id", as_index=False)
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

    # A meaningful abnormal-amount transaction requires
    # sufficient available balance.
    eligible_accounts = eligible_accounts[
        eligible_accounts["latest_balance"] > 100.0
    ].copy()

    if n_scenarios > len(eligible_accounts):
        raise ValueError(
            "n_scenarios cannot exceed the number of "
            "eligible uncompromised active accounts"
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
            customer_master["customer_id"]
            == customer_id
        ]

        if customer_rows.empty:
            continue

        customer = customer_rows.iloc[0]

        customer_history = legitimate[
            legitimate["customer_id"]
            == customer_id
        ].sort_values("timestamp")

        account_history = customer_history[
            customer_history["account_id"]
            == account_id
        ].sort_values("timestamp")

        if customer_history.empty or account_history.empty:
            continue

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

        # Select a customer-relative anomaly severity.
        target_zscore = float(
            rng.uniform(
                min_zscore,
                max_zscore,
            )
        )

        proposed_amount = (
            customer_average
            + target_zscore * customer_std
        )

        # Skip victims whose balance cannot support at least
        # the minimum requested customer-relative anomaly.
        minimum_abnormal_amount = (
            customer_average
            + min_zscore * customer_std
        )

        if current_balance < minimum_abnormal_amount:
            continue

        amount = min(
            proposed_amount,
            current_balance,
        )

        amount = round(
            max(float(amount), 0.01),
            2,
        )

        last_timestamp = pd.Timestamp(
            account_history["timestamp"].max()
        )

        fraud_timestamp = (
            last_timestamp
            + pd.Timedelta(
                hours=int(rng.integers(1, 25))
            )
        )

        if fraud_timestamp > global_max_timestamp:
            fraud_timestamp = (
                global_max_timestamp
                - pd.Timedelta(
                    hours=int(
                        rng.integers(0, 12)
                    )
                )
            )

        # Keep the transaction inside the customer's usual
        # activity window where possible.
        usual_start = int(
            customer["usual_start_hour"]
        )

        usual_end = int(
            customer["usual_end_hour"]
        )

        if usual_start <= usual_end:
            normal_hour = int(
                rng.integers(
                    usual_start,
                    usual_end + 1,
                )
            )
        else:
            normal_hours = list(
                range(usual_start, 24)
            ) + list(
                range(0, usual_end + 1)
            )

            normal_hour = int(
                rng.choice(normal_hours)
            )

        fraud_timestamp = fraud_timestamp.replace(
            hour=normal_hour,
            minute=int(rng.integers(0, 60)),
            second=int(rng.integers(0, 60)),
        )

        # Reuse a legitimate device so the primary signal
        # remains the amount anomaly.
        known_devices = (
            customer_history["device_id"]
            .dropna()
            .unique()
        )

        if len(known_devices) > 0:
            device_id = str(
                rng.choice(known_devices)
            )
        else:
            device_id = (
                f"ABNORMAL_AMOUNT_DEVICE_"
                f"{scenario_number + 1:06d}"
            )

        # Reuse plausible legitimate transaction metadata
        # rather than creating a fraud-specific merchant.
        reference_transaction = (
            customer_history.iloc[
                int(
                    rng.integers(
                        0,
                        len(customer_history),
                    )
                )
            ]
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

        scenario_id = (
            f"ABNORMAL_AMOUNT_"
            f"{scenario_number + 1:06d}"
        )

        fraudulent_rows.append(
            {
                "transaction_id": (
                    f"FRAUD_ABNORMAL_AMOUNT_"
                    f"{scenario_number + 1:06d}"
                ),
                "account_id": account_id,
                "customer_id": customer_id,
                "timestamp": fraud_timestamp,
                "amount": amount,
                "merchant": reference_transaction[
                    "merchant"
                ],
                "merchant_category": (
                    reference_transaction[
                        "merchant_category"
                    ]
                ),
                "transaction_type": (
                    reference_transaction[
                        "transaction_type"
                    ]
                ),
                "channel": reference_transaction[
                    "channel"
                ],
                "device_id": device_id,
                "province": customer["province"],
                "country": "South Africa",
                "is_international": False,
                "direction": "DEBIT",
                "balance_before": balance_before,
                "balance_after": balance_after,
                "is_fraud": 1,
                "fraud_type": (
                    FRAUD_TYPE_ABNORMAL_AMOUNT
                ),
                "fraud_scenario_id": scenario_id,
            }
        )

    return _combine_baseline_and_fraud(
        baseline=baseline,
        fraudulent_rows=fraudulent_rows,
    )
def inject_geographic_anomaly(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    customers: pd.DataFrame,
    n_scenarios: int = 40,
    min_minutes_after_previous: int = 5,
    max_minutes_after_previous: int = 45,
    seed: int = 46,
) -> pd.DataFrame:
    """
    Inject synthetic Geographic Anomaly / Impossible Travel fraud.

    Each scenario creates one fraudulent transaction in a South African
    province different from the customer's most recent legitimate domestic
    transaction, after an implausibly short time interval.

    The scenario deliberately avoids relying on new devices, abnormal
    amounts, or transaction bursts. The primary signal is the combination
    of geographic displacement and a short elapsed time.

    Accounts already involved in existing fraud scenarios are excluded.
    Existing fraud labels are preserved.
    """

    if n_scenarios <= 0:
        raise ValueError(
            "n_scenarios must be greater than zero"
        )

    if min_minutes_after_previous <= 0:
        raise ValueError(
            "min_minutes_after_previous must be greater than zero"
        )

    if (
        max_minutes_after_previous
        < min_minutes_after_previous
    ):
        raise ValueError(
            "max_minutes_after_previous must be greater than "
            "or equal to min_minutes_after_previous"
        )

    _validate_transaction_columns(transactions)

    rng = np.random.default_rng(seed)

    baseline = transactions.copy(deep=True)
    account_master = accounts.copy(deep=True)
    customer_master = customers.copy(deep=True)

    # ---------------------------------------------------------
    # Ensure fraud ground-truth columns exist
    # ---------------------------------------------------------

    if "is_fraud" not in baseline.columns:
        baseline["is_fraud"] = 0

    if "fraud_type" not in baseline.columns:
        baseline["fraud_type"] = None

    if "fraud_scenario_id" not in baseline.columns:
        baseline["fraud_scenario_id"] = None

    # ---------------------------------------------------------
    # Legitimate history only
    # ---------------------------------------------------------

    legitimate = baseline[
        baseline["is_fraud"] == 0
    ].copy()

    legitimate["timestamp"] = pd.to_datetime(
        legitimate["timestamp"]
    )

    # ---------------------------------------------------------
    # Exclude accounts already used by another fraud typology
    # ---------------------------------------------------------

    compromised_accounts = set(
        baseline.loc[
            baseline["is_fraud"] == 1,
            "account_id",
        ]
    )

    eligible_accounts = _get_eligible_accounts(
        baseline=legitimate,
        accounts=account_master,
    )

    eligible_accounts = eligible_accounts[
        ~eligible_accounts["account_id"].isin(
            compromised_accounts
        )
    ].copy()

    # ---------------------------------------------------------
    # Identify accounts with usable balances
    # ---------------------------------------------------------

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

    eligible_accounts = eligible_accounts[
        eligible_accounts["latest_balance"] > 100.0
    ].copy()

    # ---------------------------------------------------------
    # IMPORTANT:
    # Require legitimate domestic history before sampling.
    #
    # This prevents INTERNATIONAL from becoming the geographic
    # anchor for a province-level impossible-travel scenario.
    # ---------------------------------------------------------

    domestic_legitimate = legitimate[
        (
            legitimate["country"]
            == "South Africa"
        )
        & (
            legitimate["province"].notna()
        )
        & (
            legitimate["province"]
            != "INTERNATIONAL"
        )
    ].copy()

    domestic_accounts = set(
        domestic_legitimate["account_id"]
    )

    eligible_accounts = eligible_accounts[
        eligible_accounts["account_id"].isin(
            domestic_accounts
        )
    ].copy()

    if n_scenarios > len(eligible_accounts):
        raise ValueError(
            "n_scenarios cannot exceed the number of "
            "eligible uncompromised active accounts with "
            "legitimate domestic history"
        )

    # ---------------------------------------------------------
    # Select victim accounts
    # ---------------------------------------------------------

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

    south_african_provinces = [
        "Eastern Cape",
        "Free State",
        "Gauteng",
        "KwaZulu-Natal",
        "Limpopo",
        "Mpumalanga",
        "North West",
        "Northern Cape",
        "Western Cape",
    ]

    fraudulent_rows: list[dict] = []

    # ---------------------------------------------------------
    # Generate one fraud transaction per scenario
    # ---------------------------------------------------------

    for scenario_number, victim in victims.iterrows():
        account_id = victim["account_id"]
        customer_id = victim["customer_id"]

        customer_rows = customer_master[
            customer_master["customer_id"]
            == customer_id
        ]

        if customer_rows.empty:
            continue

        customer = customer_rows.iloc[0]

        # All legitimate customer history is still useful for
        # devices and normal transaction metadata.
        customer_history = legitimate[
            legitimate["customer_id"]
            == customer_id
        ].sort_values("timestamp")

        account_history = legitimate[
            legitimate["account_id"]
            == account_id
        ].sort_values("timestamp")

        if (
            customer_history.empty
            or account_history.empty
        ):
            continue

        # -----------------------------------------------------
        # Domestic geographic anchor
        # -----------------------------------------------------

        domestic_history = customer_history[
            (
                customer_history["country"]
                == "South Africa"
            )
            & (
                customer_history["province"].notna()
            )
            & (
                customer_history["province"]
                != "INTERNATIONAL"
            )
        ].sort_values("timestamp")

        if domestic_history.empty:
            continue

        previous_transaction = (
            domestic_history.iloc[-1]
        )

        previous_timestamp = pd.Timestamp(
            previous_transaction["timestamp"]
        )

        previous_province = str(
            previous_transaction["province"]
        )

        candidate_provinces = [
            province
            for province in south_african_provinces
            if province != previous_province
        ]

        if not candidate_provinces:
            continue

        fraud_province = str(
            rng.choice(
                candidate_provinces
            )
        )

        # -----------------------------------------------------
        # Create impossible-travel time gap
        # -----------------------------------------------------

        minutes_after_previous = int(
            rng.integers(
                min_minutes_after_previous,
                max_minutes_after_previous + 1,
            )
        )

        fraud_timestamp = (
            previous_timestamp
            + pd.Timedelta(
                minutes=minutes_after_previous
            )
        )

        # -----------------------------------------------------
        # Normal-looking customer-relative amount
        # -----------------------------------------------------

        customer_average = max(
            float(
                customer["avg_transaction"]
            ),
            1.0,
        )

        customer_std = max(
            float(
                customer["std_transaction"]
            ),
            1.0,
        )

        proposed_amount = float(
            rng.normal(
                loc=customer_average,
                scale=max(
                    customer_std * 0.5,
                    1.0,
                ),
            )
        )

        minimum_amount = max(
            customer_average * 0.25,
            25.0,
        )

        maximum_amount = max(
            minimum_amount,
            customer_average
            + customer_std,
        )

        proposed_amount = float(
            np.clip(
                proposed_amount,
                minimum_amount,
                maximum_amount,
            )
        )

        current_balance = float(
            account_history.iloc[-1][
                "balance_after"
            ]
        )

        if current_balance < minimum_amount:
            continue

        amount = min(
            proposed_amount,
            current_balance,
        )

        amount = round(
            max(
                float(amount),
                0.01,
            ),
            2,
        )

        # -----------------------------------------------------
        # Reuse known device
        # -----------------------------------------------------

        known_devices = (
            customer_history[
                "device_id"
            ]
            .dropna()
            .unique()
        )

        if len(known_devices) > 0:
            device_id = str(
                rng.choice(
                    known_devices
                )
            )
        else:
            device_id = (
                f"GEO_DEVICE_"
                f"{scenario_number + 1:06d}"
            )

        # -----------------------------------------------------
        # Reuse plausible legitimate transaction metadata
        # -----------------------------------------------------

        reference_transaction = (
            customer_history.iloc[
                int(
                    rng.integers(
                        0,
                        len(customer_history),
                    )
                )
            ]
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

        scenario_id = (
            f"GEO_"
            f"{scenario_number + 1:06d}"
        )

        fraudulent_rows.append(
            {
                "transaction_id": (
                    f"FRAUD_GEO_"
                    f"{scenario_number + 1:06d}"
                ),
                "account_id": account_id,
                "customer_id": customer_id,
                "timestamp": fraud_timestamp,
                "amount": amount,
                "merchant": (
                    reference_transaction[
                        "merchant"
                    ]
                ),
                "merchant_category": (
                    reference_transaction[
                        "merchant_category"
                    ]
                ),
                "transaction_type": (
                    reference_transaction[
                        "transaction_type"
                    ]
                ),
                "channel": (
                    reference_transaction[
                        "channel"
                    ]
                ),
                "device_id": device_id,
                "province": fraud_province,
                "country": "South Africa",
                "is_international": False,
                "direction": "DEBIT",
                "balance_before": balance_before,
                "balance_after": balance_after,
                "is_fraud": 1,
                "fraud_type": (
                    FRAUD_TYPE_GEOGRAPHIC_ANOMALY
                ),
                "fraud_scenario_id": scenario_id,
            }
        )

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