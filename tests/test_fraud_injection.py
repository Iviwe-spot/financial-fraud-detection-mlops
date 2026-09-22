import pandas as pd
from pandas.testing import assert_frame_equal

from src.generator.fraud_injector import (
    FRAUD_TYPE_ACCOUNT_TAKEOVER,
    FRAUD_TYPE_CARD_TESTING,
    inject_account_takeover,
    inject_card_testing,
)


def _sample_data():
    customers = pd.DataFrame(
        {
            "customer_id": [
                "C0000001",
                "C0000002",
            ],
            "province": [
                "Gauteng",
                "Western Cape",
            ],
            "avg_transaction": [
                500.0,
                1000.0,
            ],
            "std_transaction": [
                100.0,
                250.0,
            ],
            "usual_start_hour": [
                7,
                8,
            ],
            "usual_end_hour": [
                21,
                22,
            ],
        }
    )

    accounts = pd.DataFrame(
        {
            "account_id": [
                "A00000001",
                "A00000002",
            ],
            "customer_id": [
                "C0000001",
                "C0000002",
            ],
            "account_type": [
                "CHEQUE",
                "CHEQUE",
            ],
            "opening_balance": [
                50000.0,
                75000.0,
            ],
            "account_status": [
                "ACTIVE",
                "ACTIVE",
            ],
            "currency": [
                "ZAR",
                "ZAR",
            ],
        }
    )

    transactions = pd.DataFrame(
        {
            "transaction_id": [
                "T00000001",
                "T00000002",
            ],
            "account_id": [
                "A00000001",
                "A00000002",
            ],
            "customer_id": [
                "C0000001",
                "C0000002",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 11:00:00",
                ]
            ),
            "amount": [
                500.0,
                1000.0,
            ],
            "merchant": [
                "SHOP_A",
                "SHOP_B",
            ],
            "merchant_category": [
                "GROCERY",
                "FUEL",
            ],
            "transaction_type": [
                "PURCHASE",
                "PURCHASE",
            ],
            "channel": [
                "CARD",
                "CARD",
            ],
            "device_id": [
                "DEVICE_1",
                "DEVICE_2",
            ],
            "province": [
                "Gauteng",
                "Western Cape",
            ],
            "country": [
                "South Africa",
                "South Africa",
            ],
            "is_international": [
                False,
                False,
            ],
            "direction": [
                "DEBIT",
                "DEBIT",
            ],
            "balance_before": [
                50000.0,
                75000.0,
            ],
            "balance_after": [
                49500.0,
                74000.0,
            ],
        }
    )

    return customers, accounts, transactions


def test_account_takeover_adds_fraud():
    customers, accounts, transactions = _sample_data()

    result = inject_account_takeover(
        transactions,
        accounts,
        customers,
        n_scenarios=1,
        seed=42,
    )

    assert len(result) > len(transactions)
    assert result["is_fraud"].sum() >= 1


def test_legitimate_transactions_remain_legitimate():
    customers, accounts, transactions = _sample_data()

    result = inject_account_takeover(
        transactions,
        accounts,
        customers,
        n_scenarios=1,
        seed=42,
    )

    original_ids = set(
        transactions["transaction_id"]
    )

    original_rows = result[
        result["transaction_id"].isin(original_ids)
    ]

    assert (
        original_rows["is_fraud"] == 0
    ).all()

    assert original_rows[
        "fraud_type"
    ].isna().all()

    assert original_rows[
        "fraud_scenario_id"
    ].isna().all()


def test_fraud_has_correct_ground_truth():
    customers, accounts, transactions = _sample_data()

    result = inject_account_takeover(
        transactions,
        accounts,
        customers,
        n_scenarios=1,
        seed=42,
    )

    fraud = result[
        result["is_fraud"] == 1
    ]

    assert not fraud.empty

    assert (
        fraud["fraud_type"]
        == FRAUD_TYPE_ACCOUNT_TAKEOVER
    ).all()

    assert fraud[
        "fraud_scenario_id"
    ].notna().all()


def test_fraud_uses_new_device():
    customers, accounts, transactions = _sample_data()

    original_devices = set(
        transactions["device_id"]
    )

    result = inject_account_takeover(
        transactions,
        accounts,
        customers,
        n_scenarios=1,
        seed=42,
    )

    fraud_devices = set(
        result.loc[
            result["is_fraud"] == 1,
            "device_id",
        ]
    )

    assert fraud_devices.isdisjoint(
        original_devices
    )


def test_foreign_keys_remain_valid():
    customers, accounts, transactions = _sample_data()

    result = inject_account_takeover(
        transactions,
        accounts,
        customers,
        n_scenarios=1,
        seed=42,
    )

    assert set(
        result["customer_id"]
    ).issubset(
        set(customers["customer_id"])
    )

    assert set(
        result["account_id"]
    ).issubset(
        set(accounts["account_id"])
    )


def test_fraud_amounts_are_positive():
    customers, accounts, transactions = _sample_data()

    result = inject_account_takeover(
        transactions,
        accounts,
        customers,
        n_scenarios=1,
        seed=42,
    )

    fraud = result[
        result["is_fraud"] == 1
    ]

    assert (fraud["amount"] > 0).all()

    assert (
        fraud["balance_after"] >= 0
    ).all()


def test_reproducible_with_same_seed():
    customers, accounts, transactions = _sample_data()

    first = inject_account_takeover(
        transactions,
        accounts,
        customers,
        n_scenarios=1,
        seed=42,
    )

    second = inject_account_takeover(
        transactions,
        accounts,
        customers,
        n_scenarios=1,
        seed=42,
    )

    assert_frame_equal(first, second)


def test_source_transactions_are_not_modified():
    customers, accounts, transactions = _sample_data()

    original = transactions.copy(deep=True)

    inject_account_takeover(
        transactions,
        accounts,
        customers,
        n_scenarios=1,
        seed=42,
    )

    assert_frame_equal(
        transactions,
        original,
    )


def test_card_testing_adds_fraud():
    customers, accounts, transactions = _sample_data()

    result = inject_card_testing(
        transactions,
        accounts,
        customers,
        n_scenarios=1,
        seed=42,
    )

    fraud = result[
        result["is_fraud"] == 1
    ]

    assert not fraud.empty

    assert (
        fraud["fraud_type"]
        == FRAUD_TYPE_CARD_TESTING
    ).all()


def test_card_testing_uses_small_amounts():
    customers, accounts, transactions = _sample_data()

    result = inject_card_testing(
        transactions,
        accounts,
        customers,
        n_scenarios=1,
        seed=42,
    )

    fraud = result[
        result["fraud_type"]
        == FRAUD_TYPE_CARD_TESTING
    ]

    assert (
        fraud["amount"] >= 1.00
    ).all()

    assert (
        fraud["amount"] <= 20.00
    ).all()


def test_card_testing_creates_transaction_burst():
    customers, accounts, transactions = _sample_data()

    result = inject_card_testing(
        transactions,
        accounts,
        customers,
        n_scenarios=1,
        min_transactions=5,
        max_transactions=5,
        seed=42,
    )

    fraud = (
        result[
            result["fraud_type"]
            == FRAUD_TYPE_CARD_TESTING
        ]
        .sort_values("timestamp")
    )

    assert len(fraud) == 5

    gaps = (
        fraud["timestamp"]
        .diff()
        .dropna()
    )

    assert (
        gaps
        <= pd.Timedelta(seconds=90)
    ).all()

    assert (
        gaps
        >= pd.Timedelta(seconds=10)
    ).all()


def test_card_testing_uses_new_device():
    customers, accounts, transactions = _sample_data()

    original_devices = set(
        transactions["device_id"]
    )

    result = inject_card_testing(
        transactions,
        accounts,
        customers,
        n_scenarios=1,
        seed=42,
    )

    fraud_devices = set(
        result.loc[
            result["fraud_type"]
            == FRAUD_TYPE_CARD_TESTING,
            "device_id",
        ]
    )

    assert fraud_devices.isdisjoint(
        original_devices
    )


def test_card_testing_preserves_existing_fraud():
    customers, accounts, transactions = _sample_data()

    with_account_takeover = (
        inject_account_takeover(
            transactions,
            accounts,
            customers,
            n_scenarios=1,
            seed=42,
        )
    )

    account_takeover_count_before = (
        with_account_takeover["fraud_type"]
        == FRAUD_TYPE_ACCOUNT_TAKEOVER
    ).sum()

    combined = inject_card_testing(
        with_account_takeover,
        accounts,
        customers,
        n_scenarios=1,
        seed=43,
    )

    account_takeover_count_after = (
        combined["fraud_type"]
        == FRAUD_TYPE_ACCOUNT_TAKEOVER
    ).sum()

    assert (
        account_takeover_count_after
        == account_takeover_count_before
    )

    assert (
        combined["fraud_type"]
        == FRAUD_TYPE_CARD_TESTING
    ).any()