import numpy as np
import pandas as pd
import pytest

from pipelines.build_fraud_features import (
    MODEL_FEATURE_COLUMNS,
    build_point_in_time_features,
)


def make_transactions():
    return pd.DataFrame(
        [
            {
                "transaction_id": "T1",
                "account_id": "A1",
                "customer_id": "C1",
                "timestamp": "2026-01-05 09:00:00",
                "amount": 100.0,
                "merchant": "SHOP_A",
                "transaction_type": "PURCHASE",
                "channel": "CARD",
                "device_id": "DEVICE_A",
                "province": "Gauteng",
                "country": "South Africa",
                "is_international": False,
                "balance_before": 1000.0,
                "is_fraud": 0,
                "fraud_type": None,
                "fraud_scenario_id": None,
            },
            {
                "transaction_id": "T2",
                "account_id": "A1",
                "customer_id": "C1",
                "timestamp": "2026-01-05 09:30:00",
                "amount": 200.0,
                "merchant": "SHOP_A",
                "transaction_type": "PURCHASE",
                "channel": "CARD",
                "device_id": "DEVICE_A",
                "province": "Gauteng",
                "country": "South Africa",
                "is_international": False,
                "balance_before": 900.0,
                "is_fraud": 0,
                "fraud_type": None,
                "fraud_scenario_id": None,
            },
            {
                "transaction_id": "T3",
                "account_id": "A1",
                "customer_id": "C1",
                "timestamp": "2026-01-05 09:35:00",
                "amount": 300.0,
                "merchant": "SHOP_B",
                "transaction_type": "PURCHASE",
                "channel": "ONLINE",
                "device_id": "DEVICE_B",
                "province": "Western Cape",
                "country": "South Africa",
                "is_international": False,
                "balance_before": 700.0,
                "is_fraud": 1,
                "fraud_type": "TEST_FRAUD",
                "fraud_scenario_id": "S1",
            },
        ]
    )


def make_customers():
    return pd.DataFrame(
        [
            {
                "customer_id": "C1",
                "usual_start_hour": 8,
                "usual_end_hour": 18,
                "international_rate": 0.0,
            }
        ]
    )


def build():
    return build_point_in_time_features(
        make_transactions(),
        make_customers(),
    ).set_index("transaction_id")


def test_first_transaction_has_no_prior_history():
    features = build()

    first = features.loc["T1"]

    assert first["prior_transaction_count"] == 0
    assert np.isnan(
        first["prior_mean_amount"]
    )
    assert np.isnan(
        first["minutes_since_previous"]
    )

    assert first[
        "transactions_last_10min"
    ] == 0

    assert first[
        "transactions_last_1h"
    ] == 0

    assert first["spend_last_1h"] == 0.0


def test_prior_amount_statistics_exclude_current_transaction():
    features = build()

    second = features.loc["T2"]
    third = features.loc["T3"]

    assert second[
        "prior_transaction_count"
    ] == 1

    assert second[
        "prior_mean_amount"
    ] == pytest.approx(100.0)

    assert third[
        "prior_transaction_count"
    ] == 2

    assert third[
        "prior_mean_amount"
    ] == pytest.approx(150.0)

    expected_std = np.std(
        [100.0, 200.0],
        ddof=1,
    )

    assert third[
        "prior_std_amount"
    ] == pytest.approx(
        expected_std
    )

    assert third[
        "amount_vs_prior_mean"
    ] == pytest.approx(2.0)

    expected_z = (
        300.0 - 150.0
    ) / expected_std

    assert third[
        "amount_zscore"
    ] == pytest.approx(
        expected_z
    )


def test_velocity_features_use_only_prior_transactions():
    features = build()

    third = features.loc["T3"]

    assert third[
        "minutes_since_previous"
    ] == pytest.approx(5.0)

    # T2 occurred five minutes earlier.
    assert third[
        "transactions_last_10min"
    ] == 1

    # T1 and T2 both occurred within the prior hour.
    assert third[
        "transactions_last_1h"
    ] == 2

    assert third[
        "spend_last_1h"
    ] == pytest.approx(300.0)


def test_device_novelty_is_point_in_time():
    features = build()

    assert features.loc[
        "T1",
        "new_device",
    ] == 1

    assert features.loc[
        "T2",
        "new_device",
    ] == 0

    assert features.loc[
        "T2",
        "prior_device_transaction_count",
    ] == 1

    assert features.loc[
        "T3",
        "new_device",
    ] == 1

    assert features.loc[
        "T3",
        "prior_device_transaction_count",
    ] == 0


def test_merchant_novelty_is_point_in_time():
    features = build()

    assert features.loc[
        "T1",
        "new_merchant",
    ] == 1

    assert features.loc[
        "T2",
        "new_merchant",
    ] == 0

    assert features.loc[
        "T2",
        "prior_merchant_transaction_count",
    ] == 1

    assert features.loc[
        "T3",
        "new_merchant",
    ] == 1


def test_geographic_change_uses_previous_transaction():
    features = build()

    assert features.loc[
        "T1",
        "province_change",
    ] == 0

    assert features.loc[
        "T2",
        "province_change",
    ] == 0

    assert features.loc[
        "T3",
        "province_change",
    ] == 1


def test_balance_utilisation():
    features = build()

    assert features.loc[
        "T1",
        "balance_utilisation",
    ] == pytest.approx(0.1)

    assert features.loc[
        "T3",
        "balance_utilisation",
    ] == pytest.approx(
        300.0 / 700.0
    )


def test_current_transaction_does_not_create_own_history():
    transactions = make_transactions()

    transactions.loc[
        transactions["transaction_id"] == "T3",
        "amount",
    ] = 10000.0

    features = (
        build_point_in_time_features(
            transactions,
            make_customers(),
        )
        .set_index("transaction_id")
    )

    third = features.loc["T3"]

    # Changing T3 must not change statistics calculated from
    # T1 and T2.
    assert third[
        "prior_mean_amount"
    ] == pytest.approx(150.0)

    assert third[
        "transactions_last_1h"
    ] == 2

    assert third[
        "spend_last_1h"
    ] == pytest.approx(300.0)


def test_same_timestamp_transactions_do_not_leak_into_each_other():
    transactions = make_transactions()

    extra = transactions.iloc[[2]].copy()

    extra["transaction_id"] = "T4"
    extra["amount"] = 400.0
    extra["merchant"] = "SHOP_C"
    extra["device_id"] = "DEVICE_C"

    transactions = pd.concat(
        [
            transactions,
            extra,
        ],
        ignore_index=True,
    )

    features = (
        build_point_in_time_features(
            transactions,
            make_customers(),
        )
        .set_index("transaction_id")
    )

    # T3 and T4 both occur at 09:35. Neither may treat the
    # other as prior history.
    for transaction_id in [
        "T3",
        "T4",
    ]:
        row = features.loc[
            transaction_id
        ]

        assert row[
            "prior_transaction_count"
        ] == 2

        assert row[
            "transactions_last_10min"
        ] == 1

        assert row[
            "transactions_last_1h"
        ] == 2

        assert row[
            "spend_last_1h"
        ] == pytest.approx(300.0)


def test_model_feature_list_excludes_raw_identifiers_and_labels():
    prohibited = {
        "transaction_id",
        "customer_id",
        "account_id",
        "device_id",
        "merchant",
        "fraud_type",
        "fraud_scenario_id",
        "is_fraud",
        "timestamp",
    }

    assert prohibited.isdisjoint(
        MODEL_FEATURE_COLUMNS
    )


def test_velocity_window_includes_exact_boundary():
    transactions = make_transactions().iloc[[0, 1]].copy()

    transactions.loc[
        transactions["transaction_id"] == "T1",
        "timestamp",
    ] = "2026-01-05 09:00:00"

    transactions.loc[
        transactions["transaction_id"] == "T2",
        "timestamp",
    ] = "2026-01-05 09:10:00"

    features = (
        build_point_in_time_features(
            transactions,
            make_customers(),
        )
        .set_index("transaction_id")
    )

    second = features.loc["T2"]

    # Exactly 10 minutes earlier is inside [t - 10m, t).
    assert second[
        "transactions_last_10min"
    ] == 1

    assert second[
        "spend_last_1h"
    ] == pytest.approx(100.0)


def test_velocity_window_excludes_older_transaction():
    transactions = make_transactions().iloc[[0, 1]].copy()

    transactions.loc[
        transactions["transaction_id"] == "T1",
        "timestamp",
    ] = "2026-01-05 08:59:59"

    transactions.loc[
        transactions["transaction_id"] == "T2",
        "timestamp",
    ] = "2026-01-05 09:10:00"

    features = (
        build_point_in_time_features(
            transactions,
            make_customers(),
        )
        .set_index("transaction_id")
    )

    second = features.loc["T2"]

    # 10 minutes and 1 second earlier is outside the window.
    assert second[
        "transactions_last_10min"
    ] == 0
