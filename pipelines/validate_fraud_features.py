from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.build_fraud_features import MODEL_FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "transactions_with_fraud.parquet"
)

FEATURE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "fraud_features.parquet"
)


PROHIBITED_MODEL_COLUMNS = {
    "transaction_id",
    "customer_id",
    "account_id",
    "timestamp",
    "device_id",
    "merchant",
    "merchant_category",
    "fraud_type",
    "fraud_scenario_id",
    "is_fraud",
}


def check(name, condition, failures):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")

    if not condition:
        failures.append(name)


def main():
    source = pd.read_parquet(SOURCE_PATH)
    features = pd.read_parquet(FEATURE_PATH)

    failures = []

    print("=" * 78)
    print("GOLD FRAUD FEATURE VALIDATION")
    print("=" * 78)

    print("\n1. STRUCTURAL INTEGRITY")
    print("-" * 78)

    check(
        "Feature row count equals source row count",
        len(features) == len(source),
        failures,
    )

    check(
        "Transaction IDs are unique",
        not features["transaction_id"].duplicated().any(),
        failures,
    )

    check(
        "Transaction ID set matches source",
        set(features["transaction_id"])
        == set(source["transaction_id"]),
        failures,
    )

    check(
        "Target contains only 0/1",
        set(features["is_fraud"].dropna().unique())
        <= {0, 1},
        failures,
    )

    check(
        "Fraud count matches source",
        int(features["is_fraud"].sum())
        == int(source["is_fraud"].sum()),
        failures,
    )

    print("\n2. MODEL FEATURE POLICY")
    print("-" * 78)

    leaked = (
        PROHIBITED_MODEL_COLUMNS
        .intersection(MODEL_FEATURE_COLUMNS)
    )

    check(
        "No prohibited columns in MODEL_FEATURE_COLUMNS",
        len(leaked) == 0,
        failures,
    )

    check(
        "All declared model features exist",
        set(MODEL_FEATURE_COLUMNS)
        <= set(features.columns),
        failures,
    )

    print("\nDeclared model features:")
    for column in MODEL_FEATURE_COLUMNS:
        print(f"  - {column}")

    print("\n3. NUMERICAL INTEGRITY")
    print("-" * 78)

    numeric_features = [
        column
        for column in MODEL_FEATURE_COLUMNS
        if pd.api.types.is_numeric_dtype(features[column])
    ]

    infinite_columns = []

    for column in numeric_features:
        values = features[column].dropna()

        if np.isinf(values).any():
            infinite_columns.append(column)

    check(
        "No infinite numerical feature values",
        not infinite_columns,
        failures,
    )

    nonnegative_columns = [
        "amount",
        "balance_before",
        "balance_utilisation",
        "prior_transaction_count",
        "transactions_last_10min",
        "transactions_last_1h",
        "transactions_last_24h",
        "spend_last_1h",
        "spend_last_24h",
        "prior_device_transaction_count",
        "prior_merchant_transaction_count",
    ]

    for column in nonnegative_columns:
        check(
            f"{column} is nonnegative",
            bool(
                (
                    features[column].dropna()
                    >= 0
                ).all()
            ),
            failures,
        )

    print("\n4. COLD-START SEMANTICS")
    print("-" * 78)

    no_history = (
        features["prior_transaction_count"] == 0
    )

    one_prior = (
        features["prior_transaction_count"] == 1
    )

    check(
        "No-history rows have null prior_mean_amount",
        features.loc[
            no_history,
            "prior_mean_amount",
        ].isna().all(),
        failures,
    )

    check(
        "No-history rows have null amount_vs_prior_mean",
        features.loc[
            no_history,
            "amount_vs_prior_mean",
        ].isna().all(),
        failures,
    )

    check(
        "No-history rows have null minutes_since_previous",
        features.loc[
            no_history,
            "minutes_since_previous",
        ].isna().all(),
        failures,
    )

    check(
        "Rows with <2 prior transactions have null prior_std_amount",
        features.loc[
            features["prior_transaction_count"] < 2,
            "prior_std_amount",
        ].isna().all(),
        failures,
    )

    check(
        "Rows with <2 prior transactions have null amount_zscore",
        features.loc[
            features["prior_transaction_count"] < 2,
            "amount_zscore",
        ].isna().all(),
        failures,
    )

    check(
        "Rows with one prior transaction have prior mean",
        features.loc[
            one_prior,
            "prior_mean_amount",
        ].notna().all(),
        failures,
    )

    print("\nCold-start distribution:")
    print(
        features["prior_transaction_count"]
        .value_counts()
        .sort_index()
        .head(10)
        .to_string()
    )

    print("\n5. VELOCITY WINDOW CONSISTENCY")
    print("-" * 78)

    check(
        "10-minute count <= 1-hour count",
        bool(
            (
                features["transactions_last_10min"]
                <= features["transactions_last_1h"]
            ).all()
        ),
        failures,
    )

    check(
        "1-hour count <= 24-hour count",
        bool(
            (
                features["transactions_last_1h"]
                <= features["transactions_last_24h"]
            ).all()
        ),
        failures,
    )

    check(
        "1-hour spend <= 24-hour spend",
        bool(
            (
                features["spend_last_1h"]
                <= features["spend_last_24h"] + 1e-9
            ).all()
        ),
        failures,
    )

    print("\n6. BOOLEAN FEATURE INTEGRITY")
    print("-" * 78)

    binary_columns = [
        "new_device",
        "new_merchant",
        "is_weekend",
        "unusual_hour",
        "province_change",
        "is_international",
        "is_fraud",
    ]

    for column in binary_columns:
        check(
            f"{column} contains only binary values",
            set(
                features[column]
                .dropna()
                .astype(int)
                .unique()
            ) <= {0, 1},
            failures,
        )

    print("\n7. FRAUD BEHAVIOURAL COVERAGE")
    print("-" * 78)

    fraud = features[
        features["is_fraud"] == 1
    ]

    coverage = (
        fraud.groupby("fraud_type")
        .agg(
            transactions=(
                "transaction_id",
                "count",
            ),
            median_prior_count=(
                "prior_transaction_count",
                "median",
            ),
            median_amount_ratio=(
                "amount_vs_prior_mean",
                "median",
            ),
            median_amount_zscore=(
                "amount_zscore",
                "median",
            ),
            median_velocity_10m=(
                "transactions_last_10min",
                "median",
            ),
            new_device_rate=(
                "new_device",
                "mean",
            ),
            new_merchant_rate=(
                "new_merchant",
                "mean",
            ),
            province_change_rate=(
                "province_change",
                "mean",
            ),
            unusual_hour_rate=(
                "unusual_hour",
                "mean",
            ),
        )
    )

    percentage_columns = [
        "new_device_rate",
        "new_merchant_rate",
        "province_change_rate",
        "unusual_hour_rate",
    ]

    coverage[percentage_columns] *= 100

    print(
        coverage
        .round(3)
        .to_string()
    )

    print("\n8. HISTORY-DEPTH SEPARATION")
    print("-" * 78)

    history_summary = (
        features.groupby("is_fraud")[
            "prior_transaction_count"
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )
    )

    print(history_summary.round(3).to_string())

    legitimate_mean = features.loc[
        features["is_fraud"] == 0,
        "prior_transaction_count",
    ].mean()

    fraud_mean = features.loc[
        features["is_fraud"] == 1,
        "prior_transaction_count",
    ].mean()

    ratio = (
        fraud_mean / legitimate_mean
        if legitimate_mean > 0
        else np.nan
    )

    print(
        "\nFraud/legitimate mean prior-history ratio: "
        f"{ratio:.3f}x"
    )

    if ratio >= 2:
        print(
            "[WARNING] Fraud transactions have substantially "
            "more prior history than legitimate transactions."
        )
        print(
            "          Treat prior_transaction_count as a "
            "potential synthetic shortcut during modelling."
        )

    print("\n9. FINAL ASSESSMENT")
    print("=" * 78)

    if failures:
        print(
            f"OVERALL STATUS: FAIL "
            f"({len(failures)} failed checks)"
        )

        for failure in failures:
            print(f"  - {failure}")

        raise SystemExit(1)

    print("OVERALL STATUS: PASS")
    print(
        "Gold feature table satisfies the current structural, "
        "feature-policy, numerical and cold-start checks."
    )

    print(
        "History-depth separation remains a modelling risk "
        "to investigate rather than a structural validation failure."
    )


if __name__ == "__main__":
    main()