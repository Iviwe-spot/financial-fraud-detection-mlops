from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TRANSACTIONS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "transactions_with_fraud.parquet"
)

CUSTOMERS_PATH = (
    ROOT
    / "data"
    / "raw"
    / "customers.parquet"
)


EXPECTED_FRAUD_TYPES = {
    "ACCOUNT_TAKEOVER",
    "CARD_TESTING",
    "VELOCITY_ATTACK",
    "ABNORMAL_AMOUNT",
    "GEOGRAPHIC_ANOMALY",
    "BALANCE_DRAINING",
}


IDENTIFIER_COLUMNS = [
    "transaction_id",
    "account_id",
    "customer_id",
    "fraud_scenario_id",
]


CATEGORICAL_COLUMNS = [
    "merchant",
    "merchant_category",
    "transaction_type",
    "channel",
    "device_id",
    "province",
    "country",
    "direction",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRANSACTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing processed dataset: {TRANSACTIONS_PATH}"
        )

    if not CUSTOMERS_PATH.exists():
        raise FileNotFoundError(
            f"Missing customer dataset: {CUSTOMERS_PATH}"
        )

    transactions = pd.read_parquet(
        TRANSACTIONS_PATH
    )

    customers = pd.read_parquet(
        CUSTOMERS_PATH
    )

    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"]
    )

    return transactions, customers


def print_header(title: str) -> None:
    print()
    print(title)
    print("=" * 88)


def print_subheader(title: str) -> None:
    print()
    print(title)
    print("-" * 88)


def audit_class_balance(
    transactions: pd.DataFrame,
) -> None:
    print_header("1. CLASS BALANCE")

    counts = (
        transactions["is_fraud"]
        .value_counts()
        .sort_index()
    )

    rates = (
        transactions["is_fraud"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    for label in counts.index:
        name = (
            "Fraud"
            if label == 1
            else "Legitimate"
        )

        print(
            f"{name:<12} "
            f"{counts[label]:>8,} "
            f"({rates[label]:>7.3f}%)"
        )

    fraud = transactions[
        transactions["is_fraud"] == 1
    ]

    print_subheader(
        "Fraud distribution by typology"
    )

    summary = (
        fraud
        .groupby("fraud_type")
        .agg(
            transactions=(
                "transaction_id",
                "count",
            ),
            scenarios=(
                "fraud_scenario_id",
                "nunique",
            ),
            customers=(
                "customer_id",
                "nunique",
            ),
            accounts=(
                "account_id",
                "nunique",
            ),
        )
        .sort_values(
            "transactions",
            ascending=False,
        )
    )

    summary["transaction_share_pct"] = (
        summary["transactions"]
        / len(fraud)
        * 100
    )

    print(summary.to_string())


def categorical_purity_table(
    transactions: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    data = transactions[
        transactions[column].notna()
    ].copy()

    grouped = (
        data
        .groupby(column)
        .agg(
            total=(
                "transaction_id",
                "count",
            ),
            fraud=(
                "is_fraud",
                "sum",
            ),
        )
    )

    grouped["legitimate"] = (
        grouped["total"]
        - grouped["fraud"]
    )

    grouped["fraud_rate"] = (
        grouped["fraud"]
        / grouped["total"]
    )

    return grouped.sort_values(
        [
            "fraud_rate",
            "total",
        ],
        ascending=[
            False,
            False,
        ],
    )


def audit_categorical_leakage(
    transactions: pd.DataFrame,
) -> None:
    print_header(
        "2. CATEGORICAL LEAKAGE AUDIT"
    )

    for column in CATEGORICAL_COLUMNS:
        print_subheader(
            f"Feature: {column}"
        )

        table = categorical_purity_table(
            transactions,
            column,
        )

        fraud_only = table[
            (table["fraud"] > 0)
            & (table["legitimate"] == 0)
        ]

        shared = table[
            (table["fraud"] > 0)
            & (table["legitimate"] > 0)
        ]

        fraud_rows_from_fraud_only = int(
            fraud_only["fraud"].sum()
        )

        total_fraud = int(
            transactions["is_fraud"].sum()
        )

        coverage = (
            fraud_rows_from_fraud_only
            / total_fraud
            if total_fraud
            else 0.0
        )

        print(
            "Fraud-only values: "
            f"{len(fraud_only):,}"
        )

        print(
            "Shared fraud/legitimate values: "
            f"{len(shared):,}"
        )

        print(
            "Fraud transactions identifiable "
            "through fraud-only values: "
            f"{fraud_rows_from_fraud_only:,}/"
            f"{total_fraud:,} "
            f"({coverage:.1%})"
        )

        if not fraud_only.empty:
            print()
            print(
                "Top fraud-only values:"
            )

            display = (
                fraud_only
                .head(10)
                .copy()
            )

            print(
                display[
                    [
                        "total",
                        "fraud",
                        "fraud_rate",
                    ]
                ]
                .to_string()
            )


def audit_identifier_patterns(
    transactions: pd.DataFrame,
) -> None:
    print_header(
        "3. IDENTIFIER / STRING ARTEFACT AUDIT"
    )

    fraud = transactions[
        transactions["is_fraud"] == 1
    ].copy()

    legitimate = transactions[
        transactions["is_fraud"] == 0
    ].copy()

    checks = {
        "transaction_id starts FRAUD_": (
            transactions["transaction_id"]
            .astype(str)
            .str.startswith("FRAUD_")
        ),
        "device_id starts ATO_DEVICE_": (
            transactions["device_id"]
            .astype(str)
            .str.startswith(
                "ATO_DEVICE_"
            )
        ),
        "device_id starts CARD_TEST_DEVICE_": (
            transactions["device_id"]
            .astype(str)
            .str.startswith(
                "CARD_TEST_DEVICE_"
            )
        ),
        "merchant UNKNOWN_ONLINE_MERCHANT": (
            transactions["merchant"]
            .astype(str)
            .eq(
                "UNKNOWN_ONLINE_MERCHANT"
            )
        ),
        "merchant UNKNOWN_TEST_MERCHANT": (
            transactions["merchant"]
            .astype(str)
            .eq(
                "UNKNOWN_TEST_MERCHANT"
            )
        ),
    }

    rows = []

    for name, mask in checks.items():
        fraud_hits = int(
            (
                mask
                & transactions[
                    "is_fraud"
                ].eq(1)
            ).sum()
        )

        legitimate_hits = int(
            (
                mask
                & transactions[
                    "is_fraud"
                ].eq(0)
            ).sum()
        )

        rows.append(
            {
                "pattern": name,
                "fraud_hits": fraud_hits,
                "legitimate_hits": (
                    legitimate_hits
                ),
                "perfect_fraud_marker": (
                    fraud_hits > 0
                    and legitimate_hits == 0
                ),
            }
        )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False
        )
    )

    print()
    print(
        "Fraud transaction ID examples:"
    )

    print(
        fraud[
            [
                "transaction_id",
                "fraud_type",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )

    print()
    print(
        "Legitimate transaction ID examples:"
    )

    print(
        legitimate[
            ["transaction_id"]
        ]
        .head(5)
        .to_string(index=False)
    )


def build_behavioural_features(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    data = transactions.copy()

    customer_profiles = (
        customers[
            [
                "customer_id",
                "avg_transaction",
                "std_transaction",
                "usual_start_hour",
                "usual_end_hour",
            ]
        ]
        .drop_duplicates(
            "customer_id"
        )
    )

    data = data.merge(
        customer_profiles,
        on="customer_id",
        how="left",
    )

    data["amount_vs_average"] = (
        data["amount"]
        / data["avg_transaction"]
        .replace(0, np.nan)
    )

    data["amount_zscore"] = (
        (
            data["amount"]
            - data["avg_transaction"]
        )
        / data["std_transaction"]
        .replace(0, np.nan)
    )

    data["hour"] = (
        data["timestamp"].dt.hour
    )

    def hour_is_usual(row) -> bool:
        if (
            pd.isna(
                row["usual_start_hour"]
            )
            or pd.isna(
                row["usual_end_hour"]
            )
        ):
            return False

        hour = int(row["hour"])
        start = int(
            row["usual_start_hour"]
        )
        end = int(
            row["usual_end_hour"]
        )

        if start <= end:
            return (
                start
                <= hour
                <= end
            )

        return (
            hour >= start
            or hour <= end
        )

    data["unusual_hour"] = ~data.apply(
        hour_is_usual,
        axis=1,
    )

    data = data.sort_values(
        [
            "customer_id",
            "timestamp",
            "transaction_id",
        ]
    ).copy()

    data["previous_timestamp"] = (
        data
        .groupby("customer_id")[
            "timestamp"
        ]
        .shift(1)
    )

    data["minutes_since_previous"] = (
        (
            data["timestamp"]
            - data["previous_timestamp"]
        )
        .dt.total_seconds()
        / 60.0
    )

    data["previous_province"] = (
        data
        .groupby("customer_id")[
            "province"
        ]
        .shift(1)
    )

    data["province_change"] = (
        data["previous_province"]
        .notna()
        & (
            data["province"]
            != data["previous_province"]
        )
    )

    return data


def audit_behavioural_separation(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
) -> None:
    print_header(
        "4. CROSS-TYPOLOGY BEHAVIOURAL SEPARATION"
    )

    data = build_behavioural_features(
        transactions,
        customers,
    )

    fraud = data[
        data["is_fraud"] == 1
    ].copy()

    metrics = (
        fraud
        .groupby("fraud_type")
        .agg(
            transactions=(
                "transaction_id",
                "count",
            ),
            median_amount=(
                "amount",
                "median",
            ),
            median_amount_vs_average=(
                "amount_vs_average",
                "median",
            ),
            median_amount_zscore=(
                "amount_zscore",
                "median",
            ),
            unusual_hour_rate=(
                "unusual_hour",
                "mean",
            ),
            province_change_rate=(
                "province_change",
                "mean",
            ),
            median_minutes_since_previous=(
                "minutes_since_previous",
                "median",
            ),
        )
    )

    metrics[
        "unusual_hour_rate"
    ] *= 100

    metrics[
        "province_change_rate"
    ] *= 100

    print(
        metrics.round(2).to_string()
    )


def audit_account_overlap(
    transactions: pd.DataFrame,
) -> None:
    print_header(
        "5. CROSS-TYPOLOGY ACCOUNT OVERLAP"
    )

    fraud = transactions[
        transactions["is_fraud"] == 1
    ]

    account_types = (
        fraud
        .groupby("account_id")[
            "fraud_type"
        ]
        .nunique()
    )

    customer_types = (
        fraud
        .groupby("customer_id")[
            "fraud_type"
        ]
        .nunique()
    )

    overlapping_accounts = (
        account_types[
            account_types > 1
        ]
    )

    overlapping_customers = (
        customer_types[
            customer_types > 1
        ]
    )

    print(
        "Accounts appearing in multiple fraud "
        f"typologies: {len(overlapping_accounts):,}"
    )

    print(
        "Customers appearing in multiple fraud "
        f"typologies: {len(overlapping_customers):,}"
    )

    if not overlapping_accounts.empty:
        print()
        print(
            "Overlapping account details:"
        )

        details = (
            fraud[
                fraud["account_id"].isin(
                    overlapping_accounts.index
                )
            ]
            .groupby("account_id")[
                "fraud_type"
            ]
            .unique()
        )

        print(details.to_string())


def print_recommendations(
    transactions: pd.DataFrame,
) -> None:
    print_header(
        "6. PRELIMINARY ML LEAKAGE POLICY"
    )

    print(
        "Always exclude from model features:"
    )

    for column in [
        "is_fraud",
        "fraud_type",
        "fraud_scenario_id",
        "transaction_id",
    ]:
        print(
            f"  - {column}"
        )

    print()
    print(
        "Treat high-cardinality raw identifiers "
        "with caution:"
    )

    for column in [
        "customer_id",
        "account_id",
        "device_id",
    ]:
        print(
            f"  - {column}"
        )

    print()
    print(
        "The audit above determines whether "
        "merchant/category/channel/device metadata "
        "also contain synthetic shortcuts that should "
        "be redesigned before ML."
    )



def final_ml_readiness_assessment(
    transactions: pd.DataFrame,
) -> None:
    """
    Summarise synthetic leakage risk and define the feature policy
    that must be applied before model training.
    """

    print_header(
        "7. FINAL ML-READINESS ASSESSMENT"
    )

    total_fraud = int(
        transactions["is_fraud"].sum()
    )

    fraud_rate = float(
        transactions["is_fraud"].mean()
    )

    def fraud_only_coverage(
        column: str,
    ) -> tuple[int, float]:
        table = categorical_purity_table(
            transactions,
            column,
        )

        fraud_only = table[
            (table["fraud"] > 0)
            & (table["legitimate"] == 0)
        ]

        fraud_hits = int(
            fraud_only["fraud"].sum()
        )

        coverage = (
            fraud_hits / total_fraud
            if total_fraud
            else 0.0
        )

        return fraud_hits, coverage

    merchant_hits, merchant_coverage = (
        fraud_only_coverage("merchant")
    )

    category_hits, category_coverage = (
        fraud_only_coverage(
            "merchant_category"
        )
    )

    device_hits, device_coverage = (
        fraud_only_coverage("device_id")
    )

    transaction_id_mask = (
        transactions["transaction_id"]
        .astype(str)
        .str.startswith("FRAUD_")
    )

    transaction_id_fraud_hits = int(
        (
            transaction_id_mask
            & transactions["is_fraud"].eq(1)
        ).sum()
    )

    transaction_id_legitimate_hits = int(
        (
            transaction_id_mask
            & transactions["is_fraud"].eq(0)
        ).sum()
    )

    transaction_id_coverage = (
        transaction_id_fraud_hits
        / total_fraud
        if total_fraud
        else 0.0
    )

    fraud = transactions[
        transactions["is_fraud"] == 1
    ]

    account_typology_counts = (
        fraud
        .groupby("account_id")[
            "fraud_type"
        ]
        .nunique()
    )

    customer_typology_counts = (
        fraud
        .groupby("customer_id")[
            "fraud_type"
        ]
        .nunique()
    )

    overlapping_accounts = int(
        (
            account_typology_counts > 1
        ).sum()
    )

    overlapping_customers = int(
        (
            customer_typology_counts > 1
        ).sum()
    )

    def risk_label(
        coverage: float,
    ) -> str:
        if coverage >= 0.80:
            return "CRITICAL"

        if coverage >= 0.25:
            return "HIGH"

        if coverage > 0:
            return "MODERATE"

        return "NONE"

    print(
        f"Fraud rate: "
        f"{fraud_rate:.3%}"
    )

    print()

    print(
        "Synthetic leakage indicators:"
    )

    leakage_rows = [
        (
            "Fraud-only merchant values",
            merchant_hits,
            merchant_coverage,
        ),
        (
            "Fraud-only merchant-category values",
            category_hits,
            category_coverage,
        ),
        (
            "Fraud-only raw device values",
            device_hits,
            device_coverage,
        ),
        (
            "FRAUD_ transaction-ID pattern",
            transaction_id_fraud_hits,
            transaction_id_coverage,
        ),
    ]

    for (
        name,
        hits,
        coverage,
    ) in leakage_rows:
        print(
            f"  {name:<42} "
            f"{hits:>4}/{total_fraud:<4} "
            f"{coverage:>7.1%} "
            f"{risk_label(coverage):>10}"
        )

    if transaction_id_legitimate_hits == 0:
        print(
            "\n  FRAUD_ transaction-ID pattern "
            "appears in zero legitimate transactions."
        )

    print()

    print(
        "Cross-typology isolation:"
    )

    print(
        "  Accounts appearing in multiple "
        f"typologies: {overlapping_accounts}"
    )

    print(
        "  Customers appearing in multiple "
        f"typologies: {overlapping_customers}"
    )

    print()

    print(
        "MODEL FEATURE POLICY"
    )

    print(
        "  EXCLUDE TARGET/METADATA:"
    )
    print(
        "    - is_fraud from X "
        "(retain separately as y)"
    )
    print(
        "    - fraud_type"
    )
    print(
        "    - fraud_scenario_id"
    )

    print(
        "  EXCLUDE RAW IDENTIFIERS:"
    )
    print(
        "    - transaction_id"
    )
    print(
        "    - customer_id"
    )
    print(
        "    - account_id"
    )
    print(
        "    - device_id"
    )

    print(
        "  REDESIGN SYNTHETIC SHORTCUTS:"
    )
    print(
        "    - fraud-specific merchant names"
    )
    print(
        "    - fraud-specific merchant categories"
    )
    print(
        "    - recognisable attacker-device prefixes"
    )
    print(
        "    - recognisable fraud transaction-ID prefixes"
    )

    print(
        "  RETAIN / ENGINEER:"
    )
    print(
        "    - transaction_type"
    )
    print(
        "    - channel"
    )
    print(
        "    - province / country context"
    )
    print(
        "    - customer-relative amount features"
    )
    print(
        "    - point-in-time device novelty"
    )
    print(
        "    - point-in-time merchant novelty"
    )
    print(
        "    - transaction velocity features"
    )
    print(
        "    - temporal behaviour"
    )
    print(
        "    - geographic transitions"
    )
    print(
        "    - balance utilisation / depletion"
    )

    critical_or_high = any(
        coverage >= 0.25
        for _, _, coverage
        in leakage_rows
    )

    print()

    if critical_or_high:
        print(
            "OVERALL ML READINESS: NOT YET READY"
        )

        print(
            "Reason: high-severity synthetic "
            "categorical/identifier shortcuts are "
            "present in the raw transaction data."
        )

        print(
            "Next action: construct and validate a "
            "point-in-time behavioural Gold feature "
            "table that excludes raw leakage fields, "
            "then redesign generator artefacts before "
            "the final modelling benchmark."
        )

    else:
        print(
            "OVERALL ML READINESS: READY FOR "
            "CONTROLLED BASELINE MODELLING"
        )


def main() -> None:
    transactions, customers = load_data()

    print(
        "FRAUD DATASET LEAKAGE AND "
        "SEPARATION AUDIT"
    )

    print(
        "=" * 88
    )

    print(
        f"Transactions: "
        f"{len(transactions):,}"
    )

    print(
        f"Fraud: "
        f"{int(transactions['is_fraud'].sum()):,}"
    )

    print(
        f"Fraud rate: "
        f"{transactions['is_fraud'].mean():.3%}"
    )

    audit_class_balance(
        transactions
    )

    audit_categorical_leakage(
        transactions
    )

    audit_identifier_patterns(
        transactions
    )

    audit_behavioural_separation(
        transactions,
        customers,
    )

    audit_account_overlap(
        transactions
    )

    print_recommendations(
        transactions
    )

    final_ml_readiness_assessment(
        transactions
    )


if __name__ == "__main__":
    main()
