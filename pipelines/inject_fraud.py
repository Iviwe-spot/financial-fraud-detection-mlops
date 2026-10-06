#!/usr/bin/env python3
"""Inject multiple fraud typologies into the normal transaction baseline."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator.fraud_injector import (
    FRAUD_TYPE_ACCOUNT_TAKEOVER,
    FRAUD_TYPE_CARD_TESTING,
    FRAUD_TYPE_VELOCITY_ATTACK,
    inject_account_takeover,
    inject_card_testing,
    inject_velocity_attack,
)


RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SAMPLE_DIR = ROOT / "data" / "sample"

CUSTOMERS_PATH = RAW_DIR / "customers.parquet"
ACCOUNTS_PATH = RAW_DIR / "accounts.parquet"
TRANSACTIONS_PATH = RAW_DIR / "transactions.parquet"

OUTPUT_PATH = PROCESSED_DIR / "transactions_with_fraud.parquet"
SAMPLE_PATH = SAMPLE_DIR / "fraud_sample.csv"

EXPECTED_FRAUD_TYPES = {
    FRAUD_TYPE_ACCOUNT_TAKEOVER,
    FRAUD_TYPE_CARD_TESTING,
    FRAUD_TYPE_VELOCITY_ATTACK,
}


def load_source_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load the generated customer, account and transaction datasets."""
    required_files = [
        CUSTOMERS_PATH,
        ACCOUNTS_PATH,
        TRANSACTIONS_PATH,
    ]

    missing_files = [
        path for path in required_files if not path.exists()
    ]

    if missing_files:
        missing = "\n".join(
            f"  - {path.relative_to(ROOT)}"
            for path in missing_files
        )

        raise FileNotFoundError(
            "The following source files are missing:\n"
            f"{missing}\n"
            "Run the normal-data generation pipeline first."
        )

    customers = pd.read_parquet(CUSTOMERS_PATH)
    accounts = pd.read_parquet(ACCOUNTS_PATH)
    transactions = pd.read_parquet(TRANSACTIONS_PATH)

    return customers, accounts, transactions


def inject_fraud_scenarios(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the three fraud typologies sequentially."""
    output = inject_account_takeover(
        transactions=transactions,
        accounts=accounts,
        customers=customers,
        n_scenarios=30,
        min_transactions=2,
        max_transactions=5,
        seed=42,
    )

    output = inject_card_testing(
        transactions=output,
        accounts=accounts,
        customers=customers,
        n_scenarios=25,
        min_transactions=5,
        max_transactions=12,
        seed=43,
    )

    output = inject_velocity_attack(
        transactions=output,
        accounts=accounts,
        customers=customers,
        n_scenarios=25,
        min_transactions=5,
        max_transactions=10,
        seed=44,
    )

    return output


def validate_output(
    baseline: pd.DataFrame,
    transactions_with_fraud: pd.DataFrame,
    accounts: pd.DataFrame,
    customers: pd.DataFrame,
) -> None:
    """Validate structural and ground-truth properties of the output."""
    required_columns = {
        "transaction_id",
        "customer_id",
        "account_id",
        "timestamp",
        "amount",
        "device_id",
        "is_fraud",
        "fraud_type",
        "fraud_scenario_id",
    }

    missing_columns = (
        required_columns
        - set(transactions_with_fraud.columns)
    )

    if missing_columns:
        raise ValueError(
            "Output is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if transactions_with_fraud["transaction_id"].duplicated().any():
        raise ValueError(
            "Duplicate transaction IDs were found."
        )

    valid_labels = {0, 1}
    observed_labels = set(
        transactions_with_fraud["is_fraud"]
        .dropna()
        .astype(int)
        .unique()
    )

    if not observed_labels.issubset(valid_labels):
        raise ValueError(
            "The is_fraud column contains invalid labels."
        )

    fraud = transactions_with_fraud[
        transactions_with_fraud["is_fraud"] == 1
    ]

    if fraud.empty:
        raise ValueError(
            "No fraudulent transactions were generated."
        )

    observed_fraud_types = set(
        fraud["fraud_type"].dropna().unique()
    )

    missing_fraud_types = (
        EXPECTED_FRAUD_TYPES
        - observed_fraud_types
    )

    if missing_fraud_types:
        raise ValueError(
            "Expected fraud typologies were not generated: "
            f"{sorted(missing_fraud_types)}"
        )

    if fraud["fraud_scenario_id"].isna().any():
        raise ValueError(
            "Some fraudulent transactions have no scenario ID."
        )

    if (fraud["amount"] <= 0).any():
        raise ValueError(
            "Fraudulent transaction amounts must be positive."
        )

    valid_account_ids = set(accounts["account_id"])
    invalid_accounts = (
        set(transactions_with_fraud["account_id"])
        - valid_account_ids
    )

    if invalid_accounts:
        raise ValueError(
            "Some transactions reference unknown accounts."
        )

    valid_customer_ids = set(customers["customer_id"])
    invalid_customers = (
        set(transactions_with_fraud["customer_id"])
        - valid_customer_ids
    )

    if invalid_customers:
        raise ValueError(
            "Some transactions reference unknown customers."
        )

    original_ids = set(baseline["transaction_id"])
    output_original_rows = transactions_with_fraud[
        transactions_with_fraud["transaction_id"].isin(
            original_ids
        )
    ]

    if len(output_original_rows) != len(baseline):
        raise ValueError(
            "One or more baseline transactions were lost."
        )

    if (
        output_original_rows["is_fraud"]
        .fillna(0)
        .astype(int)
        .sum()
        != 0
    ):
        raise ValueError(
            "A baseline transaction was incorrectly labelled "
            "as fraud."
        )

    fraud_count = len(fraud)
    expected_rows = len(baseline) + fraud_count

    if len(transactions_with_fraud) != expected_rows:
        raise ValueError(
            "Output row count does not equal baseline rows "
            "plus injected fraud rows."
        )


def build_fraud_summary(
    transactions_with_fraud: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the fraud records and an aggregated summary."""
    fraud = transactions_with_fraud[
        transactions_with_fraud["is_fraud"] == 1
    ].copy()

    summary = (
        fraud.groupby("fraud_type")
        .agg(
            scenarios=(
                "fraud_scenario_id",
                "nunique",
            ),
            transactions=(
                "transaction_id",
                "count",
            ),
            mean_amount=(
                "amount",
                "mean",
            ),
            median_amount=(
                "amount",
                "median",
            ),
            minimum_amount=(
                "amount",
                "min",
            ),
            maximum_amount=(
                "amount",
                "max",
            ),
        )
        .sort_index()
        .round(2)
    )

    return fraud, summary


def print_examples(
    fraud: pd.DataFrame,
    fraud_type: str,
    heading: str,
) -> None:
    """Print five example records for one fraud typology."""
    columns = [
        "transaction_id",
        "customer_id",
        "timestamp",
        "amount",
        "device_id",
        "fraud_scenario_id",
    ]

    examples = (
        fraud.loc[
            fraud["fraud_type"] == fraud_type,
            columns,
        ]
        .sort_values(
            [
                "fraud_scenario_id",
                "timestamp",
            ]
        )
        .head(5)
    )

    print(f"\nExample {heading} records:")
    print(examples.to_string(index=False))


def save_output(
    transactions_with_fraud: pd.DataFrame,
    fraud: pd.DataFrame,
) -> None:
    """Write the processed dataset and a small inspection sample."""
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAMPLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    transactions_with_fraud.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    fraud_sample = (
        fraud.sort_values(
            [
                "fraud_type",
                "fraud_scenario_id",
                "timestamp",
            ]
        )
        .groupby(
            "fraud_type",
            group_keys=False,
        )
        .head(350)
    )

    fraud_sample.to_csv(
        SAMPLE_PATH,
        index=False,
    )


def main() -> None:
    customers, accounts, transactions = load_source_data()

    print(f"Customers: {len(customers):,}")
    print(f"Accounts: {len(accounts):,}")
    print(f"Baseline transactions: {len(transactions):,}")

    transactions_with_fraud = inject_fraud_scenarios(
        transactions=transactions,
        accounts=accounts,
        customers=customers,
    )

    validate_output(
        baseline=transactions,
        transactions_with_fraud=transactions_with_fraud,
        accounts=accounts,
        customers=customers,
    )

    fraud, summary = build_fraud_summary(
        transactions_with_fraud
    )

    fraud_rate = (
        transactions_with_fraud["is_fraud"].mean()
        * 100
    )

    total_scenarios = (
        fraud["fraud_scenario_id"].nunique()
    )

    print("\nFraud summary:")
    print(summary.to_string())

    print(f"\nTotal fraud scenarios: {total_scenarios:,}")
    print(
        "Total output rows: "
        f"{len(transactions_with_fraud):,}"
    )
    print(
        "Fraudulent transactions: "
        f"{len(fraud):,}"
    )
    print(f"Fraud rate: {fraud_rate:.3f}%")

    print_examples(
        fraud=fraud,
        fraud_type=FRAUD_TYPE_ACCOUNT_TAKEOVER,
        heading="Account Takeover",
    )

    print_examples(
        fraud=fraud,
        fraud_type=FRAUD_TYPE_CARD_TESTING,
        heading="Card Testing",
    )

    print_examples(
        fraud=fraud,
        fraud_type=FRAUD_TYPE_VELOCITY_ATTACK,
        heading="Velocity Attack",
    )

    save_output(
        transactions_with_fraud=transactions_with_fraud,
        fraud=fraud,
    )

    print(
        "\nSaved processed data to "
        "data/processed/transactions_with_fraud.parquet"
    )
    print(
        "Saved stratified inspection sample to "
        "data/sample/fraud_sample.csv"
    )


if __name__ == "__main__":
    main()