#!/usr/bin/env python3
"""Inject fraud scenarios into the generated transaction baseline."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator.fraud_injector import (
    inject_account_takeover,
    inject_card_testing,
)


RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SAMPLE_DIR = ROOT / "data" / "sample"


def main() -> None:
    customers = pd.read_parquet(
        RAW_DIR / "customers.parquet"
    )

    accounts = pd.read_parquet(
        RAW_DIR / "accounts.parquet"
    )

    transactions = pd.read_parquet(
        RAW_DIR / "transactions.parquet"
    )

    print("Baseline transactions:", len(transactions))

    transactions_with_fraud = inject_account_takeover(
        transactions=transactions,
        accounts=accounts,
        customers=customers,
        n_scenarios=50,
        min_transactions=2,
        max_transactions=5,
        seed=42,
    )

    transactions_with_fraud = inject_card_testing(
        transactions=transactions_with_fraud,
        accounts=accounts,
        customers=customers,
        n_scenarios=40,
        min_transactions=5,
        max_transactions=12,
        seed=43,
    )

    fraud = transactions_with_fraud[
        transactions_with_fraud["is_fraud"] == 1
    ].copy()

    summary = (
        fraud.groupby("fraud_type")
        .agg(
            scenarios=("fraud_scenario_id", "nunique"),
            transactions=("transaction_id", "count"),
            mean_amount=("amount", "mean"),
            median_amount=("amount", "median"),
            minimum_amount=("amount", "min"),
            maximum_amount=("amount", "max"),
        )
        .round(2)
    )

    fraud_rate = (
        transactions_with_fraud["is_fraud"].mean()
        * 100
    )

    print("\nFraud summary:")
    print(summary.to_string())

    print(
        f"\nTotal output rows: "
        f"{len(transactions_with_fraud):,}"
    )

    print(
        f"Fraudulent transactions: "
        f"{len(fraud):,}"
    )

    print(
        f"Fraud rate: {fraud_rate:.3f}%"
    )

    print("\nExample Account Takeover records:")

    print(
        fraud[
            fraud["fraud_type"]
            == "ACCOUNT_TAKEOVER"
        ][
            [
                "transaction_id",
                "customer_id",
                "timestamp",
                "amount",
                "device_id",
                "fraud_scenario_id",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    print("\nExample Card Testing records:")

    print(
        fraud[
            fraud["fraud_type"]
            == "CARD_TESTING"
        ][
            [
                "transaction_id",
                "customer_id",
                "timestamp",
                "amount",
                "device_id",
                "fraud_scenario_id",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAMPLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    transactions_with_fraud.to_parquet(
        PROCESSED_DIR
        / "transactions_with_fraud.parquet",
        index=False,
    )

    fraud.head(1000).to_csv(
        SAMPLE_DIR / "fraud_sample.csv",
        index=False,
    )

    print(
        "\nSaved processed data to "
        "data/processed/transactions_with_fraud.parquet"
    )

    print(
        "Saved inspection sample to "
        "data/sample/fraud_sample.csv"
    )


if __name__ == "__main__":
    main()