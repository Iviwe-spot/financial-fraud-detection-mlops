#!/usr/bin/env python3
"""Validate relational integrity and baseline behavioural distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    customers = pd.read_parquet(data_dir / "customers.parquet")
    accounts = pd.read_parquet(data_dir / "accounts.parquet")
    tx = pd.read_parquet(data_dir / "transactions.parquet")

    checks = {
        "unique_customer_ids": customers["customer_id"].is_unique,
        "unique_account_ids": accounts["account_id"].is_unique,
        "unique_transaction_ids": tx["transaction_id"].is_unique,
        "account_customer_fk": accounts["customer_id"].isin(customers["customer_id"]).all(),
        "transaction_customer_fk": tx["customer_id"].isin(customers["customer_id"]).all(),
        "transaction_account_fk": tx["account_id"].isin(accounts["account_id"]).all(),
        "positive_amounts": tx["amount"].gt(0).all(),
        "nonnegative_balances": tx[["balance_before", "balance_after"]].ge(0).all().all(),
        "no_fraud_columns": not any(c in tx for c in ["is_fraud", "fraud_rule", "fraud_scenario"]),
    }
    report = pd.Series(checks, name="passed")
    print(report.to_string())
    print("\nTransaction amount summary (ZAR):")
    print(tx["amount"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(2).to_string())
    print("\nChannel share:")
    print(tx["channel"].value_counts(normalize=True).mul(100).round(2).to_string())
    print(f"\nInternational share: {tx['is_international'].mean() * 100:.2f}%")
    if not report.all():
        raise SystemExit("Validation failed")


if __name__ == "__main__":
    main()
