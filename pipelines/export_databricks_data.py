"""
Prepare the validated synthetic fraud dataset for Databricks ingestion.

This pipeline:
1. Reads the validated 1M-scale synthetic banking dataset.
2. Partitions transactions by transaction year and month.
3. Exports customers and accounts as dimension-style Parquet datasets.
4. Creates a manifest containing lineage, schema and dataset statistics.

The source datasets are never modified.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_TRANSACTIONS = Path(
    "data/processed/transactions_with_fraud.parquet"
)
DEFAULT_CUSTOMERS = Path("data/raw/customers.parquet")
DEFAULT_ACCOUNTS = Path("data/raw/accounts.parquet")
DEFAULT_OUTPUT = Path("data/export/fraud_v2_1m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the validated synthetic fraud dataset "
            "for Databricks ingestion."
        )
    )

    parser.add_argument(
        "--transactions",
        type=Path,
        default=DEFAULT_TRANSACTIONS,
    )
    parser.add_argument(
        "--customers",
        type=Path,
        default=DEFAULT_CUSTOMERS,
    )
    parser.add_argument(
        "--accounts",
        type=Path,
        default=DEFAULT_ACCOUNTS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--generator-version",
        default="synthetic-fraud-v2",
    )

    return parser.parse_args()


def load_data(
    transactions_path: Path,
    customers_path: Path,
    accounts_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    for path in (
        transactions_path,
        customers_path,
        accounts_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required source file not found: {path}"
            )

    transactions = pd.read_parquet(transactions_path)
    customers = pd.read_parquet(customers_path)
    accounts = pd.read_parquet(accounts_path)

    return transactions, customers, accounts


def validate_export_inputs(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
) -> None:

    required_transaction_columns = {
        "transaction_id",
        "account_id",
        "customer_id",
        "timestamp",
        "amount",
        "is_fraud",
        "fraud_type",
        "fraud_scenario_id",
    }

    missing = (
        required_transaction_columns
        - set(transactions.columns)
    )

    if missing:
        raise ValueError(
            "Transaction dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    if transactions.empty:
        raise ValueError("Transaction dataset is empty.")

    if customers.empty:
        raise ValueError("Customer dataset is empty.")

    if accounts.empty:
        raise ValueError("Account dataset is empty.")

    if transactions["transaction_id"].duplicated().any():
        raise ValueError(
            "Transaction IDs are not unique."
        )

    if customers["customer_id"].duplicated().any():
        raise ValueError(
            "Customer IDs are not unique."
        )

    if accounts["account_id"].duplicated().any():
        raise ValueError(
            "Account IDs are not unique."
        )

    if not set(
        transactions["customer_id"]
    ).issubset(set(customers["customer_id"])):
        raise ValueError(
            "Transaction customer foreign keys are invalid."
        )

    if not set(
        transactions["account_id"]
    ).issubset(set(accounts["account_id"])):
        raise ValueError(
            "Transaction account foreign keys are invalid."
        )

    labels = set(
        transactions["is_fraud"]
        .dropna()
        .astype(int)
        .unique()
    )

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"Unexpected fraud labels: {sorted(labels)}"
        )


def prepare_transactions(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    result = transactions.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="raise",
    )

    result["year"] = (
        result["timestamp"]
        .dt.year
        .astype("int16")
    )

    result["month"] = (
        result["timestamp"]
        .dt.month
        .astype("int8")
    )

    return result


def export_partitioned_transactions(
    transactions: pd.DataFrame,
    output_root: Path,
) -> list[dict]:

    transaction_root = output_root / "transactions"
    transaction_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    partitions = []

    grouped = transactions.groupby(
        ["year", "month"],
        sort=True,
    )

    for (year, month), partition in grouped:

        partition_directory = (
            transaction_root
            / f"year={int(year)}"
            / f"month={int(month):02d}"
        )

        partition_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            partition_directory
            / "transactions.parquet"
        )

        # Partition keys are represented by the directory structure.
        export_frame = partition.drop(
            columns=["year", "month"]
        )

        export_frame.to_parquet(
            output_file,
            index=False,
        )

        partitions.append(
            {
                "year": int(year),
                "month": int(month),
                "rows": int(len(export_frame)),
                "file": str(
                    output_file.relative_to(output_root)
                ),
            }
        )

    return partitions


def export_dimensions(
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    output_root: Path,
) -> None:

    customer_directory = output_root / "customers"
    account_directory = output_root / "accounts"

    customer_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    account_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    customers.to_parquet(
        customer_directory / "customers.parquet",
        index=False,
    )

    accounts.to_parquet(
        account_directory / "accounts.parquet",
        index=False,
    )


def build_manifest(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    partitions: list[dict],
    seed: int,
    generator_version: str,
) -> dict:

    fraud = transactions[
        transactions["is_fraud"] == 1
    ]

    fraud_type_counts = (
        fraud["fraud_type"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    scenario_counts = (
        fraud.groupby("fraud_type")[
            "fraud_scenario_id"
        ]
        .nunique()
        .sort_index()
        .to_dict()
    )

    timestamp_min = transactions[
        "timestamp"
    ].min()

    timestamp_max = transactions[
        "timestamp"
    ].max()

    return {
        "dataset_name": (
            "synthetic_banking_fraud_v2_1m"
        ),
        "generator_version": generator_version,
        "seed": seed,
        "exported_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "validation_status": "PASS",
        "leakage_audit_status": "PASS",
        "row_counts": {
            "customers": int(len(customers)),
            "accounts": int(len(accounts)),
            "transactions": int(len(transactions)),
            "legitimate_transactions": int(
                (transactions["is_fraud"] == 0).sum()
            ),
            "fraudulent_transactions": int(
                len(fraud)
            ),
            "fraud_scenarios": int(
                fraud["fraud_scenario_id"].nunique()
            ),
        },
        "fraud_rate": float(
            transactions["is_fraud"].mean()
        ),
        "date_range": {
            "minimum_timestamp": (
                timestamp_min.isoformat()
            ),
            "maximum_timestamp": (
                timestamp_max.isoformat()
            ),
        },
        "fraud_transactions_by_type": {
            str(key): int(value)
            for key, value
            in fraud_type_counts.items()
        },
        "fraud_scenarios_by_type": {
            str(key): int(value)
            for key, value
            in scenario_counts.items()
        },
        "partitioning": {
            "dataset": "transactions",
            "columns": ["year", "month"],
            "partition_count": len(partitions),
            "partitions": partitions,
        },
        "schemas": {
            "transactions": {
                column: str(dtype)
                for column, dtype
                in transactions.drop(
                    columns=["year", "month"]
                ).dtypes.items()
            },
            "customers": {
                column: str(dtype)
                for column, dtype
                in customers.dtypes.items()
            },
            "accounts": {
                column: str(dtype)
                for column, dtype
                in accounts.dtypes.items()
            },
        },
        "model_feature_policy": {
            "target": "is_fraud",
            "exclude_from_features": [
                "fraud_type",
                "fraud_scenario_id",
                "transaction_id",
                "customer_id",
                "account_id",
                "device_id",
            ],
        },
    }


def write_manifest(
    manifest: dict,
    output_root: Path,
) -> None:

    path = output_root / "manifest.json"

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )


def main() -> None:
    args = parse_args()

    transactions, customers, accounts = load_data(
        transactions_path=args.transactions,
        customers_path=args.customers,
        accounts_path=args.accounts,
    )

    print(
        f"Transactions loaded: {len(transactions):,}"
    )
    print(
        f"Customers loaded:    {len(customers):,}"
    )
    print(
        f"Accounts loaded:     {len(accounts):,}"
    )

    validate_export_inputs(
        transactions=transactions,
        customers=customers,
        accounts=accounts,
    )

    transactions = prepare_transactions(
        transactions
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    partitions = export_partitioned_transactions(
        transactions=transactions,
        output_root=args.output,
    )

    export_dimensions(
        customers=customers,
        accounts=accounts,
        output_root=args.output,
    )

    manifest = build_manifest(
        transactions=transactions,
        customers=customers,
        accounts=accounts,
        partitions=partitions,
        seed=args.seed,
        generator_version=args.generator_version,
    )

    write_manifest(
        manifest=manifest,
        output_root=args.output,
    )

    print()
    print("Databricks export complete.")
    print(f"Output: {args.output}")
    print(
        f"Transaction partitions: {len(partitions)}"
    )
    print(
        "Transactions exported: "
        f"{len(transactions):,}"
    )
    print(
        "Fraud transactions: "
        f"{int(transactions['is_fraud'].sum()):,}"
    )
    print(
        "Fraud rate: "
        f"{transactions['is_fraud'].mean():.3%}"
    )


if __name__ == "__main__":
    main()
