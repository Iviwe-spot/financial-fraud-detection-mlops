"""
Validate the Databricks export package.

Checks:
- manifest exists and is readable
- all expected datasets exist
- transaction partitions can be read
- partition row counts reconcile
- transaction IDs remain unique
- source/export transaction IDs match
- fraud and legitimate counts reconcile
- fraud typology/scenario counts reconcile
- year/month directory partitions match timestamps
- customer/account row counts reconcile
- customer/account IDs remain unique
- transaction foreign keys remain valid
- source and export schemas remain consistent
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


EXPORT_ROOT = Path("data/export/fraud_v2_1m")

SOURCE_TRANSACTIONS = Path(
    "data/processed/transactions_with_fraud.parquet"
)
SOURCE_CUSTOMERS = Path(
    "data/raw/customers.parquet"
)
SOURCE_ACCOUNTS = Path(
    "data/raw/accounts.parquet"
)


class ValidationReport:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(
        self,
        condition: bool,
        name: str,
        detail: str = "",
    ) -> None:
        if condition:
            self.passed += 1
            prefix = "[PASS]"
        else:
            self.failed += 1
            prefix = "[FAIL]"

        if detail:
            print(f"{prefix} {name} — {detail}")
        else:
            print(f"{prefix} {name}")

    @property
    def success(self) -> bool:
        return self.failed == 0


def load_manifest() -> dict:
    path = EXPORT_ROOT / "manifest.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    with path.open(
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_exported_transactions():
    root = EXPORT_ROOT / "transactions"

    files = sorted(
        root.glob(
            "year=*/month=*/transactions.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No transaction partitions found."
        )

    frames = []
    partition_metadata = []

    pattern = re.compile(
        r"year=(\d{4})/month=(\d{2})"
    )

    for path in files:
        relative = path.relative_to(root)

        match = pattern.search(
            relative.as_posix()
        )

        if match is None:
            raise ValueError(
                f"Invalid partition path: {path}"
            )

        year = int(match.group(1))
        month = int(match.group(2))

        frame = pd.read_parquet(path)

        frame["_partition_year"] = year
        frame["_partition_month"] = month

        frames.append(frame)

        partition_metadata.append(
            {
                "path": path,
                "year": year,
                "month": month,
                "rows": len(frame),
            }
        )

    transactions = pd.concat(
        frames,
        ignore_index=True,
    )

    return transactions, partition_metadata


def main() -> None:
    report = ValidationReport()

    print("=" * 78)
    print("DATABRICKS EXPORT VALIDATION REPORT")
    print("=" * 78)

    manifest = load_manifest()

    exported_transactions, partitions = (
        load_exported_transactions()
    )

    exported_customers = pd.read_parquet(
        EXPORT_ROOT
        / "customers"
        / "customers.parquet"
    )

    exported_accounts = pd.read_parquet(
        EXPORT_ROOT
        / "accounts"
        / "accounts.parquet"
    )

    source_transactions = pd.read_parquet(
        SOURCE_TRANSACTIONS
    )

    source_customers = pd.read_parquet(
        SOURCE_CUSTOMERS
    )

    source_accounts = pd.read_parquet(
        SOURCE_ACCOUNTS
    )

    exported_transactions["timestamp"] = (
        pd.to_datetime(
            exported_transactions["timestamp"]
        )
    )

    source_transactions["timestamp"] = (
        pd.to_datetime(
            source_transactions["timestamp"]
        )
    )

    print("\n1. FILE AND PARTITION VALIDATION")
    print("-" * 78)

    report.check(
        len(partitions)
        == manifest["partitioning"][
            "partition_count"
        ],
        "Partition count",
        (
            f"{len(partitions)} files found; "
            f"{manifest['partitioning']['partition_count']} "
            "declared"
        ),
    )

    partition_rows = sum(
        item["rows"]
        for item in partitions
    )

    report.check(
        partition_rows
        == len(exported_transactions),
        "Partition row reconciliation",
        f"{partition_rows:,} rows",
    )

    expected_manifest_rows = (
        manifest["row_counts"]["transactions"]
    )

    report.check(
        len(exported_transactions)
        == expected_manifest_rows,
        "Manifest transaction count",
        f"{len(exported_transactions):,} rows",
    )

    timestamp_partition_match = (
        (
            exported_transactions[
                "timestamp"
            ].dt.year
            == exported_transactions[
                "_partition_year"
            ]
        )
        & (
            exported_transactions[
                "timestamp"
            ].dt.month
            == exported_transactions[
                "_partition_month"
            ]
        )
    ).all()

    report.check(
        timestamp_partition_match,
        "Timestamp-to-partition consistency",
    )

    print("\n2. TRANSACTION INTEGRITY")
    print("-" * 78)

    report.check(
        exported_transactions[
            "transaction_id"
        ].notna().all(),
        "Transaction IDs are non-null",
    )

    report.check(
        exported_transactions[
            "transaction_id"
        ].is_unique,
        "Transaction IDs remain unique",
        (
            f"{exported_transactions['transaction_id'].nunique():,} "
            "unique IDs"
        ),
    )

    source_ids = set(
        source_transactions["transaction_id"]
    )

    export_ids = set(
        exported_transactions[
            "transaction_id"
        ]
    )

    report.check(
        source_ids == export_ids,
        "Source/export transaction ID equality",
    )

    report.check(
        len(source_transactions)
        == len(exported_transactions),
        "Source/export transaction row count",
        (
            f"{len(source_transactions):,} source; "
            f"{len(exported_transactions):,} export"
        ),
    )

    fraud_count = int(
        exported_transactions[
            "is_fraud"
        ].sum()
    )

    report.check(
        fraud_count
        == manifest["row_counts"][
            "fraudulent_transactions"
        ],
        "Fraud transaction count",
        f"{fraud_count:,}",
    )

    legitimate_count = int(
        (
            exported_transactions[
                "is_fraud"
            ]
            == 0
        ).sum()
    )

    report.check(
        legitimate_count
        == manifest["row_counts"][
            "legitimate_transactions"
        ],
        "Legitimate transaction count",
        f"{legitimate_count:,}",
    )

    fraud = exported_transactions[
        exported_transactions["is_fraud"] == 1
    ]

    exported_type_counts = (
        fraud["fraud_type"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    manifest_type_counts = {
        key: int(value)
        for key, value
        in manifest[
            "fraud_transactions_by_type"
        ].items()
    }

    report.check(
        exported_type_counts
        == manifest_type_counts,
        "Fraud typology transaction counts",
    )

    exported_scenario_counts = (
        fraud.groupby("fraud_type")[
            "fraud_scenario_id"
        ]
        .nunique()
        .sort_index()
        .to_dict()
    )

    manifest_scenario_counts = {
        key: int(value)
        for key, value
        in manifest[
            "fraud_scenarios_by_type"
        ].items()
    }

    report.check(
        exported_scenario_counts
        == manifest_scenario_counts,
        "Fraud typology scenario counts",
    )

    print("\n3. DIMENSION INTEGRITY")
    print("-" * 78)

    report.check(
        len(exported_customers)
        == len(source_customers),
        "Customer row count",
        f"{len(exported_customers):,}",
    )

    report.check(
        len(exported_accounts)
        == len(source_accounts),
        "Account row count",
        f"{len(exported_accounts):,}",
    )

    report.check(
        exported_customers[
            "customer_id"
        ].is_unique,
        "Customer IDs remain unique",
    )

    report.check(
        exported_accounts[
            "account_id"
        ].is_unique,
        "Account IDs remain unique",
    )

    report.check(
        set(
            exported_customers[
                "customer_id"
            ]
        )
        == set(
            source_customers[
                "customer_id"
            ]
        ),
        "Source/export customer ID equality",
    )

    report.check(
        set(
            exported_accounts[
                "account_id"
            ]
        )
        == set(
            source_accounts[
                "account_id"
            ]
        ),
        "Source/export account ID equality",
    )

    print("\n4. RELATIONAL INTEGRITY")
    print("-" * 78)

    report.check(
        set(
            exported_transactions[
                "customer_id"
            ]
        ).issubset(
            set(
                exported_customers[
                    "customer_id"
                ]
            )
        ),
        "Transaction customer foreign keys",
    )

    report.check(
        set(
            exported_transactions[
                "account_id"
            ]
        ).issubset(
            set(
                exported_accounts[
                    "account_id"
                ]
            )
        ),
        "Transaction account foreign keys",
    )

    account_customer = (
        exported_accounts[
            ["account_id", "customer_id"]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "customer_id":
                    "account_owner_customer_id"
            }
        )
    )

    ownership = (
        exported_transactions[
            ["account_id", "customer_id"]
        ]
        .merge(
            account_customer,
            on="account_id",
            how="left",
        )
    )

    report.check(
        (
            ownership["customer_id"]
            == ownership[
                "account_owner_customer_id"
            ]
        ).all(),
        "Account-customer ownership",
    )

    print("\n5. SCHEMA VALIDATION")
    print("-" * 78)

    exported_schema_columns = set(
        exported_transactions.columns
    ) - {
        "_partition_year",
        "_partition_month",
    }

    source_schema_columns = set(
        source_transactions.columns
    )

    report.check(
        exported_schema_columns
        == source_schema_columns,
        "Transaction schema columns",
    )

    report.check(
        set(exported_customers.columns)
        == set(source_customers.columns),
        "Customer schema columns",
    )

    report.check(
        set(exported_accounts.columns)
        == set(source_accounts.columns),
        "Account schema columns",
    )

    print("\n6. FINAL RESULT")
    print("-" * 78)

    print(
        f"Passed checks: {report.passed}"
    )
    print(
        f"Failed checks: {report.failed}"
    )

    if report.success:
        print("\nOVERALL STATUS: PASS")
    else:
        print("\nOVERALL STATUS: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
