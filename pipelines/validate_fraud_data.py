"""
Independent validation pipeline for the synthetic fraud dataset.

This module validates the persisted fraud dataset independently of the
fraud injection functions.

Validation layers currently implemented:

1. Structural and relational integrity
2. Fraud metadata integrity
3. Fraud typology presence and scenario integrity
4. Balance Draining behavioural validation
5. Card Testing behavioural validation
6. Velocity Attack behavioural validation
7. Geographic Anomaly behavioural validation
8. Abnormal Amount behavioural validation

Account Takeover behavioural validation will be added separately.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRANSACTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_with_fraud.parquet"
)

ACCOUNTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "accounts.parquet"
)

CUSTOMERS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "customers.parquet"
)


# =====================================================================
# EXPECTED FRAUD TYPES
# =====================================================================

EXPECTED_FRAUD_TYPES = {
    "ACCOUNT_TAKEOVER",
    "CARD_TESTING",
    "VELOCITY_ATTACK",
    "ABNORMAL_AMOUNT",
    "GEOGRAPHIC_ANOMALY",
    "BALANCE_DRAINING",
}


# =====================================================================
# REQUIRED TRANSACTION SCHEMA
# =====================================================================

REQUIRED_TRANSACTION_COLUMNS = {
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
    "is_fraud",
    "fraud_type",
    "fraud_scenario_id",
}


# =====================================================================
# BALANCE DRAINING CONTRACT
# =====================================================================

BALANCE_DRAINING_MIN_DRAIN_RATIO = 0.70


# =====================================================================
# CARD TESTING CONTRACT
# =====================================================================

CARD_TESTING_MIN_TRANSACTIONS = 5
CARD_TESTING_MAX_TRANSACTIONS = 12

CARD_TESTING_MIN_AMOUNT = 1.00
CARD_TESTING_MAX_AMOUNT = 20.00

CARD_TESTING_MIN_GAP_SECONDS = 10
CARD_TESTING_MAX_GAP_SECONDS = 90


# =====================================================================
# VELOCITY ATTACK CONTRACT
# =====================================================================

VELOCITY_MIN_TRANSACTIONS = 5
VELOCITY_MAX_TRANSACTIONS = 10

VELOCITY_MIN_GAP_SECONDS = 30
VELOCITY_MAX_GAP_SECONDS = 120

VELOCITY_MIN_KNOWN_DEVICE_RATE = 0.75


# =====================================================================
# GEOGRAPHIC ANOMALY CONTRACT
# =====================================================================

GEOGRAPHIC_MIN_GAP_MINUTES = 5
GEOGRAPHIC_MAX_GAP_MINUTES = 45

SOUTH_AFRICAN_PROVINCES = {
    "Eastern Cape",
    "Free State",
    "Gauteng",
    "KwaZulu-Natal",
    "Limpopo",
    "Mpumalanga",
    "North West",
    "Northern Cape",
    "Western Cape",
}


# =====================================================================
# ABNORMAL AMOUNT CONTRACT
# =====================================================================

ABNORMAL_AMOUNT_MIN_ZSCORE = 4.0
ABNORMAL_AMOUNT_MAX_ZSCORE = 8.0

# Amounts are rounded to cents by the generator, so reconstructed
# z-scores may differ very slightly from the originally sampled value.
ABNORMAL_AMOUNT_ZSCORE_TOLERANCE = 0.01


# =====================================================================
# GENERAL NUMERICAL TOLERANCE
# =====================================================================

BALANCE_TOLERANCE = 0.01


# =====================================================================
# VALIDATION REPORT
# =====================================================================


class ValidationReport:
    """
    Collect PASS, WARN and FAIL validation results.
    """

    def __init__(self, name: str):
        self.name = name
        self.results = []

    def pass_check(
        self,
        check_name: str,
        details: str = "",
    ):
        self.results.append(
            {
                "status": "PASS",
                "check": check_name,
                "details": details,
            }
        )

    def warn_check(
        self,
        check_name: str,
        details: str = "",
    ):
        self.results.append(
            {
                "status": "WARN",
                "check": check_name,
                "details": details,
            }
        )

    def fail_check(
        self,
        check_name: str,
        details: str = "",
    ):
        self.results.append(
            {
                "status": "FAIL",
                "check": check_name,
                "details": details,
            }
        )

    @property
    def passes(self):
        return sum(
            result["status"] == "PASS"
            for result in self.results
        )

    @property
    def warnings(self):
        return sum(
            result["status"] == "WARN"
            for result in self.results
        )

    @property
    def failures(self):
        return sum(
            result["status"] == "FAIL"
            for result in self.results
        )

    def print_results(self):
        for result in self.results:
            line = (
                f"[{result['status']}] "
                f"{result['check']}"
            )

            if result["details"]:
                line += f" — {result['details']}"

            print(line)


# =====================================================================
# DATA LOADING
# =====================================================================


def load_data():
    """
    Load persisted transactions and master data.
    """

    transactions = pd.read_parquet(
        TRANSACTIONS_PATH
    )

    accounts = pd.read_parquet(
        ACCOUNTS_PATH
    )

    customers = pd.read_parquet(
        CUSTOMERS_PATH
    )

    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"]
    )

    return transactions, accounts, customers


# =====================================================================
# STRUCTURAL VALIDATION
# =====================================================================


def validate_schema(
    transactions: pd.DataFrame,
):
    report = ValidationReport(
        "Schema Validation"
    )

    missing_columns = (
        REQUIRED_TRANSACTION_COLUMNS
        - set(transactions.columns)
    )

    if missing_columns:
        report.fail_check(
            "Required transaction schema",
            (
                "Missing columns: "
                f"{sorted(missing_columns)}"
            ),
        )
    else:
        report.pass_check(
            "Required transaction schema",
            (
                f"{len(REQUIRED_TRANSACTION_COLUMNS)} "
                "required columns present"
            ),
        )

    return report


def validate_transaction_ids(
    transactions: pd.DataFrame,
):
    report = ValidationReport(
        "Transaction ID Validation"
    )

    null_count = (
        transactions["transaction_id"]
        .isna()
        .sum()
    )

    if null_count == 0:
        report.pass_check(
            "Transaction IDs are non-null"
        )
    else:
        report.fail_check(
            "Transaction IDs are non-null",
            f"{null_count} null IDs",
        )

    unique_count = (
        transactions["transaction_id"]
        .nunique()
    )

    if unique_count == len(transactions):
        report.pass_check(
            "Transaction IDs are unique",
            f"{unique_count:,} unique IDs",
        )
    else:
        duplicate_count = (
            len(transactions)
            - unique_count
        )

        report.fail_check(
            "Transaction IDs are unique",
            (
                f"{duplicate_count:,} "
                "duplicate IDs"
            ),
        )

    return report


def validate_foreign_keys(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    customers: pd.DataFrame,
):
    report = ValidationReport(
        "Foreign Key Validation"
    )

    valid_accounts = set(
        accounts["account_id"]
    )

    invalid_account_mask = (
        ~transactions["account_id"]
        .isin(valid_accounts)
    )

    invalid_account_count = (
        invalid_account_mask.sum()
    )

    if invalid_account_count == 0:
        report.pass_check(
            "Account foreign keys",
            "All transaction accounts exist",
        )
    else:
        report.fail_check(
            "Account foreign keys",
            (
                f"{invalid_account_count:,} "
                "transactions reference "
                "unknown accounts"
            ),
        )

    valid_customers = set(
        customers["customer_id"]
    )

    invalid_customer_mask = (
        ~transactions["customer_id"]
        .isin(valid_customers)
    )

    invalid_customer_count = (
        invalid_customer_mask.sum()
    )

    if invalid_customer_count == 0:
        report.pass_check(
            "Customer foreign keys",
            "All transaction customers exist",
        )
    else:
        report.fail_check(
            "Customer foreign keys",
            (
                f"{invalid_customer_count:,} "
                "transactions reference "
                "unknown customers"
            ),
        )

    return report


def validate_account_customer_relationship(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
):
    report = ValidationReport(
        "Account-Customer Relationship"
    )

    account_owner = (
        accounts[
            [
                "account_id",
                "customer_id",
            ]
        ]
        .drop_duplicates(
            subset=["account_id"]
        )
        .rename(
            columns={
                "customer_id":
                    "expected_customer_id"
            }
        )
    )

    merged = transactions.merge(
        account_owner,
        on="account_id",
        how="left",
    )

    mismatch = (
        merged["customer_id"]
        != merged["expected_customer_id"]
    )

    mismatch_count = mismatch.sum()

    if mismatch_count == 0:
        report.pass_check(
            "Account-customer relationship",
            (
                "All account ownership "
                "relationships valid"
            ),
        )
    else:
        report.fail_check(
            "Account-customer relationship",
            (
                f"{mismatch_count:,} "
                "transactions have incorrect "
                "customer ownership"
            ),
        )

    return report


def validate_fraud_labels(
    transactions: pd.DataFrame,
):
    report = ValidationReport(
        "Fraud Label Validation"
    )

    null_count = (
        transactions["is_fraud"]
        .isna()
        .sum()
    )

    if null_count == 0:
        report.pass_check(
            "Fraud labels are non-null"
        )
    else:
        report.fail_check(
            "Fraud labels are non-null",
            f"{null_count:,} null labels",
        )

    observed_labels = set(
        transactions[
            "is_fraud"
        ].dropna().unique()
    )

    if observed_labels.issubset({0, 1}):
        report.pass_check(
            "Fraud labels are binary",
            (
                "Observed labels: "
                f"{sorted(observed_labels)}"
            ),
        )
    else:
        report.fail_check(
            "Fraud labels are binary",
            (
                "Unexpected labels: "
                f"{sorted(observed_labels)}"
            ),
        )

    fraud_count = (
        transactions["is_fraud"] == 1
    ).sum()

    if fraud_count > 0:
        report.pass_check(
            "Fraud transactions exist",
            (
                f"{fraud_count:,} "
                "fraud transactions"
            ),
        )
    else:
        report.fail_check(
            "Fraud transactions exist",
            "No fraud transactions found",
        )

    return report


def validate_fraud_metadata(
    transactions: pd.DataFrame,
):
    report = ValidationReport(
        "Fraud Metadata Validation"
    )

    fraud = transactions[
        transactions["is_fraud"] == 1
    ]

    legitimate = transactions[
        transactions["is_fraud"] == 0
    ]

    missing_type = (
        fraud["fraud_type"]
        .isna()
        .sum()
    )

    if missing_type == 0:
        report.pass_check(
            "Fraud type completeness"
        )
    else:
        report.fail_check(
            "Fraud type completeness",
            (
                f"{missing_type:,} fraud "
                "transactions have no fraud type"
            ),
        )

    missing_scenario = (
        fraud["fraud_scenario_id"]
        .isna()
        .sum()
    )

    if missing_scenario == 0:
        report.pass_check(
            "Fraud scenario completeness"
        )
    else:
        report.fail_check(
            "Fraud scenario completeness",
            (
                f"{missing_scenario:,} fraud "
                "transactions have no scenario ID"
            ),
        )

    legitimate_metadata = (
        legitimate["fraud_type"].notna()
        | legitimate[
            "fraud_scenario_id"
        ].notna()
    )

    invalid_legitimate = (
        legitimate_metadata.sum()
    )

    if invalid_legitimate == 0:
        report.pass_check(
            "Legitimate fraud metadata",
            (
                "Legitimate rows have no "
                "fraud metadata"
            ),
        )
    else:
        report.fail_check(
            "Legitimate fraud metadata",
            (
                f"{invalid_legitimate:,} "
                "legitimate rows contain "
                "fraud metadata"
            ),
        )

    return report


def validate_fraud_types(
    transactions: pd.DataFrame,
):
    report = ValidationReport(
        "Fraud Typology Validation"
    )

    fraud = transactions[
        transactions["is_fraud"] == 1
    ]

    observed_types = set(
        fraud["fraud_type"]
        .dropna()
        .unique()
    )

    missing_types = (
        EXPECTED_FRAUD_TYPES
        - observed_types
    )

    unexpected_types = (
        observed_types
        - EXPECTED_FRAUD_TYPES
    )

    if not missing_types:
        report.pass_check(
            "Expected fraud typologies",
            (
                f"All "
                f"{len(EXPECTED_FRAUD_TYPES)} "
                "typologies present"
            ),
        )
    else:
        report.fail_check(
            "Expected fraud typologies",
            (
                "Missing: "
                f"{sorted(missing_types)}"
            ),
        )

    if not unexpected_types:
        report.pass_check(
            "Unexpected fraud typologies",
            "None found",
        )
    else:
        report.fail_check(
            "Unexpected fraud typologies",
            (
                "Unexpected: "
                f"{sorted(unexpected_types)}"
            ),
        )

    return report


def validate_scenario_integrity(
    transactions: pd.DataFrame,
):
    report = ValidationReport(
        "Scenario Integrity"
    )

    fraud = transactions[
        transactions["is_fraud"] == 1
    ]

    scenario_type_counts = (
        fraud
        .groupby(
            "fraud_scenario_id"
        )["fraud_type"]
        .nunique()
    )

    invalid = (
        scenario_type_counts != 1
    )

    invalid_count = invalid.sum()

    if invalid_count == 0:
        report.pass_check(
            "Scenario-to-typology mapping",
            (
                f"{len(scenario_type_counts):,} "
                "scenarios map to one "
                "typology each"
            ),
        )
    else:
        report.fail_check(
            "Scenario-to-typology mapping",
            (
                f"{invalid_count:,} scenarios "
                "map to multiple typologies"
            ),
        )

    return report


def validate_amounts_and_balances(
    transactions: pd.DataFrame,
):
    report = ValidationReport(
        "Amount and Balance Validation"
    )

    invalid_amounts = (
        transactions["amount"] <= 0
    ).sum()

    if invalid_amounts == 0:
        report.pass_check(
            "Positive transaction amounts"
        )
    else:
        report.fail_check(
            "Positive transaction amounts",
            (
                f"{invalid_amounts:,} "
                "non-positive amounts"
            ),
        )

    negative_balances = (
        (
            transactions[
                "balance_before"
            ] < 0
        )
        | (
            transactions[
                "balance_after"
            ] < 0
        )
    ).sum()

    if negative_balances == 0:
        report.pass_check(
            "Non-negative balances"
        )
    else:
        report.fail_check(
            "Non-negative balances",
            (
                f"{negative_balances:,} "
                "transactions contain "
                "negative balances"
            ),
        )

    return report


# =====================================================================
# BALANCE DRAINING VALIDATION
# =====================================================================


def validate_balance_draining(
    transactions: pd.DataFrame,
):
    report = ValidationReport(
        "Balance Draining"
    )

    fraud = transactions[
        (
            transactions["is_fraud"] == 1
        )
        & (
            transactions["fraud_type"]
            == "BALANCE_DRAINING"
        )
    ].copy()

    if fraud.empty:
        report.fail_check(
            "Balance Draining transactions",
            "No transactions found",
        )
        return report

    fraud = fraud.sort_values(
        [
            "fraud_scenario_id",
            "timestamp",
            "transaction_id",
        ]
    )

    grouped = fraud.groupby(
        "fraud_scenario_id",
        sort=False,
    )

    scenario_sizes = grouped.size()

    if (scenario_sizes > 1).all():
        report.pass_check(
            "Balance Draining multi-transaction scenarios",
            (
                f"All {len(scenario_sizes)} "
                "scenarios contain multiple "
                "transactions"
            ),
        )
    else:
        report.fail_check(
            "Balance Draining multi-transaction scenarios",
            "One or more scenarios contain only one transaction",
        )

    account_counts = (
        grouped["account_id"]
        .nunique()
    )

    if (account_counts == 1).all():
        report.pass_check(
            "Balance Draining account consistency",
            "Each scenario uses exactly one account",
        )
    else:
        report.fail_check(
            "Balance Draining account consistency",
            "One or more scenarios use multiple accounts",
        )

    customer_counts = (
        grouped["customer_id"]
        .nunique()
    )

    if (customer_counts == 1).all():
        report.pass_check(
            "Balance Draining customer consistency",
            "Each scenario uses exactly one customer",
        )
    else:
        report.fail_check(
            "Balance Draining customer consistency",
            "One or more scenarios use multiple customers",
        )

    chronological = grouped[
        "timestamp"
    ].apply(
        lambda values: (
            values.diff()
            .dropna()
            > pd.Timedelta(0)
        ).all()
    )

    if chronological.all():
        report.pass_check(
            "Balance Draining chronological sequence",
            (
                "All scenario timestamps "
                "increase strictly"
            ),
        )
    else:
        report.fail_check(
            "Balance Draining chronological sequence",
            (
                "One or more scenarios are "
                "not strictly chronological"
            ),
        )

    reconciliation_error = (
        fraud["balance_after"]
        - (
            fraud["balance_before"]
            - fraud["amount"]
        )
    ).abs()

    if (
        reconciliation_error
        <= BALANCE_TOLERANCE
    ).all():
        report.pass_check(
            "Balance Draining transaction reconciliation",
            (
                "balance_after = "
                "balance_before - amount "
                "within R0.01"
            ),
        )
    else:
        report.fail_check(
            "Balance Draining transaction reconciliation",
            (
                "One or more transactions "
                "fail balance reconciliation"
            ),
        )

    continuity_failures = 0

    for _, scenario in grouped:
        scenario = scenario.sort_values(
            [
                "timestamp",
                "transaction_id",
            ]
        )

        previous_after = (
            scenario["balance_after"]
            .shift(1)
        )

        current_before = (
            scenario["balance_before"]
        )

        mask = previous_after.notna()

        differences = (
            current_before[mask]
            - previous_after[mask]
        ).abs()

        continuity_failures += int(
            (
                differences
                > BALANCE_TOLERANCE
            ).sum()
        )

    if continuity_failures == 0:
        report.pass_check(
            "Balance Draining balance continuity",
            (
                "Each transaction starts "
                "from the previous "
                "transaction's ending balance"
            ),
        )
    else:
        report.fail_check(
            "Balance Draining balance continuity",
            (
                f"{continuity_failures:,} "
                "continuity failures"
            ),
        )

    progressive = (
        fraud["balance_after"]
        < fraud["balance_before"]
    )

    if progressive.all():
        report.pass_check(
            "Balance Draining progressive depletion",
            (
                "Every fraud transaction "
                "reduces the balance"
            ),
        )
    else:
        report.fail_check(
            "Balance Draining progressive depletion",
            (
                "One or more transactions "
                "do not reduce the balance"
            ),
        )

    drain_ratios = []

    cumulative_failures = 0

    for _, scenario in grouped:
        scenario = scenario.sort_values(
            [
                "timestamp",
                "transaction_id",
            ]
        )

        starting_balance = float(
            scenario.iloc[0][
                "balance_before"
            ]
        )

        ending_balance = float(
            scenario.iloc[-1][
                "balance_after"
            ]
        )

        total_amount = float(
            scenario["amount"].sum()
        )

        if starting_balance > 0:
            drain_ratio = (
                starting_balance
                - ending_balance
            ) / starting_balance

            drain_ratios.append(
                drain_ratio
            )

        cumulative_error = abs(
            total_amount
            - (
                starting_balance
                - ending_balance
            )
        )

        if (
            cumulative_error
            > BALANCE_TOLERANCE
        ):
            cumulative_failures += 1

    drain_ratios = np.array(
        drain_ratios,
        dtype=float,
    )

    if (
        len(drain_ratios) > 0
        and (
            drain_ratios
            >= BALANCE_DRAINING_MIN_DRAIN_RATIO
        ).all()
    ):
        report.pass_check(
            "Balance Draining minimum drain ratio",
            (
                "All scenarios drain at least "
                f"{BALANCE_DRAINING_MIN_DRAIN_RATIO:.0%} "
                "of starting balance"
            ),
        )
    else:
        report.fail_check(
            "Balance Draining minimum drain ratio",
            (
                "One or more scenarios do not "
                "meet the minimum drain ratio"
            ),
        )

    if cumulative_failures == 0:
        report.pass_check(
            "Balance Draining cumulative reconciliation",
            (
                "Scenario transaction totals "
                "equal scenario balance reduction"
            ),
        )
    else:
        report.fail_check(
            "Balance Draining cumulative reconciliation",
            (
                f"{cumulative_failures} scenarios "
                "fail cumulative reconciliation"
            ),
        )

    if len(drain_ratios) > 0:
        report.pass_check(
            "Balance Draining observed drain range",
            (
                f"{drain_ratios.min():.2%} "
                f"to {drain_ratios.max():.2%}"
            ),
        )

    return report


# =====================================================================
# CARD TESTING VALIDATION
# =====================================================================


def validate_card_testing(
    transactions: pd.DataFrame,
):
    report = ValidationReport(
        "Card Testing"
    )

    fraud = transactions[
        (
            transactions["is_fraud"] == 1
        )
        & (
            transactions["fraud_type"]
            == "CARD_TESTING"
        )
    ].copy()

    if fraud.empty:
        report.fail_check(
            "Card Testing transactions",
            "No transactions found",
        )
        return report

    fraud = fraud.sort_values(
        [
            "fraud_scenario_id",
            "timestamp",
            "transaction_id",
        ]
    )

    grouped = fraud.groupby(
        "fraud_scenario_id",
        sort=False,
    )

    scenario_sizes = grouped.size()

    valid_sizes = scenario_sizes.between(
        CARD_TESTING_MIN_TRANSACTIONS,
        CARD_TESTING_MAX_TRANSACTIONS,
    )

    if valid_sizes.all():
        report.pass_check(
            "Card Testing scenario size",
            (
                f"All {len(scenario_sizes)} "
                "scenarios contain "
                f"{CARD_TESTING_MIN_TRANSACTIONS}-"
                f"{CARD_TESTING_MAX_TRANSACTIONS} "
                "transactions"
            ),
        )
    else:
        report.fail_check(
            "Card Testing scenario size",
            "One or more scenarios have invalid transaction counts",
        )

    account_counts = (
        grouped["account_id"]
        .nunique()
    )

    if (account_counts == 1).all():
        report.pass_check(
            "Card Testing account consistency",
            "Each scenario uses exactly one account",
        )
    else:
        report.fail_check(
            "Card Testing account consistency",
            "One or more scenarios use multiple accounts",
        )

    customer_counts = (
        grouped["customer_id"]
        .nunique()
    )

    if (customer_counts == 1).all():
        report.pass_check(
            "Card Testing customer consistency",
            "Each scenario uses exactly one customer",
        )
    else:
        report.fail_check(
            "Card Testing customer consistency",
            "One or more scenarios use multiple customers",
        )

    device_counts = (
        grouped["device_id"]
        .nunique()
    )

    if (device_counts == 1).all():
        report.pass_check(
            "Card Testing device consistency",
            "Each scenario uses exactly one device",
        )
    else:
        report.fail_check(
            "Card Testing device consistency",
            "One or more scenarios use multiple devices",
        )

    valid_amounts = fraud[
        "amount"
    ].between(
        CARD_TESTING_MIN_AMOUNT,
        CARD_TESTING_MAX_AMOUNT,
    )

    if valid_amounts.all():
        report.pass_check(
            "Card Testing micro-transaction amounts",
            (
                "All transactions are between "
                f"R{CARD_TESTING_MIN_AMOUNT:.2f} "
                f"and R{CARD_TESTING_MAX_AMOUNT:.2f}"
            ),
        )
    else:
        report.fail_check(
            "Card Testing micro-transaction amounts",
            "One or more transactions fall outside the expected amount range",
        )

    chronological = grouped[
        "timestamp"
    ].apply(
        lambda values: (
            values.diff()
            .dropna()
            > pd.Timedelta(0)
        ).all()
    )

    if chronological.all():
        report.pass_check(
            "Card Testing chronological sequence",
            (
                "All scenario timestamps "
                "increase strictly"
            ),
        )
    else:
        report.fail_check(
            "Card Testing chronological sequence",
            "One or more scenarios are not strictly chronological",
        )

    gap_failures = 0
    observed_gaps = []

    for _, scenario in grouped:
        scenario = scenario.sort_values(
            [
                "timestamp",
                "transaction_id",
            ]
        )

        gaps = (
            scenario["timestamp"]
            .diff()
            .dropna()
            .dt.total_seconds()
        )

        observed_gaps.extend(
            gaps.tolist()
        )

        invalid = ~gaps.between(
            CARD_TESTING_MIN_GAP_SECONDS,
            CARD_TESTING_MAX_GAP_SECONDS,
        )

        gap_failures += int(
            invalid.sum()
        )

    if gap_failures == 0:
        report.pass_check(
            "Card Testing rapid transaction gaps",
            (
                "All inter-transaction gaps "
                f"are between "
                f"{CARD_TESTING_MIN_GAP_SECONDS} "
                f"and "
                f"{CARD_TESTING_MAX_GAP_SECONDS} "
                "seconds"
            ),
        )
    else:
        report.fail_check(
            "Card Testing rapid transaction gaps",
            (
                f"{gap_failures} transaction "
                "gaps fall outside the "
                "expected range"
            ),
        )

    reconciliation_error = (
        fraud["balance_after"]
        - (
            fraud["balance_before"]
            - fraud["amount"]
        )
    ).abs()

    if (
        reconciliation_error
        <= BALANCE_TOLERANCE
    ).all():
        report.pass_check(
            "Card Testing transaction reconciliation",
            (
                "balance_after = "
                "balance_before - amount "
                "within R0.01"
            ),
        )
    else:
        report.fail_check(
            "Card Testing transaction reconciliation",
            "One or more transactions fail balance reconciliation",
        )

    continuity_failures = 0

    for _, scenario in grouped:
        scenario = scenario.sort_values(
            [
                "timestamp",
                "transaction_id",
            ]
        )

        previous_after = (
            scenario["balance_after"]
            .shift(1)
        )

        current_before = (
            scenario["balance_before"]
        )

        mask = previous_after.notna()

        differences = (
            current_before[mask]
            - previous_after[mask]
        ).abs()

        continuity_failures += int(
            (
                differences
                > BALANCE_TOLERANCE
            ).sum()
        )

    if continuity_failures == 0:
        report.pass_check(
            "Card Testing balance continuity",
            (
                "Each transaction starts "
                "from the previous "
                "transaction's ending balance"
            ),
        )
    else:
        report.fail_check(
            "Card Testing balance continuity",
            (
                f"{continuity_failures} "
                "continuity failures"
            ),
        )

    novelty_failures = 0

    legitimate = transactions[
        transactions["is_fraud"] == 0
    ].copy()

    for _, scenario in grouped:
        first_row = scenario.iloc[0]

        customer_id = (
            first_row["customer_id"]
        )

        scenario_device = (
            first_row["device_id"]
        )

        scenario_start = (
            scenario["timestamp"].min()
        )

        prior_devices = set(
            legitimate.loc[
                (
                    legitimate["customer_id"]
                    == customer_id
                )
                & (
                    legitimate["timestamp"]
                    < scenario_start
                ),
                "device_id",
            ]
            .dropna()
            .astype(str)
        )

        if str(
            scenario_device
        ) in prior_devices:
            novelty_failures += 1

    if novelty_failures == 0:
        report.pass_check(
            "Card Testing attacker device novelty",
            (
                "Scenario devices are absent "
                "from prior legitimate "
                "customer history"
            ),
        )
    else:
        report.fail_check(
            "Card Testing attacker device novelty",
            (
                f"{novelty_failures} scenarios "
                "reuse a prior legitimate device"
            ),
        )

    report.pass_check(
        "Card Testing observed amount range",
        (
            f"R{fraud['amount'].min():.2f} "
            f"to R{fraud['amount'].max():.2f}"
        ),
    )

    if observed_gaps:
        report.pass_check(
            "Card Testing observed gap range",
            (
                f"{min(observed_gaps):.0f} "
                f"to {max(observed_gaps):.0f} "
                "seconds"
            ),
        )

    return report


# =====================================================================
# VELOCITY ATTACK VALIDATION
# =====================================================================


def validate_velocity_attack(
    transactions: pd.DataFrame,
):
    report = ValidationReport(
        "Velocity Attack"
    )

    fraud = transactions[
        (
            transactions["is_fraud"] == 1
        )
        & (
            transactions["fraud_type"]
            == "VELOCITY_ATTACK"
        )
    ].copy()

    if fraud.empty:
        report.fail_check(
            "Velocity Attack transactions",
            "No transactions found",
        )
        return report

    fraud = fraud.sort_values(
        [
            "fraud_scenario_id",
            "timestamp",
            "transaction_id",
        ]
    )

    grouped = fraud.groupby(
        "fraud_scenario_id",
        sort=False,
    )

    scenario_sizes = grouped.size()

    valid_sizes = scenario_sizes.between(
        VELOCITY_MIN_TRANSACTIONS,
        VELOCITY_MAX_TRANSACTIONS,
    )

    if valid_sizes.all():
        report.pass_check(
            "Velocity Attack scenario size",
            (
                f"All {len(scenario_sizes)} "
                "scenarios contain "
                f"{VELOCITY_MIN_TRANSACTIONS}-"
                f"{VELOCITY_MAX_TRANSACTIONS} "
                "transactions"
            ),
        )
    else:
        report.fail_check(
            "Velocity Attack scenario size",
            "One or more scenarios have invalid transaction counts",
        )

    account_counts = (
        grouped["account_id"]
        .nunique()
    )

    if (account_counts == 1).all():
        report.pass_check(
            "Velocity Attack account consistency",
            "Each scenario uses exactly one account",
        )
    else:
        report.fail_check(
            "Velocity Attack account consistency",
            "One or more scenarios use multiple accounts",
        )

    customer_counts = (
        grouped["customer_id"]
        .nunique()
    )

    if (customer_counts == 1).all():
        report.pass_check(
            "Velocity Attack customer consistency",
            "Each scenario uses exactly one customer",
        )
    else:
        report.fail_check(
            "Velocity Attack customer consistency",
            "One or more scenarios use multiple customers",
        )

    device_counts = (
        grouped["device_id"]
        .nunique()
    )

    if (device_counts == 1).all():
        report.pass_check(
            "Velocity Attack device consistency",
            "Each scenario uses exactly one device",
        )
    else:
        report.fail_check(
            "Velocity Attack device consistency",
            "One or more scenarios use multiple devices",
        )

    chronological = grouped[
        "timestamp"
    ].apply(
        lambda values: (
            values.diff()
            .dropna()
            > pd.Timedelta(0)
        ).all()
    )

    if chronological.all():
        report.pass_check(
            "Velocity Attack chronological sequence",
            (
                "All scenario timestamps "
                "increase strictly"
            ),
        )
    else:
        report.fail_check(
            "Velocity Attack chronological sequence",
            "One or more scenarios are not strictly chronological",
        )

    gap_failures = 0
    observed_gaps = []

    for _, scenario in grouped:
        scenario = scenario.sort_values(
            [
                "timestamp",
                "transaction_id",
            ]
        )

        gaps = (
            scenario["timestamp"]
            .diff()
            .dropna()
            .dt.total_seconds()
        )

        observed_gaps.extend(
            gaps.tolist()
        )

        invalid = ~gaps.between(
            VELOCITY_MIN_GAP_SECONDS,
            VELOCITY_MAX_GAP_SECONDS,
        )

        gap_failures += int(
            invalid.sum()
        )

    if gap_failures == 0:
        report.pass_check(
            "Velocity Attack rapid transaction gaps",
            (
                "All inter-transaction gaps "
                f"are between "
                f"{VELOCITY_MIN_GAP_SECONDS} "
                f"and "
                f"{VELOCITY_MAX_GAP_SECONDS} "
                "seconds"
            ),
        )
    else:
        report.fail_check(
            "Velocity Attack rapid transaction gaps",
            (
                f"{gap_failures} transaction "
                "gaps fall outside the "
                "expected range"
            ),
        )

    reconciliation_error = (
        fraud["balance_after"]
        - (
            fraud["balance_before"]
            - fraud["amount"]
        )
    ).abs()

    if (
        reconciliation_error
        <= BALANCE_TOLERANCE
    ).all():
        report.pass_check(
            "Velocity Attack transaction reconciliation",
            (
                "balance_after = "
                "balance_before - amount "
                "within R0.01"
            ),
        )
    else:
        report.fail_check(
            "Velocity Attack transaction reconciliation",
            "One or more transactions fail balance reconciliation",
        )

    continuity_failures = 0

    for _, scenario in grouped:
        scenario = scenario.sort_values(
            [
                "timestamp",
                "transaction_id",
            ]
        )

        previous_after = (
            scenario["balance_after"]
            .shift(1)
        )

        current_before = (
            scenario["balance_before"]
        )

        mask = previous_after.notna()

        differences = (
            current_before[mask]
            - previous_after[mask]
        ).abs()

        continuity_failures += int(
            (
                differences
                > BALANCE_TOLERANCE
            ).sum()
        )

    if continuity_failures == 0:
        report.pass_check(
            "Velocity Attack balance continuity",
            (
                "Each transaction starts "
                "from the previous "
                "transaction's ending balance"
            ),
        )
    else:
        report.fail_check(
            "Velocity Attack balance continuity",
            (
                f"{continuity_failures} "
                "continuity failures"
            ),
        )

    legitimate = transactions[
        transactions["is_fraud"] == 0
    ].copy()

    known_device_count = 0

    for _, scenario in grouped:
        first_row = scenario.iloc[0]

        customer_id = (
            first_row["customer_id"]
        )

        scenario_device = (
            first_row["device_id"]
        )

        scenario_start = (
            scenario["timestamp"].min()
        )

        prior_devices = set(
            legitimate.loc[
                (
                    legitimate["customer_id"]
                    == customer_id
                )
                & (
                    legitimate["timestamp"]
                    < scenario_start
                ),
                "device_id",
            ]
            .dropna()
            .astype(str)
        )

        if str(
            scenario_device
        ) in prior_devices:
            known_device_count += 1

    total_scenarios = len(
        scenario_sizes
    )

    known_device_rate = (
        known_device_count
        / total_scenarios
        if total_scenarios > 0
        else 0.0
    )

    if (
        known_device_rate
        >= VELOCITY_MIN_KNOWN_DEVICE_RATE
    ):
        report.pass_check(
            "Velocity Attack known-device behaviour",
            (
                f"{known_device_count}/"
                f"{total_scenarios} scenarios "
                f"({known_device_rate:.1%}) "
                "use a device seen in prior "
                "legitimate customer history"
            ),
        )
    else:
        report.fail_check(
            "Velocity Attack known-device behaviour",
            (
                f"Known-device rate "
                f"{known_device_rate:.1%} "
                "is below expected minimum "
                f"{VELOCITY_MIN_KNOWN_DEVICE_RATE:.0%}"
            ),
        )

    if observed_gaps:
        report.pass_check(
            "Velocity Attack observed gap range",
            (
                f"{min(observed_gaps):.0f} "
                f"to {max(observed_gaps):.0f} "
                "seconds"
            ),
        )

    report.pass_check(
        "Velocity Attack observed amount range",
        (
            f"R{fraud['amount'].min():.2f} "
            f"to R{fraud['amount'].max():.2f}"
        ),
    )

    return report


# =====================================================================
# GEOGRAPHIC ANOMALY VALIDATION
# =====================================================================


def validate_geographic_anomaly(
    transactions: pd.DataFrame,
):
    report = ValidationReport(
        "Geographic Anomaly"
    )

    fraud = transactions[
        (
            transactions["is_fraud"] == 1
        )
        & (
            transactions["fraud_type"]
            == "GEOGRAPHIC_ANOMALY"
        )
    ].copy()

    if fraud.empty:
        report.fail_check(
            "Geographic Anomaly transactions",
            "No transactions found",
        )
        return report

    fraud = fraud.sort_values(
        [
            "fraud_scenario_id",
            "timestamp",
            "transaction_id",
        ]
    )

    grouped = fraud.groupby(
        "fraud_scenario_id",
        sort=False,
    )

    scenario_sizes = grouped.size()

    if (scenario_sizes == 1).all():
        report.pass_check(
            "Geographic Anomaly scenario size",
            (
                f"All {len(scenario_sizes)} "
                "scenarios contain exactly "
                "one fraud transaction"
            ),
        )
    else:
        report.fail_check(
            "Geographic Anomaly scenario size",
            "One or more scenarios contain more than one fraud transaction",
        )

    account_counts = (
        grouped["account_id"]
        .nunique()
    )

    if (account_counts == 1).all():
        report.pass_check(
            "Geographic Anomaly account consistency",
            "Each scenario uses exactly one account",
        )
    else:
        report.fail_check(
            "Geographic Anomaly account consistency",
            "One or more scenarios use multiple accounts",
        )

    customer_counts = (
        grouped["customer_id"]
        .nunique()
    )

    if (customer_counts == 1).all():
        report.pass_check(
            "Geographic Anomaly customer consistency",
            "Each scenario uses exactly one customer",
        )
    else:
        report.fail_check(
            "Geographic Anomaly customer consistency",
            "One or more scenarios use multiple customers",
        )

    previous_rows = []

    sorted_transactions = (
        transactions
        .sort_values(
            [
                "customer_id",
                "timestamp",
                "transaction_id",
            ]
        )
        .copy()
    )

    for _, fraud_row in fraud.iterrows():
        customer_history = (
            sorted_transactions[
                (
                    sorted_transactions[
                        "customer_id"
                    ]
                    == fraud_row[
                        "customer_id"
                    ]
                )
                & (
                    sorted_transactions[
                        "timestamp"
                    ]
                    < fraud_row[
                        "timestamp"
                    ]
                )
            ]
        )

        if customer_history.empty:
            previous_rows.append(
                {
                    "transaction_id":
                        fraud_row[
                            "transaction_id"
                        ],
                    "has_previous":
                        False,
                    "previous_is_fraud":
                        np.nan,
                    "previous_province":
                        None,
                    "previous_country":
                        None,
                    "gap_minutes":
                        np.nan,
                }
            )

            continue

        previous = (
            customer_history.iloc[-1]
        )

        gap_minutes = (
            fraud_row["timestamp"]
            - previous["timestamp"]
        ).total_seconds() / 60.0

        previous_rows.append(
            {
                "transaction_id":
                    fraud_row[
                        "transaction_id"
                    ],
                "has_previous":
                    True,
                "previous_is_fraud":
                    previous[
                        "is_fraud"
                    ],
                "previous_province":
                    previous[
                        "province"
                    ],
                "previous_country":
                    previous[
                        "country"
                    ],
                "gap_minutes":
                    gap_minutes,
            }
        )

    previous_df = pd.DataFrame(
        previous_rows
    )

    analysis = fraud.merge(
        previous_df,
        on="transaction_id",
        how="left",
    )

    if analysis[
        "has_previous"
    ].all():
        report.pass_check(
            "Geographic Anomaly previous transaction",
            (
                "Every fraud transaction "
                "has a preceding customer "
                "transaction"
            ),
        )
    else:
        report.fail_check(
            "Geographic Anomaly previous transaction",
            (
                "One or more fraud "
                "transactions have no "
                "preceding customer transaction"
            ),
        )

    valid_anchor = (
        analysis[
            "previous_is_fraud"
        ] == 0
    )

    if valid_anchor.all():
        report.pass_check(
            "Geographic Anomaly legitimate anchor",
            (
                "Every scenario follows "
                "a legitimate customer "
                "transaction"
            ),
        )
    else:
        report.fail_check(
            "Geographic Anomaly legitimate anchor",
            (
                "One or more scenarios do not "
                "follow a legitimate transaction"
            ),
        )

    province_changed = (
        analysis["previous_province"]
        != analysis["province"]
    )

    if province_changed.all():
        report.pass_check(
            "Geographic Anomaly province transition",
            (
                "Every scenario changes "
                "province relative to the "
                "previous transaction"
            ),
        )
    else:
        report.fail_check(
            "Geographic Anomaly province transition",
            (
                "One or more scenarios remain "
                "in the previous province"
            ),
        )

    valid_previous_province = (
        analysis[
            "previous_province"
        ].isin(
            SOUTH_AFRICAN_PROVINCES
        )
    )

    if valid_previous_province.all():
        report.pass_check(
            "Geographic Anomaly previous province",
            (
                "All anchor transactions "
                "use valid South African "
                "provinces"
            ),
        )
    else:
        report.fail_check(
            "Geographic Anomaly previous province",
            (
                "One or more anchor "
                "transactions use an invalid "
                "province"
            ),
        )

    valid_fraud_province = (
        analysis["province"]
        .isin(
            SOUTH_AFRICAN_PROVINCES
        )
    )

    if valid_fraud_province.all():
        report.pass_check(
            "Geographic Anomaly fraud province",
            (
                "All fraud transactions "
                "use valid South African "
                "provinces"
            ),
        )
    else:
        report.fail_check(
            "Geographic Anomaly fraud province",
            (
                "One or more fraud "
                "transactions use an invalid "
                "province"
            ),
        )

    domestic = (
        (
            analysis[
                "previous_country"
            ]
            == "South Africa"
        )
        & (
            analysis["country"]
            == "South Africa"
        )
    )

    if domestic.all():
        report.pass_check(
            "Geographic Anomaly domestic transition",
            (
                "Both sides of every "
                "transition occur within "
                "South Africa"
            ),
        )
    else:
        report.fail_check(
            "Geographic Anomaly domestic transition",
            (
                "One or more scenarios "
                "contain a non-domestic "
                "transition"
            ),
        )

    valid_gap = (
        analysis["gap_minutes"]
        .between(
            GEOGRAPHIC_MIN_GAP_MINUTES,
            GEOGRAPHIC_MAX_GAP_MINUTES,
        )
    )

    if valid_gap.all():
        report.pass_check(
            "Geographic Anomaly rapid location change",
            (
                "All province changes occur "
                f"within "
                f"{GEOGRAPHIC_MIN_GAP_MINUTES}-"
                f"{GEOGRAPHIC_MAX_GAP_MINUTES} "
                "minutes"
            ),
        )
    else:
        report.fail_check(
            "Geographic Anomaly rapid location change",
            (
                "One or more province changes "
                "fall outside the expected "
                "time window"
            ),
        )

    reconciliation_error = (
        fraud["balance_after"]
        - (
            fraud["balance_before"]
            - fraud["amount"]
        )
    ).abs()

    if (
        reconciliation_error
        <= BALANCE_TOLERANCE
    ).all():
        report.pass_check(
            "Geographic Anomaly transaction reconciliation",
            (
                "balance_after = "
                "balance_before - amount "
                "within R0.01"
            ),
        )
    else:
        report.fail_check(
            "Geographic Anomaly transaction reconciliation",
            (
                "One or more transactions "
                "fail balance reconciliation"
            ),
        )

    valid_gaps = (
        analysis["gap_minutes"]
        .dropna()
    )

    if not valid_gaps.empty:
        report.pass_check(
            "Geographic Anomaly observed gap range",
            (
                f"{valid_gaps.min():.0f} "
                f"to {valid_gaps.max():.0f} "
                "minutes"
            ),
        )

    return report


# =====================================================================
# ABNORMAL AMOUNT VALIDATION
# =====================================================================


def validate_abnormal_amount(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
):
    """
    Validate the behavioural contract of Abnormal Amount fraud.

    The primary signal is an extreme transaction amount relative
    to the customer's generated behavioural profile.

    The generator samples a target z-score between 4.0 and 8.0:

        amount =
            avg_transaction
            + target_zscore * std_transaction

    The transaction is deliberately designed to otherwise resemble
    normal customer behaviour.
    """

    report = ValidationReport(
        "Abnormal Amount"
    )

    fraud = transactions[
        (
            transactions["is_fraud"] == 1
        )
        & (
            transactions["fraud_type"]
            == "ABNORMAL_AMOUNT"
        )
    ].copy()

    if fraud.empty:
        report.fail_check(
            "Abnormal Amount transactions",
            "No transactions found",
        )
        return report

    fraud = fraud.sort_values(
        [
            "fraud_scenario_id",
            "timestamp",
            "transaction_id",
        ]
    )

    grouped = fraud.groupby(
        "fraud_scenario_id",
        sort=False,
    )

    # -------------------------------------------------------------
    # Scenario structure
    # -------------------------------------------------------------

    scenario_sizes = grouped.size()

    if (scenario_sizes == 1).all():
        report.pass_check(
            "Abnormal Amount scenario size",
            (
                f"All {len(scenario_sizes)} "
                "scenarios contain exactly "
                "one fraud transaction"
            ),
        )
    else:
        report.fail_check(
            "Abnormal Amount scenario size",
            (
                "One or more scenarios contain "
                "more than one fraud transaction"
            ),
        )

    account_counts = (
        grouped["account_id"]
        .nunique()
    )

    if (account_counts == 1).all():
        report.pass_check(
            "Abnormal Amount account consistency",
            (
                "Each scenario uses exactly "
                "one account"
            ),
        )
    else:
        report.fail_check(
            "Abnormal Amount account consistency",
            (
                "One or more scenarios use "
                "multiple accounts"
            ),
        )

    customer_counts = (
        grouped["customer_id"]
        .nunique()
    )

    if (customer_counts == 1).all():
        report.pass_check(
            "Abnormal Amount customer consistency",
            (
                "Each scenario uses exactly "
                "one customer"
            ),
        )
    else:
        report.fail_check(
            "Abnormal Amount customer consistency",
            (
                "One or more scenarios use "
                "multiple customers"
            ),
        )

    # -------------------------------------------------------------
    # Customer behavioural profile
    # -------------------------------------------------------------

    required_customer_columns = {
        "customer_id",
        "avg_transaction",
        "std_transaction",
        "usual_start_hour",
        "usual_end_hour",
    }

    missing_customer_columns = (
        required_customer_columns
        - set(customers.columns)
    )

    if missing_customer_columns:
        report.fail_check(
            "Abnormal Amount customer profile schema",
            (
                "Missing customer columns: "
                f"{sorted(missing_customer_columns)}"
            ),
        )

        return report

    profile = (
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
            subset=["customer_id"]
        )
        .copy()
    )

    analysis = fraud.merge(
        profile,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    profile_complete = (
        analysis[
            [
                "avg_transaction",
                "std_transaction",
                "usual_start_hour",
                "usual_end_hour",
            ]
        ]
        .notna()
        .all(axis=1)
    )

    if profile_complete.all():
        report.pass_check(
            "Abnormal Amount customer profile",
            (
                "Every scenario has a complete "
                "customer behavioural profile"
            ),
        )
    else:
        missing_profiles = int(
            (~profile_complete).sum()
        )

        report.fail_check(
            "Abnormal Amount customer profile",
            (
                f"{missing_profiles} scenarios "
                "have incomplete customer "
                "behavioural profiles"
            ),
        )

    valid_profile_values = (
        (
            analysis[
                "avg_transaction"
            ] > 0
        )
        & (
            analysis[
                "std_transaction"
            ] > 0
        )
    )

    if valid_profile_values.all():
        report.pass_check(
            "Abnormal Amount profile parameters",
            (
                "All customer averages and "
                "standard deviations are positive"
            ),
        )
    else:
        invalid_count = int(
            (~valid_profile_values).sum()
        )

        report.fail_check(
            "Abnormal Amount profile parameters",
            (
                f"{invalid_count} scenarios "
                "contain invalid behavioural "
                "profile parameters"
            ),
        )

    # -------------------------------------------------------------
    # Customer-relative amount anomaly
    # -------------------------------------------------------------

    analysis[
        "amount_vs_average"
    ] = (
        analysis["amount"]
        / analysis["avg_transaction"]
    )

    analysis[
        "profile_zscore"
    ] = (
        (
            analysis["amount"]
            - analysis["avg_transaction"]
        )
        / analysis["std_transaction"]
    )

    lower_bound = (
        ABNORMAL_AMOUNT_MIN_ZSCORE
        - ABNORMAL_AMOUNT_ZSCORE_TOLERANCE
    )

    upper_bound = (
        ABNORMAL_AMOUNT_MAX_ZSCORE
        + ABNORMAL_AMOUNT_ZSCORE_TOLERANCE
    )

    valid_zscore = (
        analysis["profile_zscore"]
        .between(
            lower_bound,
            upper_bound,
        )
    )

    if valid_zscore.all():
        report.pass_check(
            "Abnormal Amount profile-relative z-score",
            (
                "All transactions fall within "
                f"the designed "
                f"{ABNORMAL_AMOUNT_MIN_ZSCORE:.1f}-"
                f"{ABNORMAL_AMOUNT_MAX_ZSCORE:.1f}σ "
                "customer-relative anomaly range"
            ),
        )
    else:
        invalid_count = int(
            (~valid_zscore).sum()
        )

        report.fail_check(
            "Abnormal Amount profile-relative z-score",
            (
                f"{invalid_count} transactions "
                "fall outside the expected "
                "customer-relative z-score range"
            ),
        )

    above_average = (
        analysis["amount"]
        > analysis["avg_transaction"]
    )

    if above_average.all():
        report.pass_check(
            "Abnormal Amount exceeds customer average",
            (
                "Every fraud amount exceeds "
                "the customer's expected "
                "transaction average"
            ),
        )
    else:
        invalid_count = int(
            (~above_average).sum()
        )

        report.fail_check(
            "Abnormal Amount exceeds customer average",
            (
                f"{invalid_count} transactions "
                "do not exceed the customer "
                "average"
            ),
        )

    # -------------------------------------------------------------
    # Known-device behaviour
    # -------------------------------------------------------------

    legitimate = transactions[
        transactions["is_fraud"] == 0
    ].copy()

    known_device_failures = 0
    customers_with_prior_devices = 0

    for _, fraud_row in fraud.iterrows():
        prior_history = legitimate[
            (
                legitimate["customer_id"]
                == fraud_row["customer_id"]
            )
            & (
                legitimate["timestamp"]
                < fraud_row["timestamp"]
            )
        ]

        prior_devices = set(
            prior_history[
                "device_id"
            ]
            .dropna()
            .astype(str)
        )

        if prior_devices:
            customers_with_prior_devices += 1

            if (
                str(
                    fraud_row["device_id"]
                )
                not in prior_devices
            ):
                known_device_failures += 1

    if known_device_failures == 0:
        report.pass_check(
            "Abnormal Amount known-device behaviour",
            (
                "Fraud transactions reuse a "
                "previously observed legitimate "
                "device whenever prior device "
                "history is available"
            ),
        )
    else:
        report.fail_check(
            "Abnormal Amount known-device behaviour",
            (
                f"{known_device_failures} of "
                f"{customers_with_prior_devices} "
                "scenarios with prior device "
                "history use an unknown device"
            ),
        )

    # -------------------------------------------------------------
    # Usual activity window
    # -------------------------------------------------------------

    def hour_in_usual_window(
        hour,
        start_hour,
        end_hour,
    ):
        if (
            pd.isna(hour)
            or pd.isna(start_hour)
            or pd.isna(end_hour)
        ):
            return False

        hour = int(hour)
        start_hour = int(start_hour)
        end_hour = int(end_hour)

        if start_hour <= end_hour:
            return (
                start_hour
                <= hour
                <= end_hour
            )

        return (
            hour >= start_hour
            or hour <= end_hour
        )

    transaction_hours = (
        analysis["timestamp"].dt.hour
    )

    normal_hour_mask = pd.Series(
        [
            hour_in_usual_window(
                hour,
                start,
                end,
            )
            for hour, start, end in zip(
                transaction_hours,
                analysis[
                    "usual_start_hour"
                ],
                analysis[
                    "usual_end_hour"
                ],
            )
        ],
        index=analysis.index,
    )

    if normal_hour_mask.all():
        report.pass_check(
            "Abnormal Amount usual activity window",
            (
                "All transactions occur within "
                "the customer's usual activity "
                "window"
            ),
        )
    else:
        invalid_count = int(
            (~normal_hour_mask).sum()
        )

        report.fail_check(
            "Abnormal Amount usual activity window",
            (
                f"{invalid_count} transactions "
                "occur outside the customer's "
                "usual activity window"
            ),
        )

    # -------------------------------------------------------------
    # Domestic context
    # -------------------------------------------------------------

    domestic_country = (
        analysis["country"]
        == "South Africa"
    )

    if domestic_country.all():
        report.pass_check(
            "Abnormal Amount domestic country",
            (
                "All transactions occur "
                "within South Africa"
            ),
        )
    else:
        invalid_count = int(
            (~domestic_country).sum()
        )

        report.fail_check(
            "Abnormal Amount domestic country",
            (
                f"{invalid_count} transactions "
                "are not domestic South African "
                "transactions"
            ),
        )

    non_international = (
        analysis[
            "is_international"
        ].fillna(True) == False
    )

    if non_international.all():
        report.pass_check(
            "Abnormal Amount international flag",
            (
                "All transactions are marked "
                "as non-international"
            ),
        )
    else:
        invalid_count = int(
            (~non_international).sum()
        )

        report.fail_check(
            "Abnormal Amount international flag",
            (
                f"{invalid_count} transactions "
                "are marked international"
            ),
        )

    debit_direction = (
        analysis["direction"]
        == "DEBIT"
    )

    if debit_direction.all():
        report.pass_check(
            "Abnormal Amount debit direction",
            (
                "All transactions are "
                "DEBIT transactions"
            ),
        )
    else:
        invalid_count = int(
            (~debit_direction).sum()
        )

        report.fail_check(
            "Abnormal Amount debit direction",
            (
                f"{invalid_count} transactions "
                "are not DEBIT transactions"
            ),
        )

    # -------------------------------------------------------------
    # Financial integrity
    # -------------------------------------------------------------

    positive_amount = (
        analysis["amount"] > 0
    )

    if positive_amount.all():
        report.pass_check(
            "Abnormal Amount positive amount",
            (
                "All fraud transaction "
                "amounts are positive"
            ),
        )
    else:
        invalid_count = int(
            (~positive_amount).sum()
        )

        report.fail_check(
            "Abnormal Amount positive amount",
            (
                f"{invalid_count} transactions "
                "have non-positive amounts"
            ),
        )

    reconciliation_error = (
        analysis["balance_after"]
        - (
            analysis["balance_before"]
            - analysis["amount"]
        )
    ).abs()

    if (
        reconciliation_error
        <= BALANCE_TOLERANCE
    ).all():
        report.pass_check(
            "Abnormal Amount transaction reconciliation",
            (
                "balance_after = "
                "balance_before - amount "
                "within R0.01"
            ),
        )
    else:
        invalid_count = int(
            (
                reconciliation_error
                > BALANCE_TOLERANCE
            ).sum()
        )

        report.fail_check(
            "Abnormal Amount transaction reconciliation",
            (
                f"{invalid_count} transactions "
                "fail balance reconciliation"
            ),
        )

    valid_balances = (
        (
            analysis[
                "balance_before"
            ] >= 0
        )
        & (
            analysis[
                "balance_after"
            ] >= 0
        )
    )

    if valid_balances.all():
        report.pass_check(
            "Abnormal Amount non-negative balances",
            (
                "All scenario balances "
                "remain non-negative"
            ),
        )
    else:
        invalid_count = int(
            (~valid_balances).sum()
        )

        report.fail_check(
            "Abnormal Amount non-negative balances",
            (
                f"{invalid_count} transactions "
                "contain negative balances"
            ),
        )

    # -------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------

    valid_zscores = (
        analysis[
            "profile_zscore"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if not valid_zscores.empty:
        report.pass_check(
            "Abnormal Amount observed z-score range",
            (
                f"{valid_zscores.min():.2f}σ "
                f"to "
                f"{valid_zscores.max():.2f}σ"
            ),
        )

    amount_ratios = (
        analysis[
            "amount_vs_average"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if not amount_ratios.empty:
        report.pass_check(
            "Abnormal Amount observed amount-to-average range",
            (
                f"{amount_ratios.min():.2f}x "
                f"to "
                f"{amount_ratios.max():.2f}x"
            ),
        )

    return report


# =====================================================================
# DATASET SUMMARY
# =====================================================================


def build_dataset_summary(
    transactions: pd.DataFrame,
):
    fraud = transactions[
        transactions["is_fraud"] == 1
    ]

    legitimate = transactions[
        transactions["is_fraud"] == 0
    ]

    total = len(transactions)
    fraud_count = len(fraud)

    fraud_rate = (
        fraud_count / total
        if total > 0
        else 0.0
    )

    summary = {
        "total_transactions":
            total,
        "legitimate_transactions":
            len(legitimate),
        "fraudulent_transactions":
            fraud_count,
        "fraud_scenarios":
            fraud[
                "fraud_scenario_id"
            ].nunique(),
        "fraud_rate":
            fraud_rate,
        "fraud_typologies":
            fraud[
                "fraud_type"
            ].nunique(),
    }

    return summary


def print_dataset_overview(
    transactions: pd.DataFrame,
):
    summary = build_dataset_summary(
        transactions
    )

    print(
        "FRAUD DATA VALIDATION REPORT"
    )
    print("=" * 72)

    print(
        f"Total transactions:      "
        f"{summary['total_transactions']:,}"
    )

    print(
        f"Legitimate transactions: "
        f"{summary['legitimate_transactions']:,}"
    )

    print(
        f"Fraudulent transactions: "
        f"{summary['fraudulent_transactions']:,}"
    )

    print(
        f"Fraud scenarios:         "
        f"{summary['fraud_scenarios']:,}"
    )

    print(
        f"Fraud rate:              "
        f"{summary['fraud_rate']:.3%}"
    )

    print(
        f"Fraud typologies:        "
        f"{summary['fraud_typologies']}"
    )

    print()

    fraud = transactions[
        transactions["is_fraud"] == 1
    ]

    typology_summary = (
        fraud
        .groupby("fraud_type")
        .agg(
            scenarios=(
                "fraud_scenario_id",
                "nunique",
            ),
            transactions=(
                "transaction_id",
                "count",
            ),
            customers=(
                "customer_id",
                "nunique",
            ),
            accounts=(
                "account_id",
                "nunique",
            ),
            mean_amount=(
                "amount",
                "mean",
            ),
            median_amount=(
                "amount",
                "median",
            ),
        )
        .round(2)
    )

    return typology_summary


# =====================================================================
# MAIN
# =====================================================================


def main():
    transactions, accounts, customers = (
        load_data()
    )

    typology_summary = (
        print_dataset_overview(
            transactions
        )
    )

    structural_reports = [
        validate_schema(
            transactions
        ),
        validate_transaction_ids(
            transactions
        ),
        validate_foreign_keys(
            transactions,
            accounts,
            customers,
        ),
        validate_account_customer_relationship(
            transactions,
            accounts,
        ),
        validate_fraud_labels(
            transactions
        ),
        validate_fraud_metadata(
            transactions
        ),
        validate_fraud_types(
            transactions
        ),
        validate_scenario_integrity(
            transactions
        ),
        validate_amounts_and_balances(
            transactions
        ),
    ]

    behavioural_reports = [
        validate_balance_draining(
            transactions
        ),
        validate_card_testing(
            transactions
        ),
        validate_velocity_attack(
            transactions
        ),
        validate_geographic_anomaly(
            transactions
        ),
        validate_abnormal_amount(
            transactions,
            customers,
        ),
    ]

    print(
        "1. STRUCTURAL AND "
        "RELATIONAL VALIDATION"
    )
    print("-" * 72)

    for report in structural_reports:
        report.print_results()

    print()

    print(
        "2. FRAUD TYPOLOGY SUMMARY"
    )
    print("-" * 72)

    print(
        typology_summary.to_string()
    )

    print()

    print(
        "3. BEHAVIOURAL TYPOLOGY "
        "VALIDATION"
    )
    print("-" * 72)

    for report in behavioural_reports:
        print()
        print(
            report.name.upper()
        )
        report.print_results()

    all_reports = (
        structural_reports
        + behavioural_reports
    )

    total_passes = sum(
        report.passes
        for report in all_reports
    )

    total_warnings = sum(
        report.warnings
        for report in all_reports
    )

    total_failures = sum(
        report.failures
        for report in all_reports
    )

    print()

    print(
        "4. VALIDATION RESULT"
    )
    print("-" * 72)

    print(
        f"Passed checks: {total_passes}"
    )

    print(
        f"Warnings:      {total_warnings}"
    )

    print(
        f"Failures:      {total_failures}"
    )

    print()

    if total_failures > 0:
        print(
            "OVERALL STATUS: FAIL"
        )

        raise SystemExit(1)

    if total_warnings > 0:
        print(
            "OVERALL STATUS: "
            "PASS WITH WARNINGS"
        )

        return

    print(
        "OVERALL STATUS: PASS"
    )


if __name__ == "__main__":
    main()