#!/usr/bin/env python3

"""Inject multiple fraud typologies into the normal transaction baseline."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.generator.fraud_injector import (
    FRAUD_TYPE_ABNORMAL_AMOUNT,
    FRAUD_TYPE_ACCOUNT_TAKEOVER,
    FRAUD_TYPE_CARD_TESTING,
    FRAUD_TYPE_GEOGRAPHIC_ANOMALY,
    FRAUD_TYPE_VELOCITY_ATTACK,
    FRAUD_TYPE_BALANCE_DRAINING,
    inject_balance_draining,
    inject_abnormal_amount,
    inject_account_takeover,
    inject_card_testing,
    inject_geographic_anomaly,
    inject_velocity_attack,
)


RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SAMPLE_DIR = ROOT / "data" / "sample"

CUSTOMERS_PATH = RAW_DIR / "customers.parquet"
ACCOUNTS_PATH = RAW_DIR / "accounts.parquet"
TRANSACTIONS_PATH = RAW_DIR / "transactions.parquet"

OUTPUT_PATH = (
    PROCESSED_DIR
    / "transactions_with_fraud.parquet"
)

SAMPLE_PATH = (
    SAMPLE_DIR
    / "fraud_sample.csv"
)


EXPECTED_FRAUD_TYPES = {
    FRAUD_TYPE_ACCOUNT_TAKEOVER,
    FRAUD_TYPE_CARD_TESTING,
    FRAUD_TYPE_VELOCITY_ATTACK,
    FRAUD_TYPE_ABNORMAL_AMOUNT,
    FRAUD_TYPE_GEOGRAPHIC_ANOMALY,
    FRAUD_TYPE_BALANCE_DRAINING,
}


def load_source_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load generated customer, account and transaction data."""

    required_files = [
        CUSTOMERS_PATH,
        ACCOUNTS_PATH,
        TRANSACTIONS_PATH,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
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

    customers = pd.read_parquet(
        CUSTOMERS_PATH
    )

    accounts = pd.read_parquet(
        ACCOUNTS_PATH
    )

    transactions = pd.read_parquet(
        TRANSACTIONS_PATH
    )

    return (
        customers,
        accounts,
        transactions,
    )


def inject_fraud_scenarios(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Apply all fraud typologies sequentially."""

    output = inject_account_takeover(
        transactions=transactions,
        accounts=accounts,
        customers=customers,
        n_scenarios=20,
        min_transactions=2,
        max_transactions=5,
        seed=42,
    )

    output = inject_card_testing(
        transactions=output,
        accounts=accounts,
        customers=customers,
        n_scenarios=15,
        min_transactions=5,
        max_transactions=12,
        seed=43,
    )

    output = inject_velocity_attack(
        transactions=output,
        accounts=accounts,
        customers=customers,
        n_scenarios=15,
        min_transactions=5,
        max_transactions=10,
        seed=44,
    )

    output = inject_abnormal_amount(
        transactions=output,
        accounts=accounts,
        customers=customers,
        n_scenarios=50,
        min_zscore=4.0,
        max_zscore=8.0,
        seed=45,
    )

    output = inject_geographic_anomaly(
        transactions=output,
        accounts=accounts,
        customers=customers,
        n_scenarios=35,
        min_minutes_after_previous=5,
        max_minutes_after_previous=45,
        seed=46,
    )

    output = inject_balance_draining(
        transactions=output,
        accounts=accounts,
        customers=customers,
        n_scenarios=18,
        min_transactions=4,
        max_transactions=8,
        min_drain_fraction=0.70,
        max_drain_fraction=0.95,
        min_gap_seconds=30,
        max_gap_seconds=180,
        seed=47,
    )

    return output


def validate_output(
    baseline: pd.DataFrame,
    transactions_with_fraud: pd.DataFrame,
    accounts: pd.DataFrame,
    customers: pd.DataFrame,
) -> None:
    """
    Validate structural and ground-truth properties
    of the generated fraud dataset.
    """

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
        - set(
            transactions_with_fraud.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Output is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ---------------------------------------------------------
    # Transaction ID uniqueness
    # ---------------------------------------------------------

    if (
        transactions_with_fraud[
            "transaction_id"
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate transaction IDs were found."
        )

    # ---------------------------------------------------------
    # Binary fraud labels
    # ---------------------------------------------------------

    valid_labels = {0, 1}

    observed_labels = set(
        transactions_with_fraud[
            "is_fraud"
        ]
        .dropna()
        .astype(int)
        .unique()
    )

    if not observed_labels.issubset(
        valid_labels
    ):
        raise ValueError(
            "The is_fraud column contains "
            "invalid labels."
        )

    fraud = transactions_with_fraud[
        transactions_with_fraud[
            "is_fraud"
        ]
        == 1
    ].copy()

    if fraud.empty:
        raise ValueError(
            "No fraudulent transactions "
            "were generated."
        )

    # ---------------------------------------------------------
    # Required fraud typologies
    # ---------------------------------------------------------

    observed_fraud_types = set(
        fraud[
            "fraud_type"
        ]
        .dropna()
        .unique()
    )

    missing_fraud_types = (
        EXPECTED_FRAUD_TYPES
        - observed_fraud_types
    )

    if missing_fraud_types:
        raise ValueError(
            "Expected fraud typologies "
            "were not generated: "
            f"{sorted(missing_fraud_types)}"
        )

    # ---------------------------------------------------------
    # Fraud scenario IDs
    # ---------------------------------------------------------

    if (
        fraud[
            "fraud_scenario_id"
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            "Some fraudulent transactions "
            "have no scenario ID."
        )

    # ---------------------------------------------------------
    # Positive transaction amounts
    # ---------------------------------------------------------

    if (
        fraud["amount"] <= 0
    ).any():
        raise ValueError(
            "Fraudulent transaction amounts "
            "must be positive."
        )

    # ---------------------------------------------------------
    # Non-negative balances
    # ---------------------------------------------------------

    if (
        fraud["balance_before"] < 0
    ).any():
        raise ValueError(
            "Fraudulent transactions contain "
            "negative starting balances."
        )

    if (
        fraud["balance_after"] < 0
    ).any():
        raise ValueError(
            "Fraudulent transactions contain "
            "negative ending balances."
        )

    # ---------------------------------------------------------
    # Account foreign keys
    # ---------------------------------------------------------

    valid_account_ids = set(
        accounts["account_id"]
    )

    invalid_accounts = (
        set(
            transactions_with_fraud[
                "account_id"
            ]
        )
        - valid_account_ids
    )

    if invalid_accounts:
        raise ValueError(
            "Some transactions reference "
            "unknown accounts."
        )

    # ---------------------------------------------------------
    # Customer foreign keys
    # ---------------------------------------------------------

    valid_customer_ids = set(
        customers["customer_id"]
    )

    invalid_customers = (
        set(
            transactions_with_fraud[
                "customer_id"
            ]
        )
        - valid_customer_ids
    )

    if invalid_customers:
        raise ValueError(
            "Some transactions reference "
            "unknown customers."
        )

    # ---------------------------------------------------------
    # Ensure original transactions were preserved
    # ---------------------------------------------------------

    original_ids = set(
        baseline["transaction_id"]
    )

    output_original_rows = (
        transactions_with_fraud[
            transactions_with_fraud[
                "transaction_id"
            ].isin(
                original_ids
            )
        ]
    )

    if (
        len(output_original_rows)
        != len(baseline)
    ):
        raise ValueError(
            "One or more baseline "
            "transactions were lost."
        )

    # ---------------------------------------------------------
    # Baseline transactions must remain legitimate
    # ---------------------------------------------------------

    if (
        output_original_rows[
            "is_fraud"
        ]
        .fillna(0)
        .astype(int)
        .sum()
        != 0
    ):
        raise ValueError(
            "A baseline transaction was "
            "incorrectly labelled as fraud."
        )

    # ---------------------------------------------------------
    # Row-count reconciliation
    # ---------------------------------------------------------

    fraud_count = len(fraud)

    expected_rows = (
        len(baseline)
        + fraud_count
    )

    if (
        len(transactions_with_fraud)
        != expected_rows
    ):
        raise ValueError(
            "Output row count does not equal "
            "baseline rows plus injected "
            "fraud rows."
        )


def build_fraud_summary(
    transactions_with_fraud: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Return fraud records and an aggregated summary."""

    fraud = transactions_with_fraud[
        transactions_with_fraud[
            "is_fraud"
        ]
        == 1
    ].copy()

    summary = (
        fraud
        .groupby(
            "fraud_type"
        )
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
    ]

    if fraud_type == FRAUD_TYPE_GEOGRAPHIC_ANOMALY:
        columns.append("province")

    columns.extend(
        [
            "device_id",
            "fraud_scenario_id",
        ]
    )

    if fraud_type == FRAUD_TYPE_BALANCE_DRAINING:
        columns.extend(
            [
                "balance_before",
                "balance_after",
            ]
        )

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

def validate_geographic_anomalies(
    baseline: pd.DataFrame,
    transactions_with_fraud: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate Geographic Anomaly scenarios.

    Each fraud transaction must occur 5-45 minutes after
    the customer's most recent legitimate domestic
    transaction and must occur in a different South
    African province.
    """

    south_african_provinces = {
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

    geographic_fraud = (
        transactions_with_fraud[
            transactions_with_fraud[
                "fraud_type"
            ]
            == FRAUD_TYPE_GEOGRAPHIC_ANOMALY
        ]
        .sort_values("timestamp")
        .copy()
    )

    if geographic_fraud.empty:
        raise ValueError(
            "No Geographic Anomaly fraud "
            "transactions were generated."
        )

    baseline_copy = baseline.copy()

    baseline_copy["timestamp"] = (
        pd.to_datetime(
            baseline_copy["timestamp"]
        )
    )

    validation_rows = []

    for _, fraud_row in (
        geographic_fraud.iterrows()
    ):
        fraud_timestamp = pd.Timestamp(
            fraud_row["timestamp"]
        )

        # -----------------------------------------------------
        # Only domestic legitimate transactions can serve
        # as the geographic anchor.
        # -----------------------------------------------------

        customer_history = (
            baseline_copy[
                (
                    baseline_copy[
                        "customer_id"
                    ]
                    == fraud_row[
                        "customer_id"
                    ]
                )
                & (
                    baseline_copy[
                        "timestamp"
                    ]
                    < fraud_timestamp
                )
                & (
                    baseline_copy[
                        "country"
                    ]
                    == "South Africa"
                )
                & (
                    baseline_copy[
                        "province"
                    ].notna()
                )
                & (
                    baseline_copy[
                        "province"
                    ]
                    != "INTERNATIONAL"
                )
            ]
            .sort_values("timestamp")
        )

        if customer_history.empty:
            raise ValueError(
                "A Geographic Anomaly scenario "
                "has no preceding legitimate "
                "domestic transaction."
            )

        previous = (
            customer_history.iloc[-1]
        )

        previous_timestamp = (
            pd.Timestamp(
                previous["timestamp"]
            )
        )

        previous_province = str(
            previous["province"]
        )

        fraud_province = str(
            fraud_row["province"]
        )

        gap_minutes = (
            fraud_timestamp
            - previous_timestamp
        ).total_seconds() / 60.0

        validation_rows.append(
            {
                "customer_id": (
                    fraud_row[
                        "customer_id"
                    ]
                ),
                "previous_timestamp": (
                    previous_timestamp
                ),
                "fraud_timestamp": (
                    fraud_timestamp
                ),
                "previous_province": (
                    previous_province
                ),
                "fraud_province": (
                    fraud_province
                ),
                "gap_minutes": round(
                    gap_minutes,
                    2,
                ),
                "province_changed": (
                    previous_province
                    != fraud_province
                ),
                "previous_is_domestic": (
                    previous_province
                    in south_african_provinces
                ),
                "fraud_is_domestic": (
                    fraud_province
                    in south_african_provinces
                ),
            }
        )

    geographic_validation = (
        pd.DataFrame(
            validation_rows
        )
    )

    # ---------------------------------------------------------
    # Validation 1:
    # Every generated geographic scenario must be validated.
    # ---------------------------------------------------------

    if (
        len(geographic_validation)
        != len(geographic_fraud)
    ):
        raise ValueError(
            "Not all Geographic Anomaly "
            "transactions could be validated."
        )

    # ---------------------------------------------------------
    # Validation 2:
    # Previous location must be a real SA province.
    # ---------------------------------------------------------

    if not (
        geographic_validation[
            "previous_is_domestic"
        ].all()
    ):
        raise ValueError(
            "A Geographic Anomaly uses a "
            "non-domestic previous location."
        )

    # ---------------------------------------------------------
    # Validation 3:
    # Fraud location must be a real SA province.
    # ---------------------------------------------------------

    if not (
        geographic_validation[
            "fraud_is_domestic"
        ].all()
    ):
        raise ValueError(
            "A Geographic Anomaly uses a "
            "non-domestic fraud location."
        )

    # ---------------------------------------------------------
    # Validation 4:
    # Province must change.
    # ---------------------------------------------------------

    if not (
        geographic_validation[
            "province_changed"
        ].all()
    ):
        raise ValueError(
            "At least one Geographic Anomaly "
            "did not change province."
        )

    # ---------------------------------------------------------
    # Validation 5:
    # Minimum impossible-travel interval.
    # ---------------------------------------------------------

    minimum_gap = (
        geographic_validation[
            "gap_minutes"
        ].min()
    )

    if minimum_gap < 5:
        raise ValueError(
            "A Geographic Anomaly occurred "
            "less than 5 minutes after the "
            "previous legitimate domestic "
            "transaction."
        )

    # ---------------------------------------------------------
    # Validation 6:
    # Maximum impossible-travel interval.
    # ---------------------------------------------------------

    maximum_gap = (
        geographic_validation[
            "gap_minutes"
        ].max()
    )

    if maximum_gap > 45:
        raise ValueError(
            "A Geographic Anomaly occurred "
            "more than 45 minutes after the "
            "previous legitimate domestic "
            "transaction."
        )

    return geographic_validation

def save_output(
    transactions_with_fraud: pd.DataFrame,
    fraud: pd.DataFrame,
) -> None:
    """Write processed dataset and inspection sample."""

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
        fraud
        .sort_values(
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


def print_geographic_validation(
    geographic_validation: pd.DataFrame,
) -> None:
    """Print Geographic Anomaly validation diagnostics."""

    display_columns = [
        "customer_id",
        "previous_timestamp",
        "fraud_timestamp",
        "previous_province",
        "fraud_province",
        "gap_minutes",
        "province_changed",
    ]

    print(
        "\nGeographic Anomaly validation:"
    )

    print(
        geographic_validation[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nGeographic validation summary:"
    )

    print(
        "Scenarios validated:",
        len(
            geographic_validation
        ),
    )

    print(
        "Previous locations domestic:",
        geographic_validation[
            "previous_is_domestic"
        ].all(),
    )

    print(
        "Fraud locations domestic:",
        geographic_validation[
            "fraud_is_domestic"
        ].all(),
    )

    print(
        "Province changed:",
        geographic_validation[
            "province_changed"
        ].all(),
    )

    print(
        "Minimum gap (minutes):",
        geographic_validation[
            "gap_minutes"
        ].min(),
    )

    print(
        "Maximum gap (minutes):",
        geographic_validation[
            "gap_minutes"
        ].max(),
    )


def main() -> None:
    """Run the complete fraud-injection pipeline."""

    customers, accounts, transactions = (
        load_source_data()
    )

    print(
        f"Customers: {len(customers):,}"
    )

    print(
        f"Accounts: {len(accounts):,}"
    )

    print(
        "Baseline transactions: "
        f"{len(transactions):,}"
    )

    # ---------------------------------------------------------
    # Generate fraud
    # ---------------------------------------------------------

    transactions_with_fraud = (
        inject_fraud_scenarios(
            transactions=transactions,
            accounts=accounts,
            customers=customers,
        )
    )

    # ---------------------------------------------------------
    # Structural validation
    # ---------------------------------------------------------

    validate_output(
        baseline=transactions,
        transactions_with_fraud=(
            transactions_with_fraud
        ),
        accounts=accounts,
        customers=customers,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    fraud, summary = (
        build_fraud_summary(
            transactions_with_fraud
        )
    )

    fraud_rate = (
        transactions_with_fraud[
            "is_fraud"
        ].mean()
        * 100
    )

    total_scenarios = (
        fraud[
            "fraud_scenario_id"
        ]
        .nunique()
    )

    print(
        "\nFraud summary:"
    )

    print(
        summary.to_string()
    )

    print(
        "\nTotal fraud scenarios: "
        f"{total_scenarios:,}"
    )

    print(
        "Total output rows: "
        f"{len(transactions_with_fraud):,}"
    )

    print(
        "Fraudulent transactions: "
        f"{len(fraud):,}"
    )

    print(
        f"Fraud rate: {fraud_rate:.3f}%"
    )

    # ---------------------------------------------------------
    # Examples
    # ---------------------------------------------------------

    print_examples(
        fraud=fraud,
        fraud_type=(
            FRAUD_TYPE_ACCOUNT_TAKEOVER
        ),
        heading="Account Takeover",
    )

    print_examples(
        fraud=fraud,
        fraud_type=(
            FRAUD_TYPE_CARD_TESTING
        ),
        heading="Card Testing",
    )

    print_examples(
        fraud=fraud,
        fraud_type=(
            FRAUD_TYPE_VELOCITY_ATTACK
        ),
        heading="Velocity Attack",
    )

    print_examples(
        fraud=fraud,
        fraud_type=(
            FRAUD_TYPE_ABNORMAL_AMOUNT
        ),
        heading="Abnormal Amount",
    )

    print_examples(
        fraud=fraud,
        fraud_type=(
            FRAUD_TYPE_GEOGRAPHIC_ANOMALY
        ),
        heading="Geographic Anomaly",
    )

    # ---------------------------------------------------------
    # Geographic Anomaly validation
    # ---------------------------------------------------------

    geographic_validation = (
        validate_geographic_anomalies(
            baseline=transactions,
            transactions_with_fraud=(
                transactions_with_fraud
            ),
        )
    )

    print_geographic_validation(
        geographic_validation
    )
    print_examples(
        fraud=fraud,
        fraud_type=FRAUD_TYPE_BALANCE_DRAINING,
        heading="Balance Draining",
    )

    # ---------------------------------------------------------
    # Save validated output
    # ---------------------------------------------------------

    save_output(
        transactions_with_fraud=(
            transactions_with_fraud
        ),
        fraud=fraud,
    )

    print(
        "\nSaved processed data to "
        "data/processed/"
        "transactions_with_fraud.parquet"
    )

    print(
        "Saved stratified inspection sample to "
        "data/sample/fraud_sample.csv"
    )


if __name__ == "__main__":
    main()