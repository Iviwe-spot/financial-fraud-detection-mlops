from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "transactions_with_fraud.parquet"
)

OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "fraud_features.parquet"


REQUIRED_COLUMNS = {
    "transaction_id",
    "account_id",
    "customer_id",
    "timestamp",
    "amount",
    "merchant",
    "transaction_type",
    "channel",
    "device_id",
    "province",
    "country",
    "is_international",
    "balance_before",
    "is_fraud",
    "fraud_type",
    "fraud_scenario_id",
}


METADATA_COLUMNS = [
    "transaction_id",
    "customer_id",
    "account_id",
    "timestamp",
    "fraud_type",
    "fraud_scenario_id",
]


TARGET_COLUMN = "is_fraud"


NUMERIC_FEATURE_COLUMNS = [
    "amount",
    "balance_before",
    "balance_utilisation",
    "prior_transaction_count",
    "prior_mean_amount",
    "prior_std_amount",
    "amount_vs_prior_mean",
    "amount_zscore",
    "minutes_since_previous",
    "transactions_last_10min",
    "transactions_last_1h",
    "transactions_last_24h",
    "spend_last_1h",
    "spend_last_24h",
    "new_device",
    "prior_device_transaction_count",
    "new_merchant",
    "prior_merchant_transaction_count",
    "hour",
    "day_of_week",
    "is_weekend",
    "unusual_hour",
    "province_change",
    "international_deviation",
]


CATEGORICAL_FEATURE_COLUMNS = [
    "channel",
    "transaction_type",
    "province",
    "country",
    "is_international",
]


MODEL_FEATURE_COLUMNS = (
    NUMERIC_FEATURE_COLUMNS
    + CATEGORICAL_FEATURE_COLUMNS
)


def validate_input_schema(
    transactions: pd.DataFrame,
) -> None:
    missing = REQUIRED_COLUMNS.difference(
        transactions.columns
    )

    if missing:
        raise ValueError(
            "Missing required transaction columns: "
            + ", ".join(sorted(missing))
        )


def _normalise_string(
    value: Any,
) -> str | None:
    if pd.isna(value):
        return None

    return str(value)


def _hour_is_usual(
    hour: int,
    start_hour: int | float | None,
    end_hour: int | float | None,
) -> bool:
    if (
        start_hour is None
        or end_hour is None
        or pd.isna(start_hour)
        or pd.isna(end_hour)
    ):
        return True

    start = int(start_hour)
    end = int(end_hour)

    if start <= end:
        return start <= hour <= end

    return (
        hour >= start
        or hour <= end
    )


def _remove_expired(
    history: deque,
    current_timestamp: pd.Timestamp,
    window: pd.Timedelta,
) -> None:
    cutoff = current_timestamp - window

    while history:
        timestamp = history[0][0]

        if timestamp >= cutoff:
            break

        history.popleft()


def build_point_in_time_features(
    transactions: pd.DataFrame,
    customers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build transaction-level behavioural features using only
    information available strictly before each transaction.

    Transactions sharing the same customer and timestamp are
    processed as one timestamp batch. This prevents same-time
    transactions from becoming historical information for one
    another merely because of row ordering.
    """

    validate_input_schema(transactions)

    data = transactions.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="raise",
    )

    if data["timestamp"].isna().any():
        raise ValueError(
            "Transaction timestamps may not be null."
        )

    customer_profiles: dict[str, dict[str, Any]] = {}

    if customers is not None:
        required_customer_columns = {
            "customer_id",
            "usual_start_hour",
            "usual_end_hour",
            "international_rate",
        }

        missing_customer_columns = (
            required_customer_columns.difference(
                customers.columns
            )
        )

        if missing_customer_columns:
            raise ValueError(
                "Missing required customer columns: "
                + ", ".join(
                    sorted(
                        missing_customer_columns
                    )
                )
            )

        profile_frame = (
            customers[
                [
                    "customer_id",
                    "usual_start_hour",
                    "usual_end_hour",
                    "international_rate",
                ]
            ]
            .drop_duplicates(
                subset=["customer_id"],
                keep="last",
            )
        )

        customer_profiles = (
            profile_frame
            .set_index("customer_id")
            .to_dict(orient="index")
        )

    # Stable ordering is important for reproducibility.
    data = data.sort_values(
        [
            "customer_id",
            "timestamp",
            "transaction_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    feature_rows: list[dict[str, Any]] = []

    for customer_id, customer_data in data.groupby(
        "customer_id",
        sort=False,
    ):
        prior_count = 0
        prior_amount_sum = 0.0
        prior_amount_squared_sum = 0.0

        device_counts: Counter[str] = Counter()
        merchant_counts: Counter[str] = Counter()

        last_timestamp: pd.Timestamp | None = None
        last_province: str | None = None

        international_count = 0

        history_10min: deque = deque()
        history_1h: deque = deque()
        history_24h: deque = deque()

        profile = customer_profiles.get(
            customer_id,
            {},
        )

        usual_start_hour = profile.get(
            "usual_start_hour"
        )
        usual_end_hour = profile.get(
            "usual_end_hour"
        )

        profile_international_rate = profile.get(
            "international_rate"
        )

        # Same-timestamp transactions are scored before any of
        # them are added to history.
        for timestamp, timestamp_group in (
            customer_data.groupby(
                "timestamp",
                sort=False,
            )
        ):
            timestamp = pd.Timestamp(timestamp)

            _remove_expired(
                history_10min,
                timestamp,
                pd.Timedelta(minutes=10),
            )

            _remove_expired(
                history_1h,
                timestamp,
                pd.Timedelta(hours=1),
            )

            _remove_expired(
                history_24h,
                timestamp,
                pd.Timedelta(hours=24),
            )

            if prior_count > 0:
                prior_mean = (
                    prior_amount_sum
                    / prior_count
                )
            else:
                prior_mean = np.nan

            if prior_count > 1:
                variance_numerator = (
                    prior_amount_squared_sum
                    - (
                        prior_amount_sum ** 2
                        / prior_count
                    )
                )

                prior_variance = (
                    variance_numerator
                    / (prior_count - 1)
                )

                # Floating-point arithmetic can create tiny
                # negative values around zero.
                prior_variance = max(
                    prior_variance,
                    0.0,
                )

                prior_std = float(
                    np.sqrt(prior_variance)
                )
            else:
                prior_std = np.nan

            transactions_10min = len(
                history_10min
            )

            transactions_1h = len(
                history_1h
            )

            transactions_24h = len(
                history_24h
            )

            spend_1h = float(
                sum(
                    amount
                    for _, amount
                    in history_1h
                )
            )

            spend_24h = float(
                sum(
                    amount
                    for _, amount
                    in history_24h
                )
            )

            for _, row in timestamp_group.iterrows():
                amount = float(row["amount"])

                device_id = _normalise_string(
                    row["device_id"]
                )

                merchant = _normalise_string(
                    row["merchant"]
                )

                province = _normalise_string(
                    row["province"]
                )

                current_is_international = bool(
                    row["is_international"]
                )

                if (
                    prior_count > 0
                    and prior_mean > 0
                ):
                    amount_vs_prior_mean = (
                        amount / prior_mean
                    )
                else:
                    amount_vs_prior_mean = np.nan

                if (
                    prior_count > 1
                    and pd.notna(prior_std)
                    and prior_std > 0
                ):
                    amount_zscore = (
                        amount - prior_mean
                    ) / prior_std
                else:
                    amount_zscore = np.nan

                if last_timestamp is None:
                    minutes_since_previous = np.nan
                else:
                    minutes_since_previous = (
                        (
                            timestamp
                            - last_timestamp
                        ).total_seconds()
                        / 60.0
                    )

                prior_device_count = (
                    device_counts[device_id]
                    if device_id is not None
                    else 0
                )

                new_device = int(
                    device_id is not None
                    and prior_device_count == 0
                )

                prior_merchant_count = (
                    merchant_counts[merchant]
                    if merchant is not None
                    else 0
                )

                new_merchant = int(
                    merchant is not None
                    and prior_merchant_count == 0
                )

                hour = int(timestamp.hour)
                day_of_week = int(
                    timestamp.dayofweek
                )
                is_weekend = int(
                    day_of_week >= 5
                )

                unusual_hour = int(
                    not _hour_is_usual(
                        hour,
                        usual_start_hour,
                        usual_end_hour,
                    )
                )

                province_change = int(
                    last_province is not None
                    and province is not None
                    and province != last_province
                )

                if prior_count > 0:
                    historical_international_rate = (
                        international_count
                        / prior_count
                    )
                elif (
                    profile_international_rate
                    is not None
                    and not pd.isna(
                        profile_international_rate
                    )
                ):
                    historical_international_rate = float(
                        profile_international_rate
                    )
                else:
                    historical_international_rate = 0.0

                international_deviation = abs(
                    int(current_is_international)
                    - historical_international_rate
                )

                balance_before = float(
                    row["balance_before"]
                )

                if balance_before > 0:
                    balance_utilisation = (
                        amount / balance_before
                    )
                else:
                    balance_utilisation = np.nan

                feature_rows.append(
                    {
                        # Evaluation / lineage metadata
                        "transaction_id": (
                            row["transaction_id"]
                        ),
                        "customer_id": customer_id,
                        "account_id": (
                            row["account_id"]
                        ),
                        "timestamp": timestamp,
                        "fraud_type": (
                            row["fraud_type"]
                        ),
                        "fraud_scenario_id": (
                            row[
                                "fraud_scenario_id"
                            ]
                        ),

                        # Current transaction
                        "amount": amount,
                        "balance_before": (
                            balance_before
                        ),
                        "balance_utilisation": (
                            balance_utilisation
                        ),

                        # Historical amount behaviour
                        "prior_transaction_count": (
                            prior_count
                        ),
                        "prior_mean_amount": (
                            prior_mean
                        ),
                        "prior_std_amount": (
                            prior_std
                        ),
                        "amount_vs_prior_mean": (
                            amount_vs_prior_mean
                        ),
                        "amount_zscore": (
                            amount_zscore
                        ),

                        # Velocity
                        "minutes_since_previous": (
                            minutes_since_previous
                        ),
                        "transactions_last_10min": (
                            transactions_10min
                        ),
                        "transactions_last_1h": (
                            transactions_1h
                        ),
                        "transactions_last_24h": (
                            transactions_24h
                        ),
                        "spend_last_1h": spend_1h,
                        "spend_last_24h": (
                            spend_24h
                        ),

                        # Device behaviour
                        "new_device": new_device,
                        "prior_device_transaction_count": (
                            prior_device_count
                        ),

                        # Merchant behaviour
                        "new_merchant": new_merchant,
                        "prior_merchant_transaction_count": (
                            prior_merchant_count
                        ),

                        # Temporal behaviour
                        "hour": hour,
                        "day_of_week": day_of_week,
                        "is_weekend": is_weekend,
                        "unusual_hour": unusual_hour,

                        # Geographic behaviour
                        "province_change": (
                            province_change
                        ),
                        "international_deviation": (
                            international_deviation
                        ),

                        # Safe categorical context
                        "channel": row["channel"],
                        "transaction_type": (
                            row["transaction_type"]
                        ),
                        "province": row["province"],
                        "country": row["country"],
                        "is_international": (
                            current_is_international
                        ),

                        # Target
                        "is_fraud": int(
                            row["is_fraud"]
                        ),
                    }
                )

            # Only after every transaction at this timestamp has
            # been scored do we update historical state.
            for _, row in timestamp_group.iterrows():
                amount = float(row["amount"])

                device_id = _normalise_string(
                    row["device_id"]
                )

                merchant = _normalise_string(
                    row["merchant"]
                )

                province = _normalise_string(
                    row["province"]
                )

                prior_count += 1
                prior_amount_sum += amount
                prior_amount_squared_sum += (
                    amount ** 2
                )

                if device_id is not None:
                    device_counts[
                        device_id
                    ] += 1

                if merchant is not None:
                    merchant_counts[
                        merchant
                    ] += 1

                if bool(
                    row["is_international"]
                ):
                    international_count += 1

                history_item = (
                    timestamp,
                    amount,
                )

                history_10min.append(
                    history_item
                )

                history_1h.append(
                    history_item
                )

                history_24h.append(
                    history_item
                )

            # Same-time rows deliberately share the same prior
            # timestamp/province state.
            last_timestamp = timestamp

            # If multiple rows occur simultaneously, use the
            # final deterministic row only for future geography.
            final_row = timestamp_group.iloc[-1]

            last_province = _normalise_string(
                final_row["province"]
            )

    features = pd.DataFrame(
        feature_rows
    )

    features = features.sort_values(
        [
            "timestamp",
            "transaction_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return features


def validate_feature_output(
    source: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    if len(features) != len(source):
        raise ValueError(
            "Feature row count does not match "
            "source transaction row count."
        )

    if features["transaction_id"].duplicated().any():
        raise ValueError(
            "Feature table contains duplicate "
            "transaction IDs."
        )

    if features["is_fraud"].isna().any():
        raise ValueError(
            "Target contains null values."
        )

    prohibited_model_columns = {
        "transaction_id",
        "customer_id",
        "account_id",
        "timestamp",
        "fraud_type",
        "fraud_scenario_id",
        "device_id",
        "merchant",
    }

    leaked = (
        prohibited_model_columns
        .intersection(
            MODEL_FEATURE_COLUMNS
        )
    )

    if leaked:
        raise ValueError(
            "Prohibited raw columns appear in "
            "MODEL_FEATURE_COLUMNS: "
            + ", ".join(sorted(leaked))
        )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: "
            f"{INPUT_PATH}"
        )

    print(
        "Building point-in-time fraud features..."
    )

    transactions = pd.read_parquet(
        INPUT_PATH
    )

    customers_path = (
        ROOT
        / "data"
        / "raw"
        / "customers.parquet"
    )

    customers = (
        pd.read_parquet(customers_path)
        if customers_path.exists()
        else None
    )

    print(
        f"Input transactions: "
        f"{len(transactions):,}"
    )

    features = build_point_in_time_features(
        transactions=transactions,
        customers=customers,
    )

    validate_feature_output(
        source=transactions,
        features=features,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Output feature rows: "
        f"{len(features):,}"
    )

    print(
        f"Output columns: "
        f"{len(features.columns):,}"
    )

    print(
        f"Fraud rows: "
        f"{int(features['is_fraud'].sum()):,}"
    )

    print(
        f"Fraud rate: "
        f"{features['is_fraud'].mean():.3%}"
    )

    print(
        f"Saved: "
        f"{OUTPUT_PATH.relative_to(ROOT)}"
    )

    print()
    print(
        "Model-safe feature columns:"
    )

    for column in MODEL_FEATURE_COLUMNS:
        print(
            f"  - {column}"
        )


if __name__ == "__main__":
    main()
