# Synthetic Banking Transaction Generator

The first engineering component of a South African financial-fraud detection platform. This version creates relational customer, account, and normal transaction data. It deliberately contains **no injected fraud**; this establishes a clean behavioural baseline before fraud typologies are added.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pipelines/generate_data.py --transactions 100000 --seed 42
python pipelines/validate_data.py --data-dir data/raw
pytest
```

The generator writes Parquet files to `data/raw/` and small CSV previews to `data/sample/`.

## Design

- Customers have stable behavioural profiles: income, home province, preferred channel, typical activity window, average transaction amount, transaction variation, and international propensity.
- Accounts are relational children of customers. A customer may own one or more accounts.
- Transactions are generated chronologically per account and inherit customer-specific behaviour.
- Large transactions are not automatically anomalous. Amounts are sampled relative to each customer's profile and constrained by the available account balance.
- No target label, fraud scenario, or fraud rule is included in this baseline dataset.

## Outputs

| File | Grain |
|---|---|
| `customers.parquet` | One row per customer |
| `accounts.parquet` | One row per account |
| `transactions.parquet` | One row per transaction |

## Reproducibility

All randomness flows from a single NumPy seed. Re-running the same command with the same configuration and seed produces the same records.

## Fraud injection

The project currently supports two composable fraud typologies:

| Fraud typology | Synthetic behaviour |
|---|---|
| Account Takeover | New device, unusually large customer-relative amounts, unusual activity hours and multiple rapid transactions |
| Card Testing | New device and a rapid burst of small transactions between R1 and R20 |

Generate the fraud-labelled development dataset with:

```bash
python pipelines/inject_fraud.py