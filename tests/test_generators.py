import numpy as np

from src.generator import generate_accounts, generate_customers, generate_transactions


def make_data(seed=42):
    rng = np.random.default_rng(seed)
    customers = generate_customers(100, rng, "2026-06-30")
    accounts = generate_accounts(customers, 140, rng)
    tx = generate_transactions(customers, accounts, 1000, "2025-01-01", "2026-06-30", rng)
    return customers, accounts, tx


def test_requested_row_counts_and_unique_ids():
    customers, accounts, tx = make_data()
    assert (len(customers), len(accounts), len(tx)) == (100, 140, 1000)
    assert customers.customer_id.is_unique
    assert accounts.account_id.is_unique
    assert tx.transaction_id.is_unique


def test_relational_integrity():
    customers, accounts, tx = make_data()
    assert accounts.customer_id.isin(customers.customer_id).all()
    assert tx.customer_id.isin(customers.customer_id).all()
    assert tx.account_id.isin(accounts.account_id).all()


def test_baseline_has_no_fraud_leakage():
    _, _, tx = make_data()
    assert {"is_fraud", "fraud_rule", "fraud_scenario"}.isdisjoint(tx.columns)


def test_transaction_constraints():
    _, _, tx = make_data()
    assert tx.amount.gt(0).all()
    assert tx.balance_before.ge(0).all()
    assert tx.balance_after.ge(0).all()
    assert tx.timestamp.between("2025-01-01", "2026-06-30 23:59:59").all()


def test_seed_is_reproducible():
    a = make_data(7)[2]
    b = make_data(7)[2]
    assert a.equals(b)
