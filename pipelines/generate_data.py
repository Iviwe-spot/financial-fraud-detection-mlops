#!/usr/bin/env python3
"""CLI entry point for the relational synthetic banking generator."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator import generate_accounts, generate_customers, generate_transactions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config/generator.yaml"))
    parser.add_argument("--customers", type=int)
    parser.add_argument("--accounts", type=int)
    parser.add_argument("--transactions", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    n_customers = args.customers or cfg["n_customers"]
    n_accounts = args.accounts or cfg["n_accounts"]
    n_transactions = args.transactions or cfg["n_transactions"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    rng = np.random.default_rng(seed)

    customers = generate_customers(n_customers, rng, cfg["end_date"])
    accounts = generate_accounts(customers, n_accounts, rng)
    transactions = generate_transactions(customers, accounts, n_transactions, cfg["start_date"], cfg["end_date"], rng)

    output_dir = ROOT / cfg["output_dir"]
    sample_dir = ROOT / cfg["sample_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    tables = {"customers": customers, "accounts": accounts, "transactions": transactions}
    for name, frame in tables.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
        frame.head(cfg["csv_sample_rows"]).to_csv(sample_dir / f"{name}_sample.csv", index=False)

    print(pd.DataFrame({"table": tables.keys(), "rows": [len(x) for x in tables.values()]}).to_string(index=False))
    print(f"Seed: {seed} | Parquet output: {output_dir}")


if __name__ == "__main__":
    main()
