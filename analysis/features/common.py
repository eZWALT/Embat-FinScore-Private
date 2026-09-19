"""Shared constants and DuckDB helpers for the feature store.

Family modules must import from here. Do not open a write connection.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis"
DATA = ROOT / "data"
DB_PATH = DATA / "embat.duckdb"
HOLDOUT_PATH = ANALYSIS / "splits" / "holdout_companies.csv"

AS_OF = pd.Timestamp("2026-09-01")
MONTHS = pd.date_range("2024-09-01", "2026-08-01", freq="MS")
LAST_M = MONTHS[-1]
WEEKS = pd.date_range("2024-09-02", "2026-08-31", freq="W-MON")

# Reuse Javier's transaction category grouping.
CAT_MAP = {
    "collection": "op_in",
    "bulk_collection": "op_in",
    "cash_settlement": "op_in",
    "cash_settlements": "op_in",
    "pos_settlement": "op_in",
    "collection_refund": "op_in",
    "payment": "op_out",
    "bulk_payment": "op_out",
    "utility": "op_out",
    "payment_refund": "op_out",
    "salary": "op_out",
    "social_security": "op_out",
    "tax": "op_out",
    "tax_refund": "op_out",
    "cash_withdrawal": "op_out",
    "pos_withdrawal": "op_out",
    "fee": "fin_cost",
    "interest_charge": "fin_cost",
    "debt_repayment": "debt_service",
    "transfer": "transfer",
    "investment_deployment": "invest",
    "investment_return": "invest",
}

TABLES = (
    "groups",
    "companies",
    "banking_products",
    "debt_products",
    "debt_schedule_config",
    "transactions",
    "invoices",
    "balances",
)


def connect(schema_first: str = "clean"):
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} missing — run python analysis/build_db.py")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute(f"SET search_path = '{schema_first},main'")
    return con


def load_holdout() -> set[str]:
    df = pd.read_csv(HOLDOUT_PATH)
    return set(df["company_id"].astype(str))


def train_mask(company_id: pd.Series) -> pd.Series:
    hold = load_holdout()
    return ~company_id.astype(str).isin(hold)
