"""Read-only access to the cleaned DuckDB (`clean` schema only).

Resolution: POC_CLEAN_DB, then data/clean.duckdb (clean-only export), then data/embat.duckdb
(holds a raw `main` copy too; the guard below still restricts queries to `clean.*`).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import duckdb
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CANDIDATES = [
    os.environ.get("POC_CLEAN_DB"),
    REPO / "data" / "clean.duckdb",
    REPO / "data" / "bundle_work" / "embat.duckdb",
    REPO / "data" / "embat.duckdb",
]

ALLOWED_TABLES = {
    "companies", "groups", "transactions", "invoices", "balances",
    "banking_products", "debt_products", "debt_schedule_config", "dq_log",
}
MAX_ROWS = 200
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|copy|export|import|pragma|install|load|call|set|reset|truncate|vacuum|checkpoint)\b",
    re.I,
)
_TABLE_REF = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][\w.]*)", re.I)


def db_path() -> Path:
    for c in CANDIDATES:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError("No DuckDB found. Run `python analysis/build_db.py` or set POC_CLEAN_DB.")


def connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path()), read_only=True)


class UnsafeQuery(ValueError):
    pass


def check_sql(sql: str) -> str:
    """Allow a single SELECT over clean.* tables; add a LIMIT if missing."""
    s = sql.strip().rstrip(";")
    if ";" in s:
        raise UnsafeQuery("one statement only")
    if not re.match(r"^\s*(select|with)\b", s, re.I):
        raise UnsafeQuery("SELECT only")
    if _FORBIDDEN.search(s):
        raise UnsafeQuery("read-only: statement contains a forbidden keyword")
    for ref in _TABLE_REF.findall(s):
        low = ref.lower()
        if low.startswith("clean."):
            if low.split(".", 1)[1] not in ALLOWED_TABLES:
                raise UnsafeQuery(f"unknown table {ref}")
        elif low in ALLOWED_TABLES:
            raise UnsafeQuery(f"use the clean schema: clean.{ref}")
        elif "." in low or low in ALLOWED_TABLES:
            raise UnsafeQuery(f"only clean.* tables may be queried, not {ref}")
        # else: a CTE name, allowed
    if not re.search(r"\blimit\s+\d+", s, re.I):
        s = f"{s}\nLIMIT {MAX_ROWS}"
    return s


def query(sql: str) -> pd.DataFrame:
    safe = check_sql(sql)
    con = connect()
    try:
        return con.sql(safe).df().head(MAX_ROWS)
    finally:
        con.close()
