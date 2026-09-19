"""Company × month and company × ISO-week panels plus static metadata.

A company enters the grid at the month/week of its first transaction.
No look-ahead: metadata that is a snapshot (debt outstanding, balances) is
joined as-of the extraction date and flagged; do not treat snapshot fields
as known in 2024.
"""
from __future__ import annotations

import pandas as pd

from .common import LAST_M, MONTHS, WEEKS, connect


def first_activity(con) -> pd.DataFrame:
    first = con.execute(
        """
        SELECT company_id,
               CAST(date_trunc('month', MIN("date")) AS DATE) AS first_month,
               CAST(date_trunc('week', MIN("date")) AS DATE) AS first_week
        FROM transactions
        GROUP BY 1
        """
    ).df()
    first["first_month"] = pd.to_datetime(first["first_month"])
    first["first_week"] = pd.to_datetime(first["first_week"])
    return first


def monthly_grid(con=None) -> pd.DataFrame:
    close = False
    if con is None:
        con = connect()
        close = True
    first = first_activity(con)
    grid = pd.MultiIndex.from_product(
        [first["company_id"], MONTHS], names=["company_id", "period"]
    ).to_frame(index=False)
    grid = grid.merge(first, on="company_id")
    grid = grid[grid["period"] >= grid["first_month"]].drop(columns=["first_week"])
    grid["freq"] = "M"
    if close:
        con.close()
    return grid.reset_index(drop=True)


def weekly_grid(con=None) -> pd.DataFrame:
    close = False
    if con is None:
        con = connect()
        close = True
    first = first_activity(con)
    weeks = WEEKS[WEEKS <= LAST_M + pd.offsets.MonthEnd(0)]
    grid = pd.MultiIndex.from_product(
        [first["company_id"], weeks], names=["company_id", "period"]
    ).to_frame(index=False)
    grid = grid.merge(first, on="company_id")
    grid = grid[grid["period"] >= grid["first_week"]].drop(columns=["first_month"])
    grid["freq"] = "W"
    if close:
        con.close()
    return grid.reset_index(drop=True)


def company_meta(con=None) -> pd.DataFrame:
    close = False
    if con is None:
        con = connect()
        close = True
    meta = con.execute(
        """
        SELECT c.company_id, c.group_id, c.country, c.currency, c.erp, c.created_at,
               g.n_companies_in_sample AS group_size, g.erp AS group_erp
        FROM companies c
        LEFT JOIN groups g ON c.group_id = g.group_id
        """
    ).df()
    bank = con.execute(
        """
        SELECT company_id,
               COUNT(*) AS n_banking,
               COUNT(DISTINCT bank_name) AS n_banks,
               COUNT(DISTINCT type) AS n_bank_types
        FROM banking_products
        GROUP BY 1
        """
    ).df()
    debt = con.execute(
        """
        SELECT company_id,
               COUNT(*) AS n_debt,
               COUNT(DISTINCT type) AS n_debt_types
        FROM debt_products
        GROUP BY 1
        """
    ).df()
    meta = meta.merge(bank, on="company_id", how="left").merge(debt, on="company_id", how="left")
    for c in ("n_banking", "n_banks", "n_bank_types", "n_debt", "n_debt_types"):
        meta[c] = meta[c].fillna(0).astype(int)
    meta["created_at"] = pd.to_datetime(meta["created_at"], errors="coerce")
    if close:
        con.close()
    return meta
