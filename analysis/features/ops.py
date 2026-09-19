"""Family C — operational regularity (transactions only).

No look-ahead: a row for `period` (month-start) uses booking dates with
`date <= period month-end`. Same month cut as `date_trunc('month', date) = period`
for the in-month counts. Holdout companies are scored with the same formulas;
nothing here fits percentiles or other train-only refs.

Payroll / tax / social-security flags use the raw `category` values that
`CAT_MAP` maps to `op_out` (`salary`, `tax`, `social_security`). `tax_refund`
is not treated as tax. Clean flags (`is_dup`, `is_extreme`, `product_known`)
are not used as drop filters.

`c_gap_sd` is Pérez-Salazar, Márquez & Vidal-Silva 2026 (Computers 15:135)
operational volatility: sample stdev of inter-transaction time (σ_Δt). Dates
are collapsed to unique calendar days because 94% of booking timestamps are
midnight; consecutive same-day rows would otherwise flood the series with
0-day gaps and turn σ into a size proxy (1 / n_tx).

Formulas
--------
In-month (0 if the company-month is on the grid but has no txs):

- c_n_tx            = count(*)
- c_n_days_with_tx  = count of distinct booking dates
- c_zero_in_month   = 1 if no row with amount > 0 (any category)
- c_salary_month    = 1 if any category = salary
- c_tax_month       = 1 if any category = tax
- c_ss_month        = 1 if any category = social_security

Trailing / as-of period end (only events with date <= month-end):

- c_gap_sd = sample stdev (ddof=1) of day-gaps between consecutive unique
  booking dates in (month_end - 90d, month_end]. Null if fewer than 3
  distinct days (need ≥2 gaps).
- c_zero_in_share_6 = mean of c_zero_in_month over the last ≤6 months on the
  company grid (min_periods=1).
- c_missed_salary = 1 if sum(c_salary_month) over last ≤6 months ≥ 3
  and c_salary_month = 0 this month. c_missed_tax is the same for tax.
- c_recency_days = month_end.date - last booking date ≤ month_end.
- c_last_tx_before_2026_06 = 1 if period ≥ 2026-06-01 and last booking date
  as of month_end is < 2026-06-01. Javier's extract-level count is 61
  companies; as of 2026-08-31 (no Sept-1 look-ahead) this is 62 because
  COMP_0981 is silent from 2025-04-07 until 2026-09-01.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import CAT_MAP

SOURCE_TABLES = ["transactions"]
FAMILY = "c"

# CAT_MAP keys used as presence flags (all grouped as op_out).
_SALARY = "salary"
_TAX = "tax"
_SS = "social_security"
_CUTOFF_2026_06 = pd.Timestamp("2026-06-01")
_GAP_WINDOW_DAYS = 90
_ROLL = 6
_USUAL_MIN = 3

assert _SALARY in CAT_MAP and _TAX in CAT_MAP and _SS in CAT_MAP

_INT_ZERO = (
    "c_n_tx",
    "c_n_days_with_tx",
    "c_zero_in_month",
    "c_salary_month",
    "c_tax_month",
    "c_ss_month",
    "c_missed_salary",
    "c_missed_tax",
    "c_last_tx_before_2026_06",
)


def _month_end(period: pd.Series) -> pd.Series:
    return pd.to_datetime(period) + pd.offsets.MonthEnd(0)


def _monthly_ops(con) -> pd.DataFrame:
    df = con.execute(
        f"""
        SELECT
          company_id,
          CAST(date_trunc('month', "date") AS DATE) AS month,
          COUNT(*) AS c_n_tx,
          COUNT(DISTINCT CAST("date" AS DATE)) AS c_n_days_with_tx,
          MAX(CASE WHEN amount > 0 THEN 1 ELSE 0 END) AS has_in,
          MAX(CASE WHEN category = '{_SALARY}' THEN 1 ELSE 0 END) AS c_salary_month,
          MAX(CASE WHEN category = '{_TAX}' THEN 1 ELSE 0 END) AS c_tax_month,
          MAX(CASE WHEN category = '{_SS}' THEN 1 ELSE 0 END) AS c_ss_month,
          MAX("date") AS last_tx
        FROM transactions
        WHERE "date" IS NOT NULL
        GROUP BY 1, 2
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df["period"] = pd.to_datetime(df["month"])
    df["last_tx"] = pd.to_datetime(df["last_tx"])
    return df.drop(columns=["month"])


def _unique_days(con) -> pd.DataFrame:
    days = con.execute(
        """
        SELECT company_id, CAST("date" AS DATE) AS d
        FROM transactions
        WHERE "date" IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).df()
    days["company_id"] = days["company_id"].astype(str)
    days["d"] = pd.to_datetime(days["d"])
    return days


def _gap_sd_90(days: pd.DataFrame, keys: pd.DataFrame) -> pd.Series:
    """Sample stdev of unique-day gaps in the last 90 days ending at month-end."""
    ends = _month_end(keys["period"]).to_numpy(dtype="datetime64[ns]")
    starts = ends - np.timedelta64(_GAP_WINDOW_DAYS, "D")
    out = np.full(len(keys), np.nan)
    by_co = {cid: g["d"].to_numpy(dtype="datetime64[ns]") for cid, g in days.groupby("company_id", sort=False)}
    for cid, sub in keys.groupby("company_id", sort=False):
        d = by_co.get(cid)
        if d is None or len(d) < 3:
            continue
        idx = sub.index.to_numpy()
        lo = np.searchsorted(d, starts[idx], side="right")
        hi = np.searchsorted(d, ends[idx], side="right")
        for k, a, b in zip(idx, lo, hi):
            if b - a < 3:
                continue
            gaps = np.diff(d[a:b]) / np.timedelta64(1, "D")
            out[k] = float(gaps.std(ddof=1))
    return pd.Series(out, index=keys.index, name="c_gap_sd")


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    keys = grid[["company_id", "period"]].copy()
    keys["company_id"] = keys["company_id"].astype(str)
    keys["period"] = pd.to_datetime(keys["period"])
    keys = keys.reset_index(drop=True)

    monthly = _monthly_ops(con)
    out = keys.merge(monthly, on=["company_id", "period"], how="left")
    out["c_n_tx"] = out["c_n_tx"].fillna(0).astype(np.int32)
    out["c_n_days_with_tx"] = out["c_n_days_with_tx"].fillna(0).astype(np.int32)
    out["c_salary_month"] = out["c_salary_month"].fillna(0).astype(np.int8)
    out["c_tax_month"] = out["c_tax_month"].fillna(0).astype(np.int8)
    out["c_ss_month"] = out["c_ss_month"].fillna(0).astype(np.int8)
    out["c_zero_in_month"] = (out["has_in"].fillna(0).astype(np.int8) == 0).astype(np.int8)

    out = out.sort_values(["company_id", "period"]).reset_index(drop=True)
    g = out.groupby("company_id", sort=False)
    out["last_tx"] = g["last_tx"].ffill()
    last_day = pd.to_datetime(out["last_tx"]).dt.normalize()
    month_end = _month_end(out["period"]).dt.normalize()
    out["c_recency_days"] = (month_end - last_day).dt.days.astype(np.int32)
    out["c_last_tx_before_2026_06"] = (
        (out["period"] >= _CUTOFF_2026_06) & (last_day < _CUTOFF_2026_06)
    ).astype(np.int8)

    g = out.groupby("company_id", sort=False)
    out["c_zero_in_share_6"] = g["c_zero_in_month"].transform(
        lambda s: s.rolling(_ROLL, min_periods=1).mean()
    )
    sal6 = g["c_salary_month"].transform(lambda s: s.rolling(_ROLL, min_periods=1).sum())
    tax6 = g["c_tax_month"].transform(lambda s: s.rolling(_ROLL, min_periods=1).sum())
    out["c_missed_salary"] = ((sal6 >= _USUAL_MIN) & (out["c_salary_month"] == 0)).astype(np.int8)
    out["c_missed_tax"] = ((tax6 >= _USUAL_MIN) & (out["c_tax_month"] == 0)).astype(np.int8)

    out["c_gap_sd"] = _gap_sd_90(_unique_days(con), out[["company_id", "period"]])

    c_cols = [
        "c_n_tx",
        "c_n_days_with_tx",
        "c_gap_sd",
        "c_zero_in_month",
        "c_zero_in_share_6",
        "c_salary_month",
        "c_tax_month",
        "c_ss_month",
        "c_missed_salary",
        "c_missed_tax",
        "c_recency_days",
        "c_last_tx_before_2026_06",
    ]
    for col in _INT_ZERO:
        if col in out.columns:
            out[col] = out[col].astype(int)
    return out[["company_id", "period"] + c_cols].reset_index(drop=True)
