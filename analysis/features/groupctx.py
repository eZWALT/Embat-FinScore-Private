"""Family H — group context (sibling operational flow this period).

No look-ahead: a row for `period` uses booking dates inside that period
only (`date_trunc` to the grid grain; same cut as Family A for months).
Sibling aggregates exclude the company. `h_group_size` is the static
`groups.n_companies_in_sample` snapshot (same field `company_meta` already
merges onto the assembler panel). Holdout companies are scored with the
same formulas; nothing here fits percentiles or other train-only refs.

This module does not import `cashflow.py` or `liquidity.py`. Operational
in/out reuse `CAT_MAP` from `common.py` (same groups as Family A).
Clean flags (`is_dup`, `is_extreme`) are not used as drop filters.

Formulas
--------
Company-period operational flows (0 if the company-period has no txs):

- op_in  = sum(amount | grp = op_in)
- op_out = -sum(amount | grp = op_out)
- net    = op_in - op_out
- n_tx   = count(*)

Group totals for the same period include every company in `companies`
with that `group_id` (not only rows on the caller's grid). Then:

- h_group_size         = n_companies_in_sample (static)
- h_n_siblings_active  = other group companies with n_tx > 0 this period
- h_sib_in             = sum(sibling op_in)
- h_sib_out            = sum(sibling op_out)
- h_sib_net            = h_sib_in - h_sib_out
- h_share_group_in     = company op_in / (company op_in + h_sib_in)
                         if the denominator > 0, else NaN
- h_sib_neg_share      = (# siblings with net < 0) / (h_group_size - 1)
                         if h_group_size > 1, else NaN

Solo groups (size 1): sibling sums and active-sibling count are 0;
share is 1 if the company has op_in > 0 else NaN.

On a weekly grid the same names are period-level (that ISO week), not
calendar-month aggregates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import CAT_MAP

SOURCE_TABLES = ["companies", "groups", "transactions"]
FAMILY = "h"

FEATURE_COLS = (
    "h_group_size",
    "h_n_siblings_active",
    "h_sib_in",
    "h_sib_out",
    "h_sib_net",
    "h_share_group_in",
    "h_sib_neg_share",
)

_CAT_MAP_NAME = "_feat_h_cat_map"
_GRID_NAME = "_feat_h_grid"


def _register(con, name: str, df: pd.DataFrame) -> None:
    try:
        con.unregister(name)
    except Exception:
        pass
    con.register(name, df)


def _infer_freq(period: pd.Series) -> str:
    u = pd.to_datetime(period).drop_duplicates().sort_values()
    if len(u) == 0:
        return "M"
    if len(u) >= 2:
        med = u.diff().dropna().median()
        if pd.notna(med) and med <= pd.Timedelta(days=8):
            return "W"
        return "M"
    ts = pd.Timestamp(u.iloc[0])
    return "M" if ts.is_month_start else "W"


def _max_end(period: pd.Series, freq: str) -> pd.Timestamp:
    p = pd.to_datetime(period)
    if p.empty:
        return pd.Timestamp("1970-01-01")
    if freq == "W":
        return (p.max() + pd.Timedelta(days=6)).normalize()
    return (p.max() + pd.offsets.MonthEnd(0)).normalize()


def _empty(keys: pd.DataFrame) -> pd.DataFrame:
    out = keys.copy()
    for col in FEATURE_COLS:
        out[col] = np.nan
    return out


def _panel(con, keys: pd.DataFrame, freq: str, max_end: pd.Timestamp) -> pd.DataFrame:
    """Company-period own flows + group totals (all companies, not only the grid)."""
    cm = pd.DataFrame({"category": list(CAT_MAP), "grp": list(CAT_MAP.values())})
    _register(con, _CAT_MAP_NAME, cm)
    _register(con, _GRID_NAME, keys[["company_id", "period"]])
    trunc = "week" if freq == "W" else "month"
    return con.execute(
        f"""
        WITH flows AS (
          SELECT
            t.company_id,
            CAST(date_trunc('{trunc}', t."date") AS DATE) AS period,
            SUM(CASE WHEN m.grp = 'op_in' THEN t.amount ELSE 0 END) AS op_in,
            -SUM(CASE WHEN m.grp = 'op_out' THEN t.amount ELSE 0 END) AS op_out,
            COUNT(*) AS n_tx
          FROM transactions t
          LEFT JOIN {_CAT_MAP_NAME} m ON t.category = m.category
          WHERE t."date" IS NOT NULL
            AND CAST(t."date" AS DATE) <= CAST(? AS DATE)
          GROUP BY 1, 2
        ),
        meta AS (
          SELECT c.company_id,
                 c.group_id,
                 g.n_companies_in_sample AS group_size
          FROM companies c
          LEFT JOIN groups g ON c.group_id = g.group_id
        ),
        grp AS (
          SELECT
            meta.group_id,
            f.period,
            SUM(f.op_in) AS g_in,
            SUM(f.op_out) AS g_out,
            SUM(CASE WHEN f.n_tx > 0 THEN 1 ELSE 0 END) AS g_n_active,
            SUM(CASE WHEN (f.op_in - f.op_out) < 0 THEN 1 ELSE 0 END) AS g_n_neg
          FROM flows f
          JOIN meta ON f.company_id = meta.company_id
          GROUP BY 1, 2
        )
        SELECT
          k.company_id,
          CAST(k.period AS DATE) AS period,
          meta.group_id,
          meta.group_size,
          COALESCE(f.op_in, 0) AS op_in,
          COALESCE(f.op_out, 0) AS op_out,
          COALESCE(f.n_tx, 0) AS n_tx,
          COALESCE(f.op_in, 0) - COALESCE(f.op_out, 0) AS net,
          COALESCE(grp.g_in, 0) AS g_in,
          COALESCE(grp.g_out, 0) AS g_out,
          COALESCE(grp.g_n_active, 0) AS g_n_active,
          COALESCE(grp.g_n_neg, 0) AS g_n_neg
        FROM {_GRID_NAME} k
        LEFT JOIN meta ON k.company_id = meta.company_id
        LEFT JOIN flows f
          ON k.company_id = f.company_id
         AND CAST(k.period AS DATE) = f.period
        LEFT JOIN grp
          ON meta.group_id = grp.group_id
         AND CAST(k.period AS DATE) = grp.period
        """,
        [max_end.date()],
    ).df()


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    keys = grid[["company_id", "period"]].copy()
    keys["company_id"] = keys["company_id"].astype(str)
    keys["period"] = pd.to_datetime(keys["period"]).dt.normalize()
    if keys.empty:
        return _empty(keys)

    freq = _infer_freq(keys["period"])
    out = _panel(con, keys, freq, _max_end(keys["period"], freq))
    out["company_id"] = out["company_id"].astype(str)
    out["period"] = pd.to_datetime(out["period"]).dt.normalize()

    solo = out["group_size"].fillna(0) <= 1
    sib_in = out["g_in"] - out["op_in"]
    sib_out = out["g_out"] - out["op_out"]
    n_active = out["g_n_active"] - (out["n_tx"] > 0).astype(int)
    n_neg = out["g_n_neg"] - (out["net"] < 0).astype(int)

    out["h_group_size"] = out["group_size"]
    out["h_sib_in"] = np.where(solo, 0.0, sib_in)
    out["h_sib_out"] = np.where(solo, 0.0, sib_out)
    out["h_sib_net"] = out["h_sib_in"] - out["h_sib_out"]
    out["h_n_siblings_active"] = np.where(solo, 0, n_active)

    denom = out["op_in"] + out["h_sib_in"]
    out["h_share_group_in"] = np.where(denom > 0, out["op_in"] / denom, np.nan)

    n_sib = out["h_group_size"] - 1
    out["h_sib_neg_share"] = np.where(n_sib > 0, n_neg / n_sib, np.nan)

    return out[["company_id", "period"] + list(FEATURE_COLS)].reset_index(drop=True)
