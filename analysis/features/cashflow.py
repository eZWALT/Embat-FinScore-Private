"""Family A — cash flow levels and shape (transactions only).

No look-ahead: a row for `period` (month-start) uses booking dates in the
half-open interval [period, period + 1 month). That is the same cut as
`date_trunc('month', date) = period`. Holdout companies are scored with the
same formulas; nothing here fits percentiles or other train-only refs.

Category groups come from `analysis.features.common.CAT_MAP`. Bank sign is
kept: inflows > 0, outflows < 0. Outflow groups are negated so typical months
are non-negative (same convention as `score_pipeline._monthly_flows`).
Clean flags (`is_dup`, `is_extreme`) are not used as drop filters.

Formulas
--------
Monthly levels (0 if the company-month is on the grid but has no txs):

- a_op_in        = sum(amount | grp = op_in)
- a_op_out       = -sum(amount | grp = op_out)
- a_fin_cost     = -sum(amount | grp = fin_cost)
- a_debt_service = -sum(amount | grp = debt_service)
- a_transfer     = sum(amount | grp = transfer)     # signed net
- a_invest       = sum(amount | grp = invest)       # deploy + return, signed
- a_net          = a_op_in - a_op_out
- a_n_tx         = count(*)

Trailing calendar sums on the company panel (NaN until the window is full;
`min_periods` = window, so a_in12 / a_out12 stay sparse):

- a_in3, a_in6, a_in12     = rolling sum of a_op_in over 3 / 6 / 12 months
- a_out3, a_out6, a_out12  = rolling sum of a_op_out over 3 / 6 / 12 months

Shape (NaN while the 3-month window is incomplete):

- a_net_margin = clip((in3 - out3) / in3, -1, 1) if in3 > 0 else -1
- a_io_ratio   = min(3, in3 / max(out3, 1))          # coverage in the pipeline
- a_growth_3   = clip(in3 / in3_{t-3}  - 1, -1, 1) if in3_{t-3}  > 0 else NaN
- a_growth_12  = clip(in3 / in3_{t-12} - 1, -1, 1) if in3_{t-12} > 0 else NaN
  (YoY of the trailing-3m inflow window; needs month 15+ on the company grid)

Mix:

- a_uncat_share   = share of txs whose category is `uncategorized` or not in CAT_MAP
- a_pending_share = n_pending / (n_pending + n_booked) if `status` exists
                    and the month has at least one pending or booked tx; else NaN
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import CAT_MAP

SOURCE_TABLES = ["transactions"]
FAMILY = "a"

_LEVEL_ZERO = (
    "a_op_in",
    "a_op_out",
    "a_fin_cost",
    "a_debt_service",
    "a_transfer",
    "a_invest",
    "a_n_tx",
)


def _table_columns(con, table: str) -> set[str]:
    info = con.execute(f"DESCRIBE {table}").df()
    return set(info["column_name"].astype(str))


def _monthly_flows(con) -> pd.DataFrame:
    cols = _table_columns(con, "transactions")
    has_status = "status" in cols

    cm = pd.DataFrame({"category": list(CAT_MAP), "grp": list(CAT_MAP.values())})
    con.register("_feat_a_cat_map", cm)

    pending_sql = (
        """
          SUM(CASE WHEN t.status = 'pending' THEN 1 ELSE 0 END) AS n_pending,
          SUM(CASE WHEN t.status = 'booked' THEN 1 ELSE 0 END) AS n_booked,
        """
        if has_status
        else "CAST(NULL AS BIGINT) AS n_pending, CAST(NULL AS BIGINT) AS n_booked,"
    )

    df = con.execute(
        f"""
        SELECT
          t.company_id,
          CAST(date_trunc('month', t."date") AS DATE) AS month,
          SUM(CASE WHEN c.grp = 'op_in' THEN t.amount ELSE 0 END) AS a_op_in,
          -SUM(CASE WHEN c.grp = 'op_out' THEN t.amount ELSE 0 END) AS a_op_out,
          -SUM(CASE WHEN c.grp = 'fin_cost' THEN t.amount ELSE 0 END) AS a_fin_cost,
          -SUM(CASE WHEN c.grp = 'debt_service' THEN t.amount ELSE 0 END) AS a_debt_service,
          SUM(CASE WHEN c.grp = 'transfer' THEN t.amount ELSE 0 END) AS a_transfer,
          SUM(CASE WHEN c.grp = 'invest' THEN t.amount ELSE 0 END) AS a_invest,
          COUNT(*) AS a_n_tx,
          SUM(CASE WHEN t.category = 'uncategorized' OR c.grp IS NULL THEN 1 ELSE 0 END) AS n_uncat,
          {pending_sql}
        FROM transactions t
        LEFT JOIN _feat_a_cat_map c ON t.category = c.category
        WHERE t."date" IS NOT NULL
        GROUP BY 1, 2
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df["period"] = pd.to_datetime(df["month"])
    return df.drop(columns=["month"])


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    keys = grid[["company_id", "period"]].copy()
    keys["company_id"] = keys["company_id"].astype(str)
    keys["period"] = pd.to_datetime(keys["period"])

    monthly = _monthly_flows(con)
    out = keys.merge(monthly, on=["company_id", "period"], how="left")
    for col in _LEVEL_ZERO:
        out[col] = out[col].fillna(0.0)
    out["a_net"] = out["a_op_in"] - out["a_op_out"]

    n_tx = out["a_n_tx"]
    out["a_uncat_share"] = np.where(n_tx > 0, out["n_uncat"] / n_tx, np.nan)
    denom = out["n_pending"] + out["n_booked"]
    out["a_pending_share"] = np.where(denom > 0, out["n_pending"] / denom, np.nan)

    out = out.sort_values(["company_id", "period"]).reset_index(drop=True)
    g = out.groupby("company_id", sort=False)
    out["a_in3"] = g["a_op_in"].transform(lambda s: s.rolling(3, min_periods=3).sum())
    out["a_out3"] = g["a_op_out"].transform(lambda s: s.rolling(3, min_periods=3).sum())
    out["a_in6"] = g["a_op_in"].transform(lambda s: s.rolling(6, min_periods=6).sum())
    out["a_out6"] = g["a_op_out"].transform(lambda s: s.rolling(6, min_periods=6).sum())
    out["a_in12"] = g["a_op_in"].transform(lambda s: s.rolling(12, min_periods=12).sum())
    out["a_out12"] = g["a_op_out"].transform(lambda s: s.rolling(12, min_periods=12).sum())

    in3, out3 = out["a_in3"], out["a_out3"]
    out["a_net_margin"] = np.where(
        in3.isna(),
        np.nan,
        np.where(in3.to_numpy() > 0, ((in3 - out3) / in3).clip(-1.0, 1.0), -1.0),
    )
    out["a_io_ratio"] = np.minimum(3.0, in3 / np.maximum(out3, 1.0))

    g = out.groupby("company_id", sort=False)
    in3_l3 = g["a_in3"].shift(3)
    in3_l12 = g["a_in3"].shift(12)
    out["a_growth_3"] = np.where(in3_l3 > 0, (in3 / in3_l3 - 1.0).clip(-1.0, 1.0), np.nan)
    out["a_growth_12"] = np.where(in3_l12 > 0, (in3 / in3_l12 - 1.0).clip(-1.0, 1.0), np.nan)

    a_cols = [c for c in out.columns if c.startswith("a_")]
    return out[["company_id", "period"] + a_cols].reset_index(drop=True)
