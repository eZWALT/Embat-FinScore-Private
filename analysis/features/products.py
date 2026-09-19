"""Family G — product mix and access (banking_products only).

Time-aware: a row for `period` (month-start or week-start) counts only
products whose `created_at` is before the next period start
(`[created_at] < period + 1 month` on a monthly grid, `< period + 1 week`
on a weekly grid). `created_at` null is treated as present in every period
and is flagged by `g_created_unknown_share`. Nothing here fits percentiles
or other train-only refs; holdout companies are scored with the same formulas.

`company_meta.n_banking` is a static extract count. These columns are the
as-of stock (and mix) of connected banking products.

Formulas
--------
As-of inventory (LEFT JOIN from the grid; 0 when the company has no
qualifying product yet):

- g_n_accounts = count(product_id)
- g_n_banks    = count(distinct bank_name)
- g_n_types    = count(distinct type)
- g_has_card / g_has_tpv / g_has_checking / g_has_saving / g_has_investment
  = 1 if any as-of product has that `type`, else 0

Mix / flags (shares are NaN when g_n_accounts = 0):

- g_custom_share = share with service = custom or bank_name like
  Other / customer-defined (the extract uses both together)
- g_created_unknown_share = share with created_at null
- g_created_after_snapshot = count with created_at > 2026-09-01
  (clean.dq_log: 44 banking products; the journal "63" is 44 banking + 19
  debt). On the monthly panel the last cut is created_at < 2026-09-01, so
  this count is 0 — those rows are not treated as known before the snapshot.
- g_created_after_snapshot_share = that count / g_n_accounts
- g_new_this_month = count with created_at in [period, period_next)
  (period-relative despite the name)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import AS_OF, MONTHS, WEEKS

SOURCE_TABLES = ["banking_products"]
FAMILY = "g"

FEATURE_COLS = (
    "g_n_accounts",
    "g_n_banks",
    "g_n_types",
    "g_has_card",
    "g_has_tpv",
    "g_has_checking",
    "g_has_saving",
    "g_has_investment",
    "g_custom_share",
    "g_created_unknown_share",
    "g_created_after_snapshot",
    "g_created_after_snapshot_share",
    "g_new_this_month",
)

_COUNT_ZERO = (
    "g_n_accounts",
    "g_n_banks",
    "g_n_types",
    "g_has_card",
    "g_has_tpv",
    "g_has_checking",
    "g_has_saving",
    "g_has_investment",
    "g_created_after_snapshot",
    "g_new_this_month",
)

_CUSTOM_SQL = """(
    lower(coalesce(p.service, '')) = 'custom'
    OR p.bank_name ILIKE '%customer-defined%'
    OR p.bank_name ILIKE 'Other%'
)"""


def _freq(periods: pd.Series) -> str:
    p = pd.to_datetime(periods).drop_duplicates()
    ps = set(p)
    if ps and ps <= set(MONTHS):
        return "M"
    if ps and ps <= set(WEEKS):
        return "W"
    if len(p) >= 2:
        return "W" if p.sort_values().diff().median() <= pd.Timedelta(days=8) else "M"
    if len(p) == 1 and int(p.iloc[0].day) != 1:
        return "W"
    return "M"


def _period_next(period: pd.Series, freq: str) -> pd.Series:
    period = pd.to_datetime(period)
    if freq == "W":
        return period + pd.Timedelta(weeks=1)
    return period + pd.DateOffset(months=1)


def _register(con, name: str, df: pd.DataFrame) -> None:
    try:
        con.unregister(name)
    except Exception:
        pass
    con.register(name, df)


def _empty(grid: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "company_id": grid["company_id"].astype(str) if "company_id" in grid else pd.Series(dtype=str),
            "period": pd.to_datetime(grid["period"]) if "period" in grid else pd.Series(dtype="datetime64[ns]"),
        }
    )
    for c in FEATURE_COLS:
        out[c] = np.nan
    return out


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    if grid is None or grid.empty:
        return _empty(pd.DataFrame({"company_id": [], "period": []}))

    keys = grid[["company_id", "period"]].copy()
    keys["company_id"] = keys["company_id"].astype(str)
    keys["period"] = pd.to_datetime(keys["period"]).dt.normalize()
    freq = _freq(keys["period"])
    keys["period_next"] = pd.to_datetime(_period_next(keys["period"], freq))

    _register(con, "_feat_g_grid", keys[["company_id", "period", "period_next"]])
    as_of = AS_OF.strftime("%Y-%m-%d")
    try:
        raw = con.execute(
            f"""
            SELECT
              g.company_id,
              CAST(g.period AS DATE) AS period,
              COUNT(p.product_id) AS g_n_accounts,
              COUNT(DISTINCT p.bank_name) AS g_n_banks,
              COUNT(DISTINCT p.type) AS g_n_types,
              MAX(CASE WHEN lower(p.type) = 'card' THEN 1 ELSE 0 END) AS g_has_card,
              MAX(CASE WHEN lower(p.type) = 'tpv' THEN 1 ELSE 0 END) AS g_has_tpv,
              MAX(CASE WHEN lower(p.type) = 'checking' THEN 1 ELSE 0 END) AS g_has_checking,
              MAX(CASE WHEN lower(p.type) = 'saving' THEN 1 ELSE 0 END) AS g_has_saving,
              MAX(CASE WHEN lower(p.type) = 'investment' THEN 1 ELSE 0 END) AS g_has_investment,
              COUNT(p.product_id) FILTER (WHERE {_CUSTOM_SQL}) AS n_custom,
              COUNT(p.product_id) FILTER (WHERE p.created_at IS NULL) AS n_unknown,
              COUNT(p.product_id) FILTER (WHERE p.created_at > TIMESTAMP '{as_of}')
                AS g_created_after_snapshot,
              SUM(CASE WHEN p.created_at IS NOT NULL
                        AND p.created_at >= g.period
                        AND p.created_at < g.period_next
                       THEN 1 ELSE 0 END) AS g_new_this_month
            FROM _feat_g_grid g
            LEFT JOIN banking_products p
              ON p.company_id = g.company_id
             AND (p.created_at IS NULL OR p.created_at < g.period_next)
            GROUP BY 1, 2
            """
        ).df()
    finally:
        try:
            con.unregister("_feat_g_grid")
        except Exception:
            pass

    raw["company_id"] = raw["company_id"].astype(str)
    raw["period"] = pd.to_datetime(raw["period"])

    out = keys[["company_id", "period"]].merge(raw, on=["company_id", "period"], how="left")
    for c in _COUNT_ZERO:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    n = out["g_n_accounts"].to_numpy(dtype=float)
    n_custom = pd.to_numeric(out.pop("n_custom"), errors="coerce").fillna(0).to_numpy(dtype=float)
    n_unknown = pd.to_numeric(out.pop("n_unknown"), errors="coerce").fillna(0).to_numpy(dtype=float)
    n_after = out["g_created_after_snapshot"].to_numpy(dtype=float)
    out["g_custom_share"] = np.divide(n_custom, n, out=np.full_like(n, np.nan), where=n > 0)
    out["g_created_unknown_share"] = np.divide(n_unknown, n, out=np.full_like(n, np.nan), where=n > 0)
    out["g_created_after_snapshot_share"] = np.divide(n_after, n, out=np.full_like(n, np.nan), where=n > 0)

    return out[["company_id", "period", *FEATURE_COLS]].reset_index(drop=True)
