"""Family F — debt and financing.

Time-varying columns come only from `transactions` (same category map as
`score_pipeline.fin_cost_r` / `debt_serv_r`). Snapshot columns use
`debt_products` + `debt_schedule_config` and are named so they are not
mistaken for a 2024 book: `granted` / `outstanding` are a 2026-09-01
extract, not a reconstructed path.

Formulas
--------
Flows in a period (month or ISO week), excluding `is_extreme`:

- ``f_debt_service`` = −sum(amount | category = debt_repayment)
- ``f_fin_cost`` = −sum(amount | category ∈ {fee, interest_charge})
- ``in3`` / ``ds3`` / ``fc3`` = rolling 3 months (13 weeks on a weekly grid)
- ``f_ds_r`` = ds3 / max(in3, 1) clipped to [0, 2]
- ``f_fc_r`` = fc3 / max(in3, 1) clipped to [0, 1]

`in3` is computed here from op_in categories. This module does not import
cashflow.

Facility inventory is as-of ``period_end`` via ``created_at`` (no look-ahead
on which products existed). Amount/rate/next-pay fields on those rows are
still the extraction snapshot:

- ``f_n_facilities``, ``f_n_types`` — counts of products with
  created_at ≤ period_end and not ``created_after_snapshot``
- ``f_util_snapshot`` — sum|outstanding| / sum|granted| over facilities
  with |granted| > 1 (liability sign is typically negative; granted>0 is
  almost empty). Clipped to [0, 5]. **Not** historical utilisation.
- ``f_has_loc`` / ``f_has_factoring`` / ``f_has_confirming`` — 0/1
- ``f_w_rate`` — granted_balance-weighted mean of
  ``annual_interest_rate_or_spread`` (schedule; ~87 rows / 40 companies)
- ``f_months_to_next_pay`` — days from period_end to the earliest snapshot
  ``next_payment_date``, / 30.4375
- ``f_sched_vs_obs`` — monthly-equivalent scheduled installment
  (granted_balance / total_periods, scaled by amortising frequency and
  period length) / max(f_debt_service, 1), clipped to [0, 20]. Null
  unless a schedule exists.
- ``f_outstanding_gt_granted`` — clean flag, max over as-of facilities
- ``f_new_facility`` — count of products with created_at inside the period
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.features.common import AS_OF, CAT_MAP

SOURCE_TABLES = ["debt_products", "debt_schedule_config", "transactions"]
FAMILY = "f"

_FLOW_COLS = ("f_debt_service", "f_fin_cost", "f_ds_r", "f_fc_r")
_COUNT_COLS = (
    "f_n_facilities",
    "f_n_types",
    "f_has_loc",
    "f_has_factoring",
    "f_has_confirming",
    "f_new_facility",
)
FEATURE_COLS = list(_FLOW_COLS) + [
    "f_n_facilities",
    "f_n_types",
    "f_util_snapshot",
    "f_has_loc",
    "f_has_factoring",
    "f_has_confirming",
    "f_w_rate",
    "f_months_to_next_pay",
    "f_sched_vs_obs",
    "f_outstanding_gt_granted",
    "f_new_facility",
]

_MONTH_DAYS = 365.25 / 12.0


def _infer_freq(periods: pd.Series) -> str:
    p = pd.Series(sorted(pd.to_datetime(pd.unique(periods))))
    if len(p) >= 2:
        gap = p.diff().median()
        if pd.notna(gap) and gap <= pd.Timedelta(days=9):
            return "W"
        return "M"
    ts = p.iloc[0] if len(p) else pd.NaT
    if pd.isna(ts):
        return "M"
    return "M" if ts.day == 1 else "W"


def _period_end(period: pd.Series, freq: str) -> pd.Series:
    """Last calendar day of the period (inclusive). Compare with CAST(ts AS DATE)."""
    period = pd.to_datetime(period)
    if freq == "W":
        return (period + pd.Timedelta(days=6)).dt.normalize()
    return (period + pd.offsets.MonthEnd(0)).dt.normalize()


def _register(con, name: str, df: pd.DataFrame) -> None:
    try:
        con.unregister(name)
    except Exception:
        pass
    con.register(name, df)


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["company_id"] = out["company_id"].astype(str)
    out["period"] = pd.to_datetime(out["period"]).dt.normalize()
    return out


def _empty(grid: pd.DataFrame) -> pd.DataFrame:
    out = _keys(grid[["company_id", "period"]])
    for c in FEATURE_COLS:
        out[c] = np.nan
    return out


def _flows(con, grid: pd.DataFrame, freq: str, period_end: pd.Series) -> pd.DataFrame:
    cm = pd.DataFrame({"category": list(CAT_MAP), "grp": list(CAT_MAP.values())})
    _register(con, "_f_cat_map", cm)
    trunc = "week" if freq == "W" else "month"
    max_end = pd.to_datetime(period_end).max()
    df = con.execute(
        f"""
        SELECT t.company_id,
               CAST(date_trunc('{trunc}', t."date") AS DATE) AS period,
               SUM(CASE WHEN c.grp = 'op_in' THEN t.amount ELSE 0 END) AS op_in,
               -SUM(CASE WHEN c.grp = 'fin_cost' THEN t.amount ELSE 0 END) AS fin_cost,
               -SUM(CASE WHEN c.grp = 'debt_service' THEN t.amount ELSE 0 END) AS debt_service
        FROM transactions t
        LEFT JOIN _f_cat_map c ON t.category = c.category
        WHERE CAST(t."date" AS DATE) <= CAST(? AS DATE)
          AND NOT coalesce(t.is_extreme, false)
        GROUP BY 1, 2
        """,
        [max_end.to_pydatetime()],
    ).df()
    return _keys(df)


def _facility_asof(con, g: pd.DataFrame) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT g.company_id, CAST(g.period AS DATE) AS period,
               COUNT(d.product_id) AS f_n_facilities,
               COUNT(DISTINCT d.type) AS f_n_types,
               MAX(CASE WHEN d.type = 'lineofcredit' THEN 1 ELSE 0 END) AS f_has_loc,
               MAX(CASE WHEN d.type = 'factoring' THEN 1 ELSE 0 END) AS f_has_factoring,
               MAX(CASE WHEN d.type = 'confirming' THEN 1 ELSE 0 END) AS f_has_confirming,
               SUM(CASE WHEN CAST(d.created_at AS DATE) >= CAST(g.period AS DATE)
                          AND CAST(d.created_at AS DATE) <= CAST(g.period_end AS DATE)
                         THEN 1 ELSE 0 END) AS f_new_facility,
               SUM(CASE WHEN abs(d.granted) > 1 THEN abs(d.outstanding) ELSE 0 END)
                 / NULLIF(SUM(CASE WHEN abs(d.granted) > 1 THEN abs(d.granted) ELSE 0 END), 0)
                 AS f_util_snapshot,
               MAX(CASE WHEN d.outstanding_gt_granted THEN 1 ELSE 0 END)
                 AS f_outstanding_gt_granted
        FROM _f_grid g
        LEFT JOIN debt_products d
          ON d.company_id = g.company_id
         AND CAST(d.created_at AS DATE) <= CAST(g.period_end AS DATE)
         AND NOT coalesce(d.created_after_snapshot, false)
        GROUP BY 1, 2
        """
    ).df()
    return _keys(df)


def _schedule_asof(con) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT g.company_id, CAST(g.period AS DATE) AS period,
               SUM(CASE
                     WHEN s.total_periods > 0 AND s.granted_balance > 0
                     THEN (s.granted_balance / s.total_periods)
                          / CASE lower(coalesce(s.amortising_frequency, 'monthly'))
                              WHEN 'quarterly' THEN 3.0
                              WHEN 'semiannually' THEN 6.0
                              WHEN 'semi-annually' THEN 6.0
                              WHEN 'annually' THEN 12.0
                              WHEN 'yearly' THEN 12.0
                              ELSE 1.0
                            END
                     ELSE 0 END) AS sched_monthly,
               SUM(CASE WHEN s.annual_interest_rate_or_spread IS NOT NULL
                        THEN s.annual_interest_rate_or_spread
                             * GREATEST(s.granted_balance, 0)
                        ELSE 0 END) AS rate_num,
               SUM(CASE WHEN s.annual_interest_rate_or_spread IS NOT NULL
                        THEN GREATEST(s.granted_balance, 0)
                        ELSE 0 END) AS rate_den,
               AVG(s.annual_interest_rate_or_spread) AS rate_mean,
               MIN(s.next_payment_date) AS next_pay,
               COUNT(s.product_id) AS n_sched
        FROM _f_grid g
        JOIN debt_products d
          ON d.company_id = g.company_id
         AND CAST(d.created_at AS DATE) <= CAST(g.period_end AS DATE)
         AND NOT coalesce(d.created_after_snapshot, false)
        JOIN debt_schedule_config s
          ON s.product_id = d.product_id
        GROUP BY 1, 2
        """
    ).df()
    return _keys(df)


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    if grid is None or grid.empty:
        return _empty(pd.DataFrame({"company_id": [], "period": []}))

    base = _keys(grid[["company_id", "period"]])
    freq = _infer_freq(base["period"])
    roll_w = 13 if freq == "W" else 3
    period_scale = (7.0 / _MONTH_DAYS) if freq == "W" else 1.0
    ends = _period_end(base["period"], freq)

    g = pd.DataFrame(
        {
            "company_id": base["company_id"],
            "period": base["period"],
            "period_end": pd.to_datetime(ends),
        }
    )
    _register(con, "_f_grid", g)

    out = base.copy()
    flows = _flows(con, base, freq, ends)
    out = out.merge(flows, on=["company_id", "period"], how="left")
    for c in ("op_in", "fin_cost", "debt_service"):
        out[c] = out[c].fillna(0.0)
    out = out.sort_values(["company_id", "period"], kind="mergesort").reset_index(drop=True)

    grp = out.groupby("company_id", sort=False)
    in3 = grp["op_in"].transform(lambda s: s.rolling(roll_w, min_periods=roll_w).sum())
    ds3 = grp["debt_service"].transform(lambda s: s.rolling(roll_w, min_periods=roll_w).sum())
    fc3 = grp["fin_cost"].transform(lambda s: s.rolling(roll_w, min_periods=roll_w).sum())
    out["f_debt_service"] = out["debt_service"].astype(float)
    out["f_fin_cost"] = out["fin_cost"].astype(float)
    out["f_ds_r"] = (ds3 / np.maximum(in3, 1.0)).clip(0.0, 2.0)
    out["f_fc_r"] = (fc3 / np.maximum(in3, 1.0)).clip(0.0, 1.0)

    fac = _facility_asof(con, g)
    out = out.merge(fac, on=["company_id", "period"], how="left")
    for c in _COUNT_COLS:
        out[c] = out[c].fillna(0).astype(int)
    out["f_util_snapshot"] = pd.to_numeric(out["f_util_snapshot"], errors="coerce").clip(0.0, 5.0)
    out["f_outstanding_gt_granted"] = pd.to_numeric(
        out["f_outstanding_gt_granted"], errors="coerce"
    ).fillna(0.0)

    sched = _schedule_asof(con)
    out = out.merge(sched, on=["company_id", "period"], how="left")
    rate_den = pd.to_numeric(out.get("rate_den"), errors="coerce")
    rate_num = pd.to_numeric(out.get("rate_num"), errors="coerce")
    rate_mean = pd.to_numeric(out.get("rate_mean"), errors="coerce")
    out["f_w_rate"] = np.where(rate_den > 0, rate_num / rate_den, rate_mean)

    next_pay = pd.to_datetime(out.get("next_pay"), errors="coerce")
    pend = _period_end(out["period"], freq)
    out["f_months_to_next_pay"] = (next_pay - pend).dt.total_seconds() / (86400.0 * _MONTH_DAYS)

    sched_amt = pd.to_numeric(out.get("sched_monthly"), errors="coerce") * period_scale
    n_sched = pd.to_numeric(out.get("n_sched"), errors="coerce").fillna(0)
    obs = out["f_debt_service"].to_numpy(dtype=float)
    ratio = sched_amt.to_numpy(dtype=float) / np.maximum(obs, 1.0)
    out["f_sched_vs_obs"] = np.where(n_sched > 0, np.clip(ratio, 0.0, 20.0), np.nan)

    # Snapshot amounts are the 2026-09-01 extract. Do not emit utilisation
    # or the outstanding>granted flag on 2024/early-2025 rows as if the
    # book were known then. Inventory / type / new-facility stay as-of
    # created_at. Rate and next-pay stay (schedule is sparse and static).
    snap_ok = pend >= (AS_OF - pd.Timedelta(days=1))
    out.loc[~snap_ok, "f_util_snapshot"] = np.nan
    out.loc[~snap_ok, "f_outstanding_gt_granted"] = np.nan
    # Keep a 0 on last-period companies with no facilities; NA earlier.
    out.loc[snap_ok & out["f_n_facilities"].eq(0), "f_outstanding_gt_granted"] = 0
    out.loc[snap_ok & out["f_n_facilities"].eq(0), "f_util_snapshot"] = np.nan

    keep = ["company_id", "period"] + FEATURE_COLS
    out = out[keep].drop_duplicates(["company_id", "period"], keep="last")
    return out.reset_index(drop=True)
