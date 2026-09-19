"""Y5 payment behaviour — sustained AP/AR overdue>30d and AP delay shock.

Definitions (horizon = 3 future calendar months; no single-month crash):

- y5_ap_od30_ownp80: AP share of open amount that is >30 days past due
  exceeds that company's own-history p80 for 3 consecutive future months.
  p80 uses only that company's months <= t (never future, never pooled
  with other companies, so holdout never enters a train percentile).
- y5_ar_od30_sust: AR overdue>30d share is high for 3 consecutive future
  months. "High" = above that company's own-history p80 (same rule as AP).
  A pooled train-only expanding p80 was tried first; calendar-correct
  3-month persistence sat at ~2.8% (below the 5% floor), so own-history
  is the allowed fallback. Holdout never enters a pooled percentile.
- y5_ap_delay_up15: amount-weighted AP payment delay is >15 days above
  the trailing 6-month mean (baseline at t) in each of the next 3 months.

Invoice construction follows ``score_pipeline._invoice_features``: 3-month
issuance window for open stock, 3-month payment window for delay, delay
clipped to [-30, 120], first months masked for left truncation. Overdue
is tightened to Banque de France >30 days (due < period_end - 30d), not
any-overdue.

Literature: Hirshleifer, Li, Lourie, Ruchti 2019 (NBER w25553) PastDue%;
Banque de France Bulletin 227/8 (late customer payments >30 days).

Forbidden X family: e (receivables/payables features).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.evaluate.protocol import auroc, load_holdout
from analysis.features.common import CAT_MAP, MONTHS

SOURCE_TABLES = ["invoices"]
HORIZON = 3
MIN_OWN_HIST = 6
TRAIL_DELAY = 6
DELAY_DELTA = 15.0
# score_pipeline: open features from month index 2; delay NaN for first 6 months.
OD_SHARE_START = MONTHS[2]
DELAY_START = MONTHS[6]

META = {
    "name": "y5_payment",
    "horizon": HORIZON,
    "source_tables": SOURCE_TABLES,
    "forbidden_x_families": ["e"],
    "literature": "Hirshleifer et al. 2019 PastDue%; Banque de France Bulletin 227/8 >30 days late",
    "columns": ["y5_ap_od30_ownp80", "y5_ar_od30_sust", "y5_ap_delay_up15"],
}

Y_COLS = list(META["columns"])


def _month_frame(periods: pd.DatetimeIndex) -> pd.DataFrame:
    g = pd.DataFrame({"period": pd.to_datetime(pd.Index(periods)).sort_values().unique()})
    g["e"] = g["period"] + pd.offsets.MonthEnd(0)
    g["w0"] = g["period"] - pd.DateOffset(months=2)
    g["e30"] = g["e"] - pd.Timedelta(days=30)
    return g


def _invoice_month_panel(con, periods: pd.DatetimeIndex) -> pd.DataFrame:
    """Company-month AP/AR od30 share and amount-weighted AP delay."""
    month_grid = _month_frame(periods)
    con.register("y5_month_grid", month_grid)
    od = con.execute(
        """
        WITH inv AS (
            SELECT company_id,
                   CAST(issuance_date AS DATE) AS iss,
                   GREATEST(CAST(due_date AS DATE), CAST(issuance_date AS DATE)) AS due,
                   CASE WHEN status = 'paid' THEN CAST(payment_date AS DATE) END AS paid_dt,
                   ABS(amount) AS abs_amt,
                   CASE WHEN amount > 0 THEN 'AR' ELSE 'AP' END AS side,
                   status
            FROM invoices
            WHERE document_type = 'invoice'
              AND status <> 'cancel'
              AND amount <> 0
              AND issuance_date IS NOT NULL
              AND due_date IS NOT NULL
              AND NOT is_extreme
        )
        SELECT g.period, i.company_id, i.side,
               SUM(i.abs_amt) AS open_amt,
               SUM(CASE WHEN i.due < g.e30 THEN i.abs_amt ELSE 0 END) AS od30_amt
        FROM y5_month_grid g
        JOIN inv i
          ON i.iss >= g.w0 AND i.iss <= g.e
         AND NOT (i.status = 'paid' AND i.paid_dt IS NULL)
         AND (i.paid_dt IS NULL OR i.paid_dt > g.e)
        GROUP BY 1, 2, 3
        """
    ).df()
    delay = con.execute(
        """
        WITH inv AS (
            SELECT company_id,
                   GREATEST(CAST(due_date AS DATE), CAST(issuance_date AS DATE)) AS due,
                   CAST(payment_date AS DATE) AS paid_dt,
                   ABS(amount) AS abs_amt,
                   CASE WHEN amount > 0 THEN 'AR' ELSE 'AP' END AS side
            FROM invoices
            WHERE document_type = 'invoice'
              AND status = 'paid'
              AND amount <> 0
              AND issuance_date IS NOT NULL
              AND due_date IS NOT NULL
              AND payment_date IS NOT NULL
              AND NOT is_extreme
        )
        SELECT g.period, i.company_id,
               SUM(LEAST(120, GREATEST(-30, date_diff('day', i.due, i.paid_dt))) * i.abs_amt)
                 / NULLIF(SUM(i.abs_amt), 0) AS ap_wdelay
        FROM y5_month_grid g
        JOIN inv i
          ON i.side = 'AP'
         AND i.paid_dt >= g.w0 AND i.paid_dt <= g.e
        GROUP BY 1, 2
        """
    ).df()
    try:
        con.unregister("y5_month_grid")
    except Exception:
        pass

    od["period"] = pd.to_datetime(od["period"])
    delay["period"] = pd.to_datetime(delay["period"])
    od["od30_share"] = np.where(od["open_amt"] > 0, od["od30_amt"] / od["open_amt"], np.nan)
    ap = od.loc[od["side"] == "AP", ["company_id", "period", "od30_share"]].rename(
        columns={"od30_share": "ap_od30_share"}
    )
    ar = od.loc[od["side"] == "AR", ["company_id", "period", "od30_share"]].rename(
        columns={"od30_share": "ar_od30_share"}
    )
    panel = ap.merge(ar, on=["company_id", "period"], how="outer").merge(
        delay, on=["company_id", "period"], how="outer"
    )
    panel.loc[panel["period"] < OD_SHARE_START, ["ap_od30_share", "ar_od30_share"]] = np.nan
    panel.loc[panel["period"] < DELAY_START, "ap_wdelay"] = np.nan
    return panel


def _expanding_quantile_skipna(s: pd.Series, q: float, min_periods: int) -> pd.Series:
    """Expanding percentile that counts only finite observations."""
    out = np.full(len(s), np.nan, dtype=float)
    vals: list[float] = []
    for i, v in enumerate(s.to_numpy(dtype=float)):
        if np.isfinite(v):
            vals.append(float(v))
        if len(vals) >= min_periods:
            out[i] = float(np.quantile(vals, q))
    return pd.Series(out, index=s.index)


def _calendar_shift(df: pd.DataFrame, col: str, months: int, name: str) -> pd.DataFrame:
    """Attach ``col`` observed at t+months onto the row dated t."""
    src = df[["company_id", "period", col]].dropna(subset=[col]).copy()
    src["period"] = src["period"] - pd.DateOffset(months=months)
    return src.rename(columns={col: name})


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    need = {"company_id", "period"}
    if not need <= set(grid.columns):
        raise ValueError("grid must have company_id, period")
    g = grid[["company_id", "period"]].copy()
    g["company_id"] = g["company_id"].astype(str)
    g["period"] = pd.to_datetime(g["period"])
    periods = pd.DatetimeIndex(g["period"].unique()).sort_values()

    metrics = _invoice_month_panel(con, periods)
    metrics["company_id"] = metrics["company_id"].astype(str)
    P = g.merge(metrics, on=["company_id", "period"], how="left")
    P = P.sort_values(["company_id", "period"]).reset_index(drop=True)

    # own-history p80: that company's months <= t only (holdout firms use their own past)
    P["ap_own_p80"] = P.groupby("company_id", sort=False)["ap_od30_share"].transform(
        lambda s: _expanding_quantile_skipna(s, 0.80, MIN_OWN_HIST)
    )
    P["ar_own_p80"] = P.groupby("company_id", sort=False)["ar_od30_share"].transform(
        lambda s: _expanding_quantile_skipna(s, 0.80, MIN_OWN_HIST)
    )
    P["ap_delay_base"] = (
        P.groupby("company_id", sort=False)["ap_wdelay"]
        .transform(lambda s: s.rolling(TRAIL_DELAY, min_periods=TRAIL_DELAY).mean())
    )

    parts = [("ap_od30_share", "ap_f"), ("ar_od30_share", "ar_f"), ("ap_wdelay", "dly_f")]
    for col, prefix in parts:
        for h in range(1, HORIZON + 1):
            fut = _calendar_shift(P, col, h, f"{prefix}{h}")
            P = P.merge(fut, on=["company_id", "period"], how="left")

    def _all_future(cols: list[str]) -> pd.Series:
        return P[cols].notna().all(axis=1)

    ap_f = [f"ap_f{h}" for h in range(1, HORIZON + 1)]
    ar_f = [f"ar_f{h}" for h in range(1, HORIZON + 1)]
    dly_f = [f"dly_f{h}" for h in range(1, HORIZON + 1)]

    ap_hi = pd.concat([P[c] > P["ap_own_p80"] for c in ap_f], axis=1).all(axis=1)
    P["y5_ap_od30_ownp80"] = np.where(
        _all_future(ap_f) & P["ap_own_p80"].notna(), ap_hi.astype(float), np.nan
    )

    ar_hi = pd.concat([P[c] > P["ar_own_p80"] for c in ar_f], axis=1).all(axis=1)
    P["y5_ar_od30_sust"] = np.where(
        _all_future(ar_f) & P["ar_own_p80"].notna(), ar_hi.astype(float), np.nan
    )

    dly_hi = pd.concat([P[c] > (P["ap_delay_base"] + DELAY_DELTA) for c in dly_f], axis=1).all(axis=1)
    P["y5_ap_delay_up15"] = np.where(
        _all_future(dly_f) & P["ap_delay_base"].notna(), dly_hi.astype(float), np.nan
    )

    return P[["company_id", "period", *Y_COLS]].reset_index(drop=True)


def _monthly_op_in(con) -> pd.DataFrame:
    cats = [c for c, g in CAT_MAP.items() if g == "op_in"]
    con.register("y5_op_in_cats", pd.DataFrame({"category": cats}))
    df = con.execute(
        """
        SELECT t.company_id,
               CAST(date_trunc('month', t."date") AS DATE) AS period,
               SUM(t.amount) AS op_in
        FROM transactions t
        JOIN y5_op_in_cats c ON t.category = c.category
        WHERE t."date" < TIMESTAMP '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    try:
        con.unregister("y5_op_in_cats")
    except Exception:
        pass
    df["period"] = pd.to_datetime(df["period"])
    df["company_id"] = df["company_id"].astype(str)
    return df


def acceptance_stats(con, grid: pd.DataFrame, y_panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Train-only base rates and size-proxy AUROC. Does not fit a model."""
    if y_panel is None:
        y_panel = build(con, grid)
    hold = load_holdout()
    y = y_panel.copy()
    y["company_id"] = y["company_id"].astype(str)
    y["period"] = pd.to_datetime(y["period"])
    y["is_train"] = ~y["company_id"].isin(hold)
    size = _monthly_op_in(con)
    y = y.merge(size, on=["company_id", "period"], how="left")
    y["op_in"] = y["op_in"].fillna(0.0)
    y["log_size"] = np.log1p(y["op_in"].abs())
    n_train = int(y["is_train"].sum())
    rows = []
    for col in Y_COLS:
        t = y.loc[y["is_train"]]
        labeled = t[col].notna()
        n_lab = int(labeled.sum())
        rate = float(t.loc[labeled, col].mean()) if n_lab else float("nan")
        auc = auroc(t.loc[labeled, col], t.loc[labeled, "log_size"])
        auc_abs = float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan")
        in_rate = bool(np.isfinite(rate) and 0.05 <= rate <= 0.30)
        size_ok = bool(np.isfinite(auc_abs) and auc_abs < 0.60)
        rows.append(
            {
                "y": col,
                "train_company_months": n_train,
                "train_labeled": n_lab,
                "coverage": n_lab / n_train if n_train else float("nan"),
                "base_rate": rate,
                "size_auroc": auc,
                "size_auroc_two_sided": auc_abs,
                "n_pos": int(t.loc[labeled, col].sum()) if n_lab else 0,
                "n_companies_labeled": int(t.loc[labeled, "company_id"].nunique()),
                "accepted": in_rate and size_ok,
                "verdict": "ACCEPTED" if (in_rate and size_ok) else "REJECTED",
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from analysis.features.common import connect
    from analysis.features.grid import monthly_grid

    con = connect()
    grid = monthly_grid(con)[["company_id", "period"]]
    y = build(con, grid)
    print(y.head(3).to_string(index=False))
    print("rows", len(y), "dups", int(y.duplicated(["company_id", "period"]).sum()))
    stats = acceptance_stats(con, grid, y)
    print(stats.to_string(index=False))
    con.close()
