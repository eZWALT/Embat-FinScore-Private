"""Y3 — recovery from a stressed cash state (binary).

Stressed at t: reconstructed liq < 0 OR runway < 1 (same path as Y1/Y2).
Recover within h = 6: at least one stretch of 3 consecutive months in
t+1..t+6 with runway >= 3. The AP overlay also requires the open-AP
amount share that is >30 days late to be not high (catalogue cut 0.50;
null open-AP treated as not high).

`y3_recover_cash_6m` is cash-only so models may use family E.
`y3_recover_6m` uses invoices in the label, so that variant also forbids E.

Forbidden X for the cash path: family B. Thresholds are catalogue-fixed;
they are not fit on train or holdout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.features.common import LAST_M, MONTHS
from analysis.targets.y1_forecast import _month_key, cash_month_panel

Y3_COLS = ["y3_recover_6m", "y3_recover_cash_6m"]

# Catalogue: "high" = majority of open AP is more than 30 days late.
# Not a train/holdout percentile.
AP_OVERDUE30_HIGH = 0.50
OVERDUE_30_DAYS = 30
HORIZON = 6
RECOVER_RUNWAY = 3.0
RECOVER_LEN = 3

META = {
    "name": "y3_recovery",
    "horizon": HORIZON,
    "source_tables": ["transactions", "balances", "banking_products", "invoices"],
    "forbidden_x_families": ["b"],
    "forbidden_x_families_by_column": {
        "y3_recover_6m": ["b", "e"],
        "y3_recover_cash_6m": ["b"],
    },
    "literature": (
        "Challenge brief 45->65 recovery example; FinRegLab 2025 liquidity "
        "path; Banque de France Bulletin 227/8 (only AP late >30 days moves PD). "
        "Models for y3_recover_6m must not use family E (overdue is in the label)."
    ),
    "kind": "binary",
    "columns": list(Y3_COLS),
    "ap_overdue30_high": AP_OVERDUE30_HIGH,
    "definitions": {
        "stressed": "liq < 0 OR runway < 1 at t; label is NaN if not stressed or no h=6",
        "y3_recover_cash_6m": (
            "1 if stressed at t and some 3 consecutive months in t+1..t+6 "
            f"have runway >= {RECOVER_RUNWAY:g}"
        ),
        "y3_recover_6m": (
            "cash recover AND each of those 3 months has open-AP overdue>30d "
            f"share <= {AP_OVERDUE30_HIGH:g} (no open AP counts as not high)"
        ),
    },
}

SOURCE_TABLES = META["source_tables"]


def _auroc(score, y) -> float:
    d = pd.DataFrame({"s": score, "y": y}).dropna()
    n1 = int((d["y"] == 1).sum())
    n0 = int((d["y"] == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = d["s"].rank()
    return float((r[d["y"] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _has_horizon(month: pd.Series, h: int) -> pd.Series:
    cutoff = LAST_M - pd.DateOffset(months=h)
    return month <= cutoff


def monthly_ap_overdue_30(con) -> pd.DataFrame:
    """Open-AP amount share that is more than 30 days late at each month-end.

    Same open-book rules as family E (`invoices.build` e_ap_overdue_30):
    AP is amount < 0; issued by month-end; unpaid by month-end;
    invalid payment dates dropped.
    """
    months = pd.DataFrame(
        {
            "month": MONTHS,
            "month_end": MONTHS + pd.offsets.MonthEnd(0),
        }
    )
    con.register("_y3_months", months)
    try:
        df = con.execute(
            f"""
            SELECT i.company_id,
                   p.month,
                   SUM(CASE WHEN i.due_date IS NOT NULL
                             AND date_diff('day', CAST(i.due_date AS DATE),
                                                CAST(p.month_end AS DATE)) > {OVERDUE_30_DAYS}
                            THEN abs(i.amount) ELSE 0 END)
                     / NULLIF(SUM(abs(i.amount)), 0) AS ap_overdue_30
            FROM invoices i
            JOIN _y3_months p
              ON CAST(i.issuance_date AS DATE) <= CAST(p.month_end AS DATE)
             AND (i.payment_date IS NULL
                  OR CAST(i.payment_date AS DATE) > CAST(p.month_end AS DATE))
             AND NOT coalesce(i.payment_date_invalid, FALSE)
            WHERE i.document_type = 'invoice'
              AND i.status <> 'cancel'
              AND i.amount < 0
              AND i.issuance_date IS NOT NULL
            GROUP BY 1, 2
            """
        ).df()
    finally:
        con.unregister("_y3_months")
    df["month"] = pd.to_datetime(df["month"])
    return df


def _any_recover_window(g: pd.core.groupby.generic.DataFrameGroupBy, col: str) -> pd.Series:
    """True if any 3-month window starting at t+1..t+4 is all 1.0 on `col`."""
    rec = pd.Series(False, index=g.obj.index)
    for start in range(1, HORIZON - RECOVER_LEN + 2):
        a = g[col].shift(-start)
        b = g[col].shift(-(start + 1))
        c = g[col].shift(-(start + 2))
        rec = rec | ((a == 1.0) & (b == 1.0) & (c == 1.0))
    return rec


def _future_runway_known(g: pd.core.groupby.generic.DataFrameGroupBy) -> pd.Series:
    known = pd.Series(True, index=g.obj.index)
    for k in range(1, HORIZON + 1):
        known = known & g["runway"].shift(-k).notna()
    return known


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    """Return company_id, period and the two Y3 binaries (0/1/NaN)."""
    panel = cash_month_panel(con).sort_values(["company_id", "month"]).reset_index(drop=True)
    ap = monthly_ap_overdue_30(con)
    panel = panel.merge(ap, on=["company_id", "month"], how="left")
    # No open AP => overdue share is not high.
    panel["ap_not_high"] = panel["ap_overdue_30"].fillna(0.0) <= AP_OVERDUE30_HIGH

    stressed = (panel["liq"] < 0) | (panel["runway"] < 1)
    panel["ok_cash"] = np.where(panel["runway"].isna(), np.nan, (panel["runway"] >= RECOVER_RUNWAY).astype(float))
    panel["ok_full"] = np.where(
        panel["runway"].isna(),
        np.nan,
        ((panel["runway"] >= RECOVER_RUNWAY) & panel["ap_not_high"]).astype(float),
    )

    g = panel.groupby("company_id", sort=False)
    has_h = _has_horizon(panel["month"], HORIZON)
    known = _future_runway_known(g)
    labeled = has_h & stressed & known

    cash_rec = _any_recover_window(g, "ok_cash")
    full_rec = _any_recover_window(g, "ok_full")
    panel["y3_recover_cash_6m"] = np.where(labeled, cash_rec.astype(float), np.nan)
    panel["y3_recover_6m"] = np.where(labeled, full_rec.astype(float), np.nan)

    out = grid[["company_id", "period"]].copy()
    out["period"] = pd.to_datetime(out["period"])
    out["_m"] = _month_key(out["period"])
    lab = panel[["company_id", "month", *Y3_COLS]].rename(columns={"month": "_m"})
    out = out.merge(lab, on=["company_id", "_m"], how="left").drop(columns=["_m"])
    return out.reset_index(drop=True)


def size_auroc(y: pd.Series, op_in: pd.Series) -> float:
    """AUROC of log1p(|monthly op_in|) vs a binary label."""
    return _auroc(np.log1p(pd.to_numeric(op_in, errors="coerce").abs()), y)


def train_acceptance(y3: pd.DataFrame, op_in: pd.Series, is_train: pd.Series) -> pd.DataFrame:
    """Stressed-conditional rates and size AUROC on train only (no cut search)."""
    n_all = int(is_train.sum())
    rows = []
    for col in Y3_COLS:
        y = y3.loc[is_train, col]
        ok = y.notna()
        n = int(ok.sum())
        n_pos = int((y == 1).sum())
        rate_stressed = float(y[ok].mean()) if n else float("nan")
        rate_all = float(n_pos / n_all) if n_all else float("nan")
        auc = size_auroc(y, op_in.loc[is_train])
        two = float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan")
        accepted = bool(
            n
            and 0.05 <= rate_stressed <= 0.50
            and np.isfinite(auc)
            and auc < 0.60
        )
        rows.append(
            {
                "column": col,
                "n_train_stressed": n,
                "n_pos": n_pos,
                "rate_stressed": rate_stressed,
                "rate_all": rate_all,
                "accepted": accepted,
                "size_auroc": auc,
                "size_auroc_two_sided": two,
                "size_proxy": bool(auc >= 0.60) if np.isfinite(auc) else False,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from analysis.features.common import connect, train_mask
    from analysis.features.grid import monthly_grid
    from analysis.targets.y1_forecast import cash_month_panel as _panel

    con = connect()
    grid = monthly_grid(con)
    y3 = build(con, grid)
    panel = _panel(con).rename(columns={"month": "period"})
    keys = grid[["company_id", "period"]].copy()
    keys["period"] = pd.to_datetime(keys["period"])
    tr = train_mask(keys["company_id"])
    op = keys.merge(panel[["company_id", "period", "op_in"]], on=["company_id", "period"], how="left")["op_in"]
    print("train company-months", int(tr.sum()), "companies", keys.loc[tr, "company_id"].nunique())
    acc = train_acceptance(y3, op, tr)
    print("Y3 train acceptance (rate_stressed is the relevant rate)")
    print(acc.to_string(index=False))
    con.close()
