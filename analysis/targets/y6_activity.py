"""Y6 — activity / going-concern targets (binary).

Built from transactions only (salary presence recomputed here; do not
import family C). Labels use the next 3 months unless noted.

Forbidden X for models that predict these labels: families A and C for
the same window. Thresholds are catalogue-fixed; they are not fit on
train or holdout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.features.common import LAST_M
from analysis.targets.y1_forecast import _month_key, cash_month_panel

Y6_COLS = ["y6_zero_in_3", "y6_missed_payroll", "y6_silent_60"]

HORIZON = 3
USUAL_SALARY_WINDOW = 6
USUAL_SALARY_MIN = 3
SILENT_DAYS = 60

META = {
    "name": "y6_activity",
    "horizon": HORIZON,
    "source_tables": ["transactions"],
    "forbidden_x_families": ["a", "c"],
    "literature": (
        "Yao, Levy-Chapira, Margaryan 2017 (arXiv 1707.00757): checking-account "
        "activity beats ratios for corporate default. FinRegLab 2025 bank-statement "
        "activity. Pérez-Salazar, Márquez & Vidal-Silva 2026: operational regularity "
        "(missed payroll / recency) as a solvency signal."
    ),
    "kind": "binary",
    "columns": list(Y6_COLS),
    "definitions": {
        "y6_zero_in_3": (
            "1 if t has activity (in3 > 0 or n_tx > 0) and op_in == 0 in "
            "each of t+1, t+2, t+3"
        ),
        "y6_missed_payroll": (
            "1 if the company usually pays salary (>= "
            f"{USUAL_SALARY_MIN} of the last {USUAL_SALARY_WINDOW} months "
            "ending at t have category=salary) and t+1..t+3 all have no salary"
        ),
        "y6_silent_60": (
            f"1 if last tx as of t+{HORIZON} month-end is more than "
            f"{SILENT_DAYS} days before that end, after prior activity at t "
            "(last tx on or before t is known)"
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


def month_activity(con) -> pd.DataFrame:
    """Monthly tx count, last booking date, and salary presence.

    Salary is raw category = 'salary' (same token family C uses). Not imported
    from ops.py.
    """
    df = con.execute(
        """
        SELECT
          company_id,
          CAST(date_trunc('month', "date") AS DATE) AS month,
          COUNT(*) AS n_tx,
          MAX("date") AS last_tx,
          MAX(CASE WHEN category = 'salary' THEN 1 ELSE 0 END) AS salary_month
        FROM transactions
        WHERE "date" IS NOT NULL
        GROUP BY 1, 2
        """
    ).df()
    df["month"] = pd.to_datetime(df["month"])
    df["last_tx"] = pd.to_datetime(df["last_tx"])
    return df


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    """Return company_id, period and the three Y6 binaries (0/1/NaN)."""
    panel = cash_month_panel(con).sort_values(["company_id", "month"]).reset_index(drop=True)
    act = month_activity(con)
    panel = panel.merge(act, on=["company_id", "month"], how="left")
    panel["n_tx"] = panel["n_tx"].fillna(0.0)
    panel["salary_month"] = panel["salary_month"].fillna(0.0)

    cid = panel["company_id"]
    g = panel.groupby(cid, sort=False)
    panel["in3"] = g["op_in"].transform(lambda s: s.rolling(3).sum())
    panel["last_tx"] = g["last_tx"].ffill()

    has_h = _has_horizon(panel["month"], HORIZON)
    active = (panel["in3"] > 0) | (panel["n_tx"] > 0)
    z1 = g["op_in"].shift(-1)
    z2 = g["op_in"].shift(-2)
    z3 = g["op_in"].shift(-3)
    zero3 = z1.eq(0.0) & z2.eq(0.0) & z3.eq(0.0)
    panel["y6_zero_in_3"] = np.where(has_h & active, zero3.astype(float), np.nan)

    sal6 = g["salary_month"].transform(
        lambda s: s.rolling(USUAL_SALARY_WINDOW, min_periods=1).sum()
    )
    usual = sal6 >= USUAL_SALARY_MIN
    s1 = g["salary_month"].shift(-1)
    s2 = g["salary_month"].shift(-2)
    s3 = g["salary_month"].shift(-3)
    miss3 = s1.eq(0.0) & s2.eq(0.0) & s3.eq(0.0)
    panel["y6_missed_payroll"] = np.where(has_h & usual, miss3.astype(float), np.nan)

    month_end = (panel["month"] + pd.offsets.MonthEnd(0)).dt.normalize()
    last_day = pd.to_datetime(panel["last_tx"]).dt.normalize()
    recency = (month_end - last_day).dt.days
    r3 = recency.groupby(cid).shift(-HORIZON)
    prior = last_day.notna()
    panel["y6_silent_60"] = np.where(has_h & prior, (r3 > SILENT_DAYS).astype(float), np.nan)

    out = grid[["company_id", "period"]].copy()
    out["period"] = pd.to_datetime(out["period"])
    out["_m"] = _month_key(out["period"])
    lab = panel[["company_id", "month", *Y6_COLS]].rename(columns={"month": "_m"})
    out = out.merge(lab, on=["company_id", "_m"], how="left").drop(columns=["_m"])
    return out.reset_index(drop=True)


def size_auroc(y: pd.Series, op_in: pd.Series) -> float:
    """AUROC of log1p(|monthly op_in|) vs a binary label."""
    return _auroc(np.log1p(pd.to_numeric(op_in, errors="coerce").abs()), y)


def train_acceptance(y6: pd.DataFrame, op_in: pd.Series, is_train: pd.Series) -> pd.DataFrame:
    """Base rates and size AUROC on train company-months only (no threshold search)."""
    rows = []
    for col in Y6_COLS:
        y = y6.loc[is_train, col]
        ok = y.notna()
        n = int(ok.sum())
        rate = float(y[ok].mean()) if n else float("nan")
        auc = size_auroc(y, op_in.loc[is_train])
        two = float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan")
        accepted = bool(n and 0.05 <= rate <= 0.30 and np.isfinite(auc) and auc < 0.60)
        rows.append(
            {
                "column": col,
                "n_train": n,
                "n_pos": int((y == 1).sum()),
                "base_rate": rate,
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
    y6 = build(con, grid)
    panel = _panel(con).rename(columns={"month": "period"})
    keys = grid[["company_id", "period"]].copy()
    keys["period"] = pd.to_datetime(keys["period"])
    tr = train_mask(keys["company_id"])
    op = keys.merge(panel[["company_id", "period", "op_in"]], on=["company_id", "period"], how="left")["op_in"]
    print("train company-months", int(tr.sum()), "companies", keys.loc[tr, "company_id"].nunique())
    acc = train_acceptance(y6, op, tr)
    print("Y6 train acceptance")
    print(acc.to_string(index=False))
    con.close()
