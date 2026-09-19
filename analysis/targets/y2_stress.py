"""Y2 — liquidity-stress targets (binary, sustained).

Forbidden X for models that predict these labels: family B (same cash-path
columns). Thresholds are fixed from the catalogue; they are not fit on
train or holdout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.features.common import LAST_M
from analysis.targets.y1_forecast import _month_key, cash_month_panel

Y2_COLS = ["y2_neg_2of3", "y2_runway_lt1_sust", "y2_onset_neg"]

META = {
    "name": "y2_liquidity_stress",
    "horizon": 3,
    "source_tables": ["transactions", "balances", "banking_products"],
    "forbidden_x_families": ["b"],
    "literature": (
        "FinRegLab 2025, Sharpening the Focus: low/negative ending balances "
        "and NSF-type counts are the strongest bank-statement distress "
        "indicators for SMB default."
    ),
    "kind": "binary",
    "columns": list(Y2_COLS),
    "definitions": {
        "y2_neg_2of3": "1 if reconstructed liq < 0 in at least 2 of the next 3 months",
        "y2_runway_lt1_sust": "1 if runway < 1 for the next 3 consecutive months",
        "y2_onset_neg": (
            "1 if a first negative month after >= 6 consecutive clean "
            "(liq >= 0) months falls in t+1..t+3"
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
    """t+h is inside the sample when month <= LAST_M - h months."""
    cutoff = LAST_M - pd.DateOffset(months=h)
    return month <= cutoff


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    """Return company_id, period and the three Y2 binaries (0/1/NaN)."""
    panel = cash_month_panel(con).sort_values(["company_id", "month"]).reset_index(drop=True)
    cid = panel["company_id"]
    g = panel.groupby(cid, sort=False)

    neg = pd.Series(
        np.where(panel["liq"].isna(), np.nan, (panel["liq"] < 0).astype(float)),
        index=panel.index,
    )
    clean = pd.Series(
        np.where(panel["liq"].isna(), np.nan, (panel["liq"] >= 0).astype(float)),
        index=panel.index,
    )
    six_clean = pd.Series(True, index=panel.index)
    for k in range(1, 7):
        six_clean &= clean.groupby(cid).shift(k).eq(1.0)
    onset = (neg == 1.0) & six_clean

    n1 = neg.groupby(cid).shift(-1)
    n2 = neg.groupby(cid).shift(-2)
    n3 = neg.groupby(cid).shift(-3)
    neg_count = n1 + n2 + n3
    has_h3 = _has_horizon(panel["month"], 3)
    y_neg = np.where(has_h3 & neg_count.notna(), (neg_count >= 2).astype(float), np.nan)

    r1 = g["runway"].shift(-1)
    r2 = g["runway"].shift(-2)
    r3 = g["runway"].shift(-3)
    run_ok = has_h3 & r1.notna() & r2.notna() & r3.notna()
    y_run = np.where(run_ok, ((r1 < 1.0) & (r2 < 1.0) & (r3 < 1.0)).astype(float), np.nan)

    onset_f = onset.astype(float)
    o1 = onset_f.groupby(cid).shift(-1)
    o2 = onset_f.groupby(cid).shift(-2)
    o3 = onset_f.groupby(cid).shift(-3)
    y_on = np.where(
        has_h3,
        ((o1.fillna(0.0) + o2.fillna(0.0) + o3.fillna(0.0)) > 0).astype(float),
        np.nan,
    )

    panel["y2_neg_2of3"] = y_neg
    panel["y2_runway_lt1_sust"] = y_run
    panel["y2_onset_neg"] = y_on

    out = grid[["company_id", "period"]].copy()
    out["period"] = pd.to_datetime(out["period"])
    out["_m"] = _month_key(out["period"])
    lab = panel[["company_id", "month", *Y2_COLS]].rename(columns={"month": "_m"})
    out = out.merge(lab, on=["company_id", "_m"], how="left").drop(columns=["_m"])
    return out.reset_index(drop=True)


def size_auroc(y: pd.Series, op_in: pd.Series) -> float:
    """AUROC of log1p(|monthly op_in|) vs a binary label."""
    return _auroc(np.log1p(pd.to_numeric(op_in, errors="coerce").abs()), y)


def train_acceptance(y2: pd.DataFrame, op_in: pd.Series, is_train: pd.Series) -> pd.DataFrame:
    """Base rates and size AUROC on train company-months only (no threshold search)."""
    rows = []
    for col in Y2_COLS:
        y = y2.loc[is_train, col]
        ok = y.notna()
        n = int(ok.sum())
        rate = float(y[ok].mean()) if n else float("nan")
        auc = size_auroc(y, op_in.loc[is_train])
        accepted = bool(n and 0.05 <= rate <= 0.30)
        rows.append(
            {
                "column": col,
                "n_train": n,
                "n_pos": int((y == 1).sum()),
                "base_rate": rate,
                "accepted": accepted,
                "size_auroc": auc,
                "size_proxy": bool(auc >= 0.60) if np.isfinite(auc) else False,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from analysis.features.common import connect, train_mask
    from analysis.features.grid import monthly_grid
    from analysis.targets.y1_forecast import Y1_COLS, build as build_y1, cash_month_panel

    con = connect()
    grid = monthly_grid(con)
    y1 = build_y1(con, grid)
    y2 = build(con, grid)
    panel = cash_month_panel(con).rename(columns={"month": "period"})
    keys = grid[["company_id", "period"]].copy()
    keys["period"] = pd.to_datetime(keys["period"])
    tr = train_mask(keys["company_id"])
    op = keys.merge(panel[["company_id", "period", "op_in"]], on=["company_id", "period"], how="left")["op_in"]
    print("train company-months", int(tr.sum()), "companies", keys.loc[tr, "company_id"].nunique())
    print("Y1 train coverage")
    for c in Y1_COLS:
        s = y1.loc[tr, c]
        print(f"  {c}: n={int(s.notna().sum())} cov={s.notna().mean():.4f}")
    acc = train_acceptance(y2, op, tr)
    print("Y2 train acceptance")
    print(acc.to_string(index=False))
    con.close()
