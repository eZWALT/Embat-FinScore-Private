"""Y8 — cross-source pillar deterioration (binary, horizon 6).

Cash-side (liquidez + caja: runway, net_margin, neg_liq) versus invoice-side
(cobros: e_ar_overdue_30, e_delay_coll) at t+6, and the reverse. Composites
are unit-free worse-is-higher ranks, not a 0–100 product score.

- cash_side = mean of available train-only within-month percentile ranks of
  (−runway, −net_margin, +neg_liq_3). Cash path is computed here from the
  same monthly unwind as Y1/Y2 (`cash_month_panel`).
- invoice_side = mean of available train-only ranks of e_ar_overdue_30 and
  e_delay_coll from family E (`analysis.features.invoices.build`).

A single month above own p80 is too common (~31–38%) and fails the 5–30%
band. Labels therefore require the last 3 months of the 6-month horizon
(t+4, t+5, t+6) all to exceed the company's own expanding p80 at t
(catalogue “worse than own p80” + contract sustained window).

Forbidden X (cross-source; Y source ≠ allowed X):

- y8_inv_worse_6 is built from invoice-side / cobros → forbid family E.
  Cash-side (A, B) is the intended X. An e_ column at t that already
  explains this Y (AUROC ≥ 0.70) is a persistence leak: reject the Y.
- y8_cash_worse_6 is built from cash-side / liquidez+caja → forbid A, B.
  Invoice-side (E) is the intended X.

Holdout companies never enter the rank reference. Own-history p80 is
per-company (holdout firms use only their own past).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.evaluate.protocol import assert_no_holdout, auroc
from analysis.features.common import LAST_M, train_mask
from analysis.features.invoices import build as build_e
from analysis.targets.y1_forecast import _month_key, cash_month_panel

Y8_COLS = ["y8_inv_worse_6", "y8_cash_worse_6"]

HORIZON = 6
SUSTAIN = 3  # last 3 months of the horizon must all exceed own p80
MIN_OWN_HIST = 6
OWN_P = 0.80

META = {
    "name": "y8_cross_source",
    "horizon": HORIZON,
    "source_tables": ["transactions", "balances", "banking_products", "invoices"],
    "forbidden_x_families": ["a", "b", "e"],
    "forbidden_x_by_column": {
        "y8_inv_worse_6": ["e"],
        "y8_cash_worse_6": ["a", "b"],
    },
    "literature": (
        "Javier validation 2026-09-18: liquidez/caja vs cobros pillars are "
        "nearly uncorrelated, so a 6-month move on one side is not the other "
        "side's persistence. Hirshleifer, Li, Lourie, Ruchti 2019 (NBER "
        "w25553): past-due trade credit predicts default more for low-liquidity "
        "firms — the cross-source interaction this Y isolates."
    ),
    "kind": "binary",
    "columns": list(Y8_COLS),
    "definitions": {
        "y8_inv_worse_6": (
            "1 if invoice_side at t+4, t+5 and t+6 each exceed that company's "
            "own expanding p80 of invoice_side using months <= t"
        ),
        "y8_cash_worse_6": (
            "1 if cash_side at t+4, t+5 and t+6 each exceed that company's "
            "own expanding p80 of cash_side using months <= t"
        ),
    },
}

SOURCE_TABLES = META["source_tables"]

# Single-feature leak screen (contemporaneous X vs Y). Same-source e_ on the
# invoice Y, or cash signals on the cash Y, must stay below 0.70 AUROC.
_LEAK_E = ("e_ar_overdue_30", "e_delay_coll", "e_ar_overdue")
_LEAK_CASH = ("runway", "net_margin", "neg_liq")


def _cash_components(con) -> pd.DataFrame:
    """Month-end runway / net_margin / neg_liq_3 (score_pipeline formulas)."""
    p = cash_month_panel(con).sort_values(["company_id", "month"]).reset_index(drop=True)
    p["company_id"] = p["company_id"].astype(str)
    g = p.groupby("company_id", sort=False)
    p["in3"] = g["op_in"].transform(lambda s: s.rolling(3).sum())
    in3, out3 = p["in3"], p["out3"]
    p["net_margin"] = np.where(
        in3.isna(),
        np.nan,
        np.where(in3.to_numpy() > 0, ((in3 - out3) / in3).clip(-1.0, 1.0), -1.0),
    )
    neg = pd.Series(
        np.where(p["liq"].isna(), np.nan, (p["liq"] < 0).astype(float)),
        index=p.index,
    )
    p["neg_liq"] = neg.groupby(p["company_id"], sort=False).transform(lambda s: s.rolling(3).mean())
    return p[["company_id", "month", "op_in", "runway", "net_margin", "neg_liq"]]


def _train_period_rank(values: pd.Series, period: pd.Series, is_train: pd.Series, sign: int) -> pd.Series:
    """Empirical CDF vs train companies in the same month. Higher = worse.

    ``sign`` = +1 if a larger raw value is worse, −1 if a larger raw value is better.
    Holdout rows are scored against the train reference; they never enter it.
    """
    s = sign * pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index)
    tmp = pd.DataFrame({"s": s, "period": period, "tr": is_train.astype(bool)})
    for _, sl in tmp.groupby("period", sort=False):
        ref = sl.loc[sl["tr"], "s"].dropna()
        if ref.empty:
            continue
        arr = np.sort(ref.to_numpy(dtype=float))
        xv = sl["s"].to_numpy(dtype=float)
        r = np.searchsorted(arr, xv, side="right") / float(arr.size)
        out.loc[sl.index] = np.where(np.isfinite(xv), r, np.nan)
    return out


def _expanding_p80(s: pd.Series, min_periods: int = MIN_OWN_HIST) -> pd.Series:
    """Own-history p80 counting only finite observations (Y5 style)."""
    out = np.full(len(s), np.nan, dtype=float)
    vals: list[float] = []
    for i, v in enumerate(s.to_numpy(dtype=float)):
        if np.isfinite(v):
            vals.append(float(v))
        if len(vals) >= min_periods:
            out[i] = float(np.quantile(vals, OWN_P))
    return pd.Series(out, index=s.index)


def _sustained_worse(side: pd.Series, p80: pd.Series, cid: pd.Series, month: pd.Series) -> pd.Series:
    """1 if side at t+4, t+5, t+6 all exceed own p80 at t; else 0/NaN."""
    df = pd.DataFrame({"side": side, "p80": p80, "company_id": cid, "month": month})
    df = df.sort_values(["company_id", "month"])
    g = df.groupby("company_id", sort=False)
    fut = [g["side"].shift(-(HORIZON - SUSTAIN + 1 + i)) for i in range(SUSTAIN)]
    f4, f5, f6 = fut
    cutoff = LAST_M - pd.DateOffset(months=HORIZON)
    has = df["month"] <= cutoff
    ok = has & f4.notna() & f5.notna() & f6.notna() & df["p80"].notna()
    y = np.where(
        ok,
        ((f4 > df["p80"]) & (f5 > df["p80"]) & (f6 > df["p80"])).astype(float),
        np.nan,
    )
    return pd.Series(y, index=df.index).reindex(side.index)


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    """Return company_id, period and the two Y8 binaries (0/1/NaN)."""
    out = grid[["company_id", "period"]].copy()
    out["company_id"] = out["company_id"].astype(str)
    out["period"] = pd.to_datetime(out["period"])
    out["_m"] = _month_key(out["period"])

    cash = _cash_components(con)
    month_keys = cash[["company_id", "month"]].rename(columns={"month": "period"})
    inv = build_e(con, month_keys)
    inv["company_id"] = inv["company_id"].astype(str)
    inv["period"] = pd.to_datetime(inv["period"])

    p = cash.merge(
        inv[["company_id", "period", "e_ar_overdue_30", "e_delay_coll", "e_ar_overdue"]],
        left_on=["company_id", "month"],
        right_on=["company_id", "period"],
        how="left",
    ).drop(columns=["period"])
    p = p.sort_values(["company_id", "month"]).reset_index(drop=True)
    is_tr = train_mask(p["company_id"])
    assert_no_holdout(p.loc[is_tr, "company_id"])

    # z-like ranks (train-only month CDF). Higher = worse.
    p["r_runway"] = _train_period_rank(p["runway"], p["month"], is_tr, sign=-1)
    p["r_margin"] = _train_period_rank(p["net_margin"], p["month"], is_tr, sign=-1)
    p["r_neg"] = _train_period_rank(p["neg_liq"], p["month"], is_tr, sign=1)
    p["r_od30"] = _train_period_rank(p["e_ar_overdue_30"], p["month"], is_tr, sign=1)
    p["r_delay"] = _train_period_rank(p["e_delay_coll"], p["month"], is_tr, sign=1)

    p["cash_side"] = p[["r_runway", "r_margin", "r_neg"]].mean(axis=1)
    p["invoice_side"] = p[["r_od30", "r_delay"]].mean(axis=1)

    p["cash_p80"] = p.groupby("company_id", sort=False)["cash_side"].transform(_expanding_p80)
    p["inv_p80"] = p.groupby("company_id", sort=False)["invoice_side"].transform(_expanding_p80)

    p["y8_cash_worse_6"] = _sustained_worse(p["cash_side"], p["cash_p80"], p["company_id"], p["month"])
    p["y8_inv_worse_6"] = _sustained_worse(p["invoice_side"], p["inv_p80"], p["company_id"], p["month"])

    lab = p[["company_id", "month", *Y8_COLS]].rename(columns={"month": "_m"})
    out = out.merge(lab, on=["company_id", "_m"], how="left").drop(columns=["_m"])
    return out.reset_index(drop=True)


def _max_single_auroc(y: pd.Series, frame: pd.DataFrame, cols: tuple[str, ...]) -> float:
    best = float("nan")
    for c in cols:
        if c not in frame.columns:
            continue
        auc = auroc(y, frame[c])
        if not np.isfinite(auc):
            continue
        two = float(max(auc, 1.0 - auc))
        if not np.isfinite(best) or two > best:
            best = two
    return best


def train_acceptance(con, grid: pd.DataFrame, y8: pd.DataFrame | None = None) -> pd.DataFrame:
    """Train-only rates, size AUROC, and single-feature leak screen."""
    if y8 is None:
        y8 = build(con, grid)
    keys = grid[["company_id", "period"]].copy()
    keys["company_id"] = keys["company_id"].astype(str)
    keys["period"] = pd.to_datetime(keys["period"])
    keys["_m"] = _month_key(keys["period"])
    tr = train_mask(keys["company_id"])

    cash = _cash_components(con).rename(columns={"month": "_m"})
    month_keys = cash[["company_id", "_m"]].rename(columns={"_m": "period"})
    inv = build_e(con, month_keys)
    inv["company_id"] = inv["company_id"].astype(str)
    inv["_m"] = pd.to_datetime(inv["period"])
    feat = keys.merge(cash, on=["company_id", "_m"], how="left").merge(
        inv[["company_id", "_m", "e_ar_overdue_30", "e_delay_coll", "e_ar_overdue"]],
        on=["company_id", "_m"],
        how="left",
    )
    y = y8.copy()
    y["company_id"] = y["company_id"].astype(str)
    y["period"] = pd.to_datetime(y["period"])
    panel = keys.merge(y, on=["company_id", "period"], how="left")
    panel = panel.merge(
        feat[["company_id", "period", "op_in", *_LEAK_CASH, *_LEAK_E]],
        on=["company_id", "period"],
        how="left",
    )

    rows = []
    n_train = int(tr.sum())
    for col in Y8_COLS:
        yt = panel.loc[tr, col]
        ok = yt.notna()
        n = int(ok.sum())
        rate = float(yt[ok].mean()) if n else float("nan")
        size = auroc(yt, np.log1p(pd.to_numeric(panel.loc[tr, "op_in"], errors="coerce").abs()))
        size_abs = float(max(size, 1.0 - size)) if np.isfinite(size) else float("nan")
        leak_e = _max_single_auroc(yt, panel.loc[tr], _LEAK_E)
        leak_c = _max_single_auroc(yt, panel.loc[tr], _LEAK_CASH)
        leak = leak_e if col == "y8_inv_worse_6" else leak_c
        in_rate = bool(n and 0.05 <= rate <= 0.30)
        size_ok = bool(np.isfinite(size_abs) and size_abs < 0.60)
        leak_ok = bool(np.isfinite(leak) and leak < 0.70)
        rows.append(
            {
                "column": col,
                "n_train": n,
                "n_pos": int((yt == 1).sum()),
                "coverage": n / n_train if n_train else float("nan"),
                "base_rate": rate,
                "size_auroc": size,
                "size_auroc_two_sided": size_abs,
                "single_e_auroc": leak_e,
                "single_cash_auroc": leak_c,
                "single_source_auroc": leak,
                "accepted": in_rate and size_ok and leak_ok,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from analysis.features.common import connect
    from analysis.features.grid import monthly_grid

    con = connect()
    grid = monthly_grid(con)[["company_id", "period"]]
    y8 = build(con, grid)
    print("rows", len(y8), "dups", int(y8.duplicated(["company_id", "period"]).sum()))
    acc = train_acceptance(con, grid, y8)
    print("Y8 train acceptance")
    print(acc.to_string(index=False))
    con.close()
