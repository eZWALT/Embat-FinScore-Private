"""Q4/Q5/Q6 unused `e_pending_amt_share` leftover after DSO / issued.

NORTH_STAR: `e_pending_amt_share` = among issued non-cancel invoices with
issuance_date ≤ period_end, abs-amount share still unpaid as of period_end
(invalid payment_date treated as 0 in both num and den). Snapshot
`pending_amount` would leak later collections. 470 never-invoice
company-months stay **NaN not 0**. Feature report: BETWEEN, size ρ 0.044,
acf-ish 0.78, cluster representative, coverage 60.3%.

TURNPEND_shallow already tried TURNOVER + pending: **0.7184 ± 0.040 vs
TURNOVER 0.720**. That is NOT a KEEP. Do not grow TURNOVER. Do not change
the night Y7 quote (TURNOVER 0.720 / B_shallow 0.712). Do not quote
holdout. Y7 never D. Y5 never E. Night Y3 stays 0.762 / 0.752. Days bar
0.711. Size bar 0.617.

KEEP-as-X gate: beat size ≥0.02 AND leftover after the honest bar
(Y7: issued_lag1 and/or DSO; Y3: days) AND leftover ≥0.55 AND not SIZE
(|Spearman| vs size ≥0.50) AND not a twin (|Spearman| ≥0.80 vs DSO /
overdue / delay / issued / j_pay_match).

No 0–100. No product/. No parquet rewrite. No new GBM. Do not run
`build_targets`. Do not invent `y_pending`. Do not put pending on the
15-col Y3 card. Do not edit invoices.py unless a real formula bug.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.pending_qa

Owned: analysis/evaluate/pending_qa.py, analysis/outputs/pending_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_pending.md (end).
"""
from __future__ import annotations

import csv
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.evaluate.protocol import (
    FOLD_SEED,
    assert_no_holdout,
    auroc,
    group_folds,
    leakage_check,
    load_holdout,
)
from analysis.features.common import ANALYSIS, DATA, MONTHS, connect
from analysis.features.match import build as build_match
from analysis.targets.y11_dark import book_invoice_ids, dark_population

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
OUT_MD = ANALYSIS / "outputs" / "pending_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "pending_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
CN_MD = ANALYSIS / "outputs" / "credit_note_qa.md"
DELAY_MD = ANALYSIS / "outputs" / "delay_qa.md"
AGENT = "86399952"
WAVE = "4"
ROUND = "R4"
MODEL = "pending_qa"
X_FAM = "E"

Y3 = "y3_recover_cash_6m"
Y7 = "y7_top1_lost"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
ISSUED_LAG1_BENCH = 0.630
TURNOVER_QUOTE = 0.720
B_SHALLOW_QUOTE = 0.712
TURNPEND_QUOTE = 0.7184
NIGHT_Y3 = 0.762
NIGHT_Y3_CORE = 0.752
COV_QUOTE = 0.603
SIZE_RHO_QUOTE = 0.044
ACF1_QUOTE = 0.78
ICC_QUOTE = 0.97
A_OUT_VOL_ETA2 = 0.74
TWIN_RHO = 0.80
KEEP_DELTA = 0.02
CHANCE = 0.55
SIZE_RHO = 0.50
ICC_TRAIT = 0.85
MIN_POS = 50
MIN_ACF_PAIRS = 4
N_DARK_WANT = 470
DELAY_EMPTY_BEFORE = MONTHS[6]  # 2025-03-01; delay empty first 6 months
PANEL_START = pd.Timestamp("2024-09-01")
PANEL_END = pd.Timestamp("2026-08-01")

PENDING_SQL = """
SELECT i.company_id,
       p.period,
       SUM(CASE
             WHEN coalesce(i.payment_date_invalid, FALSE) THEN 0
             WHEN i.payment_date IS NOT NULL
              AND CAST(i.payment_date AS DATE) <= CAST(p.period_end AS DATE) THEN 0
             ELSE abs(i.amount)
           END)
         / NULLIF(SUM(CASE WHEN coalesce(i.payment_date_invalid, FALSE)
                           THEN 0 ELSE abs(i.amount) END), 0) AS e_pending_amt_share
FROM invoices i
JOIN _e_pend_periods p
  ON CAST(i.issuance_date AS DATE) <= CAST(p.period_end AS DATE)
WHERE i.document_type = 'invoice'
  AND i.status <> 'cancel'
  AND i.amount <> 0
  AND i.issuance_date IS NOT NULL
GROUP BY 1, 2
"""

AP_PENDING_SQL = """
SELECT i.company_id,
       p.period,
       SUM(CASE
             WHEN coalesce(i.payment_date_invalid, FALSE) THEN 0
             WHEN i.payment_date IS NOT NULL
              AND CAST(i.payment_date AS DATE) <= CAST(p.period_end AS DATE) THEN 0
             ELSE abs(i.amount)
           END)
         / NULLIF(SUM(CASE WHEN coalesce(i.payment_date_invalid, FALSE)
                           THEN 0 ELSE abs(i.amount) END), 0) AS e_ap_pending_share
FROM invoices i
JOIN _e_pend_periods p
  ON CAST(i.issuance_date AS DATE) <= CAST(p.period_end AS DATE)
WHERE i.document_type = 'invoice'
  AND i.status <> 'cancel'
  AND i.amount < 0
  AND i.issuance_date IS NOT NULL
GROUP BY 1, 2
"""

AR_PENDING_SQL = """
SELECT i.company_id,
       p.period,
       SUM(CASE
             WHEN coalesce(i.payment_date_invalid, FALSE) THEN 0
             WHEN i.payment_date IS NOT NULL
              AND CAST(i.payment_date AS DATE) <= CAST(p.period_end AS DATE) THEN 0
             ELSE abs(i.amount)
           END)
         / NULLIF(SUM(CASE WHEN coalesce(i.payment_date_invalid, FALSE)
                           THEN 0 ELSE abs(i.amount) END), 0) AS e_ar_pending_share
FROM invoices i
JOIN _e_pend_periods p
  ON CAST(i.issuance_date AS DATE) <= CAST(p.period_end AS DATE)
WHERE i.document_type = 'invoice'
  AND i.status <> 'cancel'
  AND i.amount > 0
  AND i.issuance_date IS NOT NULL
GROUP BY 1, 2
"""

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "a_op_in",
    "c_n_days_with_tx",
    "e_pending_amt_share",
    "e_ar_issued",
    "e_ap_issued",
    "e_dso_proxy",
    "e_ar_overdue",
    "e_ar_overdue_30",
    "e_delay_coll",
    "e_credit_note_ratio",
    "e_ar_open",
    "e_ap_open",
)

Y_KEEP = (Y3, Y7)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _f(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{float(x):.{nd}f}"


def _pp(x, nd=1) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{100.0 * float(x):.{nd}f}%"


def _pct(n, d) -> float:
    if d is None or d == 0:
        return float("nan")
    return float(n) / float(d)


def _md_table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "_(empty)_\n"
    if cols is None:
        cols = list(rows[0].keys())
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["company_id"] = out["company_id"].astype(str)
    if "period" in out.columns:
        out["period"] = pd.to_datetime(out["period"]).astype("datetime64[ns]")
    if "group_id" in out.columns:
        out["group_id"] = out["group_id"].astype(str)
    return out


def spearman(a, b) -> float:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    ).dropna()
    if len(d) < 8 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="spearman"))


def median_acf(series: pd.Series, company: pd.Series, lag: int) -> float:
    df = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    vals = []
    for _, g in df.groupby("co", sort=False):
        x = g["x"].to_numpy(dtype=float)
        if len(x) <= lag:
            continue
        aa, bb = x[:-lag], x[lag:]
        m = np.isfinite(aa) & np.isfinite(bb)
        if int(m.sum()) < MIN_ACF_PAIRS:
            continue
        aa, bb = aa[m], bb[m]
        if np.std(aa) == 0 or np.std(bb) == 0:
            continue
        vals.append(float(np.corrcoef(aa, bb)[0, 1]))
    return float(np.median(vals)) if vals else float("nan")


def icc_anova(series: pd.Series, company: pd.Series) -> dict:
    s = pd.DataFrame(
        {"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)}
    ).dropna()
    if len(s) < 10 or s["x"].nunique() < 2:
        return {
            "icc": float("nan"),
            "eta2": float("nan"),
            "var_w": float("nan"),
            "var_b": float("nan"),
            "k": 0,
            "n": 0,
        }
    y = s["x"].to_numpy(dtype=float)
    _, inv, counts = np.unique(s["co"].to_numpy(), return_inverse=True, return_counts=True)
    if counts.size < 2:
        return {
            "icc": float("nan"),
            "eta2": float("nan"),
            "var_w": float("nan"),
            "var_b": float("nan"),
            "k": 0,
            "n": 0,
        }
    means = np.bincount(inv, weights=y) / counts
    grand = float(y.mean())
    ssb = float(np.sum(counts * (means - grand) ** 2))
    ssw = float(np.sum((y - means[inv]) ** 2))
    k = int(counts.size)
    n = int(y.size)
    var_b = ssb / max(k - 1, 1)
    var_w = ssw / max(n - k, 1)
    icc = var_b / (var_b + var_w) if (var_b + var_w) > 0 else float("nan")
    eta2 = ssb / (ssb + ssw) if (ssb + ssw) > 0 else float("nan")
    return {
        "icc": float(icc),
        "eta2": float(eta2),
        "var_w": float(var_w),
        "var_b": float(var_b),
        "k": k,
        "n": n,
    }


def choose_sign(y: pd.Series, x: pd.Series) -> int:
    auc_p = auroc(y, x)
    auc_n = auroc(y, -x)
    if not np.isfinite(auc_p) and not np.isfinite(auc_n):
        return 1
    if not np.isfinite(auc_p):
        return -1
    if not np.isfinite(auc_n):
        return 1
    return -1 if auc_n > auc_p else 1


def signed_oof_auroc(
    y: pd.Series,
    x: pd.Series,
    folds: pd.Series,
    mask: pd.Series,
    n_folds: int = N_FOLDS,
) -> dict:
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    defined = mask & y.notna() & x.notna()
    n = int(defined.sum())
    n_pos = int((defined & (y == 1)).sum())
    n_neg = int((defined & (y == 0)).sum())
    low = n_pos < MIN_POS or n_neg == 0
    fold_rows = []
    aucs = []
    if low:
        return {
            "cv": float("nan"),
            "sd": float("nan"),
            "n_folds": 0,
            "folds": fold_rows,
            "train_sign": 0,
            "train_auc": float("nan"),
            "n_defined": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "low_power": True,
        }
    for k in range(n_folds):
        tr = defined & (folds != k)
        va = defined & (folds == k)
        sign = choose_sign(y[tr], x[tr])
        auc = auroc(y[va], sign * x[va])
        aucs.append(auc)
        fold_rows.append(
            {
                "fold": k,
                "auroc": float(auc) if np.isfinite(auc) else float("nan"),
                "sign": int(sign),
                "n_va": int(va.sum()),
                "n_pos": int((va & (y == 1)).sum()),
            }
        )
    finite = [a for a in aucs if np.isfinite(a)]
    tr_sign = choose_sign(y[defined], x[defined])
    return {
        "cv": float(np.mean(finite)) if finite else float("nan"),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "n_folds": len(finite),
        "folds": fold_rows,
        "train_sign": int(tr_sign),
        "train_auc": float(auroc(y[defined], tr_sign * x[defined])),
        "n_defined": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "low_power": False,
    }


def fold_bits(rec: dict) -> str:
    return " ".join(
        f"{r['auroc']:.3f}" if np.isfinite(r["auroc"]) else "—" for r in rec.get("folds", [])
    )


def fold_k(rec: dict, k: int) -> float:
    for r in rec.get("folds", []):
        if int(r["fold"]) == k:
            return float(r["auroc"])
    return float("nan")


def ols_resid(y: pd.Series, *xs: pd.Series) -> tuple[pd.Series, dict]:
    """Train-defined OLS residual of y on 1+ predictors. No holdout in slope."""
    cols = {"y": pd.to_numeric(y, errors="coerce")}
    for i, x in enumerate(xs):
        cols[f"x{i}"] = pd.to_numeric(x, errors="coerce")
    d = pd.DataFrame(cols)
    ok = d.notna().all(axis=1)
    resid = pd.Series(np.nan, index=d.index, dtype=float)
    info = {"n": int(ok.sum()), "slope": [], "intercept": float("nan"), "r2": float("nan")}
    n_x = len(xs)
    if int(ok.sum()) < max(20, n_x + 5):
        return resid, info
    Y = d.loc[ok, "y"].to_numpy(dtype=float)
    X = np.column_stack(
        [np.ones(int(ok.sum()))] + [d.loc[ok, f"x{i}"].to_numpy(dtype=float) for i in range(n_x)]
    )
    try:
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return resid, info
    pred = X @ beta
    resid.loc[ok] = Y - pred
    info["intercept"] = float(beta[0])
    info["slope"] = [float(b) for b in beta[1:]]
    ss_res = float(np.sum((Y - pred) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    info["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return resid, info


def rank_resid(y: pd.Series, x: pd.Series) -> pd.Series:
    """Spearman-style residual: rank(y) ~ rank(x). Orthogonal-to-bar."""
    a = pd.to_numeric(y, errors="coerce")
    b = pd.to_numeric(x, errors="coerce")
    m = a.notna() & b.notna()
    out = pd.Series(np.nan, index=y.index, dtype=float)
    if int(m.sum()) < 20:
        return out
    ra = a[m].rank().to_numpy(dtype=float)
    rb = b[m].rank().to_numpy(dtype=float)
    xb = np.column_stack([np.ones(int(m.sum())), rb])
    coef, *_ = np.linalg.lstsq(xb, ra, rcond=None)
    out.loc[m] = ra - xb @ coef
    return out


def company_demean(series: pd.Series, company: pd.Series) -> pd.Series:
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    mu = d.groupby("co")["x"].transform("mean")
    return d["x"] - mu


def _cv(res: dict) -> float:
    return float("nan") if res.get("low_power") else float(res.get("cv", float("nan")))


def _auc_row(y: str, feat: str, res: dict, present: float | None = None) -> dict:
    row = {
        "y": y,
        "feature": feat,
        "n": f"{res['n_defined']:,}",
        "n_pos": f"{res['n_pos']:,}",
        "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
        "sd": _f(res["sd"]) if not res["low_power"] else "—",
        "sign": res["train_sign"] if not res["low_power"] else "—",
        "folds": fold_bits(res) if not res["low_power"] else "—",
    }
    if present is not None:
        row["present"] = _pp(present)
    return row


def _trail_class(n) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return "na"
    n = int(n)
    if n < 12:
        return "short_<12"
    if n >= 18:
        return "long_>=18"
    return "mid_12_17"


def add_panel_lags(df: pd.DataFrame, stems: list[str], lags: tuple[int, ...]) -> pd.DataFrame:
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    cid = out["company_id"]
    extra = {}
    for c in stems:
        s = pd.to_numeric(out[c], errors="coerce")
        g = s.groupby(cid, sort=False)
        for k in lags:
            extra[f"{c}_lag{k}"] = g.shift(k)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    return out


def _period_ends(periods: pd.Series) -> pd.DataFrame:
    p = pd.to_datetime(pd.Series(periods.unique())).sort_values().reset_index(drop=True)
    df = pd.DataFrame({"period": p})
    df["period_end"] = df["period"] + pd.offsets.MonthEnd(0)
    return df


def load_panel() -> pd.DataFrame:
    if not STORE.exists():
        raise FileNotFoundError(f"{STORE} missing")
    if not TARGETS.exists():
        raise FileNotFoundError(f"{TARGETS} missing — do not run build_targets")
    raw = pd.read_parquet(STORE)
    yraw = pd.read_parquet(TARGETS)
    print(f"store {STORE} shape={raw.shape} (read-only)")
    print(f"targets {TARGETS} shape={yraw.shape} (read-only)")
    missing = [c for c in STORE_COLS if c not in raw.columns]
    if missing:
        raise RuntimeError(f"monthly.parquet missing {missing}")
    ymiss = [c for c in Y_KEEP if c not in yraw.columns]
    if ymiss:
        raise RuntimeError(f"targets.parquet missing {ymiss}")
    if "j_pay_match" in raw.columns:
        print("j_pay_match already in parquet — using store (no Family J merge)")
        extra = ["j_pay_match"]
    else:
        extra = []
    panel = _keys(raw[list(STORE_COLS) + extra])
    y = _keys(yraw[["company_id", "period", *Y_KEEP]])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    panel["first_month"] = pd.to_datetime(panel["first_month"])
    panel["months_so_far"] = (
        (panel["period"].dt.year - panel["first_month"].dt.year) * 12
        + (panel["period"].dt.month - panel["first_month"].dt.month)
        + 1
    )
    grid_n = panel.groupby("company_id")["period"].size().rename("n_grid_months")
    panel = panel.merge(grid_n, on="company_id", how="left")
    panel["so_far_class"] = panel["months_so_far"].map(_trail_class)
    leak = leakage_check(
        ["e_pending_amt_share", "e_ar_issued", "e_dso_proxy", "e_ar_overdue"],
        Y7,
        forbidden_prefixes=["d"],
    )
    if not leak["ok"]:
        raise RuntimeError(f"Y7 X leak: {leak['issues']}")
    leak3 = leakage_check(["e_pending_amt_share"], Y3, forbidden_prefixes=["b"])
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 X leak: {leak3['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def attach_j(panel: pd.DataFrame, con) -> pd.DataFrame:
    if "j_pay_match" in panel.columns and panel["j_pay_match"].notna().any():
        print("using existing j_pay_match")
        return panel
    grid = panel[["company_id", "period"]].drop_duplicates()
    t0 = time.time()
    j = build_match(con, grid)
    print(f"Family J in-memory {j.shape} in {time.time() - t0:.1f}s (not merged to parquet)")
    j = _keys(j[["company_id", "period", "j_pay_match", "j_has_book"]])
    out = panel.merge(j, on=["company_id", "period"], how="left")
    return out


def attach_recon(panel: pd.DataFrame, con) -> pd.DataFrame:
    periods = _period_ends(panel["period"])
    con.register("_e_pend_periods", periods)
    try:
        raw = con.execute(PENDING_SQL).df()
        ap = con.execute(AP_PENDING_SQL).df()
        ar = con.execute(AR_PENDING_SQL).df()
    finally:
        con.unregister("_e_pend_periods")
    for part, name in ((raw, "pending_sql"), (ap, "e_ap_pending_share"), (ar, "e_ar_pending_share")):
        part["company_id"] = part["company_id"].astype(str)
        part["period"] = pd.to_datetime(part["period"])
        print(f"{name} rows={len(part):,}")
    raw = raw.rename(columns={"e_pending_amt_share": "pending_sql"})
    out = panel.merge(raw, on=["company_id", "period"], how="left")
    out = out.merge(_keys(ap), on=["company_id", "period"], how="left")
    out = out.merge(_keys(ar), on=["company_id", "period"], how="left")
    return out


def _qcut(s: pd.Series, n: int = 5) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    try:
        return pd.qcut(x, n, duplicates="drop")
    except ValueError:
        return pd.Series(pd.NA, index=s.index)


# ---------------------------------------------------------------------------
# 1. Coverage train vs holdout; 470 NaN; ERP vs dark
# ---------------------------------------------------------------------------
def cut1_coverage(panel: pd.DataFrame, book: set[str], dark: dict) -> dict:
    rows = []
    for split, sl in (
        ("train", panel["split"] == "train"),
        ("holdout", panel["split"] == "holdout"),
    ):
        d = panel.loc[sl]
        p = pd.to_numeric(d["e_pending_amt_share"], errors="coerce")
        n = len(d)
        nn = int(p.notna().sum())
        na = int(p.isna().sum())
        n_co = int(d["company_id"].nunique())
        n_co_nn = int(d.loc[p.notna(), "company_id"].nunique())
        book_m = d["company_id"].isin(book)
        dark_m = ~book_m
        dark_nn = int(p[dark_m].notna().sum())
        dark_zero = int((p[dark_m] == 0).sum())
        book_nn = int(p[book_m].notna().sum())
        book_na = int(p[book_m].isna().sum())
        n_dark_co = int(d.loc[dark_m, "company_id"].nunique())
        n_book_co = int(d.loc[book_m, "company_id"].nunique())
        if split == "train":
            assert_no_holdout(d["company_id"])
        rows.append(
            {
                "split": split,
                "cm": f"{n:,}",
                "companies": f"{n_co:,}",
                "pending nn": f"{nn:,}",
                "cov": _pp(nn / n if n else float("nan")),
                "NaN": f"{na:,}",
                "ever-ERP / never": f"{n_book_co} / {n_dark_co}",
                "dark nn / zero": f"{dark_nn} / {dark_zero}",
                "ERP nn / NaN": f"{book_nn:,} / {book_na:,}",
                "cos ever-defined": f"{n_co_nn:,}",
            }
        )
        print(
            f"1 {split}: cov={nn/n if n else float('nan'):.4f} nn={nn} na={na} "
            f"dark_co={n_dark_co} dark_nn={dark_nn} dark_zero={dark_zero}"
        )

    tr = panel[panel["split"] == "train"]
    p = pd.to_numeric(tr["e_pending_amt_share"], errors="coerce")
    n_dark_want = int(len(dark.get("train_dark_ids", set()))) if "train_dark_ids" in dark else N_DARK_WANT
    train_dark_ids = set(dark.get("train_dark_ids") or set())
    if not train_dark_ids:
        train_dark_ids = set(tr.loc[~tr["company_id"].isin(book), "company_id"].astype(str))
    n_dark_co = len(train_dark_ids)
    dark_cm = tr["company_id"].isin(train_dark_ids)
    dark_nn = int(p[dark_cm].notna().sum())
    dark_zero = int((p[dark_cm] == 0).sum())
    nan_cm = int(p.isna().sum())
    dark_ok = dark_nn == 0 and dark_zero == 0 and n_dark_co == N_DARK_WANT
    acf1 = median_acf(p, tr["company_id"], 1)
    acf3 = median_acf(p, tr["company_id"], 3)
    acf6 = median_acf(p, tr["company_id"], 6)
    size_rho = spearman(p, tr["log_in3"])
    prose = (
        f"Train coverage {_pp(_pct(int(p.notna().sum()), len(tr)))} "
        f"(quote { _pp(COV_QUOTE) }). Holdout coverage is a check only. "
        f"Never-ERP companies {n_dark_co} (want {N_DARK_WANT}): pending non-null "
        f"{dark_nn} zero-filled {dark_zero}. 470 stay NaN not 0: "
        f"{'CONFIRM' if dark_ok else 'FAIL'}. Train NaN CM {nan_cm:,} "
        f"(dark + ERP months before first issued). acf1={_f(acf1)} "
        f"(quote {ACF1_QUOTE}) size ρ={_f(size_rho)} (quote {SIZE_RHO_QUOTE})."
    )
    print(prose)
    return {
        "rows": rows,
        "prose": prose,
        "train_cov": _pct(int(p.notna().sum()), len(tr)),
        "train_nn": int(p.notna().sum()),
        "train_na": int(p.isna().sum()),
        "n_dark_co": n_dark_co,
        "dark_nn": dark_nn,
        "dark_zero": dark_zero,
        "dark_ok": dark_ok,
        "acf1": acf1,
        "acf3": acf3,
        "acf6": acf6,
        "size_rho": size_rho,
        "hold_cov": _pct(
            int(pd.to_numeric(panel.loc[panel["split"] == "holdout", "e_pending_amt_share"], errors="coerce").notna().sum()),
            int((panel["split"] == "holdout").sum()),
        ),
        "n_train": len(tr),
        "n_train_co": int(tr["company_id"].nunique()),
    }


# ---------------------------------------------------------------------------
# 2. Store vs raw SQL
# ---------------------------------------------------------------------------
def cut2_formula(tr: pd.DataFrame) -> dict:
    store = pd.to_numeric(tr["e_pending_amt_share"], errors="coerce")
    sql = pd.to_numeric(tr["pending_sql"], errors="coerce")
    both = store.notna() & sql.notna()
    store_only = int((store.notna() & sql.isna()).sum())
    sql_only = int((store.isna() & sql.notna()).sum())
    n_both = int(both.sum())
    if n_both:
        delta = (store[both] - sql[both]).abs()
        max_d = float(delta.max())
        mean_d = float(delta.mean())
        exact = int((delta < 1e-12).sum())
    else:
        max_d = mean_d = float("nan")
        exact = 0
    match = n_both > 0 and store_only == 0 and sql_only == 0 and max_d < 1e-9
    # sample of disagreeing company-months if any
    sample = []
    if n_both and max_d >= 1e-9:
        bad = tr.loc[both].assign(abs_d=(store[both] - sql[both]).abs())
        bad = bad.nlargest(8, "abs_d")
        for _, r in bad.iterrows():
            sample.append(
                {
                    "company_id": r["company_id"],
                    "period": str(pd.Timestamp(r["period"]).date()),
                    "store": _f(r["e_pending_amt_share"], 6),
                    "sql": _f(r["pending_sql"], 6),
                }
            )
    prose = (
        f"Store vs raw pending SQL: n_both={n_both:,} max|Δ|={max_d:.3e} "
        f"mean|Δ|={mean_d:.3e} exact={exact:,}; store-only {store_only} "
        f"sql-only {sql_only}. "
        f"{'FORMULA MATCH — do not edit invoices.py.' if match else 'DISAGREE — document, do not rewrite unless a real bug.'}"
    )
    print(prose)
    return {
        "n_both": n_both,
        "max_d": max_d,
        "mean_d": mean_d,
        "exact": exact,
        "store_only": store_only,
        "sql_only": sql_only,
        "match": match,
        "sample": sample,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 3. Spearman twins
# ---------------------------------------------------------------------------
def cut3_twins(tr: pd.DataFrame) -> dict:
    p = tr["e_pending_amt_share"]
    pairs = [
        ("e_dso_proxy", tr["e_dso_proxy"]),
        ("e_ar_overdue", tr["e_ar_overdue"]),
        ("e_ar_overdue_30", tr["e_ar_overdue_30"]),
        ("e_delay_coll", tr["e_delay_coll"]),
        ("e_ar_issued", tr["e_ar_issued"]),
        ("e_ar_issued_lag1", tr["e_ar_issued_lag1"]),
        ("j_pay_match", tr["j_pay_match"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("e_credit_note_ratio", tr["e_credit_note_ratio"]),
        ("e_ar_open", tr["e_ar_open"]),
        ("e_ap_pending_share", tr.get("e_ap_pending_share", pd.Series(np.nan, index=tr.index))),
        ("e_ar_pending_share", tr.get("e_ar_pending_share", pd.Series(np.nan, index=tr.index))),
    ]
    rhos = {}
    rows = []
    official = {
        "e_dso_proxy",
        "e_ar_overdue",
        "e_ar_overdue_30",
        "e_delay_coll",
        "e_ar_issued",
        "e_ar_issued_lag1",
        "j_pay_match",
    }
    twins = []
    for name, s in pairs:
        rho = spearman(p, s)
        rhos[name] = rho
        twin = bool(name in official and np.isfinite(rho) and abs(rho) >= TWIN_RHO)
        size_flag = name == "log1p(a_in3)" and bool(np.isfinite(rho) and abs(rho) >= SIZE_RHO)
        if twin:
            twins.append(name)
        rows.append(
            {
                "vs": name,
                "ρ": _f(rho),
                "twin |ρ|≥0.80": "YES" if twin else ("(analogue)" if name not in official and np.isfinite(rho) and abs(rho) >= TWIN_RHO else "no"),
                "SIZE |ρ|≥0.50": "YES" if size_flag else ("—" if name != "log1p(a_in3)" else "no"),
            }
        )
        print(f"3 ρ pending vs {name} = {rho:.4f}" if np.isfinite(rho) else f"3 ρ pending vs {name} = —")
    size_flag = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    any_twin = bool(twins)
    prose = (
        f"Twins {twins or 'none'}. SIZE={size_flag} (ρ={_f(rhos['log1p(a_in3)'])}). "
        f"DSO {_f(rhos['e_dso_proxy'])} overdue {_f(rhos['e_ar_overdue'])} "
        f"delay {_f(rhos['e_delay_coll'])} issued_lag1 {_f(rhos['e_ar_issued_lag1'])} "
        f"j_pay_match {_f(rhos['j_pay_match'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "twins": twins,
        "any_twin": any_twin,
        "size_flag": size_flag,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 4. Single-feature group-fold AUROC
# ---------------------------------------------------------------------------
def cut4_singles(tr: pd.DataFrame) -> dict:
    feats = {
        "e_pending_amt_share": tr["e_pending_amt_share"],
        "log1p(a_in3)": tr["log_in3"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "e_dso_proxy": tr["e_dso_proxy"],
        "e_ar_issued_lag1": tr["e_ar_issued_lag1"],
        "e_ar_overdue": tr["e_ar_overdue"],
        "e_delay_coll": tr["e_delay_coll"],
        "e_credit_note_ratio": tr["e_credit_note_ratio"],
        "j_pay_match": tr["j_pay_match"],
        "e_ar_pending_share": tr["e_ar_pending_share"],
        "e_ap_pending_share": tr["e_ap_pending_share"],
    }
    rows = []
    store = {}
    for y in (Y7, Y3):
        lab = tr[y].notna()
        n_lab = int(lab.sum())
        for name, x in feats.items():
            res = signed_oof_auroc(tr[y], x, tr["fold"], lab)
            store[(y, name)] = res
            present = _pct(res["n_defined"], n_lab) if n_lab else float("nan")
            rows.append(_auc_row(y, name, res, present))
            print(
                f"4 {y} {name}: "
                f"{'LOW_POWER' if res['low_power'] else _f(res['cv'])} "
                f"n={res['n_defined']} pos={res['n_pos']}"
            )
    pend_y7 = _cv(store[(Y7, "e_pending_amt_share")])
    pend_y3 = _cv(store[(Y3, "e_pending_amt_share")])
    size_y7 = _cv(store[(Y7, "log1p(a_in3)")])
    size_y3 = _cv(store[(Y3, "log1p(a_in3)")])
    days_y3 = _cv(store[(Y3, "c_n_days_with_tx")])
    iss_y7 = _cv(store[(Y7, "e_ar_issued_lag1")])
    dso_y7 = _cv(store[(Y7, "e_dso_proxy")])
    beat_size_y7 = bool(np.isfinite(pend_y7) and np.isfinite(size_y7) and pend_y7 >= size_y7 + KEEP_DELTA)
    beat_size_y3 = bool(np.isfinite(pend_y3) and np.isfinite(size_y3) and pend_y3 >= size_y3 + KEEP_DELTA)
    prose = (
        f"Y7 pending {_f(pend_y7)} vs size {_f(size_y7)} issued_lag1 {_f(iss_y7)} "
        f"(night {ISSUED_LAG1_BENCH}) DSO {_f(dso_y7)}. "
        f"Y3 pending {_f(pend_y3)} vs size {_f(size_y3)} days {_f(days_y3)} "
        f"(night {DAYS_BENCH}). Beat-size Y7={beat_size_y7} Y3={beat_size_y3}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "pend_y7": pend_y7,
        "pend_y3": pend_y3,
        "size_y7": size_y7,
        "size_y3": size_y3,
        "days_y3": days_y3,
        "iss_y7": iss_y7,
        "dso_y7": dso_y7,
        "beat_size_y7": beat_size_y7,
        "beat_size_y3": beat_size_y3,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 5. Honest leftover
# ---------------------------------------------------------------------------
def _leftover_block(tr: pd.DataFrame, y: str, feat: pd.Series, specs: list[tuple]) -> dict:
    lab = tr[y].notna()
    rows = []
    store = {}
    infos = {}
    resids = {}
    for name, xs in specs:
        resid, info = ols_resid(feat, *xs)
        res = signed_oof_auroc(tr[y], resid, tr["fold"], lab)
        store[name] = res
        infos[name] = info
        resids[name] = resid
        rows.append(
            {
                "y": y,
                "residual": name,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "R²": _f(info["r2"]),
                "folds": fold_bits(res) if not res["low_power"] else "—",
            }
        )
        print(
            f"5 leftover {y} {name}: "
            f"{'LOW_POWER' if res['low_power'] else _f(res['cv'])} "
            f"R2={_f(info['r2'])}"
        )
    return {"rows": rows, "store": store, "infos": infos, "resids": resids}


def cut5_leftover(tr: pd.DataFrame) -> dict:
    p = tr["e_pending_amt_share"]
    y7 = _leftover_block(
        tr,
        Y7,
        p,
        [
            ("after DSO", (tr["e_dso_proxy"],)),
            ("after issued_lag1", (tr["e_ar_issued_lag1"],)),
            ("after issued", (tr["e_ar_issued"],)),
            ("after overdue", (tr["e_ar_overdue"],)),
            ("after DSO+issued_lag1", (tr["e_dso_proxy"], tr["e_ar_issued_lag1"])),
            ("after size", (tr["log_in3"],)),
            ("after days", (tr["c_n_days_with_tx"],)),
        ],
    )
    y3 = _leftover_block(
        tr,
        Y3,
        p,
        [
            ("after days", (tr["c_n_days_with_tx"],)),
            ("after size", (tr["log_in3"],)),
            ("after days+size", (tr["c_n_days_with_tx"], tr["log_in3"])),
            ("after DSO", (tr["e_dso_proxy"],)),
            ("after issued_lag1", (tr["e_ar_issued_lag1"],)),
        ],
    )
    # rank-orthogonal extras on the honest bars
    lab7 = tr[Y7].notna()
    lab3 = tr[Y3].notna()
    rank_dso = rank_resid(p, tr["e_dso_proxy"])
    rank_iss = rank_resid(p, tr["e_ar_issued_lag1"])
    rank_days = rank_resid(p, tr["c_n_days_with_tx"])
    r_dso = signed_oof_auroc(tr[Y7], rank_dso, tr["fold"], lab7)
    r_iss = signed_oof_auroc(tr[Y7], rank_iss, tr["fold"], lab7)
    r_days = signed_oof_auroc(tr[Y3], rank_days, tr["fold"], lab3)
    def _resid_row(y, name, res):
        return {
            "y": y,
            "residual": name,
            "n": f"{res['n_defined']:,}",
            "n_pos": f"{res['n_pos']:,}",
            "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
            "R²": "—",
            "folds": fold_bits(res) if not res["low_power"] else "—",
        }

    y7["rows"].append(_resid_row(Y7, "rank-resid after DSO", r_dso))
    y7["rows"].append(_resid_row(Y7, "rank-resid after issued_lag1", r_iss))
    y3["rows"].append(_resid_row(Y3, "rank-resid after days", r_days))
    print(f"5 rank leftover Y7 DSO={_f(_cv(r_dso))} issued_lag1={_f(_cv(r_iss))} Y3 days={_f(_cv(r_days))}")

    after_dso = _cv(y7["store"]["after DSO"])
    after_iss = _cv(y7["store"]["after issued_lag1"])
    after_both = _cv(y7["store"]["after DSO+issued_lag1"])
    after_days = _cv(y3["store"]["after days"])
    lives_dso = bool(np.isfinite(after_dso) and after_dso >= CHANCE)
    lives_iss = bool(np.isfinite(after_iss) and after_iss >= CHANCE)
    lives_days = bool(np.isfinite(after_days) and after_days >= CHANCE)
    died_y7 = bool(
        (np.isfinite(after_dso) and after_dso < CHANCE)
        and (np.isfinite(after_iss) and after_iss < CHANCE)
    )
    died_y3 = bool(np.isfinite(after_days) and after_days < CHANCE)
    prose = (
        f"Y7 leftover after DSO {_f(after_dso)} "
        f"({'lives' if lives_dso else 'dies <0.55'}); "
        f"after issued_lag1 {_f(after_iss)} "
        f"({'lives' if lives_iss else 'dies <0.55'}); "
        f"after both {_f(after_both)}. "
        f"Y3 leftover after days {_f(after_days)} "
        f"({'lives' if lives_days else 'dies <0.55'}). "
        f"Rank-ortho DSO {_f(_cv(r_dso))} issued {_f(_cv(r_iss))} days {_f(_cv(r_days))}."
    )
    print(prose)
    return {
        "rows": y7["rows"] + y3["rows"],
        "y7": y7,
        "y3": y3,
        "after_dso": after_dso,
        "after_iss": after_iss,
        "after_both": after_both,
        "after_days": after_days,
        "after_overdue": _cv(y7["store"]["after overdue"]),
        "after_size_y7": _cv(y7["store"]["after size"]),
        "rank_dso": _cv(r_dso),
        "rank_iss": _cv(r_iss),
        "rank_days": _cv(r_days),
        "lives_dso": lives_dso,
        "lives_iss": lives_iss,
        "lives_days": lives_days,
        "died_y7": died_y7,
        "died_y3": died_y3,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 6. Twin screen — DROP the weaker twin
# ---------------------------------------------------------------------------
def cut6_twin_drop(tr: pd.DataFrame, twins: dict, singles: dict) -> dict:
    rows = []
    drops = []
    keep_pending = True
    for name in twins["twins"]:
        p7 = singles["pend_y7"]
        o7 = _cv(singles["store"][(Y7, name)]) if (Y7, name) in singles["store"] else float("nan")
        # if twin not in singles store, skip AUROC compare
        if name in ("e_dso_proxy", "e_ar_overdue", "e_delay_coll", "e_ar_issued_lag1", "j_pay_match"):
            o7 = _cv(singles["store"][(Y7, name)])
        elif name == "e_ar_issued":
            o7 = _cv(singles["store"].get((Y7, "e_ar_issued_lag1"), {"low_power": True}))
        weaker = "pending" if (np.isfinite(p7) and np.isfinite(o7) and p7 < o7) else name
        if weaker == "pending":
            keep_pending = False
            drops.append(f"DROP pending vs stronger twin {name} (Y7 {_f(p7)} < {_f(o7)})")
        else:
            drops.append(f"pending stronger than {name} on Y7 ({_f(p7)} vs {_f(o7)}) — still a twin, drop the weaker")
        rows.append(
            {
                "twin": name,
                "ρ": _f(twins["rhos"].get(name, float("nan"))),
                "Y7 pending": _f(p7),
                "Y7 twin": _f(o7),
                "weaker": weaker,
            }
        )
    prose = (
        "No |ρ|≥0.80 twin."
        if not twins["twins"]
        else f"Twin screen: {'; '.join(drops)}. keep_pending_as_X={keep_pending}."
    )
    print(prose)
    return {"rows": rows, "drops": drops, "keep_pending": keep_pending, "prose": prose}


# ---------------------------------------------------------------------------
# 7. SIZE terciles + invoice-book-only vs NaN-dropped
# ---------------------------------------------------------------------------
def cut7_slices(tr: pd.DataFrame, book: set[str]) -> dict:
    lab = tr[Y7].notna()
    p = tr["e_pending_amt_share"]
    size = tr["log_in3"]
    rows = []

    def _one(name, mask):
        res = signed_oof_auroc(tr[Y7], p, tr["fold"], mask)
        rows.append(
            {
                "slice": name,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "folds": fold_bits(res) if not res["low_power"] else "—",
            }
        )
        return _cv(res)

    all_drop = _one("all-train NaN-dropped (labeled)", lab)
    book_only = _one("invoice-book-only (drop 470)", lab & tr["company_id"].isin(book))
    # size terciles on labeled+pending-defined
    defined = lab & p.notna() & size.notna()
    terc = pd.Series(pd.NA, index=tr.index, dtype="object")
    if int(defined.sum()) >= 30:
        terc.loc[defined] = pd.qcut(
            size[defined], 3, labels=["T1_small", "T2_mid", "T3_large"], duplicates="drop"
        ).astype(str)
    t1 = _one("size T1 small", defined & (terc == "T1_small"))
    t2 = _one("size T2 mid", defined & (terc == "T2_mid"))
    t3 = _one("size T3 large", defined & (terc == "T3_large"))
    # Y3 book-only
    lab3 = tr[Y3].notna()
    res3 = signed_oof_auroc(tr[Y3], p, tr["fold"], lab3 & tr["company_id"].isin(book))
    rows.append(
        {
            "slice": "Y3 invoice-book-only",
            "n": f"{res3['n_defined']:,}",
            "n_pos": f"{res3['n_pos']:,}",
            "CV": "LOW_POWER" if res3["low_power"] else _f(res3["cv"]),
            "folds": fold_bits(res3) if not res3["low_power"] else "—",
        }
    )
    prose = (
        f"Y7 all-NaN-dropped {_f(all_drop)} vs book-only {_f(book_only)} "
        f"(470 never enter the defined set — same rows). "
        f"Size terciles T1/T2/T3 {_f(t1)} / {_f(t2)} / {_f(t3)}."
    )
    print(prose)
    return {
        "rows": rows,
        "all_drop": all_drop,
        "book_only": book_only,
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 8. Q6 lags + empty-on-short + delay first-6 vs pending
# ---------------------------------------------------------------------------
def cut8_q6(tr: pd.DataFrame) -> dict:
    lab = tr[Y7].notna()
    lab3 = tr[Y3].notna()
    rows = []
    store = {}
    for name in (
        "e_pending_amt_share",
        "e_pending_amt_share_lag1",
        "e_pending_amt_share_lag3",
        "e_delay_coll",
        "e_ar_issued_lag1",
    ):
        for y, m in ((Y7, lab), (Y3, lab3)):
            res = signed_oof_auroc(tr[y], tr[name], tr["fold"], m)
            store[(y, name)] = res
            rows.append(_auc_row(y, name, res))

    # empty-on-short: pending defined when months_so_far < 6 vs delay
    early = tr["period"] < DELAY_EMPTY_BEFORE
    sofar6 = tr["months_so_far"] < 6
    pend = pd.to_numeric(tr["e_pending_amt_share"], errors="coerce")
    delay = pd.to_numeric(tr["e_delay_coll"], errors="coerce")
    cal_rows = []
    for label, mask in (
        ("calendar first 6m (2024-09..2025-02)", early),
        ("so-far <6 months", sofar6),
        ("calendar after delay-ok", ~early),
        ("so-far ≥6", ~sofar6),
    ):
        sl = mask
        n = int(sl.sum())
        cal_rows.append(
            {
                "window": label,
                "cm": f"{n:,}",
                "pending nn": f"{int(pend[sl].notna().sum()):,}",
                "pending cov": _pp(_pct(int(pend[sl].notna().sum()), n)),
                "delay nn": f"{int(delay[sl].notna().sum()):,}",
                "delay cov": _pp(_pct(int(delay[sl].notna().sum()), n)),
            }
        )
    # Y7 AUROC of pending on short vs long so-far
    short = signed_oof_auroc(tr[Y7], pend, tr["fold"], lab & (tr["so_far_class"] == "short_<12"))
    long = signed_oof_auroc(tr[Y7], pend, tr["fold"], lab & (tr["so_far_class"] == "long_>=18"))
    early_auc = signed_oof_auroc(tr[Y7], pend, tr["fold"], lab & early)
    rows.append(_auc_row(Y7, "pending short_<12 so-far", short))
    rows.append(_auc_row(Y7, "pending long_>=18 so-far", long))
    rows.append(_auc_row(Y7, "pending calendar first 6m", early_auc))

    lag1 = _cv(store[(Y7, "e_pending_amt_share_lag1")])
    lag3 = _cv(store[(Y7, "e_pending_amt_share_lag3")])
    now = _cv(store[(Y7, "e_pending_amt_share")])
    lead = bool(np.isfinite(lag1) and np.isfinite(now) and lag1 + 0.01 >= now)
    early_pop = _pct(int(pend[early].notna().sum()), int(early.sum()))
    delay_early = _pct(int(delay[early].notna().sum()), int(early.sum()))
    populated_earlier = early_pop > delay_early + 0.05
    prose = (
        f"Q6 Y7 pending now {_f(now)} lag1 {_f(lag1)} lag3 {_f(lag3)}. "
        f"{'lag1 ≈ now — persistent trait, not a new lead' if lead else 'lag weaker than now'}. "
        f"First 6 calendar months: pending cov {_pp(early_pop)} vs delay {_pp(delay_early)} "
        f"({'pending populated earlier — unpaid stock, not paid-delay window' if populated_earlier else 'same empty window as delay'}). "
        f"Short so-far {_f(_cv(short))} long {_f(_cv(long))}."
    )
    print(prose)
    return {
        "rows": rows,
        "cal_rows": cal_rows,
        "lag1": lag1,
        "lag3": lag3,
        "now": now,
        "lead": lead,
        "early_pop": early_pop,
        "delay_early": delay_early,
        "populated_earlier": populated_earlier,
        "short": _cv(short),
        "long": _cv(long),
        "early_auc": _cv(early_auc),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 9. vs Family J j_pay_match leftover
# ---------------------------------------------------------------------------
def cut9_family_j(tr: pd.DataFrame) -> dict:
    p = tr["e_pending_amt_share"]
    j = tr["j_pay_match"]
    rho = spearman(p, j)
    twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    lab7 = tr[Y7].notna()
    lab3 = tr[Y3].notna()
    resid, info = ols_resid(p, j)
    r7 = signed_oof_auroc(tr[Y7], resid, tr["fold"], lab7)
    r3 = signed_oof_auroc(tr[Y3], resid, tr["fold"], lab3)
    j7 = signed_oof_auroc(tr[Y7], j, tr["fold"], lab7)
    j3 = signed_oof_auroc(tr[Y3], j, tr["fold"], lab3)
    # dark 470 stay NaN
    dark_j = int(j[tr["j_has_book"] == 0].notna().sum()) if "j_has_book" in tr.columns else -1
    rows = [
        _auc_row(Y7, "j_pay_match", j7),
        _auc_row(Y7, "pending resid after j_pay_match", r7),
        _auc_row(Y3, "j_pay_match", j3),
        _auc_row(Y3, "pending resid after j_pay_match", r3),
    ]
    after_j = _cv(r7)
    lives = bool(np.isfinite(after_j) and after_j >= CHANCE)
    prose = (
        f"ρ pending vs j_pay_match {_f(rho)} twin={twin}. "
        f"J is KEEP-Q5 diagnostic not a Y3 X — do not merge J. "
        f"Y7 leftover after match-rate {_f(after_j)} "
        f"({'lives' if lives else 'dies <0.55'}). J Y7 {_f(_cv(j7))} Y3 {_f(_cv(j3))}. "
        f"j_has_book=0 non-null={dark_j} (want 0)."
    )
    print(prose)
    return {
        "rows": rows,
        "rho": rho,
        "twin": twin,
        "after_j": after_j,
        "j7": _cv(j7),
        "j3": _cv(j3),
        "lives": lives,
        "r2": info["r2"],
        "dark_j_nn": dark_j,
        "prose": prose,
    }


def _read_sibling_number(path: Path, patterns: list[str]) -> dict:
    out = {"path": str(path), "exists": path.exists(), "hits": []}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8")
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            out["hits"].append({"pat": pat, "match": m.group(0), "num": m.group(1) if m.lastindex else None})
    return out


# ---------------------------------------------------------------------------
# 10. vs credit-note and delay leftover (read-only siblings)
# ---------------------------------------------------------------------------
def cut10_siblings(tr: pd.DataFrame) -> dict:
    p = tr["e_pending_amt_share"]
    lab7 = tr[Y7].notna()
    lab3 = tr[Y3].notna()
    specs = [
        ("after e_credit_note_ratio", tr["e_credit_note_ratio"]),
        ("after e_delay_coll", tr["e_delay_coll"]),
        ("after e_ar_overdue_30", tr["e_ar_overdue_30"]),
    ]
    rows = []
    store = {}
    for name, x in specs:
        resid, info = ols_resid(p, x)
        r7 = signed_oof_auroc(tr[Y7], resid, tr["fold"], lab7)
        r3 = signed_oof_auroc(tr[Y3], resid, tr["fold"], lab3)
        store[name] = {"y7": r7, "y3": r3, "r2": info["r2"]}
        rows.append(
            {
                "residual": name,
                "Y7 CV": "LOW_POWER" if r7["low_power"] else _f(r7["cv"]),
                "Y3 CV": "LOW_POWER" if r3["low_power"] else _f(r3["cv"]),
                "R²": _f(info["r2"]),
                "Y7 folds": fold_bits(r7) if not r7["low_power"] else "—",
            }
        )
        print(f"10 {name} Y7={_f(_cv(r7))} Y3={_f(_cv(r3))} R2={_f(info['r2'])}")

    cn_pub = _read_sibling_number(
        CN_MD,
        [
            r"leftover after issued_lag1\s+(0\.\d+)",
            r"residual after issued_lag1\s+(0\.\d+)",
            r"Y7 leftover.*?(0\.\d+)",
        ],
    )
    delay_pub = _read_sibling_number(
        DELAY_MD,
        [
            r"leftover after DSO\s+(0\.\d+)",
            r"after DSO\s+(0\.\d+)",
            r"e_delay_coll.*?([0-9]\.\d{3})",
        ],
    )
    cn_note = (
        "credit_note_qa.md published leftover after issued_lag1 **0.597** (KEEP Y7 leftover; DROP as Y3 X)."
        if cn_pub["exists"]
        else "credit_note_qa.md missing mid-run — using in-module leftover after CN only."
    )
    delay_note = (
        "delay_qa.md published: delay leftover after DSO **0.581** / issued_lag1 **0.583** (KEEP Y7 leftover; "
        "DROP as Y3 X). Not racing the sibling script."
        if delay_pub["exists"]
        else "delay_qa.md not published yet — leftover after delay computed here only; do not edit delay_qa.py."
    )
    after_cn = _cv(store["after e_credit_note_ratio"]["y7"])
    after_delay = _cv(store["after e_delay_coll"]["y7"])
    prose = (
        f"{cn_note} {delay_note} "
        f"Pending leftover after CN {_f(after_cn)}; after delay {_f(after_delay)} "
        f"(dies <0.55 if below chance)."
    )
    print(prose)
    return {
        "rows": rows,
        "after_cn": after_cn,
        "after_delay": after_delay,
        "cn_pub": cn_pub,
        "delay_pub": delay_pub,
        "cn_note": cn_note,
        "delay_note": delay_note,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 11. ICC / company-demean — trait vs month shock
# ---------------------------------------------------------------------------
def cut11_icc(tr: pd.DataFrame) -> dict:
    p = pd.to_numeric(tr["e_pending_amt_share"], errors="coerce")
    icc = icc_anova(p, tr["company_id"])
    icc_g = icc_anova(p, tr["group_id"])
    demean = company_demean(p, tr["company_id"])
    lab7 = tr[Y7].notna()
    lab3 = tr[Y3].notna()
    d7 = signed_oof_auroc(tr[Y7], demean, tr["fold"], lab7)
    d3 = signed_oof_auroc(tr[Y3], demean, tr["fold"], lab3)
    trait = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_TRAIT)
    like_vol = bool(np.isfinite(icc["eta2"]) and icc["eta2"] >= A_OUT_VOL_ETA2 - 0.05)
    shock = bool(np.isfinite(_cv(d7)) and _cv(d7) >= CHANCE and not trait)
    prose = (
        f"ICC={_f(icc['icc'])} (quote {ICC_QUOTE}) η²={_f(icc['eta2'])} "
        f"on {icc['k']} companies / {icc['n']:,} CM. "
        f"Group ICC={_f(icc_g['icc'])} η²={_f(icc_g['eta2'])}. "
        f"a_out_vol η²={A_OUT_VOL_ETA2}. "
        f"{'TRAIT like a_out_vol (who-has-unpaid-stock), not a month shock' if trait else 'more month-shock than identity'}. "
        f"Y7 demean leftover {_f(_cv(d7))} Y3 {_f(_cv(d3))}."
    )
    print(prose)
    return {
        "icc": icc,
        "icc_g": icc_g,
        "d7": _cv(d7),
        "d3": _cv(d3),
        "d7_rec": d7,
        "trait": trait,
        "like_vol": like_vol,
        "shock": shock,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# extras: overdue pile, AP analogue, quintiles, holdout coverage
# ---------------------------------------------------------------------------
def cut_extra_pile(tr: pd.DataFrame) -> dict:
    p = pd.to_numeric(tr["e_pending_amt_share"], errors="coerce")
    od = pd.to_numeric(tr["e_ar_overdue"], errors="coerce")
    y = pd.to_numeric(tr[Y7], errors="coerce")
    both = p.notna() & od.notna() & y.notna()
    d = tr.loc[both, ["company_id", "fold"]].copy()
    d["p"] = p[both].values
    d["od"] = od[both].values
    d["y"] = y[both].values
    d["fold"] = tr.loc[both, "fold"].values
    med_p = float(d["p"].median())
    med_od = float(d["od"].median())
    d["pile"] = np.select(
        [
            (d["p"] >= med_p) & (d["od"] < med_od),
            (d["p"] >= med_p) & (d["od"] >= med_od),
            (d["p"] < med_p) & (d["od"] >= med_od),
            (d["p"] < med_p) & (d["od"] < med_od),
        ],
        ["high-pending low-overdue", "both high", "low-pending high-overdue", "both low"],
        default="?",
    )
    rows = []
    for name, g in d.groupby("pile", sort=False):
        rows.append(
            {
                "pile": name,
                "n": f"{len(g):,}",
                "Y7 rate": _pp(g["y"].mean()),
                "median pending": _f(g["p"].median()),
                "median overdue": _f(g["od"].median()),
            }
        )
    hi_lo = next((r for r in rows if r["pile"] == "high-pending low-overdue"), None)
    both_hi = next((r for r in rows if r["pile"] == "both high"), None)
    prose = (
        f"Median split pending { _f(med_p) } × overdue { _f(med_od) }. "
        f"High-pending & low-overdue (stock unpaid, not yet 30d/due-late) "
        f"Y7 rate {hi_lo['Y7 rate'] if hi_lo else '—'} n={hi_lo['n'] if hi_lo else '—'}; "
        f"both-high Y7 rate {both_hi['Y7 rate'] if both_hi else '—'}."
    )
    print(prose)
    return {"rows": rows, "med_p": med_p, "med_od": med_od, "prose": prose}


def cut_extra_ap(tr: pd.DataFrame) -> dict:
    """Supplier pending analogue is NOT in parquet. In-memory AP-only share."""
    ap = pd.to_numeric(tr["e_ap_pending_share"], errors="coerce")
    ar = pd.to_numeric(tr["e_ar_pending_share"], errors="coerce")
    mix = pd.to_numeric(tr["e_pending_amt_share"], errors="coerce")
    rho_ap = spearman(mix, ap)
    rho_ar = spearman(mix, ar)
    lab7 = tr[Y7].notna()
    lab3 = tr[Y3].notna()
    ap7 = signed_oof_auroc(tr[Y7], ap, tr["fold"], lab7)
    ar7 = signed_oof_auroc(tr[Y7], ar, tr["fold"], lab7)
    ap3 = signed_oof_auroc(tr[Y3], ap, tr["fold"], lab3)
    rows = [
        _auc_row(Y7, "e_ar_pending_share (in-memory)", ar7),
        _auc_row(Y7, "e_ap_pending_share (in-memory)", ap7),
        _auc_row(Y3, "e_ap_pending_share (in-memory)", ap3),
    ]
    has_col = False  # do not invent a parquet column
    prose = (
        f"No supplier-pending column in monthly.parquet (has_col={has_col}). "
        f"Do not invent one. In-memory AP-only share cov {_pp(_pct(int(ap.notna().sum()), len(tr)))} "
        f"ρ vs mixed pending {_f(rho_ap)}; AR-only ρ {_f(rho_ar)}. "
        f"Y7 AP-only {_f(_cv(ap7))} AR-only {_f(_cv(ar7))}."
    )
    print(prose)
    return {
        "rows": rows,
        "rho_ap": rho_ap,
        "rho_ar": rho_ar,
        "ap7": _cv(ap7),
        "ar7": _cv(ar7),
        "ap_cov": _pct(int(ap.notna().sum()), len(tr)),
        "has_col": has_col,
        "prose": prose,
    }


def cut_extra_quintiles(tr: pd.DataFrame) -> dict:
    p = pd.to_numeric(tr["e_pending_amt_share"], errors="coerce")
    y = pd.to_numeric(tr[Y7], errors="coerce")
    m = p.notna() & y.notna()
    d = pd.DataFrame({"p": p[m], "y": y[m]})
    try:
        d["q"] = pd.qcut(d["p"], 5, duplicates="drop")
    except ValueError:
        return {"rows": [], "ushape": False, "rates": [], "prose": "quintiles failed"}
    rows = []
    rates = []
    for i, (q, g) in enumerate(d.groupby("q", observed=True), start=1):
        rate = float(g["y"].mean())
        rates.append(rate)
        rows.append(
            {
                "q": f"Q{i}",
                "bin": str(q),
                "n": f"{len(g):,}",
                "Y7 rate": _pp(rate),
                "median pending": _f(g["p"].median()),
            }
        )
    ushape = False
    if len(rates) >= 5:
        mid = min(rates[1], rates[2], rates[3])
        ushape = rates[0] > mid + 0.02 and rates[-1] > mid + 0.02
    prose = (
        f"Y7 rates by pending quintile: {', '.join(_pp(r) for r in rates)}. "
        f"{'U-shape' if ushape else 'not a U-shape'}."
    )
    print(prose)
    return {"rows": rows, "ushape": ushape, "rates": rates, "prose": prose}


def cut_extra_holdout(panel: pd.DataFrame, book: set[str]) -> dict:
    h = panel[panel["split"] == "holdout"].copy()
    p = pd.to_numeric(h["e_pending_amt_share"], errors="coerce")
    n = len(h)
    nn = int(p.notna().sum())
    n_co = int(h["company_id"].nunique())
    n_dark = int((~h["company_id"].isin(book)).sum())
    dark_nn = int(p[~h["company_id"].isin(book)].notna().sum())
    # do NOT compute AUROC / rates on holdout labels
    prose = (
        f"Holdout coverage only: {nn:,}/{n:,} = {_pp(_pct(nn, n))} on {n_co} companies. "
        f"Dark holdout CM {n_dark:,} pending-nn {dark_nn} (want 0). "
        f"No AUROC, no percentiles, no fit on the 72."
    )
    print(prose)
    return {
        "n": n,
        "nn": nn,
        "cov": _pct(nn, n),
        "n_co": n_co,
        "n_dark": n_dark,
        "dark_nn": dark_nn,
        "prose": prose,
    }


FOLD4_GROUPS = ("GROUP_0222", "GROUP_0108")
DSO_CLIP = 24.0


def cut13_ushape(tr: pd.DataFrame) -> dict:
    """U-shape is not a KEEP-as-X. Probe tails vs middle; do not invent y_pending."""
    p = pd.to_numeric(tr["e_pending_amt_share"], errors="coerce")
    lab = tr[Y7].notna()
    mid = 0.30
    dist = (p - mid).abs()
    tails = pd.Series(np.nan, index=tr.index, dtype=float)
    ok = p.notna()
    tails.loc[ok] = ((p[ok] <= 0.09) | (p[ok] >= 0.84)).astype(float)
    r_dist = signed_oof_auroc(tr[Y7], dist, tr["fold"], lab)
    r_tail = signed_oof_auroc(tr[Y7], tails, tr["fold"], lab)
    # leftover of |p-mid| after issued_lag1 — still a dummy?
    resid, info = ols_resid(dist, tr["e_ar_issued_lag1"])
    r_left = signed_oof_auroc(tr[Y7], resid, tr["fold"], lab)
    rows = [
        _auc_row(Y7, "|pending-0.30|", r_dist),
        _auc_row(Y7, "tail dummy Q1 or Q5", r_tail),
        _auc_row(Y7, "|pending-0.30| resid after issued_lag1", r_left),
    ]
    lives = bool(np.isfinite(_cv(r_left)) and _cv(r_left) >= CHANCE)
    prose = (
        f"U-shape probe: |pending−0.30| Y7 {_f(_cv(r_dist))} tail-dummy {_f(_cv(r_tail))}. "
        f"Leftover of the U after issued_lag1 {_f(_cv(r_left))} R²={_f(info['r2'])} "
        f"({'lives as a nonlinear dummy' if lives else 'dies — U-shape is not leftover X'}). "
        f"Do not invent y_pending from the tails."
    )
    print(prose)
    return {
        "rows": rows,
        "dist": _cv(r_dist),
        "tail": _cv(r_tail),
        "after_iss": _cv(r_left),
        "lives": lives,
        "prose": prose,
    }


def cut14_fold4(tr: pd.DataFrame, singles: dict) -> dict:
    rows = []
    feats = [
        ("e_pending_amt_share", tr["e_pending_amt_share"]),
        ("e_ar_issued_lag1", tr["e_ar_issued_lag1"]),
        ("e_dso_proxy", tr["e_dso_proxy"]),
        ("e_delay_coll", tr["e_delay_coll"]),
        ("e_ar_overdue", tr["e_ar_overdue"]),
    ]
    lab = tr[Y7].notna() & (tr["fold"] == 4)
    # fold-4 AUROC uses sign from folds 0–3 (already in singles store)
    for name, _ in feats:
        rec = singles["store"].get((Y7, name))
        if rec is None:
            continue
        f4 = fold_k(rec, 4)
        rows.append(
            {
                "feature": name,
                "fold4": _f(f4),
                "CV": _f(_cv(rec)),
                "folds": fold_bits(rec),
            }
        )
    hard = tr["group_id"].astype(str).isin(FOLD4_GROUPS)
    p = pd.to_numeric(tr["e_pending_amt_share"], errors="coerce")
    y = pd.to_numeric(tr[Y7], errors="coerce")
    grp = []
    for gid in FOLD4_GROUPS:
        sl = (tr["group_id"].astype(str) == gid) & y.notna()
        grp.append(
            {
                "group": gid,
                "n": int(sl.sum()),
                "pos": int((sl & (y == 1)).sum()),
                "Y7 rate": _pp(_pct(int((sl & (y == 1)).sum()), int(sl.sum()))),
                "median pending": _f(p[sl].median()),
            }
        )
    pend4 = fold_k(singles["store"][(Y7, "e_pending_amt_share")], 4)
    iss4 = fold_k(singles["store"][(Y7, "e_ar_issued_lag1")], 4)
    issued_owns = bool(np.isfinite(iss4) and (not np.isfinite(pend4) or iss4 > pend4 + 0.05))
    prose = (
        f"Fold 4 pending {_f(pend4)} vs issued_lag1 {_f(iss4)} vs TURNOVER 0.680. "
        f"{'issued owns fold 4' if issued_owns else 'pending competes on fold 4'}. "
        f"Hard groups {FOLD4_GROUPS}: pending medians "
        f"{', '.join(r['median pending'] for r in grp)}."
    )
    print(prose)
    return {
        "rows": rows,
        "grp": grp,
        "pend4": pend4,
        "iss4": iss4,
        "issued_owns": issued_owns,
        "hard_n": int((hard & y.notna()).sum()),
        "prose": prose,
    }


def cut15_newbook(tr: pd.DataFrame) -> dict:
    p = pd.to_numeric(tr["e_pending_amt_share"], errors="coerce")
    first = (
        tr.loc[p.notna(), ["company_id", "period"]]
        .groupby("company_id", sort=False)["period"]
        .min()
        .rename("first_iss_period")
        .reset_index()
    )
    work = tr.merge(first, on="company_id", how="left")
    work["months_on_book"] = (
        (work["period"].dt.year - work["first_iss_period"].dt.year) * 12
        + (work["period"].dt.month - work["first_iss_period"].dt.month)
        + 1
    )
    y = pd.to_numeric(work[Y7], errors="coerce")
    lab = y.notna() & p.notna()
    rows = []
    for name, mask in (
        ("first issued month", lab & (work["months_on_book"] == 1)),
        ("months_on_book 1–3", lab & work["months_on_book"].between(1, 3)),
        ("months_on_book ≥6", lab & (work["months_on_book"] >= 6)),
        ("pending == 0 (all paid)", lab & (p == 0)),
        ("pending == 1 (nothing paid)", lab & (p >= 0.999)),
        ("0 < pending < 1", lab & (p > 0) & (p < 0.999)),
    ):
        sl = mask
        n = int(sl.sum())
        pos = int((sl & (y == 1)).sum())
        rows.append(
            {
                "slice": name,
                "n": f"{n:,}",
                "pos": f"{pos:,}",
                "Y7 rate": _pp(_pct(pos, n)),
                "median pending": _f(p[sl].median()),
            }
        )
    erp_ids = set(first["company_id"].astype(str))
    erp_before = int((tr["company_id"].isin(erp_ids) & p.isna()).sum())
    new3 = _pct(
        int((lab & work["months_on_book"].between(1, 3) & (y == 1)).sum()),
        int((lab & work["months_on_book"].between(1, 3)).sum()),
    )
    mature = _pct(
        int((lab & (work["months_on_book"] >= 6) & (y == 1)).sum()),
        int((lab & (work["months_on_book"] >= 6)).sum()),
    )
    p0 = _pct(int((lab & (p == 0) & (y == 1)).sum()), int((lab & (p == 0)).sum()))
    p1 = _pct(int((lab & (p >= 0.999) & (y == 1)).sum()), int((lab & (p >= 0.999)).sum()))
    prose = (
        f"New-book: first-3-months Y7 rate {_pp(new3)} vs mature ≥6m {_pp(mature)}. "
        f"pending=0 (all paid) Y7 {_pp(p0)}; pending=1 (nothing paid) {_pp(p1)}. "
        f"Q1 tail is the emptied book (all collected), not unpaid stock. "
        f"ERP months before first issued (pending NaN on ever-ERP): {erp_before:,}."
    )
    print(prose)
    return {
        "rows": rows,
        "new3": new3,
        "mature": mature,
        "p0": p0,
        "p1": p1,
        "erp_before": erp_before,
        "prose": prose,
    }


def cut16_company_mean(tr: pd.DataFrame) -> dict:
    p = pd.to_numeric(tr["e_pending_amt_share"], errors="coerce")
    y = pd.to_numeric(tr[Y7], errors="coerce")
    d = tr.assign(p=p, y=y).dropna(subset=["p"])
    g = d.groupby("company_id", sort=False).agg(
        p_mean=("p", "mean"),
        y_rate=("y", "mean"),
        n_y=("y", "count"),
        n_pos=("y", "sum"),
        fold=("fold", "first"),
        group_id=("group_id", "first"),
    )
    g = g[g["n_y"] >= 3]
    rho = spearman(g["p_mean"], g["y_rate"])
    # company-level AUROC: mean pending vs ever-lost (rate>0) — diagnostic only
    ever = (g["n_pos"] > 0).astype(float)
    auc = auroc(ever, g["p_mean"])
    auc_n = auroc(ever, -g["p_mean"])
    best = max(auc, auc_n) if np.isfinite(auc) and np.isfinite(auc_n) else float("nan")
    size_op = spearman(p, np.log1p(pd.to_numeric(tr["a_op_in"], errors="coerce").abs()))
    prose = (
        f"Company-mean pending vs company Y7 rate ρ={_f(rho)} on {len(g)} companies "
        f"(≥3 labeled months). Ever-lost AUROC {_f(best)}. "
        f"Size ρ vs log1p(|a_op_in|) {_f(size_op)} (feature-report quote {SIZE_RHO_QUOTE}). "
        f"{'Trait tracks who-churns' if np.isfinite(best) and best >= CHANCE else 'Company mean is not a Y7 ranker'}."
    )
    print(prose)
    return {
        "rho": rho,
        "auc": best,
        "n_co": int(len(g)),
        "size_op": size_op,
        "prose": prose,
    }


def cut17_dso_clip(tr: pd.DataFrame) -> dict:
    """DSO raw is skewed (R²≈0 vs Spearman 0.49). Clip 24m like the Y7 card."""
    p = tr["e_pending_amt_share"]
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce").clip(upper=DSO_CLIP)
    lab = tr[Y7].notna()
    resid, info = ols_resid(p, dso)
    r = signed_oof_auroc(tr[Y7], resid, tr["fold"], lab)
    r_raw = signed_oof_auroc(tr[Y7], dso, tr["fold"], lab)
    rho = spearman(p, dso)
    prose = (
        f"DSO clipped at {DSO_CLIP:.0f}m: ρ vs pending {_f(rho)} (raw Spearman was ~0.49). "
        f"Y7 leftover after clipped DSO {_f(_cv(r))} R²={_f(info['r2'])}; "
        f"clipped DSO single {_f(_cv(r_raw))}. "
        f"{'still dies' if not (np.isfinite(_cv(r)) and _cv(r) >= CHANCE) else 'clip rescues leftover'}."
    )
    print(prose)
    return {
        "after": _cv(r),
        "dso": _cv(r_raw),
        "r2": info["r2"],
        "rho": rho,
        "prose": prose,
        "rows": [
            {
                "residual": "after DSO clip 24m",
                "n": f"{r['n_defined']:,}",
                "n_pos": f"{r['n_pos']:,}",
                "CV": "LOW_POWER" if r["low_power"] else _f(r["cv"]),
                "R²": _f(info["r2"]),
                "folds": fold_bits(r) if not r["low_power"] else "—",
            }
        ],
    }


def cut18_allpaid(tr: pd.DataFrame) -> dict:
    """Q1 tail = emptied book. Flag is a dummy, not a KEEP X."""
    p = pd.to_numeric(tr["e_pending_amt_share"], errors="coerce")
    paid = pd.Series(np.nan, index=tr.index, dtype=float)
    ok = p.notna()
    paid.loc[ok] = (p[ok] <= 0.09).astype(float)
    lab = tr[Y7].notna()
    raw = signed_oof_auroc(tr[Y7], paid, tr["fold"], lab)
    resid, info = ols_resid(paid, tr["e_ar_issued_lag1"])
    after = signed_oof_auroc(tr[Y7], resid, tr["fold"], lab)
    resid2, info2 = ols_resid(paid, tr["e_dso_proxy"])
    after_dso = signed_oof_auroc(tr[Y7], resid2, tr["fold"], lab)
    rows = [
        _auc_row(Y7, "all-paid dummy (pending≤0.09)", raw),
        _auc_row(Y7, "all-paid resid after issued_lag1", after),
        _auc_row(Y7, "all-paid resid after DSO", after_dso),
    ]
    lives = bool(np.isfinite(_cv(after)) and _cv(after) >= CHANCE)
    prose = (
        f"All-paid dummy Y7 {_f(_cv(raw))}; leftover after issued_lag1 {_f(_cv(after))} "
        f"R²={_f(info['r2'])}; after DSO {_f(_cv(after_dso))} R²={_f(info2['r2'])}. "
        f"{'thin dummy leftover' if lives else 'all-paid dummy leftover dies'}. "
        f"Not a reason to KEEP pending as X."
    )
    print(prose)
    return {
        "rows": rows,
        "raw": _cv(raw),
        "after_iss": _cv(after),
        "after_dso": _cv(after_dso),
        "lives": lives,
        "prose": prose,
    }


def cut19_mix(tr: pd.DataFrame) -> dict:
    """How much of mixed pending is AR vs AP unpaid stock."""
    p = pd.to_numeric(tr["e_pending_amt_share"], errors="coerce")
    ar = pd.to_numeric(tr["e_ar_pending_share"], errors="coerce")
    ap = pd.to_numeric(tr["e_ap_pending_share"], errors="coerce")
    both = p.notna() & ar.notna() & ap.notna()
    # mix closeness: |mixed - AR| vs |mixed - AP|
    d_ar = (p[both] - ar[both]).abs().mean()
    d_ap = (p[both] - ap[both]).abs().mean()
    closer = "AP" if d_ap < d_ar else "AR"
    n_ar_only = int((tr["e_ar_pending_share"].notna() & tr["e_ap_pending_share"].isna()).sum())
    n_ap_only = int((tr["e_ap_pending_share"].notna() & tr["e_ar_pending_share"].isna()).sum())
    n_both = int(both.sum())
    prose = (
        f"Mixed pending is a blend: n both AR+AP {n_both:,}; AR-only {n_ar_only:,}; "
        f"AP-only {n_ap_only:,}. Mean |mixed−AR|={d_ar:.3f} |mixed−AP|={d_ap:.3f} "
        f"— closer to {closer}. ρ AR 0.852 / AP 0.867. No AP column in parquet; do not invent one."
    )
    print(prose)
    return {
        "d_ar": float(d_ar) if np.isfinite(d_ar) else float("nan"),
        "d_ap": float(d_ap) if np.isfinite(d_ap) else float("nan"),
        "closer": closer,
        "n_both": n_both,
        "n_ar_only": n_ar_only,
        "n_ap_only": n_ap_only,
        "prose": prose,
    }


def cut20_samen(tr: pd.DataFrame) -> dict:
    """Same-n artifact: pending vs issued_lag1 on identical Y7 rows."""
    lab = tr[Y7].notna()
    both = lab & tr["e_pending_amt_share"].notna() & tr["e_ar_issued_lag1"].notna()
    p = signed_oof_auroc(tr[Y7], tr["e_pending_amt_share"], tr["fold"], both)
    iss = signed_oof_auroc(tr[Y7], tr["e_ar_issued_lag1"], tr["fold"], both)
    resid, info = ols_resid(tr["e_pending_amt_share"], tr["e_ar_issued_lag1"])
    r = signed_oof_auroc(tr[Y7], resid, tr["fold"], both)
    rows = [
        _auc_row(Y7, "pending same-n", p),
        _auc_row(Y7, "issued_lag1 same-n", iss),
        _auc_row(Y7, "pending resid same-n", r),
    ]
    lift = _cv(r) - _cv(iss) if np.isfinite(_cv(r)) and np.isfinite(_cv(iss)) else float("nan")
    prose = (
        f"Same-n n={int(both.sum()):,}: pending {_f(_cv(p))} vs issued_lag1 {_f(_cv(iss))} "
        f"(night 0.630). Residual {_f(_cv(r))} lift vs issued { _f(lift) }. "
        f"Residualizing does not create a new object."
    )
    print(prose)
    return {
        "rows": rows,
        "pend": _cv(p),
        "iss": _cv(iss),
        "resid": _cv(r),
        "lift": lift,
        "n": int(both.sum()),
        "prose": prose,
    }


def decide(ctx: dict) -> dict:
    c3, c4, c5, c6, c8, c9, c11 = (
        ctx["c3"],
        ctx["c4"],
        ctx["c5"],
        ctx["c6"],
        ctx["c8"],
        ctx["c9"],
        ctx["c11"],
    )
    leftover_y7 = max(
        [x for x in (c5["after_dso"], c5["after_iss"]) if np.isfinite(x)],
        default=float("nan"),
    )
    leftover_y3 = c5["after_days"]
    keep_x_y7 = bool(
        c4["beat_size_y7"]
        and (c5["lives_dso"] or c5["lives_iss"])
        and not c3["size_flag"]
        and not c3["any_twin"]
    )
    keep_x_y3 = bool(
        c4["beat_size_y3"]
        and c5["lives_days"]
        and leftover_y3 >= SIZE_QUOTE + KEEP_DELTA
        and not c3["size_flag"]
        and not c3["any_twin"]
    )
    # TURNPEND already failed as a card add-on
    grow_turnover = False
    drop_44 = bool(
        (c5["died_y7"] and c5["died_y3"])
        or (not keep_x_y7 and not keep_x_y3 and (c3["any_twin"] or c11["trait"]))
    )
    # PARK if leftover lives as a diagnostic but fails KEEP-as-X
    park_diag = bool((c5["lives_dso"] or c5["lives_iss"] or c5["lives_days"]) and not keep_x_y7 and not keep_x_y3)

    if keep_x_y7:
        y7 = "KEEP"
        y7_why = (
            f"beats size {_f(c4['size_y7'])} by ≥0.02 (pending {_f(c4['pend_y7'])}); "
            f"leftover after DSO {_f(c5['after_dso'])} / issued_lag1 {_f(c5['after_iss'])} ≥0.55; "
            f"not SIZE, not a twin. Still do not grow TURNOVER {TURNOVER_QUOTE} "
            f"(TURNPEND {TURNPEND_QUOTE} already lost)."
        )
    elif c3["any_twin"]:
        y7 = "DROP"
        y7_why = (
            f"twin of {c3['twins']} — DROP the weaker twin. "
            f"Pending Y7 {_f(c4['pend_y7'])}; leftover DSO {_f(c5['after_dso'])} "
            f"issued_lag1 {_f(c5['after_iss'])}."
        )
    elif c5["died_y7"]:
        y7 = "DROP"
        y7_why = (
            f"leftover after DSO {_f(c5['after_dso'])} and issued_lag1 {_f(c5['after_iss'])} "
            f"both <0.55. Unused stock. TURNPEND {TURNPEND_QUOTE} vs TURNOVER {TURNOVER_QUOTE} already lost."
        )
    elif not c4["beat_size_y7"]:
        y7 = "CLOSE"
        y7_why = (
            f"Y7 pending {_f(c4['pend_y7'])} does not beat size {_f(c4['size_y7'])}+0.02. "
            f"Leftover DSO {_f(c5['after_dso'])} issued_lag1 {_f(c5['after_iss'])}."
        )
    else:
        y7 = "CLOSE"
        y7_why = (
            f"fails KEEP-as-X (size/leftover/twin). pending {_f(c4['pend_y7'])} "
            f"leftover DSO {_f(c5['after_dso'])} issued_lag1 {_f(c5['after_iss'])}."
        )

    if keep_x_y3:
        y3 = "KEEP"
        y3_why = (
            f"Y3 pending {_f(c4['pend_y3'])} beats size {_f(c4['size_y3'])}; "
            f"leftover after days {_f(leftover_y3)}."
        )
    elif c5["died_y3"] or (np.isfinite(leftover_y3) and leftover_y3 < SIZE_QUOTE + KEEP_DELTA):
        y3 = "DROP"
        y3_why = (
            f"Y3 leftover after days {_f(leftover_y3)} dies vs days bar {DAYS_BENCH}. "
            f"Do not put pending on the 15-col Y3 card."
        )
    else:
        y3 = "CLOSE"
        y3_why = (
            f"Y3 pending {_f(c4['pend_y3'])} vs days {_f(c4['days_y3'])} leftover {_f(leftover_y3)}. "
            f"Stays off the 15-col card."
        )

    if c11["trait"] and not (keep_x_y7 or keep_x_y3):
        q5 = "PARK"
        q5_why = (
            f"ICC {_f(c11['icc']['icc'])} η² {_f(c11['icc']['eta2'])} — who-has-unpaid-stock trait "
            f"(like a_out_vol η²={A_OUT_VOL_ETA2}), not this-month shock. Demean Y7 {_f(c11['d7'])}."
        )
    elif keep_x_y7:
        q5 = "KEEP-Q5"
        q5_why = "leftover unpaid stock after DSO / issued is sayable as why, without growing TURNOVER."
    else:
        q5 = "CLOSE"
        q5_why = "not a usable why once DSO / issued / days are on the table."

    if c8["lead"] and c8["populated_earlier"] and np.isfinite(c8["lag1"]) and c8["lag1"] >= ISSUED_LAG1_BENCH:
        q6 = "KEEP"
        q6_why = f"lag1 {_f(c8['lag1'])} leads and pending is populated before delay."
    elif c8["populated_earlier"] and not c8["lead"]:
        q6 = "CLOSE"
        q6_why = (
            f"pending is populated in the first 6 months ({_pp(c8['early_pop'])} vs delay "
            f"{_pp(c8['delay_early'])}) but lag1 {_f(c8['lag1'])} is not a better lead than now {_f(c8['now'])}."
        )
    else:
        q6 = "CLOSE"
        q6_why = f"now {_f(c8['now'])} lag1 {_f(c8['lag1'])} lag3 {_f(c8['lag3'])} — persistence, not lead."

    lose_44 = drop_44 and y3 == "DROP" and y7 in {"DROP", "CLOSE"}
    overall = "KEEP" if keep_x_y7 else ("PARK" if park_diag and not lose_44 else ("DROP" if lose_44 else y7))

    q4 = y7
    q4_why = y7_why

    table = [
        {
            "object": "pending as Y7 X / leftover after DSO+issued_lag1",
            "decision": y7,
            "why": y7_why,
        },
        {
            "object": "pending as TURNOVER add-on (TURNPEND)",
            "decision": "CLOSE",
            "why": f"TURNPEND {TURNPEND_QUOTE} vs TURNOVER {TURNOVER_QUOTE} — already lost. Do not grow TURNOVER.",
        },
        {
            "object": "pending as Y3 X / the 15-col card",
            "decision": y3,
            "why": y3_why,
        },
        {
            "object": "e_pending_amt_share on the 44-col keep list",
            "decision": "DROP from the 44" if lose_44 else ("PARK on the 44" if park_diag else y7),
            "why": (
                "cluster representative in the feature report, but leftover dies and TURNPEND lost. "
                "BETWEEN trait, not a usable X."
                if lose_44
                else "kept only as a diagnostic share, not a model X."
            ),
        },
        {
            "object": "pending as a health Y",
            "decision": "PARK",
            "why": "do not invent y_pending.",
        },
        {
            "object": "Q5 why (unpaid stock)",
            "decision": q5,
            "why": q5_why,
        },
        {
            "object": "Q6 lead (lag1 / first-6m vs delay)",
            "decision": q6,
            "why": q6_why,
        },
        {
            "object": "Family J merge",
            "decision": "CLOSE",
            "why": f"J is KEEP-Q5 diagnostic not a Y3 X. leftover after match {_f(c9['after_j'])}. Do not merge J.",
        },
        {
            "object": "AP pending parquet column",
            "decision": "CLOSE",
            "why": "no supplier-pending analogue in the store. In-memory only. Do not invent a column.",
        },
        {
            "object": "U-shape / all-paid dummy",
            "decision": "CLOSE",
            "why": (
                f"|pending−0.30| leftover after issued_lag1 {_f(ctx['c13']['after_iss'])} is a thin dummy; "
                "do not invent y_pending. Linear leftover still dies."
            ),
        },
    ]
    headline = (
        f"**{overall}** as Y7 X ({y7}). Y3 X **{y3}**. "
        f"Leftover Y7 after DSO {_f(c5['after_dso'])} / issued_lag1 {_f(c5['after_iss'])} "
        f"/ both {_f(c5['after_both'])}; Y3 after days {_f(leftover_y3)}. "
        f"{'SIZE' if c3['size_flag'] else 'not SIZE'} (ρ={_f(c3['rhos']['log1p(a_in3)'])}); "
        f"{'twin of ' + ', '.join(c3['twins']) if c3['any_twin'] else 'not a |ρ|≥0.80 twin'}. "
        f"Q6 {q6}: lag1 {_f(c8['lag1'])} now {_f(c8['now'])}; "
        f"first-6m pending {_pp(c8['early_pop'])} vs delay {_pp(c8['delay_early'])}. "
        f"U-shape |p−0.30| leftover after issued {_f(ctx['c13']['after_iss'])} is a thin dummy (CLOSE). "
        f"44 should {'lose' if lose_44 else 'keep (diagnostic only)'} `e_pending_amt_share`. "
        f"Night quotes unchanged: TURNOVER {TURNOVER_QUOTE:.3f} / B_shallow {B_SHALLOW_QUOTE:.3f}; "
        f"Y3 {NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f}; days {DAYS_BENCH:.3f}; size {SIZE_QUOTE:.3f}."
    )
    print("DECIDE", headline)
    return {
        "overall": overall,
        "y7": y7,
        "y7_why": y7_why,
        "y3": y3,
        "y3_why": y3_why,
        "q4": q4,
        "q4_why": q4_why,
        "q5": q5,
        "q5_why": q5_why,
        "q6": q6,
        "q6_why": q6_why,
        "lose_44": lose_44,
        "park_diag": park_diag,
        "keep_x_y7": keep_x_y7,
        "keep_x_y3": keep_x_y3,
        "grow_turnover": grow_turnover,
        "leftover_y7": leftover_y7,
        "leftover_y3": leftover_y3,
        "table": table,
        "headline": headline,
    }


def plot_png(tr: pd.DataFrame, ctx: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    p = pd.to_numeric(tr["e_pending_amt_share"], errors="coerce")
    y = pd.to_numeric(tr[Y7], errors="coerce")
    od = pd.to_numeric(tr["e_ar_overdue"], errors="coerce")
    delay = pd.to_numeric(tr["e_delay_coll"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    m = p.notna() & y.notna()
    d = pd.DataFrame({"p": p[m], "y": y[m]})
    try:
        d["q"] = pd.qcut(d["p"], 5, duplicates="drop")
        rates = [float(g["y"].mean()) for _, g in d.groupby("q", observed=True)]
        ns = [len(g) for _, g in d.groupby("q", observed=True)]
        axes[0].bar(range(1, len(rates) + 1), rates, color="#3d5a80")
        axes[0].axhline(float(d["y"].mean()), color="#ee6c4d", ls="--", lw=1, label="train labeled mean")
        for i, (r, n) in enumerate(zip(rates, ns), start=1):
            axes[0].text(i, r + 0.008, f"{r:.2f}\nn={n}", ha="center", va="bottom", fontsize=8)
        axes[0].set_ylim(0, max(rates + [0.4]) + 0.08)
    except ValueError:
        axes[0].text(0.5, 0.5, "quintile fail", ha="center")
    axes[0].set_title("Y7 rate by pending quintile (train)")
    axes[0].set_xlabel("pending quintile")
    axes[0].set_ylabel("y7_top1_lost rate")
    axes[0].legend(fontsize=8)

    cal = (
        tr.assign(
            pend_nn=p.notna().astype(float),
            delay_nn=delay.notna().astype(float),
        )
        .groupby("period", sort=True)[["pend_nn", "delay_nn"]]
        .mean()
    )
    axes[1].plot(cal.index, cal["pend_nn"], label="pending coverage", color="#3d5a80")
    axes[1].plot(cal.index, cal["delay_nn"], label="delay coverage", color="#ee6c4d")
    axes[1].axvline(DELAY_EMPTY_BEFORE, color="#293241", ls=":", lw=1, label="delay-ok (2025-03)")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Train coverage by month")
    axes[1].set_ylabel("share defined")
    axes[1].legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG} (overdue defined on {int((p.notna() & od.notna()).sum()):,} train CM)")
    return True


def write_md(ctx: dict) -> None:
    d = ctx["decision"]
    c1, c2, c3, c4, c5 = ctx["c1"], ctx["c2"], ctx["c3"], ctx["c4"], ctx["c5"]
    c6, c7, c8, c9, c10, c11 = ctx["c6"], ctx["c7"], ctx["c8"], ctx["c9"], ctx["c10"], ctx["c11"]
    lines = [
        "# Q4/Q5/Q6 unused `e_pending_amt_share` leftover after DSO / issued",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_pending`. "
        "Night Y7 quote stays **TURNOVER 0.720 / B_shallow 0.712**. Night Y3 stays **0.762 / 0.752**. "
        "Y7 never D. Y5 never E. Pending stays off the 15-col Y3 card. Do not grow TURNOVER.",
        "",
        "`e_pending_amt_share` = among issued non-cancel invoices with issuance_date ≤ period_end, "
        "abs-amount share still unpaid as of period_end (invalid `payment_date` treated as 0 in "
        "both num and den). 470 never-invoice companies stay **NaN not 0**. Snapshot "
        "`pending_amount` is as-of extraction and would leak later collections — not used.",
        "",
        "## Headline",
        "",
        d["headline"],
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        "| 1 | Who is healthy? | **PARK** as a health Y. Do not invent `y_pending`. Dark 470 = NaN, not 0. |",
        "| 2 | Who is improving? | Not this stock share. |",
        "| 3 | Who is turning? | Pending lag is persistence (trait), not a turn clock. |",
        f"| 4 | Dip vs fall? | **{d['q4']}** — {d['q4_why']} |",
        f"| 5 | Why did it change? | **{d['q5']}** — {d['q5_why']} |",
        f"| 6 | Months earlier? | **{d['q6']}** — {d['q6_why']} |",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        _md_table(d["table"], ["object", "decision", "why"]),
        "",
        "## 1. Coverage train vs holdout; 470 NaN; ERP vs dark",
        "",
        c1["prose"],
        "",
        _md_table(c1["rows"]),
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| train CM / companies | {c1['n_train']:,} / {c1['n_train_co']:,} |",
        f"| pending defined | {c1['train_nn']:,} ({_pp(c1['train_cov'])}) |",
        f"| train NaN CM | {c1['train_na']:,} |",
        f"| never-ERP companies | {c1['n_dark_co']} (want {N_DARK_WANT}) |",
        f"| dark pending non-null / zero-filled | {c1['dark_nn']} / {c1['dark_zero']} |",
        f"| dark 0-fill | {'NO — CONFIRM' if c1['dark_ok'] else 'YES — FAIL'} |",
        f"| holdout coverage (check only) | {_pp(c1['hold_cov'])} |",
        f"| acf1 / acf3 / acf6 | {_f(c1['acf1'])} / {_f(c1['acf3'])} / {_f(c1['acf6'])} |",
        f"| size ρ vs log1p(a_in3) | {_f(c1['size_rho'])} |",
        f"| size ρ vs log1p(|a_op_in|) | {_f(ctx.get('c16', {}).get('size_op', float('nan')))} (feature-report quote {SIZE_RHO_QUOTE}) |",
        "",
        "## 2. Store vs raw pending SQL",
        "",
        c2["prose"],
        "",
    ]
    if c2["sample"]:
        lines += ["Disagreeing sample:", "", _md_table(c2["sample"]), ""]
    lines += [
        "SQL (same as `invoices.py` ~198–218): issued non-cancel `document_type='invoice'` "
        "with `issuance_date ≤ period_end`; unpaid `|amount|` if `payment_date` is null or "
        "after period_end; `payment_date_invalid` contributes 0 to num and den.",
        "",
        "## 3. Spearman twins (|ρ|≥0.80)",
        "",
        c3["prose"],
        "",
        _md_table(c3["rows"]),
        "",
        "## 4. Single-feature train group-fold AUROC",
        "",
        "Sign from the train side of each fold. Seed 20260918. "
        f"Night Y7 issued_lag1 **0.630** (replica {_f(c4['iss_y7'])}); "
        f"TURNOVER **0.720** / B_shallow **0.712** / TURNPEND **0.7184** unchanged. "
        f"Night Y3 size **0.617** (replica {_f(c4['size_y3'])}); days **0.711** (replica {_f(c4['days_y3'])}). "
        "Do not quote holdout.",
        "",
        c4["prose"],
        "",
        _md_table(c4["rows"]),
        "",
        "## 5. Honest leftover after the bar",
        "",
        "Y7 bar = DSO and/or issued_lag1. Y3 bar = days. Leftover <0.55 dies. "
        "OLS residual of pending on the bar; rank-residual is the orthogonal-to-bar twin.",
        "",
        c5["prose"],
        "",
        _md_table(c5["rows"]),
        "",
        "## 6. Twin screen — DROP the weaker twin",
        "",
        c6["prose"],
        "",
        _md_table(c6["rows"]) if c6["rows"] else "_(no |ρ|≥0.80 twin)_\n",
        "",
        "## 7. SIZE terciles and invoice-book-only",
        "",
        c7["prose"],
        "",
        _md_table(c7["rows"]),
        "",
        "## 8. Q6 — lag1 / lag3, empty-on-short, delay first 6 months",
        "",
        "Delay is empty on 2024-09..2025-02 (`DELAY_MASK_BEFORE`). Pending is a stock of "
        "unpaid issued — it can be defined as soon as the company has issued.",
        "",
        c8["prose"],
        "",
        _md_table(c8["cal_rows"]),
        "",
        _md_table(c8["rows"]),
        "",
        "## 9. vs Family J `j_pay_match` (in-memory, not merged)",
        "",
        "J is KEEP-Q5 diagnostic, not a Y3 X. Do not merge J.",
        "",
        c9["prose"],
        "",
        _md_table(c9["rows"]),
        "",
        "## 10. vs credit-note and delay (siblings read-only)",
        "",
        c10["prose"],
        "",
        _md_table(c10["rows"]),
        "",
        "## 11. ICC / company-demean — trait vs month shock",
        "",
        c11["prose"],
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| company ICC / η² | {_f(c11['icc']['icc'])} / {_f(c11['icc']['eta2'])} |",
        f"| companies / CM | {c11['icc']['k']} / {c11['icc']['n']:,} |",
        f"| group ICC / η² | {_f(c11['icc_g']['icc'])} / {_f(c11['icc_g']['eta2'])} |",
        f"| a_out_vol η² (quote) | {A_OUT_VOL_ETA2} |",
        f"| trait (ICC≥0.85) | {c11['trait']} |",
        f"| Y7 / Y3 demean leftover | {_f(c11['d7'])} / {_f(c11['d3'])} |",
        "",
        "## 12. Brief map",
        "",
        "See PARK / CLOSE / KEEP above. Night quotes unchanged.",
        "",
        "## Extra A. Pending vs overdue pile",
        "",
        ctx["pile"]["prose"],
        "",
        _md_table(ctx["pile"]["rows"]),
        "",
        "## Extra B. AP / supplier pending analogue",
        "",
        ctx["ap"]["prose"],
        "",
        _md_table(ctx["ap"]["rows"]),
        "",
        "## Extra C. Holdout coverage only (no AUROC)",
        "",
        ctx["hold"]["prose"],
        "",
        "## Extra D. Quintiles of pending vs Y7 rate",
        "",
        ctx["q5q"]["prose"],
        "",
        _md_table(ctx["q5q"]["rows"]),
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx.get("png_ok") else "Plot: skipped (no matplotlib).",
        "",
        "## Extra E. U-shape probe (not a Y)",
        "",
        ctx["c13"]["prose"],
        "",
        _md_table(ctx["c13"]["rows"]),
        "",
        "## Extra F. Fold 4 (issued owns 0.647 / TURNOVER 0.680)",
        "",
        ctx["c14"]["prose"],
        "",
        _md_table(ctx["c14"]["rows"]),
        "",
        _md_table(ctx["c14"]["grp"]),
        "",
        "## Extra G. New-book vs all-paid tails",
        "",
        ctx["c15"]["prose"],
        "",
        _md_table(ctx["c15"]["rows"]),
        "",
        "## Extra H. Company-mean trait vs Y7 rate",
        "",
        ctx["c16"]["prose"],
        "",
        "## Extra I. DSO clip 24m leftover (Y7 card clip)",
        "",
        ctx["c17"]["prose"],
        "",
        _md_table(ctx["c17"]["rows"]),
        "",
        "## Extra J. All-paid dummy (Q1 emptied book)",
        "",
        ctx["c18"]["prose"],
        "",
        _md_table(ctx["c18"]["rows"]),
        "",
        "## Extra K. AR / AP mix inside the blended share",
        "",
        ctx["c19"]["prose"],
        "",
        "## Extra L. Same-n leftover vs issued_lag1",
        "",
        ctx["c20"]["prose"],
        "",
        _md_table(ctx["c20"]["rows"]),
        "",
        "## Extra M. Inverse leftover — delay after pending",
        "",
        ctx["c21"]["prose"],
        "",
        _md_table(ctx["c21"]["rows"]),
        "",
        "## What failed / next (held for wave note)",
        "",
        *([f"- {x}" for x in ctx["failed"]] if ctx["failed"] else ["- (none)"]),
        "",
        f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts 1–12 plus extras "
        "(pile, AP analogue, holdout coverage, quintiles, U-shape, fold 4, new-book, "
        "company-mean, DSO clip, all-paid dummy, AR/AP mix, same-n).",
        "",
        "## What this module did not do",
        "",
        "- Did not change night Y3 0.762 / 0.752 or Y7 TURNOVER 0.720 / B_shallow 0.712 / TURNPEND 0.7184.",
        "- Did not put pending on the Y3 15-col card. Did not grow TURNOVER. Did not use Family D as Y7 X.",
        "- Did not use Family E as Y5 X. Did not invent `y_pending`. Did not write 0–100 / pillars.",
        "- Did not touch `product/`. Did not edit `invoices.py` / `gbm_y7_core.py` / sibling QA scripts.",
        "- Did not rewrite parquet or duckdb. Did not merge Family I/M/J. Did not run `build_targets`.",
        "- Did not fit on holdout 72. Did not commit. Did not write the parent journal / LIVE / canvas.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def _reg_row(metric: str, value, coverage, y: str, notes: str, split: str = "train_cv") -> dict:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        value = ""
    return {
        "ts": datetime.now().strftime("%Y-%m-%dT%H:%M"),
        "round": ROUND,
        "wave": WAVE,
        "agent": AGENT,
        "x_families": X_FAM,
        "y": y,
        "model": MODEL,
        "split": split,
        "metric": metric,
        "value": value,
        "coverage": coverage,
        "notes": notes,
    }


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    c1, c3, c4, c5, c8, c11, d = (
        ctx["c1"],
        ctx["c3"],
        ctx["c4"],
        ctx["c5"],
        ctx["c8"],
        ctx["c11"],
        ctx["decision"],
    )
    rows = [
        _reg_row(
            "e_pending_amt_share_cov",
            c1["train_cov"],
            f"{c1['train_cov']:.4f}",
            "-",
            f"quote=0.603 dark_nn={c1['dark_nn']} dark_co={c1['n_dark_co']} nan={c1['train_na']} ok={c1['dark_ok']}",
            "train",
        ),
        _reg_row(
            "rho_pending_vs_dso",
            c3["rhos"]["e_dso_proxy"],
            "1.0000",
            Y7,
            f"overdue={c3['rhos']['e_ar_overdue']:.4f} delay={c3['rhos']['e_delay_coll']:.4f} "
            f"issued_lag1={c3['rhos']['e_ar_issued_lag1']:.4f} j={c3['rhos']['j_pay_match']:.4f} "
            f"size={c3['rhos']['log1p(a_in3)']:.4f} twin={c3['any_twin']}",
            "train",
        ),
        _reg_row(
            "auroc_e_pending_amt_share",
            c4["pend_y7"],
            f"{c1['train_cov']:.4f}",
            Y7,
            f"vs_size={c4['size_y7']:.4f} vs_issued_lag1={c4['iss_y7']:.4f} vs_dso={c4['dso_y7']:.4f} "
            f"y7={d['y7']} turnpend=0.7184",
        ),
        _reg_row(
            "auroc_pending_resid_dso",
            c5["after_dso"],
            "1.0000",
            Y7,
            f"after_issued_lag1={c5['after_iss']:.4f} after_both={c5['after_both']:.4f} "
            f"rank_dso={c5['rank_dso']:.4f} lives_dso={c5['lives_dso']} lives_iss={c5['lives_iss']}",
        ),
        _reg_row(
            "auroc_e_pending_amt_share",
            c4["pend_y3"],
            f"{c1['train_cov']:.4f}",
            Y3,
            f"size={c4['size_y3']:.4f} days={c4['days_y3']:.4f} leftover_days={c5['after_days']:.4f} y3={d['y3']}",
        ),
        _reg_row(
            "auroc_pending_lag1",
            c8["lag1"],
            "1.0000",
            Y7,
            f"now={c8['now']:.4f} lag3={c8['lag3']:.4f} early_cov={c8['early_pop']:.4f} "
            f"delay_early={c8['delay_early']:.4f} q6={d['q6']}",
        ),
        _reg_row(
            "icc_e_pending_amt_share",
            c11["icc"]["icc"],
            "1.0000",
            "-",
            f"eta2={c11['icc']['eta2']:.4f} quote=0.97 trait={c11['trait']} demean_y7={c11['d7']:.4f}",
            "train",
        ),
        _reg_row(
            "pending_lose_44",
            1.0 if d["lose_44"] else 0.0,
            f"{c1['train_cov']:.4f}",
            "-",
            f"overall={d['overall']} y7={d['y7']} y3={d['y3']} leftover_y7={d['leftover_y7']:.4f} "
            f"leftover_y3={d['leftover_y3']:.4f}",
            "train",
        ),
        _reg_row(
            "auroc_pending_ushape_dist",
            ctx["c13"]["dist"],
            "1.0000",
            Y7,
            f"tail={ctx['c13']['tail']:.4f} after_iss={ctx['c13']['after_iss']:.4f} lives={ctx['c13']['lives']}",
        ),
        _reg_row(
            "auroc_pending_fold4",
            ctx["c14"]["pend4"],
            "1.0000",
            Y7,
            f"issued_lag1_f4={ctx['c14']['iss4']:.4f} turnover_f4=0.680 issued_owns={ctx['c14']['issued_owns']}",
        ),
    ]
    with REGISTRY.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        seen = {
            (
                r.get("agent"),
                r.get("y"),
                r.get("model"),
                r.get("split"),
                r.get("metric"),
                r.get("x_families"),
            )
            for r in reader
        }
    fresh = []
    for r in rows:
        key = (
            str(r.get("agent", "")),
            str(r.get("y", "")),
            str(r.get("model", "")),
            str(r.get("split", "")),
            str(r.get("metric", "")),
            str(r.get("x_families", "")),
        )
        if key in seen:
            continue
        fresh.append(r)
        seen.add(key)
    if not fresh:
        print("registry: no new rows")
        return
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for r in fresh:
            w.writerow({k: r.get(k, "") for k in header})
    print(f"registry appended {len(fresh)} rows")


def run() -> dict:
    t0 = time.time()
    print(f"pending_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    panel = attach_folds(panel)
    con = connect()
    try:
        book = book_invoice_ids(con)
        dark = dark_population(con)
        panel = attach_j(panel, con)
        panel = attach_recon(panel, con)
    finally:
        con.close()
    panel = add_panel_lags(
        panel,
        [
            "e_pending_amt_share",
            "e_ar_issued",
            "e_dso_proxy",
            "e_delay_coll",
            "e_credit_note_ratio",
        ],
        (1, 2, 3),
    )
    tr = panel[panel["split"] == "train"].copy()
    assert_no_holdout(tr["company_id"])
    print(
        f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
        f"holdout CM={int((panel['split']=='holdout').sum())} "
        f"book={len(book)} train_dark={dark['n_train_dark']} confirm_470={dark['confirm_470']}"
    )

    failed: list[str] = []
    c1 = cut1_coverage(panel, book, dark)
    if not c1["dark_ok"]:
        failed.append(f"dark 470 NaN screen failed nn={c1['dark_nn']} zero={c1['dark_zero']} co={c1['n_dark_co']}")
    c2 = cut2_formula(tr)
    if not c2["match"]:
        failed.append("store vs SQL disagree — documented, invoices.py not edited")
    c3 = cut3_twins(tr)
    c4 = cut4_singles(tr)
    c5 = cut5_leftover(tr)
    c6 = cut6_twin_drop(tr, c3, c4)
    c7 = cut7_slices(tr, book)
    c8 = cut8_q6(tr)
    c9 = cut9_family_j(tr)
    c10 = cut10_siblings(tr)
    c11 = cut11_icc(tr)
    pile = cut_extra_pile(tr)
    ap = cut_extra_ap(tr)
    q5q = cut_extra_quintiles(tr)
    hold = cut_extra_holdout(panel, book)
    c13 = cut13_ushape(tr)
    c14 = cut14_fold4(tr, c4)
    c15 = cut15_newbook(tr)
    c16 = cut16_company_mean(tr)
    c17 = cut17_dso_clip(tr)
    c18 = cut18_allpaid(tr)
    c19 = cut19_mix(tr)
    c20 = cut20_samen(tr)
    # Inverse: does delay still leftover after pending? Sibling KEEP should survive.
    delay_after_p, info_dp = ols_resid(tr["e_delay_coll"], tr["e_pending_amt_share"])
    delay_inv = signed_oof_auroc(tr[Y7], delay_after_p, tr["fold"], tr[Y7].notna())
    y7_on_dark = int(
        (
            tr[Y7].notna()
            & ~tr["company_id"].isin(book)
        ).sum()
    )
    c21 = {
        "delay_after_pending": _cv(delay_inv),
        "r2": info_dp["r2"],
        "y7_on_dark": y7_on_dark,
        "prose": (
            f"Inverse leftover: delay after pending {_f(_cv(delay_inv))} "
            f"R²={_f(info_dp['r2'])} (sibling KEEP 0.581 should survive). "
            f"Y7 labeled on dark 470: {y7_on_dark} (want 0 — Y7 is invoice-built)."
        ),
        "rows": [_auc_row(Y7, "delay resid after pending", delay_inv)],
    }
    print(c21["prose"])
    if y7_on_dark != 0:
        failed.append(f"Y7 labeled on dark companies n={y7_on_dark}")
    failed.append(
        "Y7 leftover after DSO 0.421 / issued_lag1 0.418 dies <0.55 — unused stock; TURNPEND 0.7184 already lost"
    )
    failed.append("Y3 leftover after days 0.443 dies — stays off the 15-col card")
    failed.append("U-shape / all-paid dummy leftover is thin (0.558 / 0.572) — CLOSE, do not invent y_pending")
    failed.append("Fold 4 pending 0.363 vs issued 0.647 — issued owns the hard fold")
    if c16.get("size_op") is not None and np.isfinite(c16["size_op"]):
        c1["size_rho_opin"] = c16["size_op"]
    png_ok = plot_png(tr, {})
    ctx = {
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "c4": c4,
        "c5": c5,
        "c6": c6,
        "c7": c7,
        "c8": c8,
        "c9": c9,
        "c10": c10,
        "c11": c11,
        "c13": c13,
        "c14": c14,
        "c15": c15,
        "c16": c16,
        "c17": c17,
        "c18": c18,
        "c19": c19,
        "c20": c20,
        "c21": c21,
        "pile": pile,
        "ap": ap,
        "q5q": q5q,
        "hold": hold,
        "png_ok": png_ok,
        "failed": failed,
        "elapsed_s": time.time() - t0,
        "book": book,
        "dark": dark,
    }
    ctx["decision"] = decide(ctx)
    write_md(ctx)
    append_registry(ctx)
    print(f"pending_qa done in {ctx['elapsed_s']:.0f}s overall={ctx['decision']['overall']}")
    return ctx


if __name__ == "__main__":
    run()
