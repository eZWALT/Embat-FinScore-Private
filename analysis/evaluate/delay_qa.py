"""Q4/Q5 delay / overdue leftover after DSO / issued_lag1.

NORTH_STAR: Javier overdue is a 3-month issuance window; the store is
all-open unpaid. Delay is SAME (ρ=1) and left-truncated 6 calendar
months. `e_delay_coll` sits on the Y7 SHAP card. DSO is SHAP#1 and
fails the short-DSO fifth. Night Y7 stays **TURNOVER 0.720 /
B_shallow 0.712**. Do not change it. Do not grow TURNOVER. Y7 never D.
Y5 never E. Night Y3 stays **0.762 / 0.752**. Days **0.711**. Delay /
overdue stay off the 15-col card. Y3 never B.

Question: leftover dip-vs-fall after DSO / issued_lag1, or a DSO twin /
truncation dummy? As Y3 X, leftover after days?

No 0–100. No product/. No parquet rewrite. No new GBM. Do not run
`build_targets`. Do not invent `y_delay`. Do not edit invoices.py
unless a real formula bug — then stop and report.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.delay_qa

Owned: analysis/evaluate/delay_qa.py, analysis/outputs/delay_qa.md,
optional one PNG, append-only registry, overnight/waves/wave4_delay.md (end).

Iteration (same module):
1. Coverage + dark NaN + Javier 3m + twins + singles + leftover + Q6
2. Fold 4 + ICC + 3m leftover + first-6 vs short books
3. Next cuts after the first green run
"""
from __future__ import annotations

import csv
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
from analysis.features.invoices import DELAY_CLIP, DELAY_MASK_BEFORE
from analysis.targets.y11_dark import book_invoice_ids

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
OUT_MD = ANALYSIS / "outputs" / "delay_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "delay_vs_dso.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_delay.md"
AGENT = "47815ca4"
WAVE = "4"
ROUND = "R4"
MODEL = "delay_qa"
X_FAM = "E"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y7 = "y7_top1_lost"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
ISSUED_LAG1_BENCH = 0.630
TURNOVER_QUOTE = 0.720
B_SHALLOW_QUOTE = 0.712
NIGHT_Y3 = 0.762
NIGHT_Y3_CORE = 0.752
TURNOVER_F4 = 0.680
TWIN_RHO = 0.80
KEEP_DELTA = 0.02
CHANCE = 0.55
SIZE_RHO = 0.50
ICC_STYLE = 0.85
MIN_POS = 50
MIN_ACF_PAIRS = 4
CHRONIC_GROUPS = ("GROUP_0158", "GROUP_0172")
CHRONIC_BELOW = 0.50
FOLD4_GROUPS = ("GROUP_0222", "GROUP_0108")
PANEL_START = pd.Timestamp("2024-09-01")
PANEL_END = pd.Timestamp("2026-08-01")
EARLY_END = pd.Timestamp("2025-02-01")  # last masked month
WRITE_WAVE = True  # last iterate writes the note

DELAY_FEATS = (
    "e_delay_coll",
    "e_delay_paid",
    "e_ar_overdue",
    "e_ap_overdue",
    "e_ar_overdue_30",
    "e_ap_overdue_30",
)

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "c_n_days_with_tx",
    "e_delay_coll",
    "e_delay_paid",
    "e_ar_overdue",
    "e_ap_overdue",
    "e_ar_overdue_30",
    "e_ap_overdue_30",
    "e_dso_proxy",
    "e_dpo_proxy",
    "e_ar_issued",
    "e_ap_issued",
    "e_ar_open",
    "e_ap_open",
    "b_below_0",
)

Y_KEEP = (Y2, Y3, Y7)


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


def spearman_n(a, b) -> tuple[float, int]:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    ).dropna()
    if len(d) < 8 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan"), int(len(d))
    return float(d["a"].corr(d["b"], method="spearman")), int(len(d))


def verdict_rho(rho: float) -> str:
    if not np.isfinite(rho):
        return "NA"
    a = abs(rho)
    if a >= 0.95:
        return "SAME"
    if a >= 0.80:
        return "CLOSE"
    return "DRIFT"


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
        return {"icc": float("nan"), "var_w": float("nan"), "var_b": float("nan"), "k": 0}
    y = s["x"].to_numpy(dtype=float)
    _, inv, counts = np.unique(s["co"].to_numpy(), return_inverse=True, return_counts=True)
    if counts.size < 2:
        return {"icc": float("nan"), "var_w": float("nan"), "var_b": float("nan"), "k": 0}
    means = np.bincount(inv, weights=y) / counts
    grand = float(y.mean())
    ssb = float(np.sum(counts * (means - grand) ** 2))
    ssw = float(np.sum((y - means[inv]) ** 2))
    k = int(counts.size)
    n = int(y.size)
    var_b = ssb / max(k - 1, 1)
    var_w = ssw / max(n - k, 1)
    icc = var_b / (var_b + var_w) if (var_b + var_w) > 0 else float("nan")
    return {"icc": float(icc), "var_w": float(var_w), "var_b": float(var_b), "k": k}


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


def period_frame() -> pd.DataFrame:
    df = pd.DataFrame({"period": pd.to_datetime(list(MONTHS))})
    df["period_end"] = df["period"] + pd.offsets.MonthEnd(0)
    df["pay_start"] = df["period"] - pd.DateOffset(months=2)
    df["delay_ok"] = df["period"] >= DELAY_MASK_BEFORE
    df["i"] = np.arange(len(df))
    return df


def reconstruct_javier(con) -> pd.DataFrame:
    """In-module Javier 3m overdue + delay. Diagnosis only — not a store write."""
    periods = period_frame()
    con.register("_dq_periods", periods[["period", "period_end", "pay_start", "delay_ok", "i"]])
    try:
        od3 = con.execute(
            """
            SELECT i.company_id,
                   p.period,
                   SUM(CASE WHEN i.amount > 0
                             AND GREATEST(CAST(i.due_date AS DATE),
                                          CAST(i.issuance_date AS DATE))
                                 < CAST(p.period_end AS DATE)
                            THEN abs(i.amount) ELSE 0 END)
                     / NULLIF(SUM(CASE WHEN i.amount > 0 THEN abs(i.amount) END), 0)
                     AS j_ar_od_3m,
                   SUM(CASE WHEN i.amount < 0
                             AND GREATEST(CAST(i.due_date AS DATE),
                                          CAST(i.issuance_date AS DATE))
                                 < CAST(p.period_end AS DATE)
                            THEN abs(i.amount) ELSE 0 END)
                     / NULLIF(SUM(CASE WHEN i.amount < 0 THEN abs(i.amount) END), 0)
                     AS j_ap_od_3m
            FROM invoices i
            JOIN _dq_periods p
              ON CAST(i.issuance_date AS DATE) >= CAST(p.pay_start AS DATE)
             AND CAST(i.issuance_date AS DATE) <= CAST(p.period_end AS DATE)
             AND (i.payment_date IS NULL
                  OR CAST(i.payment_date AS DATE) > CAST(p.period_end AS DATE)
                  OR i.status <> 'paid')
            WHERE i.document_type = 'invoice'
              AND i.status <> 'cancel'
              AND i.amount <> 0
              AND i.issuance_date IS NOT NULL
              AND i.due_date IS NOT NULL
              AND p.i >= 2
              AND NOT (i.status = 'paid'
                       AND NOT (i.payment_date >= i.issuance_date
                                AND i.payment_date <= TIMESTAMP '2026-09-01'))
            GROUP BY 1, 2
            """
        ).df()
        delay = con.execute(
            """
            SELECT i.company_id,
                   p.period,
                   CASE WHEN p.delay_ok THEN
                     SUM(CASE WHEN i.amount > 0
                              THEN date_diff('day',
                                   GREATEST(CAST(i.due_date AS DATE),
                                            CAST(i.issuance_date AS DATE)),
                                   CAST(i.payment_date AS DATE)) * abs(i.amount)
                         END)
                     / NULLIF(SUM(CASE WHEN i.amount > 0 THEN abs(i.amount) END), 0)
                   END AS j_delay_coll,
                   CASE WHEN p.delay_ok THEN
                     SUM(CASE WHEN i.amount < 0
                              THEN date_diff('day',
                                   GREATEST(CAST(i.due_date AS DATE),
                                            CAST(i.issuance_date AS DATE)),
                                   CAST(i.payment_date AS DATE)) * abs(i.amount)
                         END)
                     / NULLIF(SUM(CASE WHEN i.amount < 0 THEN abs(i.amount) END), 0)
                   END AS j_delay_paid
            FROM invoices i
            JOIN _dq_periods p
              ON i.payment_date IS NOT NULL
             AND i.status = 'paid'
             AND CAST(i.payment_date AS DATE) >= CAST(p.pay_start AS DATE)
             AND CAST(i.payment_date AS DATE) <= CAST(p.period_end AS DATE)
            WHERE i.document_type = 'invoice'
              AND i.status <> 'cancel'
              AND i.amount <> 0
              AND i.due_date IS NOT NULL
              AND i.issuance_date IS NOT NULL
              AND NOT (NOT (i.payment_date >= i.issuance_date
                            AND i.payment_date <= TIMESTAMP '2026-09-01'))
            GROUP BY 1, 2, p.delay_ok
            """
        ).df()
        store3 = con.execute(
            """
            SELECT i.company_id,
                   p.period,
                   SUM(CASE WHEN i.amount > 0
                             AND i.due_date IS NOT NULL
                             AND CAST(i.due_date AS DATE) < CAST(p.period_end AS DATE)
                            THEN abs(i.amount) ELSE 0 END)
                     / NULLIF(SUM(CASE WHEN i.amount > 0 THEN abs(i.amount) END), 0)
                     AS ar_od_3m,
                   SUM(CASE WHEN i.amount < 0
                             AND i.due_date IS NOT NULL
                             AND CAST(i.due_date AS DATE) < CAST(p.period_end AS DATE)
                            THEN abs(i.amount) ELSE 0 END)
                     / NULLIF(SUM(CASE WHEN i.amount < 0 THEN abs(i.amount) END), 0)
                     AS ap_od_3m
            FROM invoices i
            JOIN _dq_periods p
              ON CAST(i.issuance_date AS DATE) >= CAST(p.pay_start AS DATE)
             AND CAST(i.issuance_date AS DATE) <= CAST(p.period_end AS DATE)
             AND (i.payment_date IS NULL
                  OR CAST(i.payment_date AS DATE) > CAST(p.period_end AS DATE))
             AND NOT coalesce(i.payment_date_invalid, FALSE)
            WHERE i.document_type = 'invoice'
              AND i.status <> 'cancel'
              AND i.amount <> 0
              AND i.issuance_date IS NOT NULL
            GROUP BY 1, 2
            """
        ).df()
    finally:
        con.unregister("_dq_periods")

    out = None
    for part, cols in (
        (od3, ["j_ar_od_3m", "j_ap_od_3m"]),
        (delay, ["j_delay_coll", "j_delay_paid"]),
        (store3, ["ar_od_3m", "ap_od_3m"]),
    ):
        if part.empty:
            continue
        part = part.copy()
        part["company_id"] = part["company_id"].astype(str)
        part["period"] = pd.to_datetime(part["period"])
        keep = ["company_id", "period"] + [c for c in cols if c in part.columns]
        part = part[keep]
        out = part if out is None else out.merge(part, on=["company_id", "period"], how="outer")
    if out is None:
        return pd.DataFrame(
            columns=[
                "company_id",
                "period",
                "j_ar_od_3m",
                "j_ap_od_3m",
                "j_delay_coll",
                "j_delay_paid",
                "ar_od_3m",
                "ap_od_3m",
            ]
        )
    for c in ("j_delay_coll", "j_delay_paid"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").clip(*DELAY_CLIP)
    return out


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
    panel = _keys(raw[list(STORE_COLS)])
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
    panel["co_class"] = panel["n_grid_months"].map(_trail_class)
    panel["early6"] = panel["period"] < DELAY_MASK_BEFORE
    leak = leakage_check(
        ["e_delay_coll", "e_delay_paid", "e_ar_overdue", "e_dso_proxy", "e_ar_issued"],
        Y7,
        forbidden_prefixes=["d"],
    )
    if not leak["ok"]:
        raise RuntimeError(f"Y7 X leak: {leak['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def chronic_ids(tr: pd.DataFrame) -> list[str]:
    y = pd.to_numeric(tr[Y2], errors="coerce")
    below = pd.to_numeric(tr["b_below_0"], errors="coerce")
    lab = y.notna()
    hot = tr["group_id"].astype(str).isin(CHRONIC_GROUPS)
    sl = lab & hot
    if not sl.any():
        return []
    g = (
        tr.loc[sl, ["company_id"]]
        .assign(below=below[sl].values)
        .groupby("company_id")["below"]
        .agg(n="size", n_below="sum")
    )
    g["share_below"] = g["n_below"] / g["n"]
    return [str(i) for i in g.index[g["share_below"] >= CHRONIC_BELOW]]


# ---------------------------------------------------------------------------
# Pass 1 — coverage / nulls. First 6 calendar months empty. Dark 470 = NaN.
# ---------------------------------------------------------------------------
def pass1_cov(tr: pd.DataFrame, book: set[str]) -> dict:
    early = tr["early6"]
    late = ~early
    dark = ~tr["company_id"].isin(book)
    erp = tr["company_id"].isin(book)
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    n_erp_co = int(tr.loc[erp, "company_id"].nunique())
    rows = []
    store = {}
    for col in DELAY_FEATS:
        x = pd.to_numeric(tr[col], errors="coerce")
        early_nn = float(x[early].notna().mean()) if early.any() else float("nan")
        late_nn = float(x[late].notna().mean()) if late.any() else float("nan")
        dark_nn = int(x[dark].notna().sum())
        dark_zero = int((x[dark] == 0).sum())
        erp_nn = float(x[erp].notna().mean()) if erp.any() else float("nan")
        store[col] = {
            "nn": int(x.notna().sum()),
            "cov": float(x.notna().mean()),
            "early_nn": early_nn,
            "late_nn": late_nn,
            "dark_nn": dark_nn,
            "dark_zero": dark_zero,
            "erp_nn": erp_nn,
            "n_co": int(tr.loc[x.notna(), "company_id"].nunique()),
            "p50": float(x.median()) if x.notna().any() else float("nan"),
            "acf1": median_acf(x, tr["company_id"], 1),
            "acf3": median_acf(x, tr["company_id"], 3),
        }
        is_delay = col.startswith("e_delay")
        early_ok = bool(is_delay and np.isfinite(early_nn) and early_nn == 0.0)
        dark_ok = bool(dark_nn == 0 and dark_zero == 0)
        rows.append(
            {
                "col": col,
                "nn": f"{store[col]['nn']:,}",
                "cov": _pp(store[col]["cov"]),
                "early6 nn": _pp(early_nn),
                "after nn": _pp(late_nn),
                "dark nn / 0": f"{dark_nn} / {dark_zero}",
                "ERP nn": _pp(erp_nn),
                "p50": _f(store[col]["p50"], 2),
                "acf1": _f(store[col]["acf1"]),
            }
        )
        print(
            f"COV {col}: early={_pp(early_nn)} late={_pp(late_nn)} "
            f"dark_nn={dark_nn} dark_0={dark_zero}"
        )

    delay_early_empty = bool(
        store["e_delay_coll"]["early_nn"] == 0.0 and store["e_delay_paid"]["early_nn"] == 0.0
    )
    dark_nan = bool(
        all(store[c]["dark_nn"] == 0 and store[c]["dark_zero"] == 0 for c in DELAY_FEATS)
        and n_dark_co == 470
    )
    mask_ok = bool(DELAY_MASK_BEFORE == pd.Timestamp("2025-03-01"))
    y7 = pd.to_numeric(tr[Y7], errors="coerce")
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    prose = (
        f"Delay empty on first 6 calendar months (2024-09..2025-02): "
        f"{'CONFIRM' if delay_early_empty else 'FAIL'} "
        f"(coll {_pp(store['e_delay_coll']['early_nn'])} / "
        f"paid {_pp(store['e_delay_paid']['early_nn'])}; "
        f"DELAY_MASK_BEFORE={DELAY_MASK_BEFORE.date()} "
        f"{'ok' if mask_ok else 'UNEXPECTED'}). "
        f"After month 7: coll {_pp(store['e_delay_coll']['late_nn'])} "
        f"paid {_pp(store['e_delay_paid']['late_nn'])}. "
        f"Dark never-ERP {n_dark_co} (want 470): all six signals nn=0 / zero=0 "
        f"({'CONFIRM NaN not 0' if dark_nan else 'FAIL — dark filled'}). "
        f"Ever-ERP {n_erp_co}. Y7 labeled {int(y7.notna().sum()):,} "
        f"pos {int((y7 == 1).sum()):,}; Y3 {int(y3.notna().sum()):,} "
        f"pos {int((y3 == 1).sum()):,}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "delay_early_empty": delay_early_empty,
        "dark_nan": dark_nan,
        "n_dark_co": n_dark_co,
        "n_erp_co": n_erp_co,
        "n_cm": int(len(tr)),
        "n_co": int(tr["company_id"].nunique()),
        "mask_ok": mask_ok,
        "y7_n": int(y7.notna().sum()),
        "y7_pos": int((y7 == 1).sum()),
        "y7_rate": float(y7.mean()) if y7.notna().any() else float("nan"),
        "y3_n": int(y3.notna().sum()),
        "y3_pos": int((y3 == 1).sum()),
        "y3_rate": float(y3.mean()) if y3.notna().any() else float("nan"),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 2 — Javier: delay SAME; overdue vs 3m SAME; vs all-open CLOSE
# ---------------------------------------------------------------------------
def pass2_javier(tr: pd.DataFrame) -> dict:
    pairs = [
        ("delay_coll", "e_delay_coll", "j_delay_coll", None),
        ("delay_paid", "e_delay_paid", "j_delay_paid", None),
        ("ar_overdue vs store", "e_ar_overdue", "j_ar_od_3m", "CLOSE?"),
        ("ap_overdue vs store", "e_ap_overdue", "j_ap_od_3m", "CLOSE?"),
        ("ar_overdue vs 3m", "ar_od_3m", "j_ar_od_3m", "SAME?"),
        ("ap_overdue vs 3m", "ap_od_3m", "j_ap_od_3m", "SAME?"),
        ("store AR vs 3m", "e_ar_overdue", "ar_od_3m", "CLOSE"),
        ("store AP vs 3m", "e_ap_overdue", "ap_od_3m", "CLOSE"),
    ]
    rows = []
    rhos = {}
    for name, a, b, _hint in pairs:
        if a not in tr.columns or b not in tr.columns:
            rho, n = float("nan"), 0
        else:
            rho, n = spearman_n(tr[a], tr[b])
        rhos[name] = rho
        rows.append(
            {
                "pair": name,
                "ρ": _f(rho, 3),
                "n": f"{n:,}",
                "verdict": verdict_rho(rho),
            }
        )
        print(f"JAVIER {name}: ρ={_f(rho, 3)} n={n} {verdict_rho(rho)}")

    delay_same = bool(
        np.isfinite(rhos["delay_coll"])
        and abs(rhos["delay_coll"]) >= 0.95
        and np.isfinite(rhos["delay_paid"])
        and abs(rhos["delay_paid"]) >= 0.95
    )
    od3_same = bool(
        np.isfinite(rhos["ar_overdue vs 3m"])
        and abs(rhos["ar_overdue vs 3m"]) >= 0.95
        and np.isfinite(rhos["ap_overdue vs 3m"])
        and abs(rhos["ap_overdue vs 3m"]) >= 0.95
    )
    store_close = bool(
        np.isfinite(rhos["store AR vs 3m"])
        and 0.80 <= abs(rhos["store AR vs 3m"]) < 0.95
        and np.isfinite(rhos["store AP vs 3m"])
        and 0.80 <= abs(rhos["store AP vs 3m"]) < 0.95
    )
    # quote screen vs score_pipeline 0.918 / 0.883
    ar_store = rhos["ar_overdue vs store"]
    ap_store = rhos["ap_overdue vs store"]
    confirm_ar = bool(np.isfinite(ar_store) and abs(ar_store - 0.918) < 0.03)
    confirm_ap = bool(np.isfinite(ap_store) and abs(ap_store - 0.883) < 0.03)
    prose = (
        f"Delay vs Javier: coll {_f(rhos['delay_coll'])} paid {_f(rhos['delay_paid'])} "
        f"({'CONFIRM SAME ρ=1' if delay_same else 'off SAME'}). "
        f"Javier overdue vs in-module 3m: AR {_f(rhos['ar_overdue vs 3m'])} "
        f"AP {_f(rhos['ap_overdue vs 3m'])} "
        f"({'CONFIRM ρ≈1' if od3_same else 'off identity'}). "
        f"Store all-open vs 3m: AR {_f(rhos['store AR vs 3m'])} "
        f"AP {_f(rhos['store AP vs 3m'])} "
        f"({'CONFIRM CLOSE' if store_close else verdict_rho(rhos['store AR vs 3m'])}). "
        f"Javier overdue vs store all-open: AR {_f(ar_store)} "
        f"({'CONFIRM 0.918' if confirm_ar else 'off 0.918'}) "
        f"AP {_f(ap_store)} ({'CONFIRM 0.883' if confirm_ap else 'off 0.883'})."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "delay_same": delay_same,
        "od3_same": od3_same,
        "store_close": store_close,
        "confirm_ar": confirm_ar,
        "confirm_ap": confirm_ap,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 3 — Spearman twins
# ---------------------------------------------------------------------------
def pass3_twins(tr: pd.DataFrame) -> dict:
    peers = {
        "e_dso_proxy": tr["e_dso_proxy"],
        "e_ar_issued_lag1": tr["e_ar_issued_lag1"],
        "e_ar_overdue_30": tr["e_ar_overdue_30"],
        "log1p(a_in3)": tr["log_in3"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "e_ar_overdue": tr["e_ar_overdue"],
        "e_ap_overdue": tr["e_ap_overdue"],
        "e_delay_paid": tr["e_delay_paid"],
        "e_dpo_proxy": tr["e_dpo_proxy"],
    }
    stems = {
        "e_delay_coll": tr["e_delay_coll"],
        "e_delay_paid": tr["e_delay_paid"],
        "e_ar_overdue": tr["e_ar_overdue"],
        "e_ap_overdue": tr["e_ap_overdue"],
        "e_ar_overdue_30": tr["e_ar_overdue_30"],
        "e_ap_overdue_30": tr["e_ap_overdue_30"],
    }
    rows = []
    rhos = {}
    twins = []
    for sname, s in stems.items():
        for pname, p in peers.items():
            if sname == pname:
                continue
            rho, n = spearman_n(s, p)
            key = f"{sname} vs {pname}"
            rhos[key] = rho
            twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
            if twin:
                twins.append(key)
            rows.append(
                {
                    "pair": key,
                    "ρ": _f(rho, 3),
                    "n": f"{n:,}",
                    "twin?": "TWIN" if twin else "",
                }
            )
    delay_dso = rhos.get("e_delay_coll vs e_dso_proxy", float("nan"))
    delay_iss = rhos.get("e_delay_coll vs e_ar_issued_lag1", float("nan"))
    delay_od30 = rhos.get("e_delay_coll vs e_ar_overdue_30", float("nan"))
    delay_size = rhos.get("e_delay_coll vs log1p(a_in3)", float("nan"))
    delay_days = rhos.get("e_delay_coll vs c_n_days_with_tx", float("nan"))
    od_dso = rhos.get("e_ar_overdue vs e_dso_proxy", float("nan"))
    any_twin = bool(twins)
    delay_twin_dso = bool(np.isfinite(delay_dso) and abs(delay_dso) >= TWIN_RHO)
    delay_twin_iss = bool(np.isfinite(delay_iss) and abs(delay_iss) >= TWIN_RHO)
    prose = (
        f"`e_delay_coll` vs DSO {_f(delay_dso)}, issued_lag1 {_f(delay_iss)}, "
        f"ar_overdue_30 {_f(delay_od30)}, size {_f(delay_size)}, days {_f(delay_days)}. "
        f"`e_ar_overdue` vs DSO {_f(od_dso)}. "
        f"{'TWIN |ρ|≥0.80: ' + ', '.join(twins) if any_twin else 'Not a |ρ|≥0.80 twin of DSO / issued_lag1 / size / days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "twins": twins,
        "any_twin": any_twin,
        "delay_dso": delay_dso,
        "delay_iss": delay_iss,
        "delay_od30": delay_od30,
        "delay_size": delay_size,
        "delay_days": delay_days,
        "od_dso": od_dso,
        "delay_twin_dso": delay_twin_dso,
        "delay_twin_iss": delay_twin_iss,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 4 — group-fold AUROC Y7 and Y3
# ---------------------------------------------------------------------------
def pass4_auroc(tr: pd.DataFrame) -> dict:
    feats = {
        "e_delay_coll": tr["e_delay_coll"],
        "e_delay_paid": tr["e_delay_paid"],
        "e_ar_overdue": tr["e_ar_overdue"],
        "e_ap_overdue": tr["e_ap_overdue"],
        "e_ar_overdue_30": tr["e_ar_overdue_30"],
        "e_ap_overdue_30": tr["e_ap_overdue_30"],
        "e_dso_proxy": tr["e_dso_proxy"],
        "e_ar_issued_lag1": tr["e_ar_issued_lag1"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "log1p_a_in3": tr["log_in3"],
        "ar_od_3m": tr["ar_od_3m"] if "ar_od_3m" in tr.columns else pd.Series(np.nan, index=tr.index),
        "ap_od_3m": tr["ap_od_3m"] if "ap_od_3m" in tr.columns else pd.Series(np.nan, index=tr.index),
    }
    rows = []
    store = {}
    for y in (Y7, Y3):
        lab = tr[y].notna()
        for name, col in feats.items():
            res = signed_oof_auroc(tr[y], col, tr["fold"], lab)
            store[(y, name)] = res
            rows.append(_auc_row(y, name, res, present=float(col[lab].notna().mean()) if lab.any() else float("nan")))
            print(
                f"AUROC {y} {name}: "
                f"{'LOW_POWER' if res['low_power'] else _f(res['cv'])} "
                f"n_pos={res['n_pos']} sign={res['train_sign']}"
            )

    def cv(y, feat) -> float:
        return _cv(store[(y, feat)])

    delay_y7 = cv(Y7, "e_delay_coll")
    paid_y7 = cv(Y7, "e_delay_paid")
    arod_y7 = cv(Y7, "e_ar_overdue")
    apod_y7 = cv(Y7, "e_ap_overdue")
    dso_y7 = cv(Y7, "e_dso_proxy")
    iss_y7 = cv(Y7, "e_ar_issued_lag1")
    delay_y3 = cv(Y3, "e_delay_coll")
    days_y3 = cv(Y3, "c_n_days_with_tx")
    size_y3 = cv(Y3, "log1p_a_in3")
    replica_iss = bool(np.isfinite(iss_y7) and abs(iss_y7 - ISSUED_LAG1_BENCH) < 0.02)
    replica_days = bool(np.isfinite(days_y3) and abs(days_y3 - DAYS_BENCH) < 0.02)
    lose_days = bool(np.isfinite(delay_y3) and np.isfinite(days_y3) and delay_y3 < days_y3)
    lose_iss = bool(np.isfinite(delay_y7) and np.isfinite(iss_y7) and delay_y7 + 0.01 < iss_y7)
    prose = (
        f"Y7 delay_coll {_f(delay_y7)} paid {_f(paid_y7)} ar_od {_f(arod_y7)} "
        f"ap_od {_f(apod_y7)} vs DSO {_f(dso_y7)} vs issued_lag1 {_f(iss_y7)} "
        f"(night 0.630, {'CONFIRM' if replica_iss else 'off'}). "
        f"Y3 delay_coll {_f(delay_y3)} vs days {_f(days_y3)} "
        f"(night 0.711, {'CONFIRM' if replica_days else 'off'}) vs size {_f(size_y3)}. "
        f"{'Delay loses to days on Y3' if lose_days else 'Delay near days on Y3'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "delay_y7": delay_y7,
        "paid_y7": paid_y7,
        "arod_y7": arod_y7,
        "apod_y7": apod_y7,
        "arod30_y7": cv(Y7, "e_ar_overdue_30"),
        "apod30_y7": cv(Y7, "e_ap_overdue_30"),
        "dso_y7": dso_y7,
        "iss_y7": iss_y7,
        "delay_y3": delay_y3,
        "paid_y3": cv(Y3, "e_delay_paid"),
        "arod_y3": cv(Y3, "e_ar_overdue"),
        "days_y3": days_y3,
        "size_y3": size_y3,
        "od3_y7": cv(Y7, "ar_od_3m"),
        "replica_iss": replica_iss,
        "replica_days": replica_days,
        "lose_days": lose_days,
        "lose_iss": lose_iss,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 5 — residual Y7 after DSO and after issued_lag1
# ---------------------------------------------------------------------------
def pass5_resid_y7(tr: pd.DataFrame) -> dict:
    lab = tr[Y7].notna()
    stems = {
        "e_delay_coll": tr["e_delay_coll"],
        "e_delay_paid": tr["e_delay_paid"],
        "e_ar_overdue": tr["e_ar_overdue"],
        "e_ap_overdue": tr["e_ap_overdue"],
        "e_ar_overdue_30": tr["e_ar_overdue_30"],
        "e_ap_overdue_30": tr["e_ap_overdue_30"],
    }
    controls = [
        ("after DSO", (tr["e_dso_proxy"],)),
        ("after issued_lag1", (tr["e_ar_issued_lag1"],)),
        ("after DSO+issued_lag1", (tr["e_dso_proxy"], tr["e_ar_issued_lag1"])),
        ("after days", (tr["c_n_days_with_tx"],)),
        ("after size", (tr["log_in3"],)),
        ("after ar_overdue_30", (tr["e_ar_overdue_30"],)),
    ]
    rows = []
    store = {}
    infos = {}
    resids = {}
    for sname, s in stems.items():
        for cname, xs in controls:
            key = f"{sname} {cname}"
            resid, info = ols_resid(s, *xs)
            res = signed_oof_auroc(tr[Y7], resid, tr["fold"], lab)
            store[key] = res
            infos[key] = info
            if sname == "e_delay_coll" and cname in (
                "after DSO",
                "after issued_lag1",
                "after DSO+issued_lag1",
            ):
                resids[cname] = resid
            rows.append(
                {
                    "stem": sname,
                    "residual": cname,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "R²": _f(info["r2"]),
                    "folds": fold_bits(res) if not res["low_power"] else "—",
                }
            )
            print(
                f"Y7 leftover {key}: "
                f"{'LOW_POWER' if res['low_power'] else _f(res['cv'])} "
                f"R2={_f(info['r2'])}"
            )

    after_dso = _cv(store["e_delay_coll after DSO"])
    after_iss = _cv(store["e_delay_coll after issued_lag1"])
    after_both = _cv(store["e_delay_coll after DSO+issued_lag1"])
    after_od_dso = _cv(store["e_ar_overdue after DSO"])
    after_od_iss = _cv(store["e_ar_overdue after issued_lag1"])
    lives = bool(
        np.isfinite(after_dso)
        and after_dso >= CHANCE
        and np.isfinite(after_iss)
        and after_iss >= CHANCE
        and np.isfinite(after_both)
        and after_both >= CHANCE
    )
    died = bool(
        (np.isfinite(after_dso) and after_dso < CHANCE)
        or (np.isfinite(after_iss) and after_iss < CHANCE)
        or (np.isfinite(after_both) and after_both < CHANCE)
    )
    prose = (
        f"Y7 delay_coll leftover after DSO {_f(after_dso)}, "
        f"after issued_lag1 {_f(after_iss)}, after both {_f(after_both)}. "
        f"AR overdue leftover after DSO {_f(after_od_dso)} after issued_lag1 {_f(after_od_iss)}. "
        f"{'Leftover survives ≥0.55 after DSO and issued_lag1' if lives else 'Leftover dies — CLOSE as add-on'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "after_dso": after_dso,
        "after_iss": after_iss,
        "after_both": after_both,
        "after_od_dso": after_od_dso,
        "after_od_iss": after_od_iss,
        "after_paid_dso": _cv(store["e_delay_paid after DSO"]),
        "after_paid_iss": _cv(store["e_delay_paid after issued_lag1"]),
        "lives": lives,
        "died": died,
        "resid_dso": resids.get("after DSO"),
        "resid_iss": resids.get("after issued_lag1"),
        "resid_both": resids.get("after DSO+issued_lag1"),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 6 — residual Y3 after days
# ---------------------------------------------------------------------------
def pass6_resid_y3(tr: pd.DataFrame) -> dict:
    lab = tr[Y3].notna()
    stems = {
        "e_delay_coll": tr["e_delay_coll"],
        "e_delay_paid": tr["e_delay_paid"],
        "e_ar_overdue": tr["e_ar_overdue"],
        "e_ap_overdue": tr["e_ap_overdue"],
        "e_ar_overdue_30": tr["e_ar_overdue_30"],
        "e_ap_overdue_30": tr["e_ap_overdue_30"],
    }
    rows = []
    store = {}
    for sname, s in stems.items():
        for cname, xs in (
            ("after days", (tr["c_n_days_with_tx"],)),
            ("after size", (tr["log_in3"],)),
            ("after DSO", (tr["e_dso_proxy"],)),
            ("after days+size", (tr["c_n_days_with_tx"], tr["log_in3"])),
        ):
            resid, info = ols_resid(s, *xs)
            res = signed_oof_auroc(tr[Y3], resid, tr["fold"], lab)
            store[(sname, cname)] = res
            rows.append(
                {
                    "stem": sname,
                    "residual": cname,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "R²": _f(info["r2"]),
                }
            )
    after_days = _cv(store[("e_delay_coll", "after days")])
    after_size = _cv(store[("e_delay_coll", "after size")])
    size_y3 = _cv(
        signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], lab)
    )
    lives = bool(np.isfinite(after_days) and np.isfinite(size_y3) and after_days >= size_y3 + KEEP_DELTA)
    died = bool(not lives)
    prose = (
        f"Y3 delay_coll leftover after days {_f(after_days)} vs size {_f(size_y3)} "
        f"(Δ {_f(after_days - size_y3 if np.isfinite(after_days) and np.isfinite(size_y3) else float('nan'), 3)}). "
        f"{'Unexpected leftover ≥ size+0.02' if lives else 'Expect CLOSE as Y3 X — leftover after days dies'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "after_days": after_days,
        "after_size": after_size,
        "after_od_days": _cv(store[("e_ar_overdue", "after days")]),
        "size_y3": size_y3,
        "lives": lives,
        "died": died,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 7 — 3m-vs-all-open leftover
# ---------------------------------------------------------------------------
def pass7_window(tr: pd.DataFrame) -> dict:
    lab7 = tr[Y7].notna()
    lab3 = tr[Y3].notna()
    gap_ar = pd.to_numeric(tr["e_ar_overdue"], errors="coerce") - pd.to_numeric(
        tr["ar_od_3m"], errors="coerce"
    )
    gap_ap = pd.to_numeric(tr["e_ap_overdue"], errors="coerce") - pd.to_numeric(
        tr["ap_od_3m"], errors="coerce"
    )
    resid_ar, info_ar = ols_resid(tr["e_ar_overdue"], tr["ar_od_3m"])
    resid_ap, info_ap = ols_resid(tr["e_ap_overdue"], tr["ap_od_3m"])
    rows = []
    store = {}
    for y, lab in ((Y7, lab7), (Y3, lab3)):
        for name, col in (
            ("e_ar_overdue all-open", tr["e_ar_overdue"]),
            ("ar_od_3m", tr["ar_od_3m"]),
            ("store−3m AR", gap_ar),
            ("AR resid after 3m", resid_ar),
            ("e_ap_overdue all-open", tr["e_ap_overdue"]),
            ("ap_od_3m", tr["ap_od_3m"]),
            ("store−3m AP", gap_ap),
            ("AP resid after 3m", resid_ap),
        ):
            res = signed_oof_auroc(tr[y], col, tr["fold"], lab)
            store[(y, name)] = res
            rows.append(_auc_row(y, name, res))
    after_ar = _cv(store[(Y7, "AR resid after 3m")])
    after_ap = _cv(store[(Y7, "AP resid after 3m")])
    gap_ar7 = _cv(store[(Y7, "store−3m AR")])
    died = bool(
        (np.isfinite(after_ar) and after_ar < CHANCE)
        and (np.isfinite(after_ap) and after_ap < CHANCE)
    )
    lives = bool(np.isfinite(after_ar) and after_ar >= CHANCE)
    prose = (
        f"Y7 leftover of store AR after 3m-window {_f(after_ar)} "
        f"(raw store−3m {_f(gap_ar7)}, R²={_f(info_ar['r2'])}); "
        f"AP after 3m {_f(after_ap)}. "
        f"{'CLOSE window leftover lives — aging stock is a lever' if lives else 'Leftover dies — the CLOSE 3m-vs-all-open window is not a health lever'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "after_ar": after_ar,
        "after_ap": after_ap,
        "gap_ar7": gap_ar7,
        "r2_ar": info_ar["r2"],
        "r2_ap": info_ap["r2"],
        "lives": lives,
        "died": died,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 8 — Q6 lag1/lag3 short vs long. Delay empty until month 7.
# ---------------------------------------------------------------------------
def pass8_q6(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    slices = [
        ("all", tr[Y7].notna()),
        ("short_<12_sofar", tr[Y7].notna() & (tr["so_far_class"] == "short_<12")),
        ("long_>=18_sofar", tr[Y7].notna() & (tr["so_far_class"] == "long_>=18")),
        ("short_<12_company", tr[Y7].notna() & (tr["co_class"] == "short_<12")),
        ("long_>=18_company", tr[Y7].notna() & (tr["co_class"] == "long_>=18")),
        ("early6_calendar", tr[Y7].notna() & tr["early6"]),
        ("after_month7", tr[Y7].notna() & ~tr["early6"]),
    ]
    cols = (
        "e_delay_coll",
        "e_delay_coll_lag1",
        "e_delay_coll_lag3",
        "e_delay_paid",
        "e_delay_paid_lag1",
        "e_ar_issued_lag1",
        "e_ar_overdue",
        "e_ar_overdue_lag1",
    )
    for sname, mask in slices:
        n_lab = int(mask.sum())
        for col in cols:
            if col not in tr.columns:
                continue
            res = signed_oof_auroc(tr[Y7], tr[col], tr["fold"], mask)
            store[(sname, col)] = res
            present = float(tr.loc[mask, col].notna().mean()) if n_lab else float("nan")
            rows.append(
                {
                    "slice": sname,
                    "col": col,
                    "n_nn": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "present": _pp(present),
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
    short_lag = store.get(("short_<12_sofar", "e_delay_coll_lag1"))
    short_now = store.get(("short_<12_sofar", "e_delay_coll"))
    short_cov = (
        float(
            tr.loc[tr[Y7].notna() & (tr["so_far_class"] == "short_<12"), "e_delay_coll"].notna().mean()
        )
        if (tr["so_far_class"] == "short_<12").any()
        else float("nan")
    )
    early_cov = (
        float(tr.loc[tr[Y7].notna() & tr["early6"], "e_delay_coll"].notna().mean())
        if tr["early6"].any()
        else float("nan")
    )
    empty_short = bool(np.isfinite(short_cov) and short_cov < 0.15)
    empty_early = bool(np.isfinite(early_cov) and early_cov == 0.0)
    short_cv = _cv(short_lag) if short_lag else float("nan")
    keep_q6 = bool(
        not empty_short
        and not empty_early
        and np.isfinite(short_cv)
        and short_cv >= ISSUED_LAG1_BENCH - 0.03
        and short_cv >= CHANCE
    )
    prose = (
        f"Delay finite share on short so-far Y7 {_pp(short_cov)}; "
        f"on first-6 calendar Y7 {_pp(early_cov)} "
        f"({'CONFIRM empty until month 7' if empty_early else 'not empty'}). "
        f"Y7 delay lag1 short {_f(short_cv)}. "
        f"{'KEEP as Q6' if keep_q6 else 'CLOSE as Q6 — empty on the first 6 months / short books'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "short_lag": short_cv,
        "short_now": _cv(short_now) if short_now else float("nan"),
        "short_cov": short_cov,
        "early_cov": early_cov,
        "empty_short": empty_short,
        "empty_early": empty_early,
        "keep_q6": keep_q6,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 9 — fold 4 Y7 (DSO failed; TURNOVER 0.680)
# ---------------------------------------------------------------------------
def pass9_fold4(tr: pd.DataFrame, p5: dict) -> dict:
    lab = tr[Y7].notna()
    f4 = lab & (tr["fold"] == 4)
    rows = []
    f4_rows = []
    specs = [
        ("e_delay_coll", tr["e_delay_coll"]),
        ("e_delay_paid", tr["e_delay_paid"]),
        ("e_ar_overdue", tr["e_ar_overdue"]),
        ("e_ar_overdue_30", tr["e_ar_overdue_30"]),
        ("e_dso_proxy", tr["e_dso_proxy"]),
        ("e_ar_issued_lag1", tr["e_ar_issued_lag1"]),
        ("delay resid after DSO", p5.get("resid_dso")),
        ("delay resid after issued_lag1", p5.get("resid_iss")),
        ("delay resid after both", p5.get("resid_both")),
    ]
    for name, col in specs:
        if col is None:
            continue
        res = signed_oof_auroc(tr[Y7], col, tr["fold"], lab)
        rows.append(
            {
                "feature": name,
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "fold4": _f(fold_k(res, 4)),
                "folds": fold_bits(res) if not res["low_power"] else "—",
            }
        )
        trm = lab & (tr["fold"] != 4) & pd.to_numeric(col, errors="coerce").notna()
        vam = f4 & pd.to_numeric(col, errors="coerce").notna()
        sign = choose_sign(tr[Y7][trm], col[trm])
        auc = auroc(tr[Y7][vam], sign * pd.to_numeric(col[vam], errors="coerce"))
        f4_rows.append(
            {
                "feature": name,
                "n_va": int(vam.sum()),
                "n_pos": int((vam & (tr[Y7] == 1)).sum()),
                "fold4": _f(auc),
                "sign": int(sign),
            }
        )
    grp = []
    for gid in FOLD4_GROUPS:
        sl = lab & (tr["group_id"].astype(str) == gid)
        y = pd.to_numeric(tr.loc[sl, Y7], errors="coerce")
        delay = pd.to_numeric(tr.loc[sl, "e_delay_coll"], errors="coerce")
        dso = pd.to_numeric(tr.loc[sl, "e_dso_proxy"], errors="coerce")
        grp.append(
            {
                "group": gid,
                "n_lab": int(sl.sum()),
                "n_pos": int((y == 1).sum()),
                "rate": _pp(float(y.mean()) if y.notna().any() else float("nan")),
                "delay p50": _f(float(delay.median()) if delay.notna().any() else float("nan"), 2),
                "DSO p50": _f(float(dso.median()) if dso.notna().any() else float("nan"), 2),
            }
        )

    def _f4(name: str) -> float:
        hit = next((r for r in f4_rows if r["feature"] == name), None)
        if hit is None or hit["fold4"] == "—":
            return float("nan")
        return float(hit["fold4"])

    delay4 = _f4("e_delay_coll")
    dso4 = _f4("e_dso_proxy")
    iss4 = _f4("e_ar_issued_lag1")
    resid4 = _f4("delay resid after both")
    delay_saves = bool(np.isfinite(delay4) and delay4 >= TURNOVER_F4 - 0.02)
    same_hole = bool(
        np.isfinite(delay4)
        and np.isfinite(dso4)
        and abs(delay4 - dso4) < 0.04
        and delay4 < 0.60
    )
    issued_owns = bool(np.isfinite(iss4) and iss4 >= 0.60 and (not np.isfinite(delay4) or delay4 < 0.60))
    prose = (
        f"Fold 4 (sign from 0–3): delay_coll {_f(delay4)}, DSO {_f(dso4)}, "
        f"issued_lag1 {_f(iss4)}, delay resid-both {_f(resid4)}. "
        f"TURNOVER fold-4 quote 0.680. "
        + (
            "Delay saves fold 4."
            if delay_saves
            else (
                "Same short-DSO hole as DSO."
                if same_hole
                else (
                    "TURNOVER fold-4 0.680 is issued, not delay."
                    if issued_owns
                    else "Delay does not own fold 4."
                )
            )
        )
    )
    print(prose)
    return {
        "rows": rows,
        "f4_rows": f4_rows,
        "grp": grp,
        "delay4": delay4,
        "dso4": dso4,
        "iss4": iss4,
        "resid4": resid4,
        "delay_saves": delay_saves,
        "same_hole": same_hole,
        "issued_owns": issued_owns,
        "n_f4": int(f4.sum()),
        "n_f4_pos": int((f4 & (tr[Y7] == 1)).sum()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 10 — ICC / company-demean
# ---------------------------------------------------------------------------
def pass10_icc(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    iccs = {}
    for col in ("e_delay_coll", "e_delay_paid", "e_ar_overdue", "e_dso_proxy"):
        icc = icc_anova(tr[col], tr["company_id"])
        iccs[col] = icc
        demean = company_demean(tr[col], tr["company_id"])
        mean_co = tr.groupby("company_id")[col].transform("mean")
        for y in (Y7, Y3):
            lab = tr[y].notna()
            raw = signed_oof_auroc(tr[y], tr[col], tr["fold"], lab)
            de = signed_oof_auroc(tr[y], demean, tr["fold"], lab)
            mu = signed_oof_auroc(tr[y], mean_co, tr["fold"], lab)
            store[(y, col, "raw")] = raw
            store[(y, col, "demean")] = de
            store[(y, col, "co_mean")] = mu
            rows.append(_auc_row(y, f"{col} raw", raw))
            rows.append(_auc_row(y, f"{col} demean", de))
            rows.append(_auc_row(y, f"{col} company-mean", mu))
    y7_de = _cv(store[(Y7, "e_delay_coll", "demean")])
    y7_mu = _cv(store[(Y7, "e_delay_coll", "co_mean")])
    shock = bool(np.isfinite(y7_de) and y7_de >= CHANCE)
    trait = bool(np.isfinite(y7_mu) and y7_mu >= CHANCE and (not np.isfinite(y7_de) or y7_de < CHANCE))
    style = bool(np.isfinite(iccs["e_delay_coll"]["icc"]) and iccs["e_delay_coll"]["icc"] >= ICC_STYLE)
    prose = (
        f"delay_coll ICC={_f(iccs['e_delay_coll']['icc'])} k={iccs['e_delay_coll']['k']}; "
        f"DSO ICC={_f(iccs['e_dso_proxy']['icc'])}. "
        f"Y7 demean {_f(y7_de)} company-mean {_f(y7_mu)}. "
        + (
            "Within-company shock leftover lives."
            if shock
            else ("Company-mean carries the skill — style / truncation dummy." if trait else "Neither demean nor mean is leftover.")
        )
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "icc_delay": iccs["e_delay_coll"]["icc"],
        "icc_dso": iccs["e_dso_proxy"]["icc"],
        "icc_od": iccs["e_ar_overdue"]["icc"],
        "y7_de": y7_de,
        "y7_mu": y7_mu,
        "y3_de": _cv(store[(Y3, "e_delay_coll", "demean")]),
        "shock": shock,
        "trait": trait,
        "style": style,
        "acf1": median_acf(tr["e_delay_coll"], tr["company_id"], 1),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 11 — holdout coverage only
# ---------------------------------------------------------------------------
def pass11_holdout(panel: pd.DataFrame, book: set[str]) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    y7 = pd.to_numeric(ho[Y7], errors="coerce")
    y3 = pd.to_numeric(ho[Y3], errors="coerce")
    dark = ~ho["company_id"].isin(book)
    rows = []
    for col in DELAY_FEATS:
        x = pd.to_numeric(ho[col], errors="coerce")
        early = ho["early6"]
        rows.append(
            {
                "col": col,
                "n_cm": f"{len(ho):,}",
                "nn": f"{int(x.notna().sum()):,}",
                "cov": _pp(float(x.notna().mean())),
                "early6 nn": _pp(float(x[early].notna().mean()) if early.any() else float("nan")),
                "dark nn": int(x[dark].notna().sum()),
            }
        )
    y7_pos = int((y7 == 1).sum())
    delay = pd.to_numeric(ho["e_delay_coll"], errors="coerce")
    dark_nn = int(delay[dark].notna().sum())
    early_nn = float(delay[ho["early6"]].notna().mean()) if ho["early6"].any() else float("nan")
    prose = (
        f"Holdout 72 coverage only: {len(ho):,} CM / {ho['company_id'].nunique()} companies. "
        f"Y7 pos={y7_pos} (quote 122). delay_coll cov {_pp(float(delay.notna().mean()))}; "
        f"early6 {_pp(early_nn)}; dark nn={dark_nn}. No AUROC claim."
    )
    print(prose)
    return {
        "rows": rows,
        "n_cm": int(len(ho)),
        "n_co": int(ho["company_id"].nunique()),
        "y7_pos": y7_pos,
        "dark_nn": dark_nn,
        "early_nn": early_nn,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 12 — first-6 NaNs vs short books
# ---------------------------------------------------------------------------
def pass12_short(tr: pd.DataFrame) -> dict:
    delay = pd.to_numeric(tr["e_delay_coll"], errors="coerce")
    rows = []
    for sname, mask in (
        ("calendar first-6", tr["early6"]),
        ("calendar after-6", ~tr["early6"]),
        ("so_far <7", tr["months_so_far"] < 7),
        ("so_far 7-11", (tr["months_so_far"] >= 7) & (tr["months_so_far"] < 12)),
        ("so_far >=12", tr["months_so_far"] >= 12),
        ("company short_<12", tr["co_class"] == "short_<12"),
        ("company long_>=18", tr["co_class"] == "long_>=18"),
    ):
        sl = delay[mask]
        rows.append(
            {
                "slice": sname,
                "n_cm": int(mask.sum()),
                "nn": int(sl.notna().sum()),
                "cov": _pp(float(sl.notna().mean()) if mask.any() else float("nan")),
                "p50": _f(float(sl.median()) if sl.notna().any() else float("nan"), 2),
            }
        )
    # companies whose whole book sits inside the mask
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    never_delay = last["e_delay_coll"].isna()
    n_never = int(never_delay.sum())
    short_never = int((never_delay & (last["n_grid_months"] < 12)).sum())
    prose = (
        f"Companies never finite delay_coll: {n_never} / {len(last)} "
        f"(of which short-book {short_never}). "
        "Calendar mask (not company age) zeros the first 6 months for everyone."
    )
    print(prose)
    return {"rows": rows, "n_never": n_never, "short_never": short_never, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 13 — DSO quintiles / short-DSO fifth (the Y7 wound)
# ---------------------------------------------------------------------------
def pass13_dso_q(tr: pd.DataFrame) -> dict:
    sl = tr.loc[tr[Y7].notna() & tr["e_dso_proxy"].notna()].copy()
    dso = pd.to_numeric(sl["e_dso_proxy"], errors="coerce").clip(upper=24)
    sl["dso_q"] = pd.qcut(dso.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    rows = []
    store = {}
    for q, part in sl.groupby("dso_q", observed=False):
        idx = part.index
        for name, col in (
            ("e_delay_coll", tr["e_delay_coll"]),
            ("e_dso_proxy", tr["e_dso_proxy"]),
            ("e_ar_issued_lag1", tr["e_ar_issued_lag1"]),
        ):
            mask = pd.Series(False, index=tr.index)
            mask.loc[idx] = True
            mask = mask & tr[Y7].notna()
            res = signed_oof_auroc(tr[Y7], col, tr["fold"], mask)
            store[(str(q), name)] = res
        y = pd.to_numeric(part[Y7], errors="coerce")
        delay = pd.to_numeric(part["e_delay_coll"], errors="coerce")
        rows.append(
            {
                "DSO q": str(q),
                "n": int(len(part)),
                "rate": _pp(float(y.mean()) if y.notna().any() else float("nan")),
                "DSO p50": _f(float(dso.loc[idx].median()), 2),
                "delay nn": _pp(float(delay.notna().mean())),
                "delay CV": "LOW_POWER" if store[(str(q), "e_delay_coll")]["low_power"] else _f(store[(str(q), "e_delay_coll")]["cv"]),
                "DSO CV": "LOW_POWER" if store[(str(q), "e_dso_proxy")]["low_power"] else _f(store[(str(q), "e_dso_proxy")]["cv"]),
                "issued CV": "LOW_POWER" if store[(str(q), "e_ar_issued_lag1")]["low_power"] else _f(store[(str(q), "e_ar_issued_lag1")]["cv"]),
            }
        )
    q1 = store[("Q1", "e_delay_coll")]
    q1_dso = store[("Q1", "e_dso_proxy")]
    q1_delay = _cv(q1)
    same_wound = bool(np.isfinite(q1_delay) and q1_delay < 0.55)
    prose = (
        f"Short-DSO fifth (Q1): delay CV {_f(q1_delay)} vs DSO {_f(_cv(q1_dso))}. "
        f"{'Same short-DSO hole — delay does not save Q1' if same_wound else 'Delay lifts the short-DSO fifth'}."
    )
    print(prose)
    return {
        "rows": rows,
        "q1_delay": q1_delay,
        "q1_dso": _cv(q1_dso),
        "same_wound": same_wound,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 14 — delay vs overdue_30 (Banque de France twin?)
# ---------------------------------------------------------------------------
def pass14_od30(tr: pd.DataFrame, p3: dict) -> dict:
    rho = p3["delay_od30"]
    twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    lab = tr[Y7].notna()
    resid, info = ols_resid(tr["e_delay_coll"], tr["e_ar_overdue_30"])
    res = signed_oof_auroc(tr[Y7], resid, tr["fold"], lab)
    prose = (
        f"delay_coll vs ar_overdue_30 ρ={_f(rho)} "
        f"({'TWIN' if twin else 'not a twin'}). "
        f"Y7 leftover after overdue_30 {_f(_cv(res))} R²={_f(info['r2'])}."
    )
    print(prose)
    return {
        "rho": rho,
        "twin": twin,
        "after": _cv(res),
        "r2": info["r2"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 15 — 12 chronic Y2 names, Y3 only
# ---------------------------------------------------------------------------
def pass15_chronic(tr: pd.DataFrame) -> dict:
    ids = chronic_ids(tr)
    drop = ~tr["company_id"].astype(str).isin(set(ids))
    lab = tr[Y3].notna()
    full = signed_oof_auroc(tr[Y3], tr["e_delay_coll"], tr["fold"], lab)
    rest = signed_oof_auroc(tr[Y3], tr["e_delay_coll"], tr["fold"], lab & drop)
    days_f = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], lab)
    days_r = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], lab & drop)
    flip = False
    if not full["low_power"] and not rest["low_power"]:
        flip = abs(full["cv"] - rest["cv"]) >= 0.03
    rows = [
        {
            "slice": "Y3 all",
            "delay": "LOW_POWER" if full["low_power"] else _f(full["cv"]),
            "days": "LOW_POWER" if days_f["low_power"] else _f(days_f["cv"]),
        },
        {
            "slice": "Y3 drop-12",
            "delay": "LOW_POWER" if rest["low_power"] else _f(rest["cv"]),
            "days": "LOW_POWER" if days_r["low_power"] else _f(days_r["cv"]),
        },
    ]
    prose = (
        f"Chronic 12 Y2 names: {len(ids)}. Y3 delay {_f(_cv(full))} → drop-12 {_f(_cv(rest))} "
        f"({'flips ≥0.03' if flip else 'does not flip'}). No Y2 AUROC claim."
    )
    print(prose)
    return {
        "n_ids": len(ids),
        "rows": rows,
        "y3_full": _cv(full),
        "y3_drop": _cv(rest),
        "flip": flip,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 16 — Y7 / Y3 quintile rates
# ---------------------------------------------------------------------------
def pass16_quintiles(tr: pd.DataFrame) -> dict:
    rows = []
    for y in (Y7, Y3):
        sl = tr.loc[tr[y].notna() & tr["e_delay_coll"].notna()].copy()
        if sl.empty:
            continue
        sl["q"] = pd.qcut(
            pd.to_numeric(sl["e_delay_coll"], errors="coerce").rank(method="first"),
            5,
            labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        )
        for q, part in sl.groupby("q", observed=False):
            s = pd.to_numeric(part[y], errors="coerce")
            rows.append(
                {
                    "y": y,
                    "delay q": str(q),
                    "n": int(len(part)),
                    "n_pos": int((s == 1).sum()),
                    "rate": _pp(float(s.mean()) if s.notna().any() else float("nan")),
                    "delay p50": _f(float(pd.to_numeric(part["e_delay_coll"], errors="coerce").median()), 2),
                }
            )
    prose = "Y7 / Y3 rates by delay_coll quintile (train labeled, delay finite)."
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 17 — leftover after DSO on ERP-only / after-month-7
# ---------------------------------------------------------------------------
def pass17_slices(tr: pd.DataFrame, book: set[str], p5: dict) -> dict:
    lab = tr[Y7].notna()
    resid = p5.get("resid_both")
    rows = []
    for sname, mask in (
        ("all labeled", lab),
        ("after month 7", lab & ~tr["early6"]),
        ("ever ERP", lab & tr["company_id"].isin(book)),
        ("delay finite", lab & tr["e_delay_coll"].notna()),
    ):
        raw = signed_oof_auroc(tr[Y7], tr["e_delay_coll"], tr["fold"], mask)
        if resid is not None:
            lef = signed_oof_auroc(tr[Y7], resid, tr["fold"], mask)
        else:
            lef = {"low_power": True, "cv": float("nan"), "n_defined": 0, "n_pos": 0, "sd": float("nan"), "train_sign": 0, "folds": []}
        rows.append(
            {
                "slice": sname,
                "raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "leftover both": "LOW_POWER" if lef["low_power"] else _f(lef["cv"]),
                "n": f"{raw['n_defined']:,}",
                "n_pos": f"{raw['n_pos']:,}",
            }
        )
    prose = "Y7 delay raw vs leftover-after-DSO+issued_lag1 inside ERP / post-truncation slices."
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 18 — delay clip pile (−30 / 120)
# ---------------------------------------------------------------------------
def pass18_clip(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["e_delay_coll"], errors="coerce")
    nn = x.dropna()
    lo = float(DELAY_CLIP[0])
    hi = float(DELAY_CLIP[1])
    n_lo = int((nn <= lo + 1e-9).sum())
    n_hi = int((nn >= hi - 1e-9).sum())
    rows = [
        {"edge": f"clip lo {lo:g}", "n": n_lo, "share_nn": _pp(_pct(n_lo, len(nn)))},
        {"edge": f"clip hi {hi:g}", "n": n_hi, "share_nn": _pp(_pct(n_hi, len(nn)))},
        {"edge": "interior", "n": int(len(nn) - n_lo - n_hi), "share_nn": _pp(_pct(len(nn) - n_lo - n_hi, len(nn)))},
    ]
    pile = bool(_pct(n_lo + n_hi, len(nn)) >= 0.15) if len(nn) else False
    prose = (
        f"delay_coll clip pile: lo {n_lo:,} hi {n_hi:,} / {len(nn):,} defined "
        f"({'clip is a pile' if pile else 'clip is a tail'})."
    )
    print(prose)
    return {"rows": rows, "n_lo": n_lo, "n_hi": n_hi, "pile": pile, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 19 — same-n raw vs leftover (is leftover just the raw 0.57?)
# ---------------------------------------------------------------------------
def pass19_samen(tr: pd.DataFrame, p5: dict) -> dict:
    lab = (
        tr[Y7].notna()
        & tr["e_delay_coll"].notna()
        & tr["e_dso_proxy"].notna()
        & tr["e_ar_issued_lag1"].notna()
    )
    raw = signed_oof_auroc(tr[Y7], tr["e_delay_coll"], tr["fold"], lab)
    iss = signed_oof_auroc(tr[Y7], tr["e_ar_issued_lag1"], tr["fold"], lab)
    dso = signed_oof_auroc(tr[Y7], tr["e_dso_proxy"], tr["fold"], lab)
    rows = [_auc_row(Y7, "delay same-n", raw), _auc_row(Y7, "issued_lag1 same-n", iss), _auc_row(Y7, "DSO same-n", dso)]
    leftover_cv = float("nan")
    if p5.get("resid_both") is not None:
        lef = signed_oof_auroc(tr[Y7], p5["resid_both"], tr["fold"], lab)
        rows.append(_auc_row(Y7, "leftover both same-n", lef))
        leftover_cv = _cv(lef)
    raw_cv = _cv(raw)
    artifact = bool(np.isfinite(raw_cv) and np.isfinite(leftover_cv) and abs(raw_cv - leftover_cv) < 0.02)
    prose = (
        f"Same-n (delay+DSO+issued_lag1 finite): raw delay {_f(raw_cv)} leftover {_f(leftover_cv)} "
        f"issued {_f(_cv(iss))} DSO {_f(_cv(dso))}. "
        f"{'Leftover ≈ raw — residualizing DSO/issued does not create a new object.' if artifact else 'Leftover moves vs raw.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "raw": raw_cv,
        "leftover": leftover_cv,
        "iss": _cv(iss),
        "dso": _cv(dso),
        "artifact": artifact,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 20 — company-mean vs demean leftover after DSO
# ---------------------------------------------------------------------------
def pass20_style(tr: pd.DataFrame) -> dict:
    delay = pd.to_numeric(tr["e_delay_coll"], errors="coerce")
    mu = delay.groupby(tr["company_id"]).transform("mean")
    de = delay - mu
    resid_mu, info_mu = ols_resid(mu, tr["e_dso_proxy"])
    resid_de, info_de = ols_resid(de, tr["e_dso_proxy"])
    lab = tr[Y7].notna()
    rows = []
    store = {}
    for name, col in (
        ("company-mean delay", mu),
        ("demean delay", de),
        ("mean after DSO", resid_mu),
        ("demean after DSO", resid_de),
    ):
        res = signed_oof_auroc(tr[Y7], col, tr["fold"], lab)
        store[name] = res
        rows.append(_auc_row(Y7, name, res))
    mean_cv = _cv(store["company-mean delay"])
    de_cv = _cv(store["demean delay"])
    style = bool(np.isfinite(mean_cv) and mean_cv >= CHANCE and (not np.isfinite(de_cv) or de_cv < CHANCE))
    prose = (
        f"Y7 company-mean delay {_f(mean_cv)} demean {_f(de_cv)}; "
        f"mean|DSO {_f(_cv(store['mean after DSO']))} demean|DSO {_f(_cv(store['demean after DSO']))}. "
        f"{'BETWEEN who-pays-late style — not a month dip-vs-fall shock.' if style else 'Within-company shock leftover lives.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "mean_cv": mean_cv,
        "de_cv": de_cv,
        "style": style,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 21 — fold-wise leftover
# ---------------------------------------------------------------------------
def pass21_folds(tr: pd.DataFrame, p5: dict) -> dict:
    lab = tr[Y7].notna()
    rows = []
    for name, col in (
        ("e_delay_coll", tr["e_delay_coll"]),
        ("e_ar_issued_lag1", tr["e_ar_issued_lag1"]),
        ("e_dso_proxy", tr["e_dso_proxy"]),
        ("leftover both", p5.get("resid_both")),
    ):
        if col is None:
            continue
        res = signed_oof_auroc(tr[Y7], col, tr["fold"], lab)
        rows.append(
            {
                "feature": name,
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "folds": fold_bits(res) if not res["low_power"] else "—",
                "fold4": _f(fold_k(res, 4)),
            }
        )
    prose = "Y7 fold-wise delay vs leftover vs issued_lag1 / DSO."
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 22 — Q6 after month 7 only (honest lead among defined delay)
# ---------------------------------------------------------------------------
def pass22_q6_late(tr: pd.DataFrame) -> dict:
    mask = tr[Y7].notna() & ~tr["early6"]
    rows = []
    store = {}
    for col in (
        "e_delay_coll",
        "e_delay_coll_lag1",
        "e_delay_coll_lag3",
        "e_ar_issued_lag1",
        "e_ar_overdue",
        "e_ar_overdue_lag1",
    ):
        if col not in tr.columns:
            continue
        res = signed_oof_auroc(tr[Y7], tr[col], tr["fold"], mask)
        store[col] = res
        present = float(tr.loc[mask, col].notna().mean()) if mask.any() else float("nan")
        rows.append(_auc_row(Y7, col, res, present=present))
    now = _cv(store.get("e_delay_coll", {"low_power": True}))
    lag1 = _cv(store.get("e_delay_coll_lag1", {"low_power": True}))
    iss = _cv(store.get("e_ar_issued_lag1", {"low_power": True}))
    keep = bool(np.isfinite(lag1) and lag1 >= ISSUED_LAG1_BENCH - 0.03 and lag1 >= CHANCE and np.isfinite(now) and now >= CHANCE)
    prose = (
        f"After month 7 only: delay now {_f(now)} lag1 {_f(lag1)} vs issued_lag1 {_f(iss)}. "
        f"{'KEEP as Q6 among defined delay' if keep else 'CLOSE as Q6 — lag1 does not hold issued_lag1 0.630 even after the mask'}."
    )
    print(prose)
    return {
        "rows": rows,
        "now": now,
        "lag1": lag1,
        "iss": iss,
        "keep": keep,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 23 — who-is-late company terciles
# ---------------------------------------------------------------------------
def pass23_who(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", sort=False)
    mu = last["e_delay_coll"].mean().rename("delay_mu")
    mu = mu.dropna()
    if mu.empty or mu.nunique() < 3:
        return {"rows": [], "prose": "Too few companies with finite delay mean.", "spread": float("nan")}
    terc = pd.qcut(mu, 3, labels=["late-low", "late-mid", "late-high"], duplicates="drop")
    m = tr.merge(terc.rename("late_t").reset_index(), on="company_id", how="left")
    rows = []
    for y in (Y7, Y3):
        for labv, part in m.groupby("late_t", observed=False):
            s = pd.to_numeric(part[y], errors="coerce")
            rows.append(
                {
                    "y": y,
                    "tercile": str(labv),
                    "n_cm": int(len(part)),
                    "n_co": int(part["company_id"].nunique()),
                    "n_lab": int(s.notna().sum()),
                    "n_pos": int((s == 1).sum()),
                    "rate": _pp(float(s.mean()) if s.notna().any() else float("nan")),
                }
            )
    y7 = [r for r in rows if r["y"] == Y7]
    rates = []
    for r in y7:
        try:
            rates.append(float(r["rate"].rstrip("%")) / 100.0)
        except (TypeError, ValueError):
            rates.append(float("nan"))
    spread = (max(rates) - min(rates)) if rates and all(np.isfinite(r) for r in rates) else float("nan")
    prose = (
        f"Company-mean delay terciles: Y7 rate spread {_pp(spread)}. "
        "A BETWEEN late-payer style would show a monotone Y7 gap."
    )
    print(prose)
    return {"rows": rows, "spread": spread, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 24 — overdue as Y3 leftover (near size 0.617)
# ---------------------------------------------------------------------------
def pass24_od_y3(tr: pd.DataFrame, p4: dict, p6: dict) -> dict:
    arod = p4["arod_y3"]
    days = p4["days_y3"]
    size = p4["size_y3"]
    after = p6["after_od_days"]
    lose = bool(np.isfinite(arod) and np.isfinite(days) and arod < days)
    leftover_die = bool(not np.isfinite(after) or after < size + KEEP_DELTA)
    prose = (
        f"Y3 ar_overdue {_f(arod)} vs days {_f(days)} vs size {_f(size)}; "
        f"leftover after days {_f(after)}. "
        f"{'CLOSE / DROP from the 44' if lose or leftover_die else 'unexpected leftover'}."
    )
    print(prose)
    return {
        "arod": arod,
        "after": after,
        "lose": lose,
        "leftover_die": leftover_die,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 25 — delay_paid / overdue Y7 leftover split
# ---------------------------------------------------------------------------
def pass25_split(p4: dict, p5: dict) -> dict:
    paid_die = bool(not np.isfinite(p5["after_paid_dso"]) or p5["after_paid_dso"] < CHANCE)
    od_die = bool(not np.isfinite(p5["after_od_dso"]) or p5["after_od_dso"] < CHANCE)
    prose = (
        f"Y7 delay_paid leftover after DSO {_f(p5['after_paid_dso'])} "
        f"({'CLOSE' if paid_die else 'lives'}); "
        f"AR overdue leftover after DSO {_f(p5['after_od_dso'])} "
        f"({'CLOSE' if od_die else 'lives'}). "
        f"Raw delay_paid {_f(p4['paid_y7'])} ar_od {_f(p4['arod_y7'])}."
    )
    print(prose)
    return {
        "paid_die": paid_die,
        "od_die": od_die,
        "paid_y7": p4["paid_y7"],
        "arod_y7": p4["arod_y7"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 26 — leftover vs issued same-n; late-joiner delay
# ---------------------------------------------------------------------------
def pass26_issued(tr: pd.DataFrame, p19: dict) -> dict:
    delta = (
        p19["leftover"] - p19["iss"]
        if np.isfinite(p19["leftover"]) and np.isfinite(p19["iss"])
        else float("nan")
    )
    beats = bool(np.isfinite(delta) and delta >= KEEP_DELTA)
    late = (~tr["early6"]) & (tr["months_so_far"] < 7)
    x = pd.to_numeric(tr.loc[late, "e_delay_coll"], errors="coerce")
    late_cov = float(x.notna().mean()) if late.any() else float("nan")
    prose = (
        f"Same-n leftover − issued_lag1 Δ={_f(delta, 3)} "
        f"({'leftover beats issued by ≥0.02' if beats else 'leftover does not beat issued on the overlap'}). "
        f"Late-joiner so_far<7 after calendar month 7: delay cov {_pp(late_cov)} "
        "(calendar mask ≠ company age)."
    )
    print(prose)
    return {
        "delta": delta,
        "beats": beats,
        "late_cov": late_cov,
        "n_late": int(late.sum()),
        "prose": prose,
    }


def make_png(tr: pd.DataFrame, p1: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    months = sorted(tr["period"].unique())
    cov = []
    for m in months:
        sl = tr[tr["period"] == m]
        x = pd.to_numeric(sl["e_delay_coll"], errors="coerce")
        cov.append((pd.Timestamp(m), 100.0 * float(x.notna().mean()) if len(sl) else np.nan))

    sl = tr.loc[tr[Y7].notna() & tr["e_delay_coll"].notna()].copy()
    sl["dq"] = pd.qcut(
        pd.to_numeric(sl["e_delay_coll"], errors="coerce").rank(method="first"),
        5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )
    sl2 = tr.loc[tr[Y7].notna() & tr["e_dso_proxy"].notna()].copy()
    sl2["sq"] = pd.qcut(
        pd.to_numeric(sl2["e_dso_proxy"], errors="coerce").clip(upper=24).rank(method="first"),
        5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )

    def _rates(df, qcol):
        out = []
        for q, part in df.groupby(qcol, observed=False):
            s = pd.to_numeric(part[Y7], errors="coerce")
            out.append((str(q), 100.0 * float(s.mean()) if s.notna().any() else np.nan))
        return out

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    xs = np.arange(len(cov))
    colors = ["#9e6b4a" if t < DELAY_MASK_BEFORE else "#1f4e79" for t, _ in cov]
    ax.bar(xs, [v for _, v in cov], color=colors)
    ax.set_xticks(xs[::3])
    ax.set_xticklabels([t.strftime("%Y-%m") for t, _ in cov][::3], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("% train CM with finite delay_coll")
    ax.set_title("Left truncation — empty until 2025-03")
    ax.axvline(5.5, color="#9e6b4a", ls="--", lw=0.8)

    ax = axes[1]
    dr = _rates(sl, "dq")
    sr = _rates(sl2, "sq")
    w = 0.38
    ax.bar(np.arange(5) - w / 2, [v for _, v in dr], width=w, color="#1f4e79", label="delay_coll")
    ax.bar(np.arange(5) + w / 2, [v for _, v in sr], width=w, color="#9e6b4a", label="DSO")
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels([x[0] for x in dr])
    ax.set_ylabel("% y7_top1_lost")
    ax.set_title("Y7 rate by quintile (train labeled)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def decide(p3, p4, p5, p6, p7, p8, p9, p10, p19=None, p20=None, p22=None, p25=None) -> dict:
    """KEEP leftover / CLOSE add-on / DROP-from-44 / PARK Y."""
    twin = bool(p3["delay_twin_dso"] or p3["delay_twin_iss"])
    leftover_ok = bool(p5["lives"] and not twin)
    leftover_dies = bool(p5["died"] or twin)
    style = bool(p10.get("trait") or (p20 is not None and p20.get("style")))
    artifact = bool(p19 is not None and p19.get("artifact"))
    if leftover_ok:
        y7 = "KEEP"
        extra = ""
        if artifact:
            extra = " Leftover ≈ same-n raw — residualizing does not create a new object."
        if style:
            extra += " ICC / demean: BETWEEN who-pays-late style, not a month shock."
        y7_why = (
            f"residual after DSO {_f(p5['after_dso'])} and after issued_lag1 {_f(p5['after_iss'])} "
            f"(both {_f(p5['after_both'])}) survives and is not a twin "
            f"(ρ DSO {_f(p3['delay_dso'])}, issued_lag1 {_f(p3['delay_iss'])}). "
            f"Still do not grow TURNOVER or change 0.720.{extra}"
        )
    else:
        y7 = "CLOSE"
        y7_why = (
            f"twin or leftover dies (after DSO {_f(p5['after_dso'])}, "
            f"after issued_lag1 {_f(p5['after_iss'])}, after both {_f(p5['after_both'])}, "
            f"ρ DSO {_f(p3['delay_dso'])}). CLOSE as Y7 add-on — do not grow TURNOVER."
        )
    if p4["lose_days"] or p6["died"]:
        y3 = "CLOSE / DROP from the 44"
        y3_why = (
            f"Y3 delay_coll {_f(p4['delay_y3'])} loses to days {_f(p4['days_y3'])} "
            f"(leftover after days {_f(p6['after_days'])})."
        )
    else:
        y3 = "KEEP"
        y3_why = (
            f"Y3 delay_coll {_f(p4['delay_y3'])} leftover after days {_f(p6['after_days'])} "
            f"≥ size+0.02 — unexpected."
        )
    q6_late = bool(p22 is not None and p22.get("keep"))
    q6 = "KEEP" if (p8["keep_q6"] or q6_late) else "CLOSE"
    window = "KEEP leftover" if p7["lives"] else "CLOSE"
    paid = "CLOSE" if (p25 is None or p25.get("paid_die")) else "KEEP"
    od = "CLOSE" if (p25 is None or p25.get("od_die")) else "KEEP"
    f4 = (
        "delay saves fold 4"
        if p9["delay_saves"]
        else (
            "same short-DSO hole"
            if p9["same_hole"]
            else ("issued owns fold 4" if p9["issued_owns"] else "delay does not own fold 4")
        )
    )
    return {
        "y7": y7,
        "y7_why": y7_why,
        "y3": y3,
        "y3_why": y3_why,
        "q6": q6,
        "window": window,
        "paid": paid,
        "od": od,
        "park_y": True,
        "leftover_ok": leftover_ok,
        "leftover_dies": leftover_dies,
        "f4": f4,
        "twin": twin,
        "style": style,
        "artifact": artifact,
    }


def write_md(ctx: dict) -> None:
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
    p6, p7, p8, p9, p10, p11 = (
        ctx["p6"],
        ctx["p7"],
        ctx["p8"],
        ctx["p9"],
        ctx["p10"],
        ctx["p11"],
    )
    d = ctx["decision"]
    lines = [
        "# Q4/Q5 delay / overdue leftover after DSO / issued_lag1",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_delay`. "
        "Night Y7 quote stays **TURNOVER 0.720 / B_shallow 0.712**. Night Y3 stays **0.762 / 0.752**. "
        "Y7 never D. Y5 never E. Delay / overdue stay off the 15-col card. Y3 never B.",
        "",
        "`e_delay_coll` / `e_delay_paid` = amount-weighted (paid − due) days on invoices paid "
        "in the trailing 3 months, clipped [−30, 120], **null the first 6 calendar months**. "
        "`e_ar_overdue` / `e_ap_overdue` = overdue |amount| / all-open unpaid. "
        "Javier overdue uses a 3-month issuance window.",
        "",
        "## Headline",
        "",
        ctx["headline"],
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        "| 1 | Who is healthy? | **PARK** as a health Y. Do not invent `y_delay`. Dark 470 = NaN, not 0. |",
        "| 2 | Who is improving? | Not this clock. |",
        "| 3 | Who is turning? | Delay lag is empty until month 7 — not a turn clock on short books. |",
        f"| 4 | Dip vs fall? | Y7 leftover after DSO / issued_lag1 **{d['y7']}** — {d['y7_why']} |",
        f"| 5 | Why did it change? | Delay on the SHAP card is Q5-shaped only if leftover lives. Fold 4: {d['f4']}. 3m window leftover **{d['window']}**. |",
        f"| 6 | Months earlier? | Delay empty first 6 calendar months — **{d['q6']}** as Q6 on short companies. |",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| delay_coll as Y7 leftover after DSO + issued_lag1 | **{d['y7']}** | {d['y7_why']} |",
        "| delay / overdue as Y7 add-on / grow TURNOVER | **CLOSE** | do not grow TURNOVER; night quote stays 0.720 / 0.712 |",
        f"| delay / overdue as Y3 X / the 44 | **{d['y3']}** | {d['y3_why']} |",
        "| delay / overdue as a health Y | **PARK** | do not invent `y_delay` |",
        f"| Q6 delay lag1/lag3 on short books | **{d['q6']}** | {p8['prose']} |",
        f"| 3m-vs-all-open leftover | **{d['window']}** | {p7['prose']} |",
        f"| delay_paid as Y7 leftover | **{d['paid']}** | leftover after DSO {_f(p5['after_paid_dso'])}; raw {_f(p4['paid_y7'])} |",
        f"| ar_overdue as Y7 leftover | **{d['od']}** | leftover after DSO {_f(p5['after_od_dso'])}; raw {_f(p4['arod_y7'])} |",
        f"| Fold 4 Y7 | **{d['f4']}** | delay {_f(p9['delay4'])} vs DSO {_f(p9['dso4'])} vs issued_lag1 {_f(p9['iss4'])} vs TURNOVER 0.680 |",
        "",
        "## 1. Coverage / nulls (first 6 empty; dark = NaN not 0)",
        "",
        p1["prose"],
        "",
        _md_table(p1["rows"]),
        "",
        f"Train CM {p1['n_cm']:,} / companies {p1['n_co']:,}. "
        f"Y7 labeled {p1['y7_n']:,} pos {p1['y7_pos']:,} rate {_pp(p1['y7_rate'])}. "
        f"Y3 stressed {p1['y3_n']:,} pos {p1['y3_pos']:,} rate {_pp(p1['y3_rate'])}.",
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx["png_ok"] else "Plot: skipped (no matplotlib).",
        "",
        "## 2. Javier confirm (delay SAME; overdue 3m SAME / all-open CLOSE)",
        "",
        p2["prose"],
        "",
        _md_table(p2["rows"]),
        "",
        "## 3. Spearman twins (|ρ|≥0.80)",
        "",
        p3["prose"],
        "",
        _md_table(p3["rows"]),
        "",
        "## 4. Single-feature train group-fold AUROC",
        "",
        p4["prose"],
        "",
        "Sign from the train side of each fold. Seed 20260918. "
        f"Night quotes: issued_lag1 0.630 (replica {_f(p4['iss_y7'])}); "
        f"days 0.711 (replica {_f(p4['days_y3'])}).",
        "",
        _md_table(p4["rows"]),
        "",
        "## 5. Residual Y7 after DSO and after issued_lag1",
        "",
        p5["prose"],
        "",
        "KEEP leftover only if residual after DSO **and** after issued_lag1 survives (≥0.55) "
        "**and** not a twin. Still do **not** grow TURNOVER.",
        "",
        _md_table(p5["rows"]),
        "",
        "## 6. Residual Y3 after days",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        "## 7. 3m-vs-all-open leftover",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 8. Q6 — lag1/lag3 on short vs long",
        "",
        p8["prose"],
        "",
        _md_table(p8["rows"]),
        "",
        "## 9. Fold 4 Y7 (DSO failed)",
        "",
        p9["prose"],
        "",
        _md_table(p9["rows"]),
        "",
        "Fold-4-only (sign from folds 0–3):",
        "",
        _md_table(p9["f4_rows"]),
        "",
        "Problem groups:",
        "",
        _md_table(p9["grp"]),
        "",
        "## 10. ICC / company-demean",
        "",
        p10["prose"],
        "",
        _md_table(p10["rows"]),
        "",
        "## 11. Holdout coverage only (no AUROC)",
        "",
        p11["prose"],
        "",
        _md_table(p11["rows"]),
        "",
        "## Extra 12 — first-6 NaNs vs short books",
        "",
        ctx["p12"]["prose"],
        "",
        _md_table(ctx["p12"]["rows"]),
        "",
        "## Extra 13 — short-DSO fifth",
        "",
        ctx["p13"]["prose"],
        "",
        _md_table(ctx["p13"]["rows"]),
        "",
        "## Extra 14 — delay vs overdue_30",
        "",
        ctx["p14"]["prose"],
        "",
        "## Extra 15 — chronic 12 (Y3 only)",
        "",
        ctx["p15"]["prose"],
        "",
        _md_table(ctx["p15"]["rows"]),
        "",
        "## Extra 16 — quintile rates",
        "",
        ctx["p16"]["prose"],
        "",
        _md_table(ctx["p16"]["rows"]),
        "",
        "## Extra 17 — leftover slices",
        "",
        ctx["p17"]["prose"],
        "",
        _md_table(ctx["p17"]["rows"]),
        "",
        "## Extra 18 — clip pile",
        "",
        ctx["p18"]["prose"],
        "",
        _md_table(ctx["p18"]["rows"]),
        "",
        "## Extra 19 — same-n raw vs leftover",
        "",
        ctx["p19"]["prose"],
        "",
        _md_table(ctx["p19"]["rows"]),
        "",
        "## Extra 20 — company-mean vs demean leftover",
        "",
        ctx["p20"]["prose"],
        "",
        _md_table(ctx["p20"]["rows"]),
        "",
        "## Extra 21 — fold-wise leftover",
        "",
        ctx["p21"]["prose"],
        "",
        _md_table(ctx["p21"]["rows"]),
        "",
        "## Extra 22 — Q6 after month 7 only",
        "",
        ctx["p22"]["prose"],
        "",
        _md_table(ctx["p22"]["rows"]),
        "",
        "## Extra 23 — who-is-late terciles",
        "",
        ctx["p23"]["prose"],
        "",
        _md_table(ctx["p23"]["rows"]),
        "",
        "## Extra 24 — overdue as Y3 leftover",
        "",
        ctx["p24"]["prose"],
        "",
        "## Extra 25 — delay_paid / overdue Y7 split",
        "",
        ctx["p25"]["prose"],
        "",
        "## Extra 26 — leftover vs issued same-n / late joiners",
        "",
        ctx["p26"]["prose"],
        "",
        "## What failed / next (held for wave note)",
        "",
        *([f"- {x}" for x in ctx["failed"]] if ctx["failed"] else ["- (none)"]),
        "",
        f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts: coverage, Javier 3m, twins, singles, "
        "Y7 leftover after DSO/issued_lag1, Y3 leftover after days, 3m window leftover, "
        "Q6 short, fold 4, ICC, holdout, first-6 vs short books, short-DSO fifth, overdue_30, "
        "chronic-12, quintiles, leftover slices, clip, same-n leftover, "
        "company-mean vs demean, fold-wise leftover, Q6 after-month-7, who-is-late, "
        "overdue Y3 leftover, delay_paid/overdue split, leftover vs issued same-n.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    p1, p2, p3, p4, p5, p6, p7, p8, d = (
        ctx["p1"],
        ctx["p2"],
        ctx["p3"],
        ctx["p4"],
        ctx["p5"],
        ctx["p6"],
        ctx["p7"],
        ctx["p8"],
        ctx["decision"],
    )
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    rows = [
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y7,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_e_delay_coll",
            "value": p4["delay_y7"],
            "coverage": f"{p1['store']['e_delay_coll']['cov']:.4f}",
            "notes": f"iss={p4['iss_y7']:.4f} dso={p4['dso_y7']:.4f} leftover_both={p5['after_both']:.4f} y7={d['y7']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y7,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_delay_resid_dso_issued",
            "value": p5["after_both"],
            "coverage": f"{p1['store']['e_delay_coll']['cov']:.4f}",
            "notes": f"after_dso={p5['after_dso']:.4f} after_iss={p5['after_iss']:.4f} lives={p5['lives']} twin_dso={p3['delay_twin_dso']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_e_delay_coll",
            "value": p4["delay_y3"],
            "coverage": f"{p1['store']['e_delay_coll']['cov']:.4f}",
            "notes": f"days={p4['days_y3']:.4f} leftover_days={p6['after_days']:.4f} y3={d['y3']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": "-",
            "model": MODEL,
            "split": "train",
            "metric": "rho_delay_coll_vs_dso",
            "value": p3["delay_dso"],
            "coverage": f"{p1['store']['e_delay_coll']['cov']:.4f}",
            "notes": f"issued_lag1={p3['delay_iss']:.4f} od30={p3['delay_od30']:.4f} size={p3['delay_size']:.4f} days={p3['delay_days']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y7,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_ar_overdue_resid_3m",
            "value": p7["after_ar"],
            "coverage": f"{p1['store']['e_ar_overdue']['cov']:.4f}",
            "notes": f"window={d['window']} r2={p7['r2_ar']:.4f} javier_ar={p2['rhos'].get('store AR vs 3m', float('nan')):.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y7,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_e_delay_coll_lag1_short",
            "value": p8["short_lag"],
            "coverage": f"{p8['short_cov']:.4f}" if np.isfinite(p8["short_cov"]) else "",
            "notes": f"q6={d['q6']} early_cov={p8['early_cov']:.4f} empty_early={p8['empty_early']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": "-",
            "model": MODEL,
            "split": "train",
            "metric": "delay_early6_finite_share",
            "value": p1["store"]["e_delay_coll"]["early_nn"],
            "coverage": "1.0000",
            "notes": f"dark_nan={p1['dark_nan']} n_dark={p1['n_dark_co']} mask_ok={p1['mask_ok']} delay_same={p2['delay_same']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y7,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_delay_coll_demean",
            "value": ctx["p10"]["y7_de"],
            "coverage": f"{p1['store']['e_delay_coll']['cov']:.4f}",
            "notes": f"icc={ctx['p10']['icc_delay']:.4f} co_mean={ctx['p10']['y7_mu']:.4f} style={ctx['p20']['style']} samen={ctx['p19']['leftover']:.4f}",
        },
    ]
    with REGISTRY.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        seen = {
            (
                r.get("agent"),
                r.get("x_families"),
                r.get("y"),
                r.get("model"),
                r.get("split"),
                r.get("metric"),
            )
            for r in reader
        }
    fresh = []
    for r in rows:
        key = (
            str(r.get("agent", "")),
            str(r.get("x_families", "")),
            str(r.get("y", "")),
            str(r.get("model", "")),
            str(r.get("split", "")),
            str(r.get("metric", "")),
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


def write_wave(ctx: dict) -> None:
    d = ctx["decision"]
    p3, p4, p5, p6, p7, p8, p9 = (
        ctx["p3"],
        ctx["p4"],
        ctx["p5"],
        ctx["p6"],
        ctx["p7"],
        ctx["p8"],
        ctx["p9"],
    )
    text = (
        f"# Wave 4 — delay / overdue leftover\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/delay_qa.py`\n"
        f"- `analysis/outputs/delay_qa.md`\n"
        f"- `analysis/outputs/delay_vs_dso.png`\n"
        f"- append-only `analysis/experiments/registry.csv` "
        f"(skip key agent+x_families+y+model+split+metric)\n"
        f"- this note\n\n"
        f"Did not touch `invoices.py`, `gbm_y7_core.py`, `credit_note_qa.py`, "
        f"`zero_in_qa.py`, `growth_qa.py`, `product/`, parquet / duckdb, "
        f"`build_targets`, parent journal, TURNOVER, or the 15-col card. "
        f"Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. "
        f"Night Y3 stays **0.762 / 0.752**. Days 0.711 untouched.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| delay_coll as Y7 leftover | **{d['y7']}** |\n"
        f"| delay / overdue as Y7 add-on | **CLOSE** (do not grow TURNOVER) |\n"
        f"| delay_paid as Y7 leftover | **{d['paid']}** |\n"
        f"| ar_overdue as Y7 leftover | **{d['od']}** |\n"
        f"| delay / overdue as Y3 X / the 44 | **{d['y3']}** |\n"
        f"| delay as a health Y | **PARK** |\n"
        f"| Q6 delay on short books | **{d['q6']}** |\n"
        f"| 3m-vs-all-open leftover | **{d['window']}** |\n\n"
        f"ρ delay vs DSO {_f(p3['delay_dso'])}. Y7 leftover after DSO {_f(p5['after_dso'])} "
        f"/ issued_lag1 {_f(p5['after_iss'])} / both {_f(p5['after_both'])}. "
        f"Same-n leftover {_f(ctx['p19']['leftover'])} ≈ raw {_f(ctx['p19']['raw'])} "
        f"≈ issued {_f(ctx['p19']['iss'])}. ICC {_f(ctx['p10']['icc_delay'])} "
        f"demean {_f(ctx['p10']['y7_de'])}. "
        f"Y3 delay {_f(p4['delay_y3'])} leftover-after-days {_f(p6['after_days'])} "
        f"vs days {_f(p4['days_y3'])}. 3m leftover {_f(p7['after_ar'])}. "
        f"Q6 short lag1 {_f(p8['short_lag'])} early cov {_pp(p8['early_cov'])} "
        f"after-month-7 lag1 {_f(ctx['p22']['lag1'])} vs issued {_f(ctx['p22']['iss'])}. "
        f"Fold 4 delay {_f(p9['delay4'])} vs DSO {_f(p9['dso4'])} vs issued {_f(p9['iss4'])}.\n\n"
        f"## What failed / next\n\n"
        + "\n".join(f"- {x}" for x in ctx["failed"])
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.parent.mkdir(parents=True, exist_ok=True)
    WAVE_NOTE.write_text(text, encoding="utf-8")
    print(f"wrote {WAVE_NOTE}")


def run() -> dict:
    t0 = time.time()
    print(f"delay_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(
        panel,
        [
            "e_delay_coll",
            "e_delay_paid",
            "e_ar_overdue",
            "e_ap_overdue",
            "e_ar_issued",
        ],
        (1, 3),
    )
    tr = panel[panel["split"] == "train"].copy()
    assert_no_holdout(tr["company_id"])
    print(
        f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
        f"holdout CM={(panel['split']=='holdout').sum()}"
    )

    con = connect()
    try:
        book = book_invoice_ids(con)
        print("reconstruct Javier 3m + delay")
        recon = reconstruct_javier(con)
    finally:
        con.close()
    tr = tr.merge(recon, on=["company_id", "period"], how="left")
    panel = panel.merge(recon, on=["company_id", "period"], how="left")

    print("pass 1 coverage")
    p1 = pass1_cov(tr, book)
    print("pass 2 Javier")
    p2 = pass2_javier(tr)
    print("pass 3 twins")
    p3 = pass3_twins(tr)
    print("pass 4 singles")
    p4 = pass4_auroc(tr)
    print("pass 5 residual Y7")
    p5 = pass5_resid_y7(tr)
    print("pass 6 residual Y3")
    p6 = pass6_resid_y3(tr)
    print("pass 7 3m window")
    p7 = pass7_window(tr)
    print("pass 8 Q6")
    p8 = pass8_q6(tr)
    print("pass 9 fold 4")
    p9 = pass9_fold4(tr, p5)
    print("pass 10 ICC")
    p10 = pass10_icc(tr)
    print("pass 11 holdout")
    p11 = pass11_holdout(panel, book)
    print("pass 12 short books")
    p12 = pass12_short(tr)
    print("pass 13 DSO quintiles")
    p13 = pass13_dso_q(tr)
    print("pass 14 overdue_30")
    p14 = pass14_od30(tr, p3)
    print("pass 15 chronic")
    p15 = pass15_chronic(tr)
    print("pass 16 quintiles")
    p16 = pass16_quintiles(tr)
    print("pass 17 leftover slices")
    p17 = pass17_slices(tr, book, p5)
    print("pass 18 clip")
    p18 = pass18_clip(tr)
    print("pass 19 same-n leftover")
    p19 = pass19_samen(tr, p5)
    print("pass 20 style leftover")
    p20 = pass20_style(tr)
    print("pass 21 fold-wise leftover")
    p21 = pass21_folds(tr, p5)
    print("pass 22 Q6 after month 7")
    p22 = pass22_q6_late(tr)
    print("pass 23 who-is-late")
    p23 = pass23_who(tr)
    print("pass 24 overdue Y3")
    p24 = pass24_od_y3(tr, p4, p6)
    print("pass 25 delay_paid / overdue split")
    p25 = pass25_split(p4, p5)
    print("pass 26 leftover vs issued")
    p26 = pass26_issued(tr, p19)

    decision = decide(p3, p4, p5, p6, p7, p8, p9, p10, p19, p20, p22, p25)
    png_ok = make_png(tr, p1)
    headline = (
        f"Delay vs DSO ρ={_f(p3['delay_dso'])} "
        f"({'TWIN' if p3['delay_twin_dso'] else 'not a twin'}). "
        f"Y7 delay_coll {_f(p4['delay_y7'])} leftover after DSO {_f(p5['after_dso'])} "
        f"/ issued_lag1 {_f(p5['after_iss'])} / both {_f(p5['after_both'])}. "
        f"Y3 {_f(p4['delay_y3'])} leftover-days {_f(p6['after_days'])} vs days {_f(p4['days_y3'])}. "
        f"3m leftover {_f(p7['after_ar'])}. Q6 short {_f(p8['short_lag'])}. "
        f"Y7 leftover **{decision['y7']}**. Y3 X **{decision['y3']}**. "
        f"Q6 **{decision['q6']}**. PARK as health Y. Do not grow TURNOVER 0.720."
    )
    print(headline)
    failed = []
    if not p1["delay_early_empty"]:
        failed.append("delay not empty on first 6 calendar months — inspect invoices.py")
    if not p1["dark_nan"]:
        failed.append(f"dark {p1['n_dark_co']} filled — want 470 NaN not 0")
    if not p2["delay_same"]:
        failed.append(f"delay vs Javier not SAME (coll {_f(p2['rhos']['delay_coll'])})")
    if abs(p4["days_y3"] - DAYS_BENCH) > 0.02 if np.isfinite(p4["days_y3"]) else True:
        failed.append(f"Y3 days replica {_f(p4['days_y3'])} vs night 0.711")
    if abs(p4["iss_y7"] - ISSUED_LAG1_BENCH) > 0.02 if np.isfinite(p4["iss_y7"]) else True:
        failed.append(f"Y7 issued_lag1 replica {_f(p4['iss_y7'])} vs night 0.630")
    if p5["died"]:
        failed.append(
            f"Y7 leftover dies after DSO {_f(p5['after_dso'])} / issued_lag1 {_f(p5['after_iss'])}"
        )
    if p6["died"]:
        failed.append(f"Y3 leftover after days {_f(p6['after_days'])} — CLOSE as Y3 X")
    if p8["empty_early"]:
        failed.append("CLOSE as Q6 — delay empty until month 7")
    if p7["died"]:
        failed.append("3m-vs-all-open leftover dies — CLOSE window is not a health lever")
    if p9["same_hole"] or not p9["delay_saves"]:
        failed.append(f"fold 4 delay {_f(p9['delay4'])} does not save the short-DSO hole")
    if p19["artifact"]:
        failed.append(
            f"same-n leftover {_f(p19['leftover'])} ≈ raw {_f(p19['raw'])} — not a new object"
        )
    if p20["style"]:
        failed.append(
            f"BETWEEN style: company-mean {_f(p20['mean_cv'])} demean {_f(p20['de_cv'])} ICC={_f(p10['icc_delay'])}"
        )
    if p25["paid_die"]:
        failed.append(f"delay_paid leftover dies {_f(p5['after_paid_dso'])} — CLOSE")
    if p25["od_die"]:
        failed.append(f"AR overdue leftover after DSO dies {_f(p5['after_od_dso'])} — CLOSE")
    if not p26["beats"]:
        failed.append(
            f"same-n leftover − issued Δ={_f(p26['delta'], 3)} — do not grow TURNOVER"
        )
    elapsed = time.time() - t0
    ctx = {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "p6": p6,
        "p7": p7,
        "p8": p8,
        "p9": p9,
        "p10": p10,
        "p11": p11,
        "p12": p12,
        "p13": p13,
        "p14": p14,
        "p15": p15,
        "p16": p16,
        "p17": p17,
        "p18": p18,
        "p19": p19,
        "p20": p20,
        "p21": p21,
        "p22": p22,
        "p23": p23,
        "p24": p24,
        "p25": p25,
        "p26": p26,
        "decision": decision,
        "headline": headline,
        "png_ok": png_ok,
        "failed": failed,
        "elapsed_s": elapsed,
    }
    write_md(ctx)
    append_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    print(f"elapsed {elapsed:.1f}s")
    return ctx


if __name__ == "__main__":
    run()
