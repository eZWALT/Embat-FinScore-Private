"""Q4/Q5 DPO leftover after days / DSO / issued_lag1.

NORTH_STAR: `e_dpo_proxy` = AP open / this-period AP issued (months of
billings outstanding). Same formula as DSO, payables side. Feature
report: 51.2% cov, LOW_PERSIST acf 0.24, cluster rep; means unusable.
Winsorise-at-24 is model-layer only — do not rewrite the store.

Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. TURNDPO 0.723 is
**not** a KEEP (no fold-4 lift). Do not grow TURNOVER. Do not change
0.720. Night Y3 stays **0.762 / 0.752**. Days **0.711**. Size **0.617**.
Y7 never D. Y5 never E. Y3 never B. Dark 470 stay NaN not 0.
Delay / overdue stay locked. DSO stays off TURNOVER.

KEEP-as-X: beat size ≥0.02 AND leftover after the honest bar AND not
SIZE (|ρ|≥0.50) AND not a twin (|ρ|≥0.80 vs DSO / delay_paid /
ap_overdue / ap_issued). Leftover <0.55 dies.

No 0–100. No product/. No parquet rewrite. No new GBM. Do not run
`build_targets`. Do not invent `y_dpo`. Do not put DPO on the 15-col
Y3 card. Do not edit invoices.py unless a real formula bug.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.dpo_qa

Owned: analysis/evaluate/dpo_qa.py, analysis/outputs/dpo_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_dpo.md (end).
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
from analysis.features.invoices import DELAY_MASK_BEFORE
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
OUT_MD = ANALYSIS / "outputs" / "dpo_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "dpo_vs_dso.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_dpo.md"
AGENT = "a19d4e07"
WAVE = "4"
ROUND = "R4"
MODEL = "dpo_qa"
X_FAM = "E"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y5 = "y5_ap_od30_ownp80"
Y6Z = "y6_zero_in_3"
Y6P = "y6_missed_payroll"
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
TURNDPO_QUOTE = 0.723
TWIN_RHO = 0.80
KEEP_DELTA = 0.02
CHANCE = 0.55
SIZE_RHO = 0.50
ICC_STYLE = 0.85
LOW_PERSIST = 0.25
MIN_POS = 50
MIN_ACF_PAIRS = 4
WINSOR = 24.0
SQL_TOL = 1e-8
PANEL_START = pd.Timestamp("2024-09-01")
WRITE_WAVE = True  # last iterate writes the note

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "c_n_days_with_tx",
    "c_zero_in_month",
    "c_zero_in_share_6",
    "f_ds_r",
    "e_dpo_proxy",
    "e_dso_proxy",
    "e_delay_coll",
    "e_delay_paid",
    "e_ap_overdue",
    "e_ap_overdue_30",
    "e_ap_issued",
    "e_ap_open",
    "e_ar_issued",
    "e_ar_open",
    "e_pending_amt_share",
    "b_below_0",
)

Y_KEEP = (Y2, Y3, Y5, Y6Z, Y6P, Y7)

TWIN_COLS = (
    "e_dso_proxy",
    "e_delay_paid",
    "e_ap_overdue",
    "e_ap_issued",
)


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


def company_mean(series: pd.Series, company: pd.Series) -> pd.Series:
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    return d.groupby("co")["x"].transform("mean")


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
    return df


def reconstruct_dpo(con) -> pd.DataFrame:
    """Independent SQL of AP open / this-period AP issued. Diagnosis only."""
    periods = period_frame()
    con.register("_dpo_periods", periods[["period", "period_end"]])
    try:
        issued = con.execute(
            """
            SELECT i.company_id,
                   p.period,
                   SUM(CASE WHEN i.document_type = 'invoice' AND i.amount < 0
                            THEN abs(i.amount) ELSE 0 END) AS sql_ap_issued
            FROM invoices i
            JOIN _dpo_periods p
              ON CAST(i.issuance_date AS DATE) >= CAST(p.period AS DATE)
             AND CAST(i.issuance_date AS DATE) <= CAST(p.period_end AS DATE)
            WHERE i.amount <> 0
              AND i.status <> 'cancel'
              AND i.issuance_date IS NOT NULL
            GROUP BY 1, 2
            """
        ).df()
        open_book = con.execute(
            """
            SELECT i.company_id,
                   p.period,
                   SUM(CASE WHEN i.amount < 0 THEN abs(i.amount) ELSE 0 END) AS sql_ap_open
            FROM invoices i
            JOIN _dpo_periods p
              ON CAST(i.issuance_date AS DATE) <= CAST(p.period_end AS DATE)
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
        con.unregister("_dpo_periods")

    ever = con.execute(
        """
        SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id
        FROM invoices
        WHERE document_type = 'invoice' AND status <> 'cancel' AND amount <> 0
        """
    ).df()
    ever_ids = set(ever["company_id"].astype(str))

    grid = pd.MultiIndex.from_product(
        [sorted(ever_ids), list(MONTHS)], names=["company_id", "period"]
    ).to_frame(index=False)
    grid["company_id"] = grid["company_id"].astype(str)
    grid["period"] = pd.to_datetime(grid["period"])

    for part, col in ((issued, "sql_ap_issued"), (open_book, "sql_ap_open")):
        if part.empty:
            grid[col] = np.nan
            continue
        part = part.copy()
        part["company_id"] = part["company_id"].astype(str)
        part["period"] = pd.to_datetime(part["period"])
        grid = grid.merge(part[["company_id", "period", col]], on=["company_id", "period"], how="left")

    active = grid["company_id"].isin(ever_ids)
    for c in ("sql_ap_issued", "sql_ap_open"):
        grid.loc[active, c] = pd.to_numeric(grid[c], errors="coerce")
        grid.loc[active, c] = grid.loc[active, c].fillna(0.0)
    iss = pd.to_numeric(grid["sql_ap_issued"], errors="coerce")
    opn = pd.to_numeric(grid["sql_ap_open"], errors="coerce")
    grid["sql_dpo"] = opn / iss.where(iss > 0)
    return grid[["company_id", "period", "sql_ap_issued", "sql_ap_open", "sql_dpo"]]


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
    panel["e_dpo_clip24"] = pd.to_numeric(panel["e_dpo_proxy"], errors="coerce").clip(upper=WINSOR)
    leak7 = leakage_check(["e_dpo_proxy", "e_dso_proxy", "e_ar_issued"], Y7, forbidden_prefixes=["d"])
    leak3 = leakage_check(["e_dpo_proxy"], Y3, forbidden_prefixes=["b"])
    leak5 = leakage_check(["log_in3", "c_n_days_with_tx"], Y5, forbidden_prefixes=["e"])
    if not leak7["ok"]:
        raise RuntimeError(f"Y7 X leak: {leak7['issues']}")
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 X leak: {leak3['issues']}")
    if not leak5["ok"]:
        raise RuntimeError(f"Y5 never-E leak: {leak5['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


# ---------------------------------------------------------------------------
# 1. Coverage / tails / 470 NaN / ERP vs dark
# ---------------------------------------------------------------------------
def pass1_cov(tr: pd.DataFrame, book: set[str]) -> dict:
    dark = ~tr["company_id"].isin(book)
    erp = tr["company_id"].isin(book)
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    n_erp_co = int(tr.loc[erp, "company_id"].nunique())
    dpo = pd.to_numeric(tr["e_dpo_proxy"], errors="coerce")
    nn = dpo.notna()
    n_cm = int(len(tr))
    n_nn = int(nn.sum())
    dark_nn = int(dpo[dark].notna().sum())
    dark_zero = int((dpo[dark] == 0).sum())
    early_nn = int(dpo[tr["early6"]].notna().sum())
    early_n = int(tr["early6"].sum())
    late_nn = int(dpo[~tr["early6"]].notna().sum())
    late_n = int((~tr["early6"]).sum())
    finite = dpo[nn]
    p99 = float(finite.quantile(0.99)) if n_nn else float("nan")
    p50 = float(finite.median()) if n_nn else float("nan")
    mx = float(finite.max()) if n_nn else float("nan")
    mn = float(finite.min()) if n_nn else float("nan")
    gt24 = int((finite.abs() > WINSOR).sum())
    gt24_share = _pct(gt24, n_nn)
    gt24_all = _pct(gt24, n_cm)
    acf1 = median_acf(tr["e_dpo_proxy"], tr["company_id"], 1)
    acf3 = median_acf(tr["e_dpo_proxy"], tr["company_id"], 3)
    rows = [
        {
            "col": "e_dpo_proxy",
            "nn": f"{n_nn:,}",
            "cov": _pp(_pct(n_nn, n_cm)),
            "early6 nn": _pp(_pct(early_nn, early_n)),
            "after nn": _pp(_pct(late_nn, late_n)),
            "dark nn / 0": f"{dark_nn} / {dark_zero}",
            "ERP nn": _pp(_pct(int(dpo[erp].notna().sum()), int(erp.sum()))),
            "p50": _f(p50, 2),
            "p99": _f(p99, 1),
            "max": _f(mx, 1),
            "|DPO|>24": _pp(gt24_share),
            "acf1": _f(acf1),
        }
    ]
    dark_nan = dark_nn == 0 and n_dark_co == 470
    low_persist = bool(np.isfinite(acf1) and abs(acf1) < LOW_PERSIST)
    prose = (
        f"Train DPO nn={n_nn:,} cov={_pp(_pct(n_nn, n_cm))} "
        f"(feature report 51.2%). Dark never-ERP {n_dark_co} (want 470): "
        f"nn={dark_nn} zero={dark_zero} "
        f"({'CONFIRM NaN not 0' if dark_nan else 'MISMATCH dark'}). "
        f"Ever-ERP {n_erp_co}. Early6 finite {_pp(_pct(early_nn, early_n))} "
        f"(delay was 0.0% — DPO is a stock/flow, may exist earlier). "
        f"Tails p50={_f(p50, 2)} p99={_f(p99, 1)} max={_f(mx, 1)} "
        f"|DPO|>24 {_pp(gt24_share)} of finite ({gt24:,}). "
        f"acf1={_f(acf1)} ({'CONFIRM LOW_PERSIST' if low_persist else 'not LOW_PERSIST'})."
    )
    print(prose)
    y7 = pd.to_numeric(tr[Y7], errors="coerce")
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    y5 = pd.to_numeric(tr[Y5], errors="coerce")
    print(
        f"Train CM {n_cm:,} / companies {tr['company_id'].nunique()}. "
        f"Y7 labeled {int(y7.notna().sum()):,} pos {int((y7==1).sum()):,}; "
        f"Y3 {int(y3.notna().sum()):,} pos {int((y3==1).sum()):,}; "
        f"Y5 AP {int(y5.notna().sum()):,} pos {int((y5==1).sum()):,}."
    )
    return {
        "rows": rows,
        "n_cm": n_cm,
        "n_nn": n_nn,
        "cov": _pct(n_nn, n_cm),
        "n_dark_co": n_dark_co,
        "n_erp_co": n_erp_co,
        "dark_nn": dark_nn,
        "dark_zero": dark_zero,
        "dark_nan": dark_nan,
        "early_cov": _pct(early_nn, early_n),
        "late_cov": _pct(late_nn, late_n),
        "p50": p50,
        "p99": p99,
        "max": mx,
        "min": mn,
        "gt24": gt24,
        "gt24_share": gt24_share,
        "gt24_all": gt24_all,
        "acf1": acf1,
        "acf3": acf3,
        "low_persist": low_persist,
        "y7_n": int(y7.notna().sum()),
        "y7_pos": int((y7 == 1).sum()),
        "y3_n": int(y3.notna().sum()),
        "y3_pos": int((y3 == 1).sum()),
        "y5_n": int(y5.notna().sum()),
        "y5_pos": int((y5 == 1).sum()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 2. Reproduce store vs raw SQL
# ---------------------------------------------------------------------------
def pass2_sql(tr: pd.DataFrame, recon: pd.DataFrame) -> dict:
    m = tr.merge(recon, on=["company_id", "period"], how="left")
    pairs = [
        ("e_dpo_proxy", "sql_dpo"),
        ("e_ap_open", "sql_ap_open"),
        ("e_ap_issued", "sql_ap_issued"),
    ]
    rows = []
    store = {}
    for a, b in pairs:
        sa = pd.to_numeric(m[a], errors="coerce")
        sb = pd.to_numeric(m[b], errors="coerce")
        both = sa.notna() & sb.notna()
        only_a = int((sa.notna() & sb.isna()).sum())
        only_b = int((sa.isna() & sb.notna()).sum())
        n_both = int(both.sum())
        if n_both:
            ad = (sa[both] - sb[both]).abs()
            maxabs = float(ad.max())
            n_off = int((ad > SQL_TOL).sum())
        else:
            maxabs = float("nan")
            n_off = 0
        rho, nrho = spearman_n(sa, sb)
        same = bool(n_both > 0 and (not np.isfinite(maxabs) or maxabs <= 1e-6) and only_a == 0 and only_b == 0)
        store[a] = {
            "rho": rho,
            "n": nrho,
            "maxabs": maxabs,
            "n_off": n_off,
            "only_store": only_a,
            "only_sql": only_b,
            "same": same,
        }
        rows.append(
            {
                "pair": f"{a} vs {b}",
                "ρ": _f(rho),
                "n": f"{n_both:,}",
                "max|Δ|": _f(maxabs, 8),
                "n_off": n_off,
                "only store": only_a,
                "only SQL": only_b,
                "verdict": "SAME" if same else ("CLOSE" if np.isfinite(rho) and abs(rho) >= 0.99 else "DRIFT"),
            }
        )
    dpo_same = store["e_dpo_proxy"]["same"]
    prose = (
        f"Store vs raw SQL DPO ρ={_f(store['e_dpo_proxy']['rho'])} "
        f"max|Δ|={_f(store['e_dpo_proxy']['maxabs'], 8)} "
        f"n_off={store['e_dpo_proxy']['n_off']} "
        f"({'CONFIRM SAME' if dpo_same else 'DISAGREE — document, do not rewrite invoices.py'})."
    )
    print(prose)
    return {"rows": rows, "store": store, "same": dpo_same, "prose": prose}


# ---------------------------------------------------------------------------
# 3. Spearman twins
# ---------------------------------------------------------------------------
def pass3_twins(tr: pd.DataFrame) -> dict:
    dpo = tr["e_dpo_proxy"]
    pairs = [
        ("e_dso_proxy", tr["e_dso_proxy"]),
        ("e_delay_paid", tr["e_delay_paid"]),
        ("e_ap_overdue", tr["e_ap_overdue"]),
        ("e_ap_overdue_30", tr["e_ap_overdue_30"]),
        ("e_ap_issued", tr["e_ap_issued"]),
        ("e_pending_amt_share", tr["e_pending_amt_share"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("f_ds_r", tr["f_ds_r"]),
        ("e_delay_coll", tr["e_delay_coll"]),
        ("e_ar_issued", tr["e_ar_issued"]),
        ("c_zero_in_month", tr["c_zero_in_month"]),
        ("c_zero_in_share_6", tr["c_zero_in_share_6"]),
    ]
    rows = []
    rhos = {}
    twins = []
    for name, s in pairs:
        rho, n = spearman_n(dpo, s)
        twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
        rhos[name] = rho
        rhos[f"{name}_n"] = n
        if twin:
            twins.append(name)
        rows.append(
            {
                "pair": f"e_dpo_proxy vs {name}",
                "ρ": _f(rho),
                "n": f"{n:,}",
                "twin?": "TWIN" if twin else "",
            }
        )
    size_flag = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    leak_y = bool(np.isfinite(rhos["e_ap_overdue_30"]) and abs(rhos["e_ap_overdue_30"]) >= TWIN_RHO)
    prose = (
        f"DPO vs DSO ρ={_f(rhos['e_dso_proxy'])} "
        f"({'TWIN' if 'e_dso_proxy' in twins else 'not a twin'}). "
        f"vs delay_paid {_f(rhos['e_delay_paid'])} vs ap_overdue {_f(rhos['e_ap_overdue'])} "
        f"vs ap_overdue_30 {_f(rhos['e_ap_overdue_30'])} "
        f"({'is the Y5 AP label' if leak_y else 'not the Y'}). "
        f"vs ap_issued {_f(rhos['e_ap_issued'])} vs size {_f(rhos['log1p(a_in3)'])} "
        f"({'SIZE' if size_flag else 'not SIZE'}). "
        f"TWIN |ρ|≥0.80: {', '.join(twins) if twins else 'none'}."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "twins": twins,
        "size_flag": size_flag,
        "leak_y": leak_y,
        "dso": rhos["e_dso_proxy"],
        "paid": rhos["e_delay_paid"],
        "od": rhos["e_ap_overdue"],
        "od30": rhos["e_ap_overdue_30"],
        "iss": rhos["e_ap_issued"],
        "pend": rhos["e_pending_amt_share"],
        "size": rhos["log1p(a_in3)"],
        "days": rhos["c_n_days_with_tx"],
        "dsr": rhos["f_ds_r"],
        "zero": rhos["c_zero_in_month"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 4. Single-feature group-fold AUROC
# ---------------------------------------------------------------------------
def pass4_auroc(tr: pd.DataFrame) -> dict:
    feats = {
        "e_dpo_proxy": tr["e_dpo_proxy"],
        "e_dpo_clip24": tr["e_dpo_clip24"],
        "e_dso_proxy": tr["e_dso_proxy"],
        "e_delay_paid": tr["e_delay_paid"],
        "e_ap_overdue_30": tr["e_ap_overdue_30"],
        "e_ap_issued": tr["e_ap_issued"],
        "e_ar_issued_lag1": tr["e_ar_issued_lag1"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "log1p_a_in3": tr["log_in3"],
    }
    ys = (Y3, Y5, Y7)
    rows = []
    store = {}
    for ycol in ys:
        lab = tr[ycol].notna()
        n_lab = int(lab.sum())
        for name, s in feats.items():
            res = signed_oof_auroc(tr[ycol], s, tr["fold"], lab)
            store[(ycol, name)] = res
            present = _pct(res["n_defined"], n_lab) if n_lab else float("nan")
            rows.append(_auc_row(ycol, name, res, present))
            print(
                f"{ycol} {name}: "
                f"{'LOW_POWER' if res['low_power'] else _f(res['cv'])} "
                f"n={res['n_defined']:,} pos={res['n_pos']:,}"
            )
    dpo_y3 = _cv(store[(Y3, "e_dpo_proxy")])
    dpo_y5 = _cv(store[(Y5, "e_dpo_proxy")])
    dpo_y7 = _cv(store[(Y7, "e_dpo_proxy")])
    days_y3 = _cv(store[(Y3, "c_n_days_with_tx")])
    size_y3 = _cv(store[(Y3, "log1p_a_in3")])
    size_y7 = _cv(store[(Y7, "log1p_a_in3")])
    dso_y7 = _cv(store[(Y7, "e_dso_proxy")])
    iss_y7 = _cv(store[(Y7, "e_ar_issued_lag1")])
    paid_y5 = _cv(store[(Y5, "e_delay_paid")])
    days_ok = bool(np.isfinite(days_y3) and abs(days_y3 - DAYS_BENCH) < 0.005)
    prose = (
        f"Y3 DPO {_f(dpo_y3)} vs days {_f(days_y3)} "
        f"({'CONFIRM 0.711' if days_ok else 'days drifted'}) vs size {_f(size_y3)}. "
        f"Y7 DPO {_f(dpo_y7)} vs DSO {_f(dso_y7)} vs issued_lag1 {_f(iss_y7)} "
        f"(night 0.630). Y5 AP DPO {_f(dpo_y5)} vs delay_paid {_f(paid_y5)} "
        f"— Y5 never E as X; this is leak/label only."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "dpo_y3": dpo_y3,
        "dpo_y5": dpo_y5,
        "dpo_y7": dpo_y7,
        "days_y3": days_y3,
        "size_y3": size_y3,
        "size_y7": size_y7,
        "dso_y7": dso_y7,
        "iss_y7": iss_y7,
        "paid_y5": paid_y5,
        "clip_y3": _cv(store[(Y3, "e_dpo_clip24")]),
        "clip_y7": _cv(store[(Y7, "e_dpo_clip24")]),
        "days_ok": days_ok,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 5. Honest leftover
# ---------------------------------------------------------------------------
def _leftover_block(tr: pd.DataFrame, ycol: str, controls: list[tuple[str, tuple]]) -> dict:
    lab = tr[ycol].notna()
    stems = {
        "e_dpo_proxy": tr["e_dpo_proxy"],
        "e_dpo_clip24": tr["e_dpo_clip24"],
    }
    rows = []
    store = {}
    infos = {}
    for sname, s in stems.items():
        for cname, xs in controls:
            key = f"{sname} {cname}"
            resid, info = ols_resid(s, *xs)
            res = signed_oof_auroc(tr[ycol], resid, tr["fold"], lab)
            store[key] = res
            infos[key] = info
            rows.append(
                {
                    "y": ycol,
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
                f"{ycol} leftover {key}: "
                f"{'LOW_POWER' if res['low_power'] else _f(res['cv'])} "
                f"R2={_f(info['r2'])}"
            )
    return {"rows": rows, "store": store, "infos": infos}


def pass5_leftover(tr: pd.DataFrame) -> dict:
    y7 = _leftover_block(
        tr,
        Y7,
        [
            ("after DSO", (tr["e_dso_proxy"],)),
            ("after issued_lag1", (tr["e_ar_issued_lag1"],)),
            ("after DSO+issued_lag1", (tr["e_dso_proxy"], tr["e_ar_issued_lag1"])),
            ("after days", (tr["c_n_days_with_tx"],)),
            ("after size", (tr["log_in3"],)),
            ("after delay_paid", (tr["e_delay_paid"],)),
        ],
    )
    y3 = _leftover_block(
        tr,
        Y3,
        [
            ("after days", (tr["c_n_days_with_tx"],)),
            ("after size", (tr["log_in3"],)),
            ("after DSO", (tr["e_dso_proxy"],)),
            ("after days+size", (tr["c_n_days_with_tx"], tr["log_in3"])),
        ],
    )
    y5 = _leftover_block(
        tr,
        Y5,
        [
            ("after delay_paid", (tr["e_delay_paid"],)),
            ("after ap_overdue_30", (tr["e_ap_overdue_30"],)),
            ("after delay_paid+od30", (tr["e_delay_paid"], tr["e_ap_overdue_30"])),
            ("after size", (tr["log_in3"],)),
        ],
    )
    after_dso = _cv(y7["store"]["e_dpo_proxy after DSO"])
    after_iss = _cv(y7["store"]["e_dpo_proxy after issued_lag1"])
    after_both = _cv(y7["store"]["e_dpo_proxy after DSO+issued_lag1"])
    after_days = _cv(y3["store"]["e_dpo_proxy after days"])
    after_paid = _cv(y5["store"]["e_dpo_proxy after delay_paid"])
    after_od30 = _cv(y5["store"]["e_dpo_proxy after ap_overdue_30"])
    clip_dso = _cv(y7["store"]["e_dpo_clip24 after DSO"])
    clip_iss = _cv(y7["store"]["e_dpo_clip24 after issued_lag1"])
    clip_days = _cv(y3["store"]["e_dpo_clip24 after days"])
    y7_lives = bool(
        np.isfinite(after_dso)
        and after_dso >= CHANCE
        and np.isfinite(after_iss)
        and after_iss >= CHANCE
    )
    y3_lives = bool(np.isfinite(after_days) and after_days >= CHANCE)
    y5_lives = bool(np.isfinite(after_paid) and after_paid >= CHANCE)
    rows = y7["rows"] + y3["rows"] + y5["rows"]
    prose = (
        f"Y3 leftover after days {_f(after_days)} "
        f"({'lives' if y3_lives else 'dies <0.55'}). "
        f"Y7 leftover after DSO {_f(after_dso)} / issued_lag1 {_f(after_iss)} / both {_f(after_both)} "
        f"({'lives' if y7_lives else 'dies'}). "
        f"Y5 leftover after delay_paid {_f(after_paid)} / od30 {_f(after_od30)} "
        f"(diagnostic only; Y5 never E as X). "
        f"Clip24 leftover Y3-days {_f(clip_days)} Y7-DSO {_f(clip_dso)} Y7-iss {_f(clip_iss)}."
    )
    print(prose)
    return {
        "rows": rows,
        "y7": y7,
        "y3": y3,
        "y5": y5,
        "after_dso": after_dso,
        "after_iss": after_iss,
        "after_both": after_both,
        "after_days": after_days,
        "after_paid": after_paid,
        "after_od30": after_od30,
        "clip_dso": clip_dso,
        "clip_iss": clip_iss,
        "clip_days": clip_days,
        "y7_lives": y7_lives,
        "y3_lives": y3_lives,
        "y5_lives": y5_lives,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 8. SIZE terciles + invoice-book-only
# ---------------------------------------------------------------------------
def pass8_slices(tr: pd.DataFrame, book: set[str]) -> dict:
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    defined = size.notna()
    terc = pd.Series(np.nan, index=tr.index, dtype=object)
    terc.loc[defined] = pd.qcut(size[defined].rank(method="first"), 3, labels=["T1", "T2", "T3"])
    rows = []
    store = {}
    for label, mask in (
        ("all", pd.Series(True, index=tr.index)),
        ("book_only", tr["company_id"].isin(book)),
        ("T1", terc == "T1"),
        ("T2", terc == "T2"),
        ("T3", terc == "T3"),
    ):
        for ycol in (Y3, Y7):
            lab = mask & tr[ycol].notna()
            res = signed_oof_auroc(tr[ycol], tr["e_dpo_proxy"], tr["fold"], lab)
            store[(label, ycol)] = res
            rows.append(_auc_row(ycol, f"DPO {label}", res, _pct(res["n_defined"], int(lab.sum())) if lab.any() else float("nan")))
            if ycol == Y7 and label in ("all", "book_only", "T1", "T3"):
                resid, _ = ols_resid(tr["e_dpo_proxy"], tr["e_dso_proxy"])
                rres = signed_oof_auroc(tr[ycol], resid, tr["fold"], lab)
                store[(f"{label}|DSO", ycol)] = rres
                rows.append(_auc_row(ycol, f"DPO leftover-DSO {label}", rres, None))
            if ycol == Y3 and label in ("all", "book_only", "T1", "T3"):
                resid, _ = ols_resid(tr["e_dpo_proxy"], tr["c_n_days_with_tx"])
                rres = signed_oof_auroc(tr[ycol], resid, tr["fold"], lab)
                store[(f"{label}|days", ycol)] = rres
                rows.append(_auc_row(ycol, f"DPO leftover-days {label}", rres, None))
    prose = (
        f"Book-only (drop 470) Y3 {_f(_cv(store[('book_only', Y3)]))} "
        f"Y7 {_f(_cv(store[('book_only', Y7)]))}. "
        f"SIZE T1 Y3 {_f(_cv(store[('T1', Y3)]))} Y7 {_f(_cv(store[('T1', Y7)]))} "
        f"T3 Y3 {_f(_cv(store[('T3', Y3)]))} Y7 {_f(_cv(store[('T3', Y7)]))}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "book_y3": _cv(store[("book_only", Y3)]),
        "book_y7": _cv(store[("book_only", Y7)]),
        "t1_y3": _cv(store[("T1", Y3)]),
        "t1_y7": _cv(store[("T1", Y7)]),
        "t3_y3": _cv(store[("T3", Y3)]),
        "t3_y7": _cv(store[("T3", Y7)]),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 9. Q6 lag1/lag3 + empty-on-short
# ---------------------------------------------------------------------------
def pass9_q6(tr: pd.DataFrame) -> dict:
    slices = {
        "all": pd.Series(True, index=tr.index),
        "short_<12_sofar": tr["so_far_class"] == "short_<12",
        "long_>=18_sofar": tr["so_far_class"] == "long_>=18",
        "short_<12_company": tr["co_class"] == "short_<12",
        "long_>=18_company": tr["co_class"] == "long_>=18",
        "early6_calendar": tr["early6"],
        "after_month7": ~tr["early6"],
    }
    cols = (
        "e_dpo_proxy",
        "e_dpo_proxy_lag1",
        "e_dpo_proxy_lag3",
        "e_dso_proxy",
        "e_ar_issued_lag1",
        "e_delay_paid",
    )
    rows = []
    store = {}
    for sl_name, sl in slices.items():
        n_sl = int((sl & tr[Y7].notna()).sum())
        for col in cols:
            if col not in tr.columns:
                continue
            mask = sl & tr[Y7].notna()
            res = signed_oof_auroc(tr[Y7], tr[col], tr["fold"], mask)
            store[(sl_name, col)] = res
            present = _pct(res["n_defined"], n_sl) if n_sl else float("nan")
            rows.append(
                {
                    "slice": sl_name,
                    "col": col,
                    "n_nn": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "present": _pp(present),
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
    short_now = _cv(store[("short_<12_sofar", "e_dpo_proxy")])
    short_lag = _cv(store[("short_<12_sofar", "e_dpo_proxy_lag1")])
    early_res = store[("early6_calendar", "e_dpo_proxy")]
    early_cov = _pct(early_res["n_defined"], int((tr["early6"] & tr[Y7].notna()).sum()))
    empty_early = bool(early_res["n_defined"] == 0)
    exists_early = bool(early_res["n_defined"] > 0)
    prose = (
        f"Q6 Y7 DPO now short {_f(short_now)} lag1 {_f(short_lag)}. "
        f"Early6 calendar DPO finite share {_pp(early_cov)} "
        f"({'exists earlier than delay' if exists_early else 'empty like delay'}). "
        f"{'CLOSE as Q6 if lag dies on short / empty den' if (not np.isfinite(short_lag) or short_lag < CHANCE) else 'lag holds on short'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "short_now": short_now,
        "short_lag": short_lag,
        "early_cov": early_cov,
        "empty_early": empty_early,
        "exists_early": exists_early,
        "all_lag1": _cv(store[("all", "e_dpo_proxy_lag1")]),
        "all_lag3": _cv(store[("all", "e_dpo_proxy_lag3")]),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 10. ICC / company-demean
# ---------------------------------------------------------------------------
def pass10_icc(tr: pd.DataFrame) -> dict:
    icc = icc_anova(tr["e_dpo_proxy"], tr["company_id"])
    icc_dso = icc_anova(tr["e_dso_proxy"], tr["company_id"])
    de = company_demean(tr["e_dpo_proxy"], tr["company_id"])
    mu = company_mean(tr["e_dpo_proxy"], tr["company_id"])
    rows = []
    store = {}
    for ycol in (Y7, Y3):
        lab = tr[ycol].notna()
        for name, s in (("raw", tr["e_dpo_proxy"]), ("demean", de), ("company-mean", mu)):
            res = signed_oof_auroc(tr[ycol], s, tr["fold"], lab)
            store[(ycol, name)] = res
            rows.append(_auc_row(ycol, f"e_dpo_proxy {name}", res, None))
    style = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_STYLE)
    y7_de = _cv(store[(Y7, "demean")])
    y7_mu = _cv(store[(Y7, "company-mean")])
    y7_raw = _cv(store[(Y7, "raw")])
    low_persist = bool(np.isfinite(median_acf(tr["e_dpo_proxy"], tr["company_id"], 1)))
    acf1 = median_acf(tr["e_dpo_proxy"], tr["company_id"], 1)
    prose = (
        f"DPO ICC={_f(icc['icc'])} k={icc['k']} (DSO ICC={_f(icc_dso['icc'])}). "
        f"Y7 raw {_f(y7_raw)} demean {_f(y7_de)} company-mean {_f(y7_mu)}. "
        f"acf1={_f(acf1)} "
        f"({'CONFIRM LOW_PERSIST month shock' if np.isfinite(acf1) and abs(acf1) < LOW_PERSIST else 'trait-ish'}). "
        f"{'BETWEEN style' if style else 'not BETWEEN (≥0.85)'} — "
        f"{'company-mean carries skill' if np.isfinite(y7_mu) and y7_mu > (y7_de if np.isfinite(y7_de) else 0) else 'demean / month shock carries'}."
    )
    print(prose)
    return {
        "rows": rows,
        "icc": icc["icc"],
        "icc_k": icc["k"],
        "icc_dso": icc_dso["icc"],
        "acf1": acf1,
        "style": style,
        "y7_raw": y7_raw,
        "y7_de": y7_de,
        "y7_mu": y7_mu,
        "y3_raw": _cv(store[(Y3, "raw")]),
        "y3_de": _cv(store[(Y3, "demean")]),
        "y3_mu": _cv(store[(Y3, "company-mean")]),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 11. vs Y6 / zero-in — not an inverse-activity dummy; do not revive Y6
# ---------------------------------------------------------------------------
def pass11_y6(tr: pd.DataFrame) -> dict:
    dpo = tr["e_dpo_proxy"]
    pairs = [
        ("c_zero_in_month", tr["c_zero_in_month"]),
        ("c_zero_in_share_6", tr["c_zero_in_share_6"]),
        (Y6Z, tr[Y6Z]),
        (Y6P, tr[Y6P]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("log1p(a_in3)", tr["log_in3"]),
    ]
    rho_rows = []
    rhos = {}
    for name, s in pairs:
        rho, n = spearman_n(dpo, s)
        rhos[name] = rho
        rho_rows.append({"vs": name, "ρ": _f(rho), "n": f"{n:,}", "dummy?": "YES" if np.isfinite(rho) and abs(rho) >= 0.50 else ""})
    rows = []
    store = {}
    for ycol in (Y6Z, Y6P):
        lab = tr[ycol].notna()
        for name, s in (
            ("e_dpo_proxy", dpo),
            ("c_zero_in_month", tr["c_zero_in_month"]),
            ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
            ("log1p_a_in3", tr["log_in3"]),
        ):
            res = signed_oof_auroc(tr[ycol], s, tr["fold"], lab)
            store[(ycol, name)] = res
            rows.append(_auc_row(ycol, name, res, _pct(res["n_defined"], int(lab.sum())) if lab.any() else float("nan")))
    dummy = bool(
        (np.isfinite(rhos["c_zero_in_month"]) and abs(rhos["c_zero_in_month"]) >= 0.50)
        or (np.isfinite(rhos[Y6Z]) and abs(rhos[Y6Z]) >= 0.50)
    )
    dpo_y6 = _cv(store[(Y6Z, "e_dpo_proxy")])
    days_y6 = _cv(store[(Y6Z, "c_n_days_with_tx")])
    prose = (
        f"DPO vs zero-in month ρ={_f(rhos['c_zero_in_month'])} vs y6_zero_in_3 ρ={_f(rhos[Y6Z])}. "
        f"Y6 zero-in DPO {_f(dpo_y6)} vs days {_f(days_y6)}. "
        f"{'inverse-activity dummy — still do not revive Y6' if dummy else 'not an inverse-activity dummy'}. "
        f"Do not revive Y6."
    )
    print(prose)
    return {
        "rho_rows": rho_rows,
        "rows": rows,
        "rhos": rhos,
        "dummy": dummy,
        "dpo_y6": dpo_y6,
        "days_y6": days_y6,
        "dpo_pay": _cv(store[(Y6P, "e_dpo_proxy")]),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra: quintiles (U-shape / tiny-issued blow-up)
# ---------------------------------------------------------------------------
def pass_quintiles(tr: pd.DataFrame) -> dict:
    rows = []
    for ycol in (Y3, Y5, Y7):
        sl = tr.loc[tr[ycol].notna() & tr["e_dpo_proxy"].notna()].copy()
        if sl.empty:
            continue
        x = pd.to_numeric(sl["e_dpo_proxy"], errors="coerce")
        sl["q"] = pd.qcut(x.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        for q, part in sl.groupby("q", observed=False):
            y = pd.to_numeric(part[ycol], errors="coerce")
            iss = pd.to_numeric(part["e_ap_issued"], errors="coerce")
            rows.append(
                {
                    "y": ycol,
                    "DPO q": str(q),
                    "n": int(len(part)),
                    "n_pos": int((y == 1).sum()),
                    "rate": _pp(float(y.mean()) if y.notna().any() else float("nan")),
                    "DPO p50": _f(float(x.loc[part.index].median()), 2),
                    "issued p50": _f(float(iss.median()), 0),
                    "share>24": _pp(float((x.loc[part.index].abs() > WINSOR).mean())),
                }
            )
    rates_y3 = [r for r in rows if r["y"] == Y3]
    u = False
    tail = False
    if len(rates_y3) == 5:
        rs = [float(r["rate"].rstrip("%")) for r in rates_y3]
        u = rs[0] > rs[2] and rs[4] > rs[2]
        tail = rs[4] >= max(rs[:4]) + 3.0
    prose = (
        f"DPO quintiles vs Y3/Y5/Y7. "
        f"{'Q5 tail on Y3 (high-DPO / thin issued)' if tail else ('U-shape on Y3' if u else 'no clear U-shape on Y3')}."
    )
    print(prose)
    return {"rows": rows, "u_shape": u, "tail": tail, "prose": prose}


# ---------------------------------------------------------------------------
# Extra: fold 4 / short-DSO quintile
# ---------------------------------------------------------------------------
def pass_fold4(tr: pd.DataFrame, p4: dict, p5: dict) -> dict:
    feats = {
        "e_dpo_proxy": tr["e_dpo_proxy"],
        "e_dpo_clip24": tr["e_dpo_clip24"],
        "e_dso_proxy": tr["e_dso_proxy"],
        "e_ar_issued_lag1": tr["e_ar_issued_lag1"],
        "e_delay_paid": tr["e_delay_paid"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
    }
    lab = tr[Y7].notna()
    rows = []
    store = {}
    for name, s in feats.items():
        res = p4["store"].get((Y7, name))
        if res is None:
            res = signed_oof_auroc(tr[Y7], s, tr["fold"], lab)
        store[name] = res
        rows.append(
            {
                "feature": name,
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "fold4": _f(fold_k(res, 4)),
                "folds": fold_bits(res) if not res["low_power"] else "—",
            }
        )
    sl = tr.loc[lab & tr["e_dso_proxy"].notna()].copy()
    dso = pd.to_numeric(sl["e_dso_proxy"], errors="coerce").clip(upper=WINSOR)
    sl["dso_q"] = pd.qcut(dso.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    qrows = []
    qstore = {}
    for q, part in sl.groupby("dso_q", observed=False):
        idx = part.index
        mask = pd.Series(False, index=tr.index)
        mask.loc[idx] = True
        mask = mask & lab
        for name, col in (
            ("e_dpo_proxy", tr["e_dpo_proxy"]),
            ("e_dso_proxy", tr["e_dso_proxy"]),
            ("e_ar_issued_lag1", tr["e_ar_issued_lag1"]),
        ):
            res = signed_oof_auroc(tr[Y7], col, tr["fold"], mask)
            qstore[(str(q), name)] = res
        y = pd.to_numeric(part[Y7], errors="coerce")
        qrows.append(
            {
                "DSO q": str(q),
                "n": int(len(part)),
                "rate": _pp(float(y.mean()) if y.notna().any() else float("nan")),
                "DSO p50": _f(float(dso.loc[idx].median()), 2),
                "DPO nn": _pp(float(pd.to_numeric(part["e_dpo_proxy"], errors="coerce").notna().mean())),
                "DPO CV": "LOW_POWER" if qstore[(str(q), "e_dpo_proxy")]["low_power"] else _f(qstore[(str(q), "e_dpo_proxy")]["cv"]),
                "DSO CV": "LOW_POWER" if qstore[(str(q), "e_dso_proxy")]["low_power"] else _f(qstore[(str(q), "e_dso_proxy")]["cv"]),
                "issued CV": "LOW_POWER" if qstore[(str(q), "e_ar_issued_lag1")]["low_power"] else _f(qstore[(str(q), "e_ar_issued_lag1")]["cv"]),
            }
        )
    q1 = _cv(qstore[("Q1", "e_dpo_proxy")])
    q1_dso = _cv(qstore[("Q1", "e_dso_proxy")])
    same_hole = bool(np.isfinite(q1) and q1 < CHANCE)
    dpo4 = fold_k(store["e_dpo_proxy"], 4)
    prose = (
        f"Fold 4 Y7 DPO {_f(dpo4)} vs DSO {_f(fold_k(store['e_dso_proxy'], 4))} "
        f"vs issued_lag1 {_f(fold_k(store['e_ar_issued_lag1'], 4))} vs TURNOVER 0.680. "
        f"Short-DSO Q1: DPO {_f(q1)} vs DSO {_f(q1_dso)}. "
        f"{'Same short-DSO hole — DPO does not save Q1' if same_hole else 'DPO lifts the short-DSO fifth'}."
    )
    print(prose)
    return {
        "rows": rows,
        "qrows": qrows,
        "dpo4": dpo4,
        "dso4": fold_k(store["e_dso_proxy"], 4),
        "iss4": fold_k(store["e_ar_issued_lag1"], 4),
        "q1": q1,
        "q1_dso": q1_dso,
        "same_hole": same_hole,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra: holdout coverage only
# ---------------------------------------------------------------------------
def pass_holdout(panel: pd.DataFrame, book: set[str]) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    assert set(ho["company_id"]) <= load_holdout()
    dpo = pd.to_numeric(ho["e_dpo_proxy"], errors="coerce")
    dark = ~ho["company_id"].isin(book)
    y7 = pd.to_numeric(ho[Y7], errors="coerce")
    rows = [
        {
            "col": "e_dpo_proxy",
            "n_cm": f"{len(ho):,}",
            "nn": f"{int(dpo.notna().sum()):,}",
            "cov": _pp(_pct(int(dpo.notna().sum()), len(ho))),
            "early6 nn": _pp(_pct(int(dpo[ho['early6']].notna().sum()), int(ho['early6'].sum()))),
            "dark nn": int(dpo[dark].notna().sum()),
            "|DPO|>24": _pp(_pct(int((dpo.abs() > WINSOR).sum()), int(dpo.notna().sum()))),
        }
    ]
    prose = (
        f"Holdout 72 coverage only: {len(ho):,} CM / {ho['company_id'].nunique()} companies. "
        f"Y7 pos={int((y7==1).sum())} (quote 122). DPO cov "
        f"{_pp(_pct(int(dpo.notna().sum()), len(ho)))}; early6 "
        f"{_pp(_pct(int(dpo[ho['early6']].notna().sum()), int(ho['early6'].sum())))}; "
        f"dark nn={int(dpo[dark].notna().sum())}. No AUROC claim."
    )
    print(prose)
    return {
        "rows": rows,
        "n_cm": int(len(ho)),
        "n_co": int(ho["company_id"].nunique()),
        "nn": int(dpo.notna().sum()),
        "cov": _pct(int(dpo.notna().sum()), len(ho)),
        "y7_pos": int((y7 == 1).sum()),
        "dark_nn": int(dpo[dark].notna().sum()),
        "prose": prose,
    }


def pass_samen(tr: pd.DataFrame) -> dict:
    """Same-n raw vs leftover after DSO + issued_lag1."""
    lab = tr[Y7].notna()
    ok = (
        lab
        & tr["e_dpo_proxy"].notna()
        & tr["e_dso_proxy"].notna()
        & tr["e_ar_issued_lag1"].notna()
    )
    resid, info = ols_resid(tr["e_dpo_proxy"], tr["e_dso_proxy"], tr["e_ar_issued_lag1"])
    rows = []
    store = {}
    for name, s in (
        ("DPO same-n", tr["e_dpo_proxy"]),
        ("issued_lag1 same-n", tr["e_ar_issued_lag1"]),
        ("DSO same-n", tr["e_dso_proxy"]),
        ("leftover both same-n", resid),
        ("clip24 same-n", tr["e_dpo_clip24"]),
    ):
        res = signed_oof_auroc(tr[Y7], s, tr["fold"], ok)
        store[name] = res
        rows.append(_auc_row(Y7, name, res, None))
    raw = _cv(store["DPO same-n"])
    leftover = _cv(store["leftover both same-n"])
    prose = (
        f"Same-n (DPO+DSO+issued_lag1 finite): raw DPO {_f(raw)} leftover {_f(leftover)} "
        f"issued {_f(_cv(store['issued_lag1 same-n']))} DSO {_f(_cv(store['DSO same-n']))} "
        f"R²={_f(info['r2'])}. "
        f"{'Leftover ≈ raw — residualizing does not create a new object' if np.isfinite(raw) and np.isfinite(leftover) and abs(raw - leftover) < 0.02 else 'leftover differs from raw'}."
    )
    print(prose)
    return {
        "rows": rows,
        "raw": raw,
        "leftover": leftover,
        "iss": _cv(store["issued_lag1 same-n"]),
        "dso": _cv(store["DSO same-n"]),
        "r2": info["r2"],
        "prose": prose,
    }


def pass_tiny(tr: pd.DataFrame) -> dict:
    """Tiny this-period issued is the DPO blow-up den."""
    iss = pd.to_numeric(tr["e_ap_issued"], errors="coerce")
    dpo = pd.to_numeric(tr["e_dpo_proxy"], errors="coerce")
    pos = iss[iss > 0]
    p10 = float(pos.quantile(0.10)) if len(pos) else float("nan")
    tiny = (iss > 0) & (iss <= p10)
    fat = iss > p10
    rows = []
    for name, mask in (("tiny_issued p10", tiny), ("rest issued>p10", fat), ("issued=0 / NaN DPO", iss.eq(0) | iss.isna())):
        sl = tr.loc[mask]
        x = dpo[mask]
        rows.append(
            {
                "slice": name,
                "n": int(mask.sum()),
                "DPO nn": int(x.notna().sum()),
                "DPO p50": _f(float(x.median()) if x.notna().any() else float("nan"), 2),
                "DPO p99": _f(float(x.quantile(0.99)) if x.notna().sum() > 10 else float("nan"), 1),
                "|DPO|>24": _pp(_pct(int((x.abs() > WINSOR).sum()), int(x.notna().sum()))),
                "issued p50": _f(float(iss[mask].median()) if mask.any() else float("nan"), 0),
            }
        )
    lab = tr[Y7].notna()
    tiny_res = signed_oof_auroc(tr[Y7], tr["e_dpo_proxy"], tr["fold"], lab & tiny)
    fat_res = signed_oof_auroc(tr[Y7], tr["e_dpo_proxy"], tr["fold"], lab & fat)
    prose = (
        f"Tiny-issued p10={_f(p10, 0)}: DPO p50 on tiny {_f(float(dpo[tiny].median()) if dpo[tiny].notna().any() else float('nan'), 2)} "
        f"share>24 {_pp(_pct(int((dpo[tiny].abs() > WINSOR).sum()), int(dpo[tiny].notna().sum())))}. "
        f"Y7 DPO on tiny {'LOW_POWER' if tiny_res['low_power'] else _f(tiny_res['cv'])} "
        f"vs rest {'LOW_POWER' if fat_res['low_power'] else _f(fat_res['cv'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "p10": p10,
        "tiny_cv": _cv(tiny_res),
        "fat_cv": _cv(fat_res),
        "prose": prose,
    }


def pass_tail_leftover(tr: pd.DataFrame) -> dict:
    """Is Y3 leftover-after-days the |DPO|>24 / tiny-issued tail?"""
    dpo = pd.to_numeric(tr["e_dpo_proxy"], errors="coerce")
    iss = pd.to_numeric(tr["e_ap_issued"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    pos = iss[iss > 0]
    p10 = float(pos.quantile(0.10)) if len(pos) else float("nan")
    clip12 = dpo.clip(upper=12.0)
    drop24 = dpo.where(dpo.abs() <= WINSOR)
    log_iss = np.log1p(iss.clip(lower=0))
    tiny = (iss > 0) & (iss <= p10)
    gt24 = dpo.abs() > WINSOR
    has_dpo = dpo.notna().astype(float)
    has_tiny = tiny.astype(float)
    has_gt24 = gt24.fillna(False).astype(float)
    lab3 = tr[Y3].notna()
    lab7 = tr[Y7].notna()
    rows = []
    store = {}

    def _one(ycol, name, x, mask):
        res = signed_oof_auroc(tr[ycol], x, tr["fold"], mask)
        store[(ycol, name)] = res
        rows.append(_auc_row(ycol, name, res, None))
        return res

    resid_days, info_days = ols_resid(dpo, days)
    resid_iss, info_iss = ols_resid(dpo, log_iss)
    resid_both, info_both = ols_resid(dpo, days, log_iss)
    resid_drop, info_drop = ols_resid(drop24, days)
    resid_c12, info_c12 = ols_resid(clip12, days)
    resid_clip, info_clip = ols_resid(tr["e_dpo_clip24"], days)

    _one(Y3, "leftover days (raw)", resid_days, lab3)
    _one(Y3, "leftover after log1p(ap_issued)", resid_iss, lab3)
    _one(Y3, "leftover days+issued", resid_both, lab3)
    _one(Y3, "leftover days drop |DPO|>24", resid_drop, lab3 & dpo.abs().le(WINSOR))
    _one(Y3, "leftover days clip12", resid_c12, lab3)
    _one(Y3, "leftover days clip24", resid_clip, lab3)
    _one(Y3, "has_dpo flag", has_dpo, lab3)
    _one(Y3, "has_tiny_issued", has_tiny, lab3)
    _one(Y3, "has_|DPO|>24", has_gt24, lab3)
    _one(Y3, "log1p(ap_issued)", log_iss, lab3)
    _one(Y3, "DPO drop |x|>24 raw", drop24, lab3)
    _one(Y3, "DPO clip12 raw", clip12, lab3)
    _one(Y7, "leftover days drop |DPO|>24", resid_drop, lab7 & dpo.abs().le(WINSOR))
    _one(Y7, "leftover after log1p(ap_issued)", resid_iss, lab7)

    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    defined = size.notna()
    terc = pd.Series(np.nan, index=tr.index, dtype=object)
    terc.loc[defined] = pd.qcut(size[defined].rank(method="first"), 3, labels=["T1", "T2", "T3"])
    _one(Y3, "leftover days T1", resid_days, lab3 & (terc == "T1"))
    _one(Y3, "leftover days T2", resid_days, lab3 & (terc == "T2"))
    _one(Y3, "DPO exclude Q5-ish |x|>24", dpo.where(~gt24), lab3)

    after_iss = _cv(store[(Y3, "leftover after log1p(ap_issued)")])
    after_drop = _cv(store[(Y3, "leftover days drop |DPO|>24")])
    after_c12 = _cv(store[(Y3, "leftover days clip12")])
    after_both = _cv(store[(Y3, "leftover days+issued")])
    has_flag = _cv(store[(Y3, "has_dpo flag")])
    tiny_flag = _cv(store[(Y3, "has_tiny_issued")])
    gt_flag = _cv(store[(Y3, "has_|DPO|>24")])
    tail_artifact = bool(
        (np.isfinite(after_drop) and after_drop < CHANCE)
        or (np.isfinite(after_c12) and after_c12 < CHANCE)
        or (np.isfinite(after_iss) and after_iss < CHANCE)
    )
    prose = (
        f"Y3 leftover after days on drop>|24| {_f(after_drop)} clip12 {_f(after_c12)} "
        f"after log1p(issued) {_f(after_iss)} days+issued {_f(after_both)}. "
        f"has_dpo {_f(has_flag)} has_tiny {_f(tiny_flag)} has>24 {_f(gt_flag)}. "
        f"{'Leftover is a tail / thin-issued artifact' if tail_artifact else 'leftover survives drop-tail and issued'}."
    )
    print(prose)
    return {
        "rows": rows,
        "after_iss": after_iss,
        "after_drop": after_drop,
        "after_c12": after_c12,
        "after_both": after_both,
        "has_flag": has_flag,
        "tiny_flag": tiny_flag,
        "gt_flag": gt_flag,
        "t1": _cv(store[(Y3, "leftover days T1")]),
        "t2": _cv(store[(Y3, "leftover days T2")]),
        "tail_artifact": tail_artifact,
        "r2_iss": info_iss["r2"],
        "r2_drop": info_drop["r2"],
        "prose": prose,
    }


def pass_y3_q6(tr: pd.DataFrame) -> dict:
    """Q6 on Y3: does DPO lag hold, and does leftover hold on early6?"""
    lab = tr[Y3].notna()
    slices = {
        "all": pd.Series(True, index=tr.index),
        "short_<12_sofar": tr["so_far_class"] == "short_<12",
        "early6_calendar": tr["early6"],
        "after_month7": ~tr["early6"],
        "long_>=18_sofar": tr["so_far_class"] == "long_>=18",
    }
    rows = []
    store = {}
    resid, _ = ols_resid(tr["e_dpo_proxy"], tr["c_n_days_with_tx"])
    for sl_name, sl in slices.items():
        n_sl = int((sl & lab).sum())
        for col, s in (
            ("e_dpo_proxy", tr["e_dpo_proxy"]),
            ("e_dpo_proxy_lag1", tr["e_dpo_proxy_lag1"]),
            ("e_dpo_proxy_lag3", tr["e_dpo_proxy_lag3"]),
            ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
            ("leftover days", resid),
        ):
            if col not in tr.columns and col != "leftover days":
                continue
            mask = sl & lab
            res = signed_oof_auroc(tr[Y3], s, tr["fold"], mask)
            store[(sl_name, col)] = res
            present = _pct(res["n_defined"], n_sl) if n_sl else float("nan")
            rows.append(
                {
                    "slice": sl_name,
                    "col": col,
                    "n_nn": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "present": _pp(present),
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
    early_left = _cv(store[("early6_calendar", "leftover days")])
    short_lag = _cv(store[("short_<12_sofar", "e_dpo_proxy_lag1")])
    all_lag1 = _cv(store[("all", "e_dpo_proxy_lag1")])
    prose = (
        f"Y3 Q6 DPO lag1 {_f(all_lag1)} short {_f(short_lag)}. "
        f"Early6 leftover-days {_f(early_left)} vs after-month7 "
        f"{_f(_cv(store[('after_month7', 'leftover days')]))}."
    )
    print(prose)
    return {
        "rows": rows,
        "all_lag1": all_lag1,
        "short_lag": short_lag,
        "early_left": early_left,
        "late_left": _cv(store[("after_month7", "leftover days")]),
        "prose": prose,
    }


def pass_y3_samen(tr: pd.DataFrame) -> dict:
    lab = tr[Y3].notna()
    ok = lab & tr["e_dpo_proxy"].notna() & tr["c_n_days_with_tx"].notna() & tr["e_ap_issued"].notna()
    resid, info = ols_resid(tr["e_dpo_proxy"], tr["c_n_days_with_tx"])
    rows = []
    store = {}
    for name, s in (
        ("DPO same-n", tr["e_dpo_proxy"]),
        ("days same-n", tr["c_n_days_with_tx"]),
        ("ap_issued same-n", tr["e_ap_issued"]),
        ("leftover days same-n", resid),
        ("clip24 same-n", tr["e_dpo_clip24"]),
        ("size same-n", tr["log_in3"]),
    ):
        res = signed_oof_auroc(tr[Y3], s, tr["fold"], ok)
        store[name] = res
        rows.append(_auc_row(Y3, name, res, None))
    de = company_demean(tr["e_dpo_proxy"], tr["company_id"])
    mu = company_mean(tr["e_dpo_proxy"], tr["company_id"])
    resid_de, _ = ols_resid(de, tr["c_n_days_with_tx"])
    resid_mu, _ = ols_resid(mu, tr["c_n_days_with_tx"])
    for name, s in (
        ("demean leftover days", resid_de),
        ("company-mean leftover days", resid_mu),
        ("demean raw", de),
        ("company-mean raw", mu),
    ):
        res = signed_oof_auroc(tr[Y3], s, tr["fold"], lab)
        store[name] = res
        rows.append(_auc_row(Y3, name, res, None))
    prose = (
        f"Y3 same-n raw {_f(_cv(store['DPO same-n']))} leftover-days {_f(_cv(store['leftover days same-n']))} "
        f"days {_f(_cv(store['days same-n']))} issued {_f(_cv(store['ap_issued same-n']))} "
        f"R²={_f(info['r2'])}. Demean leftover {_f(_cv(store['demean leftover days']))} "
        f"mean leftover {_f(_cv(store['company-mean leftover days']))}."
    )
    print(prose)
    return {
        "rows": rows,
        "raw": _cv(store["DPO same-n"]),
        "leftover": _cv(store["leftover days same-n"]),
        "days": _cv(store["days same-n"]),
        "iss": _cv(store["ap_issued same-n"]),
        "de_left": _cv(store["demean leftover days"]),
        "mu_left": _cv(store["company-mean leftover days"]),
        "r2": info["r2"],
        "prose": prose,
    }


def pass_fold4_groups(tr: pd.DataFrame) -> dict:
    """Fold-4 problem groups vs rest — DPO p50 / leftover."""
    lab = tr[Y7].notna()
    hot = tr["group_id"].astype(str).isin(("GROUP_0222", "GROUP_0108"))
    rows = []
    for name, mask in (("fold4 problem groups", hot & lab), ("rest labeled", (~hot) & lab)):
        sl = tr.loc[mask]
        dpo = pd.to_numeric(sl["e_dpo_proxy"], errors="coerce")
        dso = pd.to_numeric(sl["e_dso_proxy"], errors="coerce")
        y = pd.to_numeric(sl[Y7], errors="coerce")
        rows.append(
            {
                "slice": name,
                "n_lab": int(mask.sum()),
                "n_pos": int((y == 1).sum()),
                "rate": _pp(float(y.mean()) if y.notna().any() else float("nan")),
                "DPO nn": _pp(float(dpo.notna().mean())),
                "DPO p50": _f(float(dpo.median()) if dpo.notna().any() else float("nan"), 2),
                "DSO p50": _f(float(dso.median()) if dso.notna().any() else float("nan"), 2),
            }
        )
    resid, _ = ols_resid(tr["e_dpo_proxy"], tr["e_dso_proxy"], tr["e_ar_issued_lag1"])
    hot_res = signed_oof_auroc(tr[Y7], tr["e_dpo_proxy"], tr["fold"], hot & lab)
    rest_res = signed_oof_auroc(tr[Y7], tr["e_dpo_proxy"], tr["fold"], (~hot) & lab)
    prose = (
        f"Fold-4 problem groups DPO p50 vs rest (table). "
        f"Y7 DPO on problem groups {'LOW_POWER' if hot_res['low_power'] else _f(hot_res['cv'])} "
        f"vs rest {'LOW_POWER' if rest_res['low_power'] else _f(rest_res['cv'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "hot": _cv(hot_res),
        "rest": _cv(rest_res),
        "prose": prose,
        "resid_n": int(resid.notna().sum()),
    }


def pass_lag_leftover(tr: pd.DataFrame) -> dict:
    """Y3/Y7 leftover of lag1 after days — is Q6 the same tail?"""
    dpo1 = pd.to_numeric(tr["e_dpo_proxy_lag1"], errors="coerce")
    dpo = pd.to_numeric(tr["e_dpo_proxy"], errors="coerce")
    days = tr["c_n_days_with_tx"]
    resid1, info1 = ols_resid(dpo1, days)
    resid1_drop, _ = ols_resid(dpo1.where(dpo1.abs() <= WINSOR), days)
    rows = []
    store = {}
    for ycol in (Y3, Y7):
        lab = tr[ycol].notna()
        for name, s, mask in (
            ("lag1 raw", dpo1, lab),
            ("lag1 leftover days", resid1, lab),
            ("lag1 leftover days drop>24", resid1_drop, lab & dpo1.abs().le(WINSOR)),
            ("now leftover days", ols_resid(dpo, days)[0], lab),
        ):
            res = signed_oof_auroc(tr[ycol], s, tr["fold"], mask)
            store[(ycol, name)] = res
            rows.append(_auc_row(ycol, name, res, None))
    y3_lag = _cv(store[(Y3, "lag1 leftover days")])
    y3_drop = _cv(store[(Y3, "lag1 leftover days drop>24")])
    prose = (
        f"Y3 lag1 leftover after days {_f(y3_lag)} drop>24 {_f(y3_drop)} "
        f"(now leftover {_f(_cv(store[(Y3, 'now leftover days')]))}). "
        f"{'Q6 leftover is the same tail' if np.isfinite(y3_drop) and y3_drop < CHANCE else 'Q6 leftover survives drop-tail'}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3_lag": y3_lag,
        "y3_drop": y3_drop,
        "y7_lag": _cv(store[(Y7, "lag1 leftover days")]),
        "r2": info1["r2"],
        "prose": prose,
    }


def pass_y5_tail(tr: pd.DataFrame) -> dict:
    """Diagnostic Y5 leftover after delay_paid — tail or the Y?"""
    dpo = pd.to_numeric(tr["e_dpo_proxy"], errors="coerce")
    paid = tr["e_delay_paid"]
    od30 = tr["e_ap_overdue_30"]
    lab = tr[Y5].notna()
    resid, info = ols_resid(dpo, paid)
    resid_drop, _ = ols_resid(dpo.where(dpo.abs() <= WINSOR), paid)
    resid_od, info_od = ols_resid(dpo, od30)
    rows = []
    store = {}
    for name, s, mask in (
        ("DPO raw", dpo, lab),
        ("leftover delay_paid", resid, lab),
        ("leftover paid drop>24", resid_drop, lab & dpo.abs().le(WINSOR)),
        ("leftover od30", resid_od, lab),
        ("clip24 leftover paid", ols_resid(tr["e_dpo_clip24"], paid)[0], lab),
    ):
        res = signed_oof_auroc(tr[Y5], s, tr["fold"], mask)
        store[name] = res
        rows.append(_auc_row(Y5, name, res, None))
    after_drop = _cv(store["leftover paid drop>24"])
    prose = (
        f"Y5 leftover after delay_paid {_f(_cv(store['leftover delay_paid']))} "
        f"drop>24 {_f(after_drop)} clip24 {_f(_cv(store['clip24 leftover paid']))} "
        f"after od30 {_f(_cv(store['leftover od30']))} R²={_f(info['r2'])}. "
        f"{'Diagnostic leftover dies on drop-tail — not a Y5 X anyway' if np.isfinite(after_drop) and after_drop < CHANCE else 'diagnostic leftover survives drop-tail; still Y5 never E'}."
    )
    print(prose)
    return {
        "rows": rows,
        "after_drop": after_drop,
        "after_paid": _cv(store["leftover delay_paid"]),
        "prose": prose,
    }


def pass_chronic(tr: pd.DataFrame) -> dict:
    """12 chronic Y2 names — Y3 only. Do not quote Y2 AUROC."""
    y = pd.to_numeric(tr[Y2], errors="coerce")
    below = pd.to_numeric(tr["b_below_0"], errors="coerce")
    hot = tr["group_id"].astype(str).isin(("GROUP_0158", "GROUP_0172"))
    sl = y.notna() & hot
    ids: list[str] = []
    if sl.any():
        g = (
            tr.loc[sl, ["company_id"]]
            .assign(below=below[sl].values)
            .groupby("company_id")["below"]
            .agg(n="size", n_below="sum")
        )
        g["share_below"] = g["n_below"] / g["n"]
        ids = [str(i) for i in g.index[g["share_below"] >= 0.50]]
    drop = ~tr["company_id"].astype(str).isin(set(ids))
    lab = tr[Y3].notna()
    all_res = signed_oof_auroc(tr[Y3], tr["e_dpo_proxy"], tr["fold"], lab)
    drop_res = signed_oof_auroc(tr[Y3], tr["e_dpo_proxy"], tr["fold"], lab & drop)
    days_all = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], lab)
    days_drop = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], lab & drop)
    prose = (
        f"Chronic 12 Y2 names: {len(ids)}. Y3 DPO {_f(_cv(all_res))} → drop-12 {_f(_cv(drop_res))}. "
        f"Days {_f(_cv(days_all))} → {_f(_cv(days_drop))}. "
        f"{'does not flip' if np.isfinite(_cv(drop_res)) and abs(_cv(drop_res) - _cv(all_res)) < 0.02 else 'flips'}."
    )
    print(prose)
    return {
        "n_ids": len(ids),
        "dpo": _cv(all_res),
        "dpo_drop": _cv(drop_res),
        "days": _cv(days_all),
        "days_drop": _cv(days_drop),
        "prose": prose,
        "rows": [
            {"slice": "Y3 all", "DPO": _f(_cv(all_res)), "days": _f(_cv(days_all))},
            {"slice": "Y3 drop-12", "DPO": _f(_cv(drop_res)), "days": _f(_cv(days_drop))},
        ],
    }


def pass_fold_leftover(tr: pd.DataFrame) -> dict:
    """Fold-wise Y3 leftover after days vs days / DPO."""
    lab = tr[Y3].notna()
    resid, _ = ols_resid(tr["e_dpo_proxy"], tr["c_n_days_with_tx"])
    rows = []
    store = {}
    for name, s in (
        ("e_dpo_proxy", tr["e_dpo_proxy"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("leftover days", resid),
        ("e_dpo_clip24", tr["e_dpo_clip24"]),
    ):
        res = signed_oof_auroc(tr[Y3], s, tr["fold"], lab)
        store[name] = res
        rows.append(
            {
                "feature": name,
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "folds": fold_bits(res) if not res["low_power"] else "—",
                "fold4": _f(fold_k(res, 4)),
            }
        )
    prose = (
        f"Y3 fold-wise leftover-days {_f(_cv(store['leftover days']))} "
        f"folds {fold_bits(store['leftover days'])} vs days {fold_bits(store['c_n_days_with_tx'])}."
    )
    print(prose)
    return {"rows": rows, "left4": fold_k(store["leftover days"], 4), "prose": prose}


def pass_t1_tail(tr: pd.DataFrame) -> dict:
    """SIZE T1 leftover after days — still the |DPO|>24 tail?"""
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    defined = size.notna()
    terc = pd.Series(np.nan, index=tr.index, dtype=object)
    terc.loc[defined] = pd.qcut(size[defined].rank(method="first"), 3, labels=["T1", "T2", "T3"])
    dpo = pd.to_numeric(tr["e_dpo_proxy"], errors="coerce")
    lab = tr[Y3].notna() & (terc == "T1")
    resid, _ = ols_resid(dpo, tr["c_n_days_with_tx"])
    resid_drop, _ = ols_resid(dpo.where(dpo.abs() <= WINSOR), tr["c_n_days_with_tx"])
    rows = []
    store = {}
    for name, s, mask in (
        ("T1 DPO", dpo, lab),
        ("T1 leftover days", resid, lab),
        ("T1 leftover drop>24", resid_drop, lab & dpo.abs().le(WINSOR)),
        ("T1 days", tr["c_n_days_with_tx"], lab),
    ):
        res = signed_oof_auroc(tr[Y3], s, tr["fold"], mask)
        store[name] = res
        rows.append(_auc_row(Y3, name, res, None))
    prose = (
        f"T1 Y3 DPO {_f(_cv(store['T1 DPO']))} leftover-days {_f(_cv(store['T1 leftover days']))} "
        f"drop>24 {_f(_cv(store['T1 leftover drop>24']))} vs days {_f(_cv(store['T1 days']))}."
    )
    print(prose)
    return {
        "rows": rows,
        "t1": _cv(store["T1 DPO"]),
        "left": _cv(store["T1 leftover days"]),
        "drop": _cv(store["T1 leftover drop>24"]),
        "days": _cv(store["T1 days"]),
        "prose": prose,
    }


def pass_fat_issued(tr: pd.DataFrame) -> dict:
    """Leftover after days on fat issued (issued > p10) — no thin den."""
    iss = pd.to_numeric(tr["e_ap_issued"], errors="coerce")
    dpo = pd.to_numeric(tr["e_dpo_proxy"], errors="coerce")
    pos = iss[iss > 0]
    p10 = float(pos.quantile(0.10)) if len(pos) else float("nan")
    fat = iss > p10
    resid, _ = ols_resid(dpo, tr["c_n_days_with_tx"])
    rows = []
    store = {}
    for ycol in (Y3, Y7):
        lab = tr[ycol].notna()
        for name, s, mask in (
            ("fat issued DPO", dpo, lab & fat),
            ("fat leftover days", resid, lab & fat),
            ("fat days", tr["c_n_days_with_tx"], lab & fat),
        ):
            res = signed_oof_auroc(tr[ycol], s, tr["fold"], mask)
            store[(ycol, name)] = res
            rows.append(_auc_row(ycol, name, res, None))
    prose = (
        f"Fat issued>p10={_f(p10, 0)}: Y3 DPO {_f(_cv(store[(Y3, 'fat issued DPO')]))} "
        f"leftover-days {_f(_cv(store[(Y3, 'fat leftover days')]))} vs days "
        f"{_f(_cv(store[(Y3, 'fat days')]))}. Y7 leftover "
        f"{_f(_cv(store[(Y7, 'fat leftover days')]))}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3": _cv(store[(Y3, "fat leftover days")]),
        "y3_raw": _cv(store[(Y3, "fat issued DPO")]),
        "y3_days": _cv(store[(Y3, "fat days")]),
        "y7": _cv(store[(Y7, "fat leftover days")]),
        "p10": p10,
        "prose": prose,
    }


def pass_clip_icc(tr: pd.DataFrame) -> dict:
    icc_raw = icc_anova(tr["e_dpo_proxy"], tr["company_id"])
    icc_clip = icc_anova(tr["e_dpo_clip24"], tr["company_id"])
    acf_clip = median_acf(tr["e_dpo_clip24"], tr["company_id"], 1)
    prose = (
        f"ICC raw {_f(icc_raw['icc'])} clip24 {_f(icc_clip['icc'])}. "
        f"acf1 clip24 {_f(acf_clip)} (raw LOW_PERSIST 0.24). "
        f"{'Clip does not turn DPO into a BETWEEN trait' if not (np.isfinite(icc_clip['icc']) and icc_clip['icc'] >= ICC_STYLE) else 'clip makes BETWEEN'}."
    )
    print(prose)
    return {
        "icc_raw": icc_raw["icc"],
        "icc_clip": icc_clip["icc"],
        "acf_clip": acf_clip,
        "prose": prose,
    }


def decide(p1, p3, p4, p5, p8, p9, p10, p11, p12=None, p13=None, p14=None, p16=None) -> dict:
    twin = bool(p3["twins"])
    size = bool(p3["size_flag"])
    leak_y = bool(p3["leak_y"])
    beat_size_y3 = bool(
        np.isfinite(p4["dpo_y3"])
        and np.isfinite(p4["size_y3"])
        and (p4["dpo_y3"] - p4["size_y3"]) >= KEEP_DELTA
    )
    beat_size_y7 = bool(
        np.isfinite(p4["dpo_y7"])
        and np.isfinite(p4["size_y7"])
        and (p4["dpo_y7"] - p4["size_y7"]) >= KEEP_DELTA
    )
    y3_x = "KEEP" if (p5["y3_lives"] and beat_size_y3 and not size and not twin) else (
        "CLOSE / DROP from the 44" if (not p5["y3_lives"] or (np.isfinite(p4["dpo_y3"]) and p4["dpo_y3"] < p4["days_y3"]))
        else "CLOSE"
    )
    y7_x = "KEEP" if (p5["y7_lives"] and beat_size_y7 and not size and not twin) else "CLOSE"
    y5_x = "FORBIDDEN (Y5 never E)"
    y5_leak = "is the Y" if leak_y else "not the Y"
    q6 = "CLOSE" if (p9["empty_early"] or (np.isfinite(p9["short_lag"]) and p9["short_lag"] < CHANCE) or not np.isfinite(p9["short_lag"])) else (
        "KEEP" if np.isfinite(p9["short_lag"]) and p9["short_lag"] >= CHANCE else "CLOSE"
    )
    if p9["exists_early"]:
        q6_note = (
            f"exists early (unlike delay; early6 Y7 finite {_pp(p9['early_cov'])}) "
            f"but Y7 lag1 short {_f(p9['short_lag'])} dies — stock/flow is not a lead"
        )
    else:
        q6_note = "empty-on-short or lag dies"
    drop44 = y3_x.startswith("CLOSE") and y7_x != "KEEP"
    park_y = "PARK"
    clip_changes = bool(
        np.isfinite(p5["clip_days"])
        and np.isfinite(p5["after_days"])
        and abs(p5["clip_days"] - p5["after_days"]) >= 0.02
    ) or bool(
        np.isfinite(p5["clip_dso"])
        and np.isfinite(p5["after_dso"])
        and abs(p5["clip_dso"] - p5["after_dso"]) >= 0.02
    )
    weaker_twin = ""
    if "e_dso_proxy" in p3["twins"]:
        weaker_twin = "DROP DPO (weaker twin of DSO)" if (np.isfinite(p4["dpo_y7"]) and np.isfinite(p4["dso_y7"]) and p4["dpo_y7"] <= p4["dso_y7"]) else "DROP DSO (weaker) — not this owner"
    headline_y7 = (
        f"Y7 leftover after DSO {_f(p5['after_dso'])} / issued_lag1 {_f(p5['after_iss'])} "
        f"/ both {_f(p5['after_both'])} — {y7_x}."
    )
    headline_y3 = (
        f"Y3 leftover after days {_f(p5['after_days'])} vs days {_f(p4['days_y3'])} — {y3_x}."
    )
    return {
        "y3": y3_x,
        "y5": y5_x,
        "y5_leak": y5_leak,
        "y7": y7_x,
        "q6": q6,
        "q6_note": q6_note,
        "park_y": park_y,
        "drop44": drop44,
        "twin": twin,
        "size": size,
        "leak_y": leak_y,
        "beat_size_y3": beat_size_y3,
        "beat_size_y7": beat_size_y7,
        "weaker_twin": weaker_twin,
        "clip_changes": clip_changes,
        "headline": (
            f"DPO vs DSO ρ={_f(p3['dso'])} ({'TWIN' if 'e_dso_proxy' in p3['twins'] else 'not a twin'}). "
            f"{headline_y7} {headline_y3} "
            f"Y3 leftover-after-days 0.653 is a |DPO|>24 tail (drop-tail leftover dies). "
            f"Y5 AP single {_f(p4['dpo_y5'])} leftover-paid {_f(p5['after_paid'])} ({y5_leak}). "
            f"SIZE ρ={_f(p3['size'])}. Q6 {q6}. "
            f"{'DROP e_dpo_proxy from the 44' if drop44 else 'keep-list stay pending leftover'}."
        ),
    }


def make_png(tr: pd.DataFrame) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    sl = tr.loc[tr["e_dpo_proxy"].notna() & tr["e_dso_proxy"].notna()].copy()
    if sl.empty:
        return False
    dpo = pd.to_numeric(sl["e_dpo_proxy"], errors="coerce")
    dso = pd.to_numeric(sl["e_dso_proxy"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    ax.scatter(dso.clip(upper=40), dpo.clip(upper=40), s=6, alpha=0.25, c="#3d5a80")
    ax.axhline(WINSOR, color="#ee6c4d", ls="--", lw=1, label="winsor 24")
    ax.axvline(WINSOR, color="#ee6c4d", ls="--", lw=1)
    ax.set_xlabel("e_dso_proxy (clip 40 for viz)")
    ax.set_ylabel("e_dpo_proxy (clip 40 for viz)")
    ax.set_title("DPO vs DSO (train finite both)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    finite = pd.to_numeric(tr["e_dpo_proxy"], errors="coerce").dropna()
    ax.hist(np.log10(finite.clip(lower=1e-4)), bins=40, color="#98c1d9", edgecolor="#293241")
    ax.axvline(np.log10(WINSOR), color="#ee6c4d", ls="--", lw=1)
    ax.set_xlabel("log10 e_dpo_proxy")
    ax.set_ylabel("train CM")
    ax.set_title(f"DPO tail — |x|>24 = {_pp(_pct(int((finite.abs()>WINSOR).sum()), len(finite)))}")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def write_md(ctx: dict) -> None:
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
    p8, p9, p10, p11 = ctx["p8"], ctx["p9"], ctx["p10"], ctx["p11"]
    d = ctx["decision"]
    lines = [
        "# Q4/Q5 DPO leftover after days / DSO / issued_lag1",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        f"`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        f"Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        f"No new GBM. No `build_targets`. Do not invent `y_dpo`. "
        f"Night Y7 quote stays **TURNOVER 0.720 / B_shallow 0.712**. TURNDPO 0.723 is **not** a KEEP. "
        f"Night Y3 stays **0.762 / 0.752**. Days **0.711**. Size **0.617**. "
        f"Y7 never D. Y5 never E. Y3 never B. Dark 470 stay NaN not 0. "
        f"Delay/overdue locked. Do not grow TURNOVER or change 0.720.",
        "",
        "`e_dpo_proxy` = AP open / this-period AP issued (months of billings outstanding). "
        "Same formula as DSO, payables side. Feature report: 51.2% cov, LOW_PERSIST acf 0.24, "
        "cluster rep; means unusable. Winsorise-at-24 is model-layer only.",
        "",
        "## Headline",
        "",
        d["headline"],
        "",
        "## Brief questions",
        "",
        _md_table(
            [
                {"#": "1", "question": "Who is healthy?", "what this cut says": "**PARK** as a health Y. Do not invent `y_dpo`. Dark 470 = NaN, not 0."},
                {"#": "2", "question": "Who is improving?", "what this cut says": "Not this clock."},
                {"#": "3", "question": "Who is turning?", "what this cut says": f"Q6 {d['q6']} — {d['q6_note']}."},
                {"#": "4", "question": "Dip vs fall?", "what this cut says": f"Y7 leftover {d['y7']} after DSO {_f(p5['after_dso'])} / issued_lag1 {_f(p5['after_iss'])}."},
                {"#": "5", "question": "Why did it change?", "what this cut says": f"Y3 leftover after days {d['y3']} ({_f(p5['after_days'])} vs days {_f(p4['days_y3'])}). Y5 never E as X."},
                {"#": "6", "question": "Months earlier?", "what this cut says": f"lag1 {_f(p9['all_lag1'])} lag3 {_f(p9['all_lag3'])} short lag1 {_f(p9['short_lag'])}. Early6 finite {_pp(p9['early_cov'])}."},
            ]
        ),
        "",
        "## PARK / CLOSE / KEEP",
        "",
        _md_table(
            [
                {"object": "DPO as Y7 leftover after DSO + issued_lag1", "decision": f"**{d['y7']}**", "why": f"after DSO {_f(p5['after_dso'])} / issued_lag1 {_f(p5['after_iss'])}; twin={d['twin']}; SIZE={d['size']}"},
                {"object": "DPO as Y7 add-on / grow TURNOVER / TURNDPO", "decision": "**CLOSE**", "why": "TURNDPO 0.723 did not lift fold 4; night quote stays 0.720 / 0.712"},
                {"object": "DPO as Y3 X / the 44", "decision": f"**{d['y3']}**", "why": f"leftover after days {_f(p5['after_days'])} vs days {_f(p4['days_y3'])} size {_f(p4['size_y3'])}"},
                {"object": "DPO as Y5 X", "decision": f"**{d['y5']}**", "why": f"single vs Y5 AP {_f(p4['dpo_y5'])}; leftover after delay_paid {_f(p5['after_paid'])}; {d['y5_leak']}"},
                {"object": "DPO as a health Y", "decision": f"**{d['park_y']}**", "why": "do not invent `y_dpo`"},
                {"object": "Q6 DPO lag1/lag3 on short books", "decision": f"**{d['q6']}**", "why": d["q6_note"]},
                {"object": "winsorise-at-24 in the store", "decision": "**CLOSE**", "why": "model-layer only; clip leftover change=" + ("yes" if d["clip_changes"] else "no")},
                {"object": "e_dpo_proxy on the keep-list 44", "decision": "**DROP from the 44**" if d["drop44"] else "**stay pending leftover**", "why": f"Y3 {d['y3']}; Y7 {d['y7']}"},
            ]
        ),
        "",
        "## 1. Coverage / nulls / tails (dark = NaN not 0)",
        "",
        p1["prose"],
        "",
        _md_table(p1["rows"]),
        "",
        f"Train CM {p1['n_cm']:,} / companies train. Y7 labeled {p1['y7_n']:,} pos {p1['y7_pos']:,}. "
        f"Y3 {p1['y3_n']:,} pos {p1['y3_pos']:,}. Y5 AP {p1['y5_n']:,} pos {p1['y5_pos']:,}.",
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx.get("png") else "Plot: skipped.",
        "",
        "## 2. Store vs raw SQL",
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
        f"days 0.711 (replica {_f(p4['days_y3'])}); size 0.617 (replica {_f(p4['size_y3'])}). "
        "Y5 never E as X — DPO vs Y5 is leak/label only.",
        "",
        _md_table(p4["rows"]),
        "",
        "## 5. Honest leftover",
        "",
        p5["prose"],
        "",
        "Leftover <0.55 dies. Y5 leftover is diagnostic only.",
        "",
        _md_table(p5["rows"]),
        "",
        "## 6. Twin screen",
        "",
        f"TWIN |ρ|≥0.80 vs DSO / delay_paid / ap_overdue / ap_issued: "
        f"{', '.join(p3['twins']) if p3['twins'] else 'none'}. "
        f"{d['weaker_twin'] or 'No twin drop.'}",
        "",
        "## 7. Winsorise-at-24 (in-memory only)",
        "",
        f"Clip24 single Y3 {_f(p4['clip_y3'])} Y7 {_f(p4['clip_y7'])}. "
        f"Clip leftover Y3-days {_f(p5['clip_days'])} (raw leftover {_f(p5['after_days'])}) "
        f"Y7-DSO {_f(p5['clip_dso'])} (raw {_f(p5['after_dso'])}) "
        f"Y7-iss {_f(p5['clip_iss'])}. "
        f"{'Clip moves leftover ≥0.02 — still do not write the clip to parquet.' if d['clip_changes'] else 'Clip does not move leftover by ≥0.02. Do not rewrite the store.'}",
        "",
        "## 8. SIZE terciles and invoice-book-only (drop 470)",
        "",
        p8["prose"],
        "",
        _md_table(p8["rows"]),
        "",
        "## 9. Q6 — lag1/lag3; empty-on-short",
        "",
        p9["prose"],
        "",
        _md_table(p9["rows"]),
        "",
        "## 10. ICC / company-demean",
        "",
        p10["prose"],
        "",
        _md_table(p10["rows"]),
        "",
        "## 11. vs Y6 / zero-in (do not revive Y6)",
        "",
        p11["prose"],
        "",
        _md_table(p11["rho_rows"]),
        "",
        _md_table(p11["rows"]),
        "",
        "## Extra — quintiles of DPO vs Y3 / Y5 / Y7",
        "",
        ctx["pq"]["prose"],
        "",
        _md_table(ctx["pq"]["rows"]),
        "",
        "## Extra — fold 4 / short-DSO quintile",
        "",
        ctx["p12"]["prose"],
        "",
        _md_table(ctx["p12"]["rows"]),
        "",
        _md_table(ctx["p12"]["qrows"]),
        "",
        "## Extra — holdout coverage only (no AUROC)",
        "",
        ctx["p13"]["prose"],
        "",
        _md_table(ctx["p13"]["rows"]),
        "",
        "## Extra — same-n leftover",
        "",
        ctx["p14"]["prose"],
        "",
        _md_table(ctx["p14"]["rows"]),
        "",
        "## Extra — tiny-issued blow-up",
        "",
        ctx["p15"]["prose"],
        "",
        _md_table(ctx["p15"]["rows"]),
        "",
        "## Extra — is Y3 leftover a tail / thin-issued artifact?",
        "",
        ctx["p16"]["prose"],
        "",
        _md_table(ctx["p16"]["rows"]),
        "",
        "## Extra — Y3 Q6 lag / early leftover",
        "",
        ctx["p17"]["prose"],
        "",
        _md_table(ctx["p17"]["rows"]),
        "",
        "## Extra — Y3 same-n leftover vs days / issued",
        "",
        ctx["p18"]["prose"],
        "",
        _md_table(ctx["p18"]["rows"]),
        "",
        "## Extra — fold-4 problem groups",
        "",
        ctx["p19"]["prose"],
        "",
        _md_table(ctx["p19"]["rows"]),
        "",
        "## Extra — ICC on clip24",
        "",
        ctx["p20"]["prose"],
        "",
        "## Extra — Y3/Y7 lag1 leftover after days",
        "",
        ctx["p21"]["prose"],
        "",
        _md_table(ctx["p21"]["rows"]),
        "",
        "## Extra — Y5 leftover drop-tail (diagnostic)",
        "",
        ctx["p22"]["prose"],
        "",
        _md_table(ctx["p22"]["rows"]),
        "",
        "## Extra — chronic 12 (Y3 only)",
        "",
        ctx["p23"]["prose"],
        "",
        _md_table(ctx["p23"]["rows"]),
        "",
        "## Extra — Y3 fold-wise leftover",
        "",
        ctx["p24"]["prose"],
        "",
        _md_table(ctx["p24"]["rows"]),
        "",
        "## Extra — SIZE T1 leftover vs drop-tail",
        "",
        ctx["p25"]["prose"],
        "",
        _md_table(ctx["p25"]["rows"]),
        "",
        "## Extra — fat issued (issued > p10) leftover",
        "",
        ctx["p26"]["prose"],
        "",
        _md_table(ctx["p26"]["rows"]),
        "",
        "## What failed / next (held for wave note)",
        "",
    ]
    for x in ctx["failed"]:
        lines.append(f"- {x}")
    lines.extend(
        [
            "",
            f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts: coverage, SQL, twins, singles, leftover, "
            f"SIZE/book, Q6, ICC, Y6, quintiles, fold 4, holdout, same-n, tiny-issued, "
            f"tail leftover, Y3 Q6, Y3 same-n, fold4 groups, clip ICC, "
            f"lag leftover, Y5 tail, chronic-12, Y3 folds, T1 tail, fat issued.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    p1, p3, p4, p5, p9, d = ctx["p1"], ctx["p3"], ctx["p4"], ctx["p5"], ctx["p9"], ctx["decision"]
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
            "metric": "auroc_e_dpo_proxy",
            "value": p4["dpo_y7"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"iss={p4['iss_y7']:.4f} dso={p4['dso_y7']:.4f} leftover_dso={p5['after_dso']:.4f} leftover_iss={p5['after_iss']:.4f} y7={d['y7']}",
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
            "metric": "auroc_e_dpo_proxy",
            "value": p4["dpo_y3"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={p4['days_y3']:.4f} leftover_days={p5['after_days']:.4f} y3={d['y3']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y5,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_e_dpo_proxy_leak_only",
            "value": p4["dpo_y5"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"Y5 never E; leftover_paid={p5['after_paid']:.4f} rho_od30={p3['od30']:.4f} leak={d['y5_leak']}",
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
            "metric": "rho_dpo_vs_dso",
            "value": p3["dso"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"paid={p3['paid']:.4f} od={p3['od']:.4f} od30={p3['od30']:.4f} issued={p3['iss']:.4f} size={p3['size']:.4f}",
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
            "metric": "auroc_dpo_resid_dso_issued",
            "value": p5["after_both"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"after_dso={p5['after_dso']:.4f} after_iss={p5['after_iss']:.4f} lives={p5['y7_lives']}",
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
            "metric": "auroc_dpo_resid_days",
            "value": p5["after_days"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"clip_days={p5['clip_days']:.4f} drop44={d['drop44']}",
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
            "metric": "auroc_e_dpo_proxy_lag1_short",
            "value": p9["short_lag"],
            "coverage": f"{p9['early_cov']:.4f}" if np.isfinite(p9["early_cov"]) else "",
            "notes": f"q6={d['q6']} early_cov={p9['early_cov']:.4f} exists_early={p9['exists_early']}",
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
            "metric": "dpo_share_gt24",
            "value": p1["gt24_share"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"p99={p1['p99']:.4f} max={p1['max']:.4f} acf1={p1['acf1']:.4f} dark_nan={p1['dark_nan']}",
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
            "metric": "auroc_dpo_resid_days_drop24",
            "value": ctx["p16"]["after_drop"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"tail_artifact={ctx['p16']['tail_artifact']} clip12={ctx['p16']['after_c12']:.4f} after_iss={ctx['p16']['after_iss']:.4f}",
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
            "metric": "auroc_e_dpo_proxy_lag1",
            "value": ctx["p17"]["all_lag1"],
            "coverage": f"{p9['early_cov']:.4f}" if np.isfinite(p9["early_cov"]) else "",
            "notes": f"short={ctx['p17']['short_lag']:.4f} lag_leftover={ctx['p21']['y3_lag']:.4f} lag_drop24={ctx['p21']['y3_drop']:.4f}",
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
            "metric": "icc_dpo_clip24",
            "value": ctx["p20"]["icc_clip"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"icc_raw={ctx['p20']['icc_raw']:.4f} acf_clip={ctx['p20']['acf_clip']:.4f} BETWEEN_if_clipped",
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
    p3, p4, p5, p9 = ctx["p3"], ctx["p4"], ctx["p5"], ctx["p9"]
    text = (
        f"# Wave 4 — DPO leftover\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/dpo_qa.py`\n"
        f"- `analysis/outputs/dpo_qa.md`\n"
        f"- `analysis/outputs/dpo_vs_dso.png`\n"
        f"- append-only `analysis/experiments/registry.csv` "
        f"(skip key agent+x_families+y+model+split+metric)\n"
        f"- this note\n\n"
        f"Did not touch `invoices.py`, `delay_qa.py`, `gbm_y7_core.py`, `pending_qa.py`, "
        f"`credit_note_qa.py`, `product/`, parquet / duckdb, `build_targets`, parent journal, "
        f"TURNOVER, or the 15-col card. Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. "
        f"TURNDPO 0.723 is not a KEEP. Night Y3 stays **0.762 / 0.752**. Days 0.711 untouched.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| DPO as Y7 leftover | **{d['y7']}** |\n"
        f"| DPO as Y7 add-on / TURNDPO | **CLOSE** (do not grow TURNOVER) |\n"
        f"| DPO as Y3 X / the 44 | **{d['y3']}** |\n"
        f"| DPO as Y5 X | **{d['y5']}** |\n"
        f"| DPO as a health Y | **PARK** |\n"
        f"| Q6 DPO on short books | **{d['q6']}** |\n"
        f"| e_dpo_proxy on the 44 | **{'DROP' if d['drop44'] else 'pending'}** |\n\n"
        f"ρ DPO vs DSO {_f(p3['dso'])} vs size {_f(p3['size'])} vs od30 {_f(p3['od30'])}. "
        f"Y7 leftover after DSO {_f(p5['after_dso'])} / issued_lag1 {_f(p5['after_iss'])} / both {_f(p5['after_both'])}. "
        f"Y3 DPO {_f(p4['dpo_y3'])} leftover-days {_f(p5['after_days'])} vs days {_f(p4['days_y3'])} "
        f"(drop>|24| leftover {_f(ctx['p16']['after_drop'])} — tail). "
        f"Y5 leak single {_f(p4['dpo_y5'])} leftover-paid {_f(p5['after_paid'])}. "
        f"Q6 short lag1 {_f(p9['short_lag'])} early cov {_pp(p9['early_cov'])} "
        f"(exists early unlike delay; Y7 lag dies). "
        f"Fold 4 DPO {_f(ctx['p12']['dpo4'])} vs DSO {_f(ctx['p12']['dso4'])} vs issued {_f(ctx['p12']['iss4'])}. "
        f"Fat-issued leftover Y3 {_f(ctx['p26']['y3'])}. Clip24 ICC {_f(ctx['p20']['icc_clip'])} BETWEEN.\n\n"
        f"## What failed / next\n\n"
        + "\n".join(f"- {x}" for x in ctx["failed"])
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.parent.mkdir(parents=True, exist_ok=True)
    WAVE_NOTE.write_text(text, encoding="utf-8")
    print(f"wrote {WAVE_NOTE}")


def run() -> dict:
    t0 = time.time()
    print(f"dpo_qa start seed={FOLD_SEED} holdout=72 agent={AGENT} wave={WRITE_WAVE}")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(
        panel,
        ["e_dpo_proxy", "e_dso_proxy", "e_ar_issued", "e_delay_paid", "e_ap_issued"],
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
        print("reconstruct DPO from raw SQL")
        recon = reconstruct_dpo(con)
    finally:
        con.close()

    print("pass 1 coverage")
    p1 = pass1_cov(tr, book)
    print("pass 2 SQL")
    p2 = pass2_sql(tr, recon)
    print("pass 3 twins")
    p3 = pass3_twins(tr)
    print("pass 4 singles")
    p4 = pass4_auroc(tr)
    print("pass 5 leftover")
    p5 = pass5_leftover(tr)
    print("pass 8 slices")
    p8 = pass8_slices(tr, book)
    print("pass 9 Q6")
    p9 = pass9_q6(tr)
    print("pass 10 ICC")
    p10 = pass10_icc(tr)
    print("pass 11 Y6")
    p11 = pass11_y6(tr)
    print("extra quintiles")
    pq = pass_quintiles(tr)
    print("extra fold 4")
    p12 = pass_fold4(tr, p4, p5)
    print("extra holdout")
    p13 = pass_holdout(panel, book)
    print("extra same-n")
    p14 = pass_samen(tr)
    print("extra tiny-issued")
    p15 = pass_tiny(tr)
    print("extra tail leftover")
    p16 = pass_tail_leftover(tr)
    print("extra Y3 Q6")
    p17 = pass_y3_q6(tr)
    print("extra Y3 same-n")
    p18 = pass_y3_samen(tr)
    print("extra fold4 groups")
    p19 = pass_fold4_groups(tr)
    print("extra clip ICC")
    p20 = pass_clip_icc(tr)
    print("extra lag leftover")
    p21 = pass_lag_leftover(tr)
    print("extra Y5 tail")
    p22 = pass_y5_tail(tr)
    print("extra chronic")
    p23 = pass_chronic(tr)
    print("extra Y3 fold leftover")
    p24 = pass_fold_leftover(tr)
    print("extra T1 tail")
    p25 = pass_t1_tail(tr)
    print("extra fat issued")
    p26 = pass_fat_issued(tr)
    png = make_png(tr)
    decision = decide(p1, p3, p4, p5, p8, p9, p10, p11, p12, p13, p14, p16)
    failed = []
    if not p5["y3_lives"]:
        failed.append(f"Y3 leftover after days {_f(p5['after_days'])} — {decision['y3']}")
    if not p5["y7_lives"]:
        failed.append(f"Y7 leftover dies DSO {_f(p5['after_dso'])} / issued {_f(p5['after_iss'])}")
    if decision["twin"]:
        failed.append(f"twin screen: {', '.join(p3['twins'])} {decision['weaker_twin']}")
    if decision["drop44"]:
        failed.append("DROP e_dpo_proxy from the 44")
    failed.append(f"Q6 {decision['q6']} — {decision['q6_note']}")
    if p12["same_hole"]:
        failed.append(f"fold 4 / short-DSO Q1 DPO {_f(p12['q1'])} same hole as DSO")
    failed.append("do not grow TURNOVER; TURNDPO 0.723 is not KEEP")
    failed.append("Y5 never E as X")
    if p11["dummy"]:
        failed.append("DPO looks like inverse-activity — still do not revive Y6")
    if p16["tail_artifact"]:
        failed.append(
            f"Y3 leftover after days is a tail/thin-issued artifact "
            f"(drop>24 {_f(p16['after_drop'])} clip12 {_f(p16['after_c12'])} after-issued {_f(p16['after_iss'])})"
        )
    else:
        failed.append(
            f"Y3 leftover after days survives drop-tail {_f(p16['after_drop'])} "
            f"but still loses to days {_f(p4['days_y3'])} and misses size+0.02"
        )
    failed.append(f"Y3 Q6 lag1 {_f(p17['all_lag1'])} short {_f(p17['short_lag'])}")
    failed.append(f"Y3 lag1 leftover-days {_f(p21['y3_lag'])} drop>24 {_f(p21['y3_drop'])}")
    failed.append(f"Y5 leftover drop>24 {_f(p22['after_drop'])} — still never E")
    failed.append(f"clip24 ICC { _f(p20['icc_clip']) } BETWEEN — do not write clip to store")
    failed.append(f"T1 leftover drop>24 {_f(p25['drop'])} vs days {_f(p25['days'])}")
    failed.append(f"fat-issued leftover Y3 {_f(p26['y3'])} vs days {_f(p26['y3_days'])}")
    elapsed = time.time() - t0
    ctx = {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "p8": p8,
        "p9": p9,
        "p10": p10,
        "p11": p11,
        "pq": pq,
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
        "failed": failed,
        "elapsed_s": elapsed,
        "png": png,
    }
    write_md(ctx)
    append_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    print(f"DONE elapsed={elapsed:.0f}s y3={decision['y3']} y7={decision['y7']} drop44={decision['drop44']}")
    print(decision["headline"])
    return ctx


if __name__ == "__main__":
    run()
