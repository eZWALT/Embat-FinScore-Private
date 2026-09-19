"""Q4/Q5 unused leftover of `e_dso_proxy` on the 44.

NORTH_STAR: `e_dso_proxy` = AR open / this-period AR issued (months of
billings outstanding). Feature report: 40.6% cov, acf1 0.25, ICC 0.63;
means unusable. Winsorise-at-24 is model-layer only — do not rewrite
the store.

DSO is already CLOSE as a TURNOVER stem. The even card **drops DSO**.
Do **not** put DSO back on TURNOVER. Do not change 0.720. Night Y7
stays **TURNOVER 0.720 / B_shallow 0.712**. Fold 4 TURNOVER 0.680 is
issued, not DSO. DSO is SHAP #1 and fails the short-DSO quintile
(B_shallow OOF **0.410**). Night Y3 stays **0.762 / 0.752**. Days
**0.711**. Size **0.617**. Y7 never D. Y5 never E. Y3 never B. Dark
470 stay NaN not 0. DPO just DROPPED from the 44. Delay leftover KEEP
after DSO (0.581); delay not a DSO twin ρ 0.214.

This ticket is the unused leftover: after days as Y3 X, after
issued_lag1 as Y7, leftover of DSO after delay_coll (is SHAP #1 just
delay?), and whether the 44 should lose `e_dso_proxy` the same way
DPO left.

KEEP-as-X: beat size ≥0.02 AND leftover after the honest bar AND not
SIZE (|ρ|≥0.50) AND not a twin (|ρ|≥0.80 vs DPO / delay_coll /
ar_overdue / issued). Leftover <0.55 dies.

No 0–100. No product/. No parquet rewrite. No new GBM. Do not run
`build_targets`. Do not invent `y_dso`. Do not put DSO on the 15-col
Y3 card. Do not add DSO back to TURNOVER. Do not edit invoices.py
unless a real formula bug.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.dso_qa

Owned: analysis/evaluate/dso_qa.py, analysis/outputs/dso_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_dso.md (end).
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
    rolling_origins,
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
OUT_MD = ANALYSIS / "outputs" / "dso_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "dso_leftover.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_dso.md"
AGENT = "e8b2c0d4"
WAVE = "4"
ROUND = "R4"
MODEL = "dso_qa"
X_FAM = "E"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y5 = "y5_ap_od30_ownp80"
Y6Z = "y6_zero_in_3"
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
DSO_F4_QUOTE = 0.342
ISSUED_F4_QUOTE = 0.647
SHORT_Q1_QUOTE = 0.410
DPO_GT24 = 0.090
DPO_P99 = 802.4
DELAY_LEFTOVER_DSO = 0.581
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
    "e_dso_proxy",
    "e_dpo_proxy",
    "e_delay_coll",
    "e_delay_paid",
    "e_ar_overdue",
    "e_ar_overdue_30",
    "e_ap_overdue",
    "e_ar_issued",
    "e_ar_open",
    "e_ap_issued",
    "e_pending_amt_share",
)

Y_KEEP = (Y2, Y3, Y5, Y6Z, Y7)

TWIN_COLS = (
    "e_dpo_proxy",
    "e_delay_coll",
    "e_ar_overdue",
    "e_ar_issued",
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


def reconstruct_dso(con) -> pd.DataFrame:
    """Independent SQL of AR open / this-period AR issued. Diagnosis only."""
    periods = period_frame()
    con.register("_dso_periods", periods[["period", "period_end"]])
    try:
        issued = con.execute(
            """
            SELECT i.company_id,
                   p.period,
                   SUM(CASE WHEN i.document_type = 'invoice' AND i.amount > 0
                            THEN abs(i.amount) ELSE 0 END) AS sql_ar_issued
            FROM invoices i
            JOIN _dso_periods p
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
                   SUM(CASE WHEN i.amount > 0 THEN abs(i.amount) ELSE 0 END) AS sql_ar_open
            FROM invoices i
            JOIN _dso_periods p
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
        con.unregister("_dso_periods")

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

    for part, col in ((issued, "sql_ar_issued"), (open_book, "sql_ar_open")):
        if part.empty:
            grid[col] = np.nan
            continue
        part = part.copy()
        part["company_id"] = part["company_id"].astype(str)
        part["period"] = pd.to_datetime(part["period"])
        grid = grid.merge(part[["company_id", "period", col]], on=["company_id", "period"], how="left")

    active = grid["company_id"].isin(ever_ids)
    for c in ("sql_ar_issued", "sql_ar_open"):
        grid.loc[active, c] = pd.to_numeric(grid[c], errors="coerce")
        grid.loc[active, c] = grid.loc[active, c].fillna(0.0)
    iss = pd.to_numeric(grid["sql_ar_issued"], errors="coerce")
    opn = pd.to_numeric(grid["sql_ar_open"], errors="coerce")
    grid["sql_dso"] = opn / iss.where(iss > 0)
    return grid[["company_id", "period", "sql_ar_issued", "sql_ar_open", "sql_dso"]]


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
    dso = pd.to_numeric(panel["e_dso_proxy"], errors="coerce")
    panel["e_dso_clip24"] = dso.clip(upper=WINSOR)
    panel["e_dso_clip12"] = dso.clip(upper=12.0)
    leak7 = leakage_check(["e_dso_proxy", "e_ar_issued_lag1", "e_delay_coll"], Y7, forbidden_prefixes=["d"])
    leak3 = leakage_check(["e_dso_proxy"], Y3, forbidden_prefixes=["b"])
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
# 1. Coverage / tails / 470 NaN / |DSO|>24 vs DPO 9.0% / p99 802
# ---------------------------------------------------------------------------
def pass1_cov(tr: pd.DataFrame, book: set[str]) -> dict:
    dark = ~tr["company_id"].isin(book)
    erp = tr["company_id"].isin(book)
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    n_erp_co = int(tr.loc[erp, "company_id"].nunique())
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    dpo = pd.to_numeric(tr["e_dpo_proxy"], errors="coerce")
    nn = dso.notna()
    n_cm = int(len(tr))
    n_nn = int(nn.sum())
    dark_nn = int(dso[dark].notna().sum())
    dark_zero = int((dso[dark] == 0).sum())
    early_nn = int(dso[tr["early6"]].notna().sum())
    early_n = int(tr["early6"].sum())
    late_nn = int(dso[~tr["early6"]].notna().sum())
    late_n = int((~tr["early6"]).sum())
    finite = dso[nn]
    p99 = float(finite.quantile(0.99)) if n_nn else float("nan")
    p50 = float(finite.median()) if n_nn else float("nan")
    mx = float(finite.max()) if n_nn else float("nan")
    mn = float(finite.min()) if n_nn else float("nan")
    gt24 = int((finite.abs() > WINSOR).sum())
    gt24_share = _pct(gt24, n_nn)
    dpo_nn = int(dpo.notna().sum())
    dpo_gt24 = _pct(int((dpo[dpo.notna()].abs() > WINSOR).sum()), dpo_nn)
    dpo_p99 = float(dpo[dpo.notna()].quantile(0.99)) if dpo_nn else float("nan")
    acf1 = median_acf(tr["e_dso_proxy"], tr["company_id"], 1)
    acf3 = median_acf(tr["e_dso_proxy"], tr["company_id"], 3)
    rows = [
        {
            "col": "e_dso_proxy",
            "nn": f"{n_nn:,}",
            "cov": _pp(_pct(n_nn, n_cm)),
            "early6 nn": _pp(_pct(early_nn, early_n)),
            "after nn": _pp(_pct(late_nn, late_n)),
            "dark nn / 0": f"{dark_nn} / {dark_zero}",
            "ERP nn": _pp(_pct(int(dso[erp].notna().sum()), int(erp.sum()))),
            "p50": _f(p50, 2),
            "p99": _f(p99, 1),
            "max": _f(mx, 1),
            "|DSO|>24": _pp(gt24_share),
            "acf1": _f(acf1),
        },
        {
            "col": "e_dpo_proxy (locked)",
            "nn": f"{dpo_nn:,}",
            "cov": _pp(_pct(dpo_nn, n_cm)),
            "early6 nn": "—",
            "after nn": "—",
            "dark nn / 0": "—",
            "ERP nn": "—",
            "p50": _f(float(dpo[dpo.notna()].median()) if dpo_nn else float("nan"), 2),
            "p99": _f(dpo_p99, 1),
            "max": _f(float(dpo[dpo.notna()].max()) if dpo_nn else float("nan"), 1),
            "|DSO|>24": _pp(dpo_gt24),
            "acf1": "0.244",
        },
    ]
    dark_nan = dark_nn == 0 and n_dark_co == 470
    low_persist = bool(np.isfinite(acf1) and abs(acf1) < LOW_PERSIST)
    vs_dpo = (
        f"|DSO|>24 {_pp(gt24_share)} vs locked DPO 9.0% "
        f"(this-run DPO {_pp(dpo_gt24)}); DSO p99={_f(p99, 1)} vs DPO p99 802 "
        f"(this-run {_f(dpo_p99, 1)})."
    )
    prose = (
        f"Train DSO nn={n_nn:,} cov={_pp(_pct(n_nn, n_cm))} "
        f"(feature report 40.6%). Dark never-ERP {n_dark_co} (want 470): "
        f"nn={dark_nn} zero={dark_zero} "
        f"({'CONFIRM NaN not 0' if dark_nan else 'MISMATCH dark'}). "
        f"Ever-ERP {n_erp_co}. Early6 finite {_pp(_pct(early_nn, early_n))} "
        f"(delay was 0.0% — DSO is a stock/flow, may exist earlier like DPO). "
        f"Tails p50={_f(p50, 2)} p99={_f(p99, 1)} max={_f(mx, 1)} "
        f"|DSO|>24 {_pp(gt24_share)} of finite ({gt24:,}). "
        f"{vs_dpo} acf1={_f(acf1)} "
        f"({'CONFIRM LOW_PERSIST' if low_persist else 'not LOW_PERSIST / borderline'})."
    )
    print(prose)
    y7 = pd.to_numeric(tr[Y7], errors="coerce")
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    print(
        f"Train CM {n_cm:,} / companies {tr['company_id'].nunique()}. "
        f"Y7 labeled {int(y7.notna().sum()):,} pos {int((y7==1).sum()):,}; "
        f"Y3 {int(y3.notna().sum()):,} pos {int((y3==1).sum()):,}."
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
        "dpo_gt24": dpo_gt24,
        "dpo_p99": dpo_p99,
        "acf1": acf1,
        "acf3": acf3,
        "low_persist": low_persist,
        "y7_n": int(y7.notna().sum()),
        "y7_pos": int((y7 == 1).sum()),
        "y3_n": int(y3.notna().sum()),
        "y3_pos": int((y3 == 1).sum()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 2. Store vs raw SQL
# ---------------------------------------------------------------------------
def pass2_sql(tr: pd.DataFrame, recon: pd.DataFrame) -> dict:
    m = tr.merge(recon, on=["company_id", "period"], how="left")
    pairs = [
        ("e_dso_proxy", "sql_dso"),
        ("e_ar_open", "sql_ar_open"),
        ("e_ar_issued", "sql_ar_issued"),
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
    dso_same = store["e_dso_proxy"]["same"]
    prose = (
        f"Store vs raw SQL DSO ρ={_f(store['e_dso_proxy']['rho'])} "
        f"max|Δ|={_f(store['e_dso_proxy']['maxabs'], 8)} "
        f"n_off={store['e_dso_proxy']['n_off']} "
        f"({'CONFIRM SAME' if dso_same else 'DISAGREE — document, do not rewrite invoices.py'})."
    )
    print(prose)
    return {"rows": rows, "store": store, "same": dso_same, "prose": prose}


# ---------------------------------------------------------------------------
# 3. Spearman twins (|ρ|≥0.80 vs DPO / delay_coll / ar_overdue / issued)
# ---------------------------------------------------------------------------
def pass3_twins(tr: pd.DataFrame) -> dict:
    dso = tr["e_dso_proxy"]
    pairs = [
        ("e_dpo_proxy", tr["e_dpo_proxy"]),
        ("e_delay_coll", tr["e_delay_coll"]),
        ("e_ar_overdue", tr["e_ar_overdue"]),
        ("e_ar_overdue_30", tr["e_ar_overdue_30"]),
        ("e_ar_issued", tr["e_ar_issued"]),
        ("e_ar_issued_lag1", tr["e_ar_issued_lag1"]),
        ("e_ar_open", tr["e_ar_open"]),
        ("e_pending_amt_share", tr["e_pending_amt_share"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("e_delay_paid", tr["e_delay_paid"]),
        ("e_ap_overdue", tr["e_ap_overdue"]),
        ("c_zero_in_month", tr["c_zero_in_month"]),
        ("f_ds_r", tr["f_ds_r"]),
    ]
    rows = []
    rhos = {}
    twins = []
    for name, s in pairs:
        rho, n = spearman_n(dso, s)
        twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
        rhos[name] = rho
        rhos[f"{name}_n"] = n
        if twin:
            twins.append(name)
        rows.append(
            {
                "pair": f"e_dso_proxy vs {name}",
                "ρ": _f(rho),
                "n": f"{n:,}",
                "twin?": "TWIN" if twin else "",
            }
        )
    size_flag = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    gate_twins = [t for t in twins if t in TWIN_COLS or t == "e_ar_issued_lag1"]
    prose = (
        f"DSO vs DPO ρ={_f(rhos['e_dpo_proxy'])} "
        f"({'TWIN' if 'e_dpo_proxy' in twins else 'not a twin — locked 0.456'}). "
        f"vs delay_coll {_f(rhos['e_delay_coll'])} "
        f"({'TWIN' if 'e_delay_coll' in twins else 'not a twin — delay locked 0.214'}). "
        f"vs ar_overdue {_f(rhos['e_ar_overdue'])} vs ar_overdue_30 {_f(rhos['e_ar_overdue_30'])} "
        f"vs issued {_f(rhos['e_ar_issued'])} vs issued_lag1 {_f(rhos['e_ar_issued_lag1'])} "
        f"vs days {_f(rhos['c_n_days_with_tx'])} vs size {_f(rhos['log1p(a_in3)'])} "
        f"({'SIZE' if size_flag else 'not SIZE'}). "
        f"TWIN |ρ|≥0.80: {', '.join(twins) if twins else 'none'}."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "twins": twins,
        "gate_twins": gate_twins,
        "size_flag": size_flag,
        "dpo": rhos["e_dpo_proxy"],
        "delay": rhos["e_delay_coll"],
        "od": rhos["e_ar_overdue"],
        "od30": rhos["e_ar_overdue_30"],
        "iss": rhos["e_ar_issued"],
        "iss1": rhos["e_ar_issued_lag1"],
        "open": rhos["e_ar_open"],
        "pend": rhos["e_pending_amt_share"],
        "size": rhos["log1p(a_in3)"],
        "days": rhos["c_n_days_with_tx"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 4. Single-feature group-fold AUROC + short-DSO quintile
# ---------------------------------------------------------------------------
def pass4_auroc(tr: pd.DataFrame) -> dict:
    feats = {
        "e_dso_proxy": tr["e_dso_proxy"],
        "e_dso_clip24": tr["e_dso_clip24"],
        "e_dso_clip12": tr["e_dso_clip12"],
        "e_dpo_proxy": tr["e_dpo_proxy"],
        "e_delay_coll": tr["e_delay_coll"],
        "e_ar_overdue": tr["e_ar_overdue"],
        "e_ar_issued": tr["e_ar_issued"],
        "e_ar_issued_lag1": tr["e_ar_issued_lag1"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "log1p_a_in3": tr["log_in3"],
    }
    ys = (Y3, Y7)
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
    dso_y3 = _cv(store[(Y3, "e_dso_proxy")])
    dso_y7 = _cv(store[(Y7, "e_dso_proxy")])
    days_y3 = _cv(store[(Y3, "c_n_days_with_tx")])
    size_y3 = _cv(store[(Y3, "log1p_a_in3")])
    size_y7 = _cv(store[(Y7, "log1p_a_in3")])
    iss_y7 = _cv(store[(Y7, "e_ar_issued_lag1")])
    delay_y7 = _cv(store[(Y7, "e_delay_coll")])
    dpo_y7 = _cv(store[(Y7, "e_dpo_proxy")])
    days_ok = bool(np.isfinite(days_y3) and abs(days_y3 - DAYS_BENCH) < 0.005)
    iss_ok = bool(np.isfinite(iss_y7) and abs(iss_y7 - ISSUED_LAG1_BENCH) < 0.005)
    size_ok = bool(np.isfinite(size_y3) and abs(size_y3 - SIZE_QUOTE) < 0.005)
    prose = (
        f"Y3 DSO {_f(dso_y3)} vs days {_f(days_y3)} "
        f"({'CONFIRM 0.711' if days_ok else 'days drifted'}) vs size {_f(size_y3)} "
        f"({'CONFIRM 0.617' if size_ok else 'size drifted'}). "
        f"Y7 DSO {_f(dso_y7)} vs issued_lag1 {_f(iss_y7)} "
        f"({'CONFIRM 0.630' if iss_ok else 'issued drifted'}) vs delay_coll {_f(delay_y7)} "
        f"vs DPO {_f(dpo_y7)} vs size {_f(size_y7)}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "dso_y3": dso_y3,
        "dso_y7": dso_y7,
        "days_y3": days_y3,
        "size_y3": size_y3,
        "size_y7": size_y7,
        "iss_y7": iss_y7,
        "delay_y7": delay_y7,
        "dpo_y7": dpo_y7,
        "clip_y3": _cv(store[(Y3, "e_dso_clip24")]),
        "clip_y7": _cv(store[(Y7, "e_dso_clip24")]),
        "clip12_y3": _cv(store[(Y3, "e_dso_clip12")]),
        "clip12_y7": _cv(store[(Y7, "e_dso_clip12")]),
        "days_ok": days_ok,
        "iss_ok": iss_ok,
        "size_ok": size_ok,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 5. Honest leftover: Y3 after days; Y7 after issued_lag1; after delay_coll
# ---------------------------------------------------------------------------
def _leftover_block(tr: pd.DataFrame, ycol: str, controls: list[tuple[str, tuple]]) -> dict:
    lab = tr[ycol].notna()
    stems = {
        "e_dso_proxy": tr["e_dso_proxy"],
        "e_dso_clip24": tr["e_dso_clip24"],
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
            ("after issued_lag1", (tr["e_ar_issued_lag1"],)),
            ("after delay_coll", (tr["e_delay_coll"],)),
            ("after issued_lag1+delay", (tr["e_ar_issued_lag1"], tr["e_delay_coll"])),
            ("after days", (tr["c_n_days_with_tx"],)),
            ("after size", (tr["log_in3"],)),
            ("after DPO", (tr["e_dpo_proxy"],)),
            ("after ar_overdue", (tr["e_ar_overdue"],)),
            ("after issued", (tr["e_ar_issued"],)),
        ],
    )
    y3 = _leftover_block(
        tr,
        Y3,
        [
            ("after days", (tr["c_n_days_with_tx"],)),
            ("after size", (tr["log_in3"],)),
            ("after days+size", (tr["c_n_days_with_tx"], tr["log_in3"])),
            ("after issued_lag1", (tr["e_ar_issued_lag1"],)),
            ("after delay_coll", (tr["e_delay_coll"],)),
        ],
    )
    after_iss = _cv(y7["store"]["e_dso_proxy after issued_lag1"])
    after_delay = _cv(y7["store"]["e_dso_proxy after delay_coll"])
    after_both = _cv(y7["store"]["e_dso_proxy after issued_lag1+delay"])
    after_days = _cv(y3["store"]["e_dso_proxy after days"])
    after_size = _cv(y3["store"]["e_dso_proxy after size"])
    clip_iss = _cv(y7["store"]["e_dso_clip24 after issued_lag1"])
    clip_delay = _cv(y7["store"]["e_dso_clip24 after delay_coll"])
    clip_days = _cv(y3["store"]["e_dso_clip24 after days"])
    y7_lives = bool(np.isfinite(after_iss) and after_iss >= CHANCE)
    y3_lives = bool(np.isfinite(after_days) and after_days >= CHANCE)
    delay_is_shap = bool(np.isfinite(after_delay) and after_delay < CHANCE)
    rows = y7["rows"] + y3["rows"]
    prose = (
        f"Y3 leftover after days {_f(after_days)} "
        f"({'lives' if y3_lives else 'dies <0.55'}). "
        f"Y7 leftover after issued_lag1 {_f(after_iss)} "
        f"({'lives' if y7_lives else 'dies'}). "
        f"Y7 leftover after delay_coll {_f(after_delay)} / both {_f(after_both)} "
        f"({'SHAP #1 is delay — leftover dies' if delay_is_shap else 'SHAP #1 is not just delay'}). "
        f"Clip24 leftover Y3-days {_f(clip_days)} Y7-iss {_f(clip_iss)} Y7-delay {_f(clip_delay)}."
    )
    print(prose)
    return {
        "rows": rows,
        "y7": y7,
        "y3": y3,
        "after_iss": after_iss,
        "after_delay": after_delay,
        "after_both": after_both,
        "after_days": after_days,
        "after_size": after_size,
        "clip_iss": clip_iss,
        "clip_delay": clip_delay,
        "clip_days": clip_days,
        "y7_lives": y7_lives,
        "y3_lives": y3_lives,
        "delay_is_shap": delay_is_shap,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 6. |DSO|>24 drop-tail leftover (DPO's 0.653 was a tail)
# ---------------------------------------------------------------------------
def pass6_tail(tr: pd.DataFrame) -> dict:
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    pos = iss[iss > 0]
    p10 = float(pos.quantile(0.10)) if len(pos) else float("nan")
    drop24 = dso.where(dso.abs() <= WINSOR)
    clip12 = dso.clip(upper=12.0)
    log_iss = np.log1p(iss.clip(lower=0))
    rows = []
    store = {}

    def _one(ycol, name, x, mask=None):
        lab = tr[ycol].notna() if mask is None else (tr[ycol].notna() & mask)
        res = signed_oof_auroc(tr[ycol], x, tr["fold"], lab)
        store[name] = res
        rows.append(_auc_row(ycol, name, res, None))
        return res

    resid_days, _ = ols_resid(dso, days)
    resid_drop, _ = ols_resid(drop24, days)
    resid_c12, _ = ols_resid(clip12, days)
    resid_iss, _ = ols_resid(dso, log_iss)
    resid_days_iss, _ = ols_resid(dso, days, log_iss)
    resid_y7_iss, _ = ols_resid(dso, tr["e_ar_issued_lag1"])
    resid_y7_drop, _ = ols_resid(drop24, tr["e_ar_issued_lag1"])

    _one(Y3, "leftover days (raw)", resid_days)
    _one(Y3, "leftover days drop |DSO|>24", resid_drop)
    _one(Y3, "leftover days clip12", resid_c12)
    _one(Y3, "leftover days clip24", ols_resid(tr["e_dso_clip24"], days)[0])
    _one(Y3, "leftover after log1p(ar_issued)", resid_iss)
    _one(Y3, "leftover days+issued", resid_days_iss)
    _one(Y3, "DSO drop |x|>24 raw", drop24)
    _one(Y3, "has_|DSO|>24", (dso.abs() > WINSOR).astype(float))
    _one(Y3, "has_tiny_issued", ((iss > 0) & (iss <= p10)).astype(float))
    _one(Y3, "has_dso flag", dso.notna().astype(float))
    _one(Y7, "leftover issued_lag1 (raw)", resid_y7_iss)
    _one(Y7, "leftover issued_lag1 drop |DSO|>24", resid_y7_drop)

    after_drop = _cv(store["leftover days drop |DSO|>24"])
    after_raw = _cv(store["leftover days (raw)"])
    after_c12 = _cv(store["leftover days clip12"])
    y7_drop = _cv(store["leftover issued_lag1 drop |DSO|>24"])
    tail_artifact = bool(
        np.isfinite(after_raw)
        and after_raw >= CHANCE
        and np.isfinite(after_drop)
        and after_drop < CHANCE
    )
    prose = (
        f"Y3 leftover after days on drop>|24| {_f(after_drop)} "
        f"(raw leftover {_f(after_raw)}) clip12 {_f(after_c12)} "
        f"after log1p(issued) {_f(_cv(store['leftover after log1p(ar_issued)']))}. "
        f"Y7 leftover-iss drop>|24| {_f(y7_drop)}. "
        f"{'Y3 leftover is a |DSO|>24 tail (DPO pattern)' if tail_artifact else 'Y3 leftover is not only the |DSO|>24 tail'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "after_raw": after_raw,
        "after_drop": after_drop,
        "after_c12": after_c12,
        "after_iss": _cv(store["leftover after log1p(ar_issued)"]),
        "y7_drop": y7_drop,
        "tail_artifact": tail_artifact,
        "p10": p10,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 7. SIZE terciles + invoice-book-only (drop 470)
# ---------------------------------------------------------------------------
def pass7_slices(tr: pd.DataFrame, book: set[str]) -> dict:
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
            res = signed_oof_auroc(tr[ycol], tr["e_dso_proxy"], tr["fold"], lab)
            store[(label, ycol)] = res
            rows.append(
                _auc_row(
                    ycol,
                    f"DSO {label}",
                    res,
                    _pct(res["n_defined"], int(lab.sum())) if lab.any() else float("nan"),
                )
            )
            if ycol == Y7 and label in ("all", "book_only", "T1", "T3"):
                resid, _ = ols_resid(tr["e_dso_proxy"], tr["e_ar_issued_lag1"])
                rres = signed_oof_auroc(tr[ycol], resid, tr["fold"], lab)
                store[(f"{label}|iss", ycol)] = rres
                rows.append(_auc_row(ycol, f"DSO leftover-iss {label}", rres, None))
            if ycol == Y3 and label in ("all", "book_only", "T1", "T3"):
                resid, _ = ols_resid(tr["e_dso_proxy"], tr["c_n_days_with_tx"])
                rres = signed_oof_auroc(tr[ycol], resid, tr["fold"], lab)
                store[(f"{label}|days", ycol)] = rres
                rows.append(_auc_row(ycol, f"DSO leftover-days {label}", rres, None))
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
# 8. Q6 lag1/lag3; empty-on-short; exists early like DPO?
# ---------------------------------------------------------------------------
def pass8_q6(tr: pd.DataFrame) -> dict:
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
        "e_dso_proxy",
        "e_dso_proxy_lag1",
        "e_dso_proxy_lag3",
        "e_ar_issued_lag1",
        "e_delay_coll",
        "e_dpo_proxy",
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
    early_res = store[("early6_calendar", "e_dso_proxy")]
    early_n_lab = int((tr["early6"] & tr[Y7].notna()).sum())
    early_cov = _pct(early_res["n_defined"], early_n_lab) if early_n_lab else float("nan")
    short_lag = _cv(store[("short_<12_sofar", "e_dso_proxy_lag1")])
    all_lag1 = _cv(store[("all", "e_dso_proxy_lag1")])
    all_lag3 = _cv(store[("all", "e_dso_proxy_lag3")])
    empty_early = bool(np.isfinite(early_cov) and early_cov < 0.05)
    exists_early = bool(np.isfinite(early_cov) and early_cov >= 0.50)
    delay_early = store[("early6_calendar", "e_delay_coll")]
    delay_early_empty = delay_early["n_defined"] == 0
    prose = (
        f"Q6 Y7 DSO now {_f(_cv(store[('all', 'e_dso_proxy')]))} "
        f"lag1 {_f(all_lag1)} lag3 {_f(all_lag3)} short lag1 {_f(short_lag)}. "
        f"Early6 calendar DSO finite share {_pp(early_cov)} "
        f"({'exists earlier than delay' if exists_early else 'empty-on-early'}). "
        f"Delay early6 nn={delay_early['n_defined']} "
        f"({'CONFIRM empty until month 7' if delay_early_empty else 'delay not empty'})."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "early_cov": early_cov,
        "short_lag": short_lag,
        "all_lag1": all_lag1,
        "all_lag3": all_lag3,
        "empty_early": empty_early,
        "exists_early": exists_early,
        "delay_early_empty": delay_early_empty,
        "all_now": _cv(store[("all", "e_dso_proxy")]),
        "short_now": _cv(store[("short_<12_sofar", "e_dso_proxy")]),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 9. ICC / company-demean (DPO LOW_PERSIST acf1 0.244 ICC 0.807)
# ---------------------------------------------------------------------------
def pass9_icc(tr: pd.DataFrame) -> dict:
    icc = icc_anova(tr["e_dso_proxy"], tr["company_id"])
    icc_dpo = icc_anova(tr["e_dpo_proxy"], tr["company_id"])
    de = company_demean(tr["e_dso_proxy"], tr["company_id"])
    mu = company_mean(tr["e_dso_proxy"], tr["company_id"])
    rows = []
    store = {}
    for ycol in (Y7, Y3):
        lab = tr[ycol].notna()
        for name, s in (("raw", tr["e_dso_proxy"]), ("demean", de), ("company-mean", mu)):
            res = signed_oof_auroc(tr[ycol], s, tr["fold"], lab)
            store[(ycol, name)] = res
            rows.append(_auc_row(ycol, f"e_dso_proxy {name}", res, None))
    style = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_STYLE)
    y7_de = _cv(store[(Y7, "demean")])
    y7_mu = _cv(store[(Y7, "company-mean")])
    y7_raw = _cv(store[(Y7, "raw")])
    acf1 = median_acf(tr["e_dso_proxy"], tr["company_id"], 1)
    prose = (
        f"DSO ICC={_f(icc['icc'])} k={icc['k']} (DPO ICC={_f(icc_dpo['icc'])} locked 0.807). "
        f"Y7 raw {_f(y7_raw)} demean {_f(y7_de)} company-mean {_f(y7_mu)}. "
        f"acf1={_f(acf1)} "
        f"({'CONFIRM LOW_PERSIST month shock' if np.isfinite(acf1) and abs(acf1) < LOW_PERSIST else 'trait-ish / borderline'}). "
        f"{'BETWEEN style' if style else 'not BETWEEN (≥0.85)'} — "
        f"{'company-mean carries skill' if np.isfinite(y7_mu) and y7_mu > (y7_de if np.isfinite(y7_de) else 0) else 'demean / month shock carries'}."
    )
    print(prose)
    return {
        "rows": rows,
        "icc": icc["icc"],
        "icc_k": icc["k"],
        "icc_dpo": icc_dpo["icc"],
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
# 10. Fold 4 / short-DSO quintile — issued owns fold 4 (0.647 vs DSO 0.342)
# ---------------------------------------------------------------------------
def pass10_fold4(tr: pd.DataFrame, p4: dict) -> dict:
    lab = tr[Y7].notna()
    feats = {
        "e_dso_proxy": tr["e_dso_proxy"],
        "e_dso_clip24": tr["e_dso_clip24"],
        "e_ar_issued_lag1": tr["e_ar_issued_lag1"],
        "e_delay_coll": tr["e_delay_coll"],
        "e_dpo_proxy": tr["e_dpo_proxy"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
    }
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
            ("e_dso_proxy", tr["e_dso_proxy"]),
            ("e_ar_issued_lag1", tr["e_ar_issued_lag1"]),
            ("e_delay_coll", tr["e_delay_coll"]),
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
                "delay nn": _pp(
                    float(pd.to_numeric(part["e_delay_coll"], errors="coerce").notna().mean())
                ),
                "DSO CV": "LOW_POWER"
                if qstore[(str(q), "e_dso_proxy")]["low_power"]
                else _f(qstore[(str(q), "e_dso_proxy")]["cv"]),
                "issued CV": "LOW_POWER"
                if qstore[(str(q), "e_ar_issued_lag1")]["low_power"]
                else _f(qstore[(str(q), "e_ar_issued_lag1")]["cv"]),
                "delay CV": "LOW_POWER"
                if qstore[(str(q), "e_delay_coll")]["low_power"]
                else _f(qstore[(str(q), "e_delay_coll")]["cv"]),
            }
        )
    q1_dso = _cv(qstore[("Q1", "e_dso_proxy")])
    q1_iss = _cv(qstore[("Q1", "e_ar_issued_lag1")])
    q1_n = int((sl["dso_q"] == "Q1").sum())
    dso4 = fold_k(store["e_dso_proxy"], 4)
    iss4 = fold_k(store["e_ar_issued_lag1"], 4)
    issued_owns = bool(np.isfinite(iss4) and np.isfinite(dso4) and iss4 - dso4 >= 0.20)
    q1_hole = bool(np.isfinite(q1_dso) and q1_dso < CHANCE)
    q1_near_410 = bool(np.isfinite(q1_dso) and abs(q1_dso - SHORT_Q1_QUOTE) < 0.08)
    prose = (
        f"Fold 4 Y7 DSO {_f(dso4)} (quote 0.342) vs issued_lag1 {_f(iss4)} "
        f"(quote 0.647) vs TURNOVER 0.680. "
        f"{'CONFIRM issued owns fold 4' if issued_owns else 'fold 4 not issued-owned'}. "
        f"Short-DSO Q1 n={q1_n} (B_shallow card 1331 / OOF 0.410): "
        f"univariate DSO {_f(q1_dso)} issued {_f(q1_iss)}. "
        f"{'CONFIRM short-DSO hole' if q1_hole else 'Q1 not a hole'}"
        f"{' (near locked 0.410)' if q1_near_410 else ''}."
    )
    print(prose)
    return {
        "rows": rows,
        "qrows": qrows,
        "dso4": dso4,
        "iss4": iss4,
        "delay4": fold_k(store["e_delay_coll"], 4),
        "q1_dso": q1_dso,
        "q1_iss": q1_iss,
        "q1_n": q1_n,
        "issued_owns": issued_owns,
        "q1_hole": q1_hole,
        "q1_near_410": q1_near_410,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 11. leftover of DSO after delay_coll (is SHAP #1 just delay?)
#     plus delay leftover KEEP confirm (do not reopen delay ticket)
# ---------------------------------------------------------------------------
def pass11_delay(tr: pd.DataFrame, p5: dict) -> dict:
    lab = tr[Y7].notna()
    resid_dso_after_delay, info_d = ols_resid(tr["e_dso_proxy"], tr["e_delay_coll"])
    resid_delay_after_dso, info_r = ols_resid(tr["e_delay_coll"], tr["e_dso_proxy"])
    ok = lab & tr["e_dso_proxy"].notna() & tr["e_delay_coll"].notna()
    rows = []
    store = {}
    for name, s in (
        ("DSO same-n delay", tr["e_dso_proxy"]),
        ("delay same-n DSO", tr["e_delay_coll"]),
        ("DSO leftover after delay", resid_dso_after_delay),
        ("delay leftover after DSO", resid_delay_after_dso),
        ("issued_lag1 same-n", tr["e_ar_issued_lag1"]),
    ):
        res = signed_oof_auroc(tr[Y7], s, tr["fold"], ok)
        store[name] = res
        rows.append(_auc_row(Y7, name, res, None))
    dso_after = _cv(store["DSO leftover after delay"])
    delay_after = _cv(store["delay leftover after DSO"])
    dso_raw = _cv(store["DSO same-n delay"])
    delay_raw = _cv(store["delay same-n DSO"])
    shap_is_delay = bool(np.isfinite(dso_after) and dso_after < CHANCE)
    delay_keep = bool(np.isfinite(delay_after) and delay_after >= CHANCE)
    prose = (
        f"Same-n DSO+delay: raw DSO {_f(dso_raw)} leftover-after-delay {_f(dso_after)} "
        f"R²={_f(info_d['r2'])}. Raw delay {_f(delay_raw)} leftover-after-DSO {_f(delay_after)} "
        f"(locked KEEP 0.581, R²={_f(info_r['r2'])}). "
        f"{'SHAP #1 is just delay — DSO leftover dies' if shap_is_delay else 'SHAP #1 is not just delay (leftover of DSO after delay lives or DSO itself is already <0.55)'}. "
        f"Delay leftover after DSO {'CONFIRM KEEP' if delay_keep else 'drifted from 0.581'}."
    )
    print(prose)
    return {
        "rows": rows,
        "dso_after": dso_after,
        "delay_after": delay_after,
        "dso_raw": dso_raw,
        "delay_raw": delay_raw,
        "r2_dso": info_d["r2"],
        "r2_delay": info_r["r2"],
        "shap_is_delay": shap_is_delay,
        "delay_keep": delay_keep,
        "iss_samen": _cv(store["issued_lag1 same-n"]),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra: holdout coverage only (no AUROC)
# ---------------------------------------------------------------------------
def pass_holdout(panel: pd.DataFrame, book: set[str]) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    assert set(ho["company_id"]) <= load_holdout()
    dso = pd.to_numeric(ho["e_dso_proxy"], errors="coerce")
    dark = ~ho["company_id"].isin(book)
    y7 = pd.to_numeric(ho[Y7], errors="coerce")
    rows = [
        {
            "col": "e_dso_proxy",
            "n_cm": f"{len(ho):,}",
            "nn": f"{int(dso.notna().sum()):,}",
            "cov": _pp(_pct(int(dso.notna().sum()), len(ho))),
            "early6 nn": _pp(_pct(int(dso[ho["early6"]].notna().sum()), int(ho["early6"].sum()))),
            "dark nn": int(dso[dark].notna().sum()),
            "|DSO|>24": _pp(_pct(int((dso.abs() > WINSOR).sum()), int(dso.notna().sum()))),
        }
    ]
    prose = (
        f"Holdout 72 coverage only: {len(ho):,} CM / {ho['company_id'].nunique()} companies. "
        f"Y7 pos={int((y7==1).sum())} (quote 122). DSO cov "
        f"{_pp(_pct(int(dso.notna().sum()), len(ho)))}; early6 "
        f"{_pp(_pct(int(dso[ho['early6']].notna().sum()), int(ho['early6'].sum())))}; "
        f"dark nn={int(dso[dark].notna().sum())}. No AUROC claim."
    )
    print(prose)
    return {
        "rows": rows,
        "n_cm": int(len(ho)),
        "n_co": int(ho["company_id"].nunique()),
        "nn": int(dso.notna().sum()),
        "cov": _pct(int(dso.notna().sum()), len(ho)),
        "y7_pos": int((y7 == 1).sum()),
        "dark_nn": int(dso[dark].notna().sum()),
        "prose": prose,
    }


def pass_samen(tr: pd.DataFrame) -> dict:
    """Same-n raw vs leftover after issued_lag1."""
    lab = tr[Y7].notna()
    ok = lab & tr["e_dso_proxy"].notna() & tr["e_ar_issued_lag1"].notna()
    resid, info = ols_resid(tr["e_dso_proxy"], tr["e_ar_issued_lag1"])
    rows = []
    store = {}
    for name, s in (
        ("DSO same-n", tr["e_dso_proxy"]),
        ("issued_lag1 same-n", tr["e_ar_issued_lag1"]),
        ("leftover iss same-n", resid),
        ("clip24 same-n", tr["e_dso_clip24"]),
        ("delay same-n", tr["e_delay_coll"]),
    ):
        res = signed_oof_auroc(tr[Y7], s, tr["fold"], ok)
        store[name] = res
        rows.append(_auc_row(Y7, name, res, None))
    raw = _cv(store["DSO same-n"])
    leftover = _cv(store["leftover iss same-n"])
    prose = (
        f"Same-n (DSO+issued_lag1 finite): raw DSO {_f(raw)} leftover {_f(leftover)} "
        f"issued {_f(_cv(store['issued_lag1 same-n']))} "
        f"R²={_f(info['r2'])}. "
        f"{'Leftover ≈ raw — residualizing does not create a new object' if np.isfinite(raw) and np.isfinite(leftover) and abs(raw - leftover) < 0.02 else 'leftover differs from raw'}."
    )
    print(prose)
    return {
        "rows": rows,
        "raw": raw,
        "leftover": leftover,
        "iss": _cv(store["issued_lag1 same-n"]),
        "r2": info["r2"],
        "prose": prose,
    }


def pass_quintiles(tr: pd.DataFrame) -> dict:
    rows = []
    for ycol in (Y3, Y7):
        sl = tr.loc[tr[ycol].notna() & tr["e_dso_proxy"].notna()].copy()
        if sl.empty:
            continue
        x = pd.to_numeric(sl["e_dso_proxy"], errors="coerce")
        sl["q"] = pd.qcut(x.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        for q, part in sl.groupby("q", observed=False):
            y = pd.to_numeric(part[ycol], errors="coerce")
            iss = pd.to_numeric(part["e_ar_issued"], errors="coerce")
            rows.append(
                {
                    "y": ycol,
                    "DSO q": str(q),
                    "n": int(len(part)),
                    "n_pos": int((y == 1).sum()),
                    "rate": _pp(float(y.mean()) if y.notna().any() else float("nan")),
                    "DSO p50": _f(float(x.loc[part.index].median()), 2),
                    "issued p50": _f(float(iss.median()), 0),
                    "share>24": _pp(float((x.loc[part.index].abs() > WINSOR).mean())),
                }
            )
    rates_y3 = [r for r in rows if r["y"] == Y3]
    tail = False
    if len(rates_y3) == 5:
        rs = [float(r["rate"].rstrip("%")) for r in rates_y3]
        tail = rs[4] >= max(rs[:4]) + 3.0
    prose = (
        f"DSO quintiles vs Y3/Y7. "
        f"{'Q5 tail on Y3 (high-DSO / thin issued)' if tail else 'no clear Q5 tail on Y3'}."
    )
    print(prose)
    return {"rows": rows, "tail": tail, "prose": prose}


def pass_tiny(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    pos = iss[iss > 0]
    p10 = float(pos.quantile(0.10)) if len(pos) else float("nan")
    tiny = (iss > 0) & (iss <= p10)
    fat = iss > p10
    rows = []
    for name, mask in (
        ("tiny_issued p10", tiny),
        ("rest issued>p10", fat),
        ("issued=0 / NaN DSO", iss.eq(0) | iss.isna()),
    ):
        x = dso[mask]
        rows.append(
            {
                "slice": name,
                "n": int(mask.sum()),
                "DSO nn": int(x.notna().sum()),
                "DSO p50": _f(float(x.median()) if x.notna().any() else float("nan"), 2),
                "DSO p99": _f(float(x.quantile(0.99)) if x.notna().sum() > 10 else float("nan"), 1),
                "|DSO|>24": _pp(_pct(int((x.abs() > WINSOR).sum()), int(x.notna().sum()))),
                "issued p50": _f(float(iss[mask].median()) if mask.any() else float("nan"), 0),
            }
        )
    lab = tr[Y7].notna()
    tiny_res = signed_oof_auroc(tr[Y7], tr["e_dso_proxy"], tr["fold"], lab & tiny)
    fat_res = signed_oof_auroc(tr[Y7], tr["e_dso_proxy"], tr["fold"], lab & fat)
    prose = (
        f"Tiny-issued p10={_f(p10, 0)}: DSO p50 on tiny "
        f"{_f(float(dso[tiny].median()) if dso[tiny].notna().any() else float('nan'), 2)} "
        f"share>24 {_pp(_pct(int((dso[tiny].abs() > WINSOR).sum()), int(dso[tiny].notna().sum())))}. "
        f"Y7 DSO on tiny {'LOW_POWER' if tiny_res['low_power'] else _f(tiny_res['cv'])} "
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


def pass_y3_samen(tr: pd.DataFrame) -> dict:
    lab = tr[Y3].notna()
    ok = lab & tr["e_dso_proxy"].notna() & tr["c_n_days_with_tx"].notna()
    resid, info = ols_resid(tr["e_dso_proxy"], tr["c_n_days_with_tx"])
    rows = []
    store = {}
    for name, s in (
        ("DSO same-n", tr["e_dso_proxy"]),
        ("days same-n", tr["c_n_days_with_tx"]),
        ("issued_lag1 same-n", tr["e_ar_issued_lag1"]),
        ("leftover days same-n", resid),
        ("clip24 same-n", tr["e_dso_clip24"]),
        ("size same-n", tr["log_in3"]),
    ):
        res = signed_oof_auroc(tr[Y3], s, tr["fold"], ok)
        store[name] = res
        rows.append(_auc_row(Y3, name, res, None))
    prose = (
        f"Y3 same-n raw {_f(_cv(store['DSO same-n']))} leftover-days "
        f"{_f(_cv(store['leftover days same-n']))} days {_f(_cv(store['days same-n']))} "
        f"issued {_f(_cv(store['issued_lag1 same-n']))} R²={_f(info['r2'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "raw": _cv(store["DSO same-n"]),
        "leftover": _cv(store["leftover days same-n"]),
        "days": _cv(store["days same-n"]),
        "r2": info["r2"],
        "prose": prose,
    }


def pass_fold4_groups(tr: pd.DataFrame) -> dict:
    lab = tr[Y7].notna()
    problem = tr["group_id"].isin(["GROUP_0222", "GROUP_0108"])
    rows = []
    for name, mask in (("fold4 problem groups", problem & lab), ("rest labeled", (~problem) & lab)):
        sl = tr.loc[mask]
        dso = pd.to_numeric(sl["e_dso_proxy"], errors="coerce")
        iss = pd.to_numeric(sl["e_ar_issued_lag1"], errors="coerce")
        y = pd.to_numeric(sl[Y7], errors="coerce")
        rows.append(
            {
                "slice": name,
                "n_lab": int(mask.sum()),
                "n_pos": int((y == 1).sum()),
                "rate": _pp(float(y.mean()) if y.notna().any() else float("nan")),
                "DSO nn": _pp(_pct(int(dso.notna().sum()), int(mask.sum()))),
                "DSO p50": _f(float(dso.median()) if dso.notna().any() else float("nan"), 2),
                "issued p50": _f(float(iss.median()) if iss.notna().any() else float("nan"), 0),
            }
        )
    res_p = signed_oof_auroc(tr[Y7], tr["e_dso_proxy"], tr["fold"], problem & lab)
    res_r = signed_oof_auroc(tr[Y7], tr["e_dso_proxy"], tr["fold"], (~problem) & lab)
    prose = (
        f"Fold-4 problem groups DSO p50 vs rest (table). "
        f"Y7 DSO on problem groups {'LOW_POWER' if res_p['low_power'] else _f(res_p['cv'])} "
        f"vs rest {'LOW_POWER' if res_r['low_power'] else _f(res_r['cv'])}."
    )
    print(prose)
    return {"rows": rows, "prob": _cv(res_p), "rest": _cv(res_r), "prose": prose}


def pass_clip_icc(tr: pd.DataFrame) -> dict:
    icc_raw = icc_anova(tr["e_dso_proxy"], tr["company_id"])
    icc_clip = icc_anova(tr["e_dso_clip24"], tr["company_id"])
    acf_clip = median_acf(tr["e_dso_clip24"], tr["company_id"], 1)
    acf_raw = median_acf(tr["e_dso_proxy"], tr["company_id"], 1)
    prose = (
        f"ICC raw {_f(icc_raw['icc'])} clip24 {_f(icc_clip['icc'])}. "
        f"acf1 clip24 {_f(acf_clip)} (raw {_f(acf_raw)}). "
        f"{'clip makes BETWEEN' if np.isfinite(icc_clip['icc']) and icc_clip['icc'] >= ICC_STYLE else 'clip does not make BETWEEN'}."
    )
    print(prose)
    return {
        "icc_raw": icc_raw["icc"],
        "icc_clip": icc_clip["icc"],
        "acf_clip": acf_clip,
        "acf_raw": acf_raw,
        "prose": prose,
    }


def pass_y3_q6(tr: pd.DataFrame) -> dict:
    slices = {
        "all": pd.Series(True, index=tr.index),
        "short_<12_sofar": tr["so_far_class"] == "short_<12",
        "early6_calendar": tr["early6"],
        "after_month7": ~tr["early6"],
    }
    cols = (
        "e_dso_proxy",
        "e_dso_proxy_lag1",
        "e_dso_proxy_lag3",
        "c_n_days_with_tx",
    )
    rows = []
    store = {}
    resid, _ = ols_resid(tr["e_dso_proxy"], tr["c_n_days_with_tx"])
    for sl_name, sl in slices.items():
        n_sl = int((sl & tr[Y3].notna()).sum())
        for col in cols:
            mask = sl & tr[Y3].notna()
            res = signed_oof_auroc(tr[Y3], tr[col], tr["fold"], mask)
            store[(sl_name, col)] = res
            rows.append(
                {
                    "slice": sl_name,
                    "col": col,
                    "n_nn": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "present": _pp(_pct(res["n_defined"], n_sl) if n_sl else float("nan")),
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
        rres = signed_oof_auroc(tr[Y3], resid, tr["fold"], sl & tr[Y3].notna())
        store[(sl_name, "leftover days")] = rres
        rows.append(
            {
                "slice": sl_name,
                "col": "leftover days",
                "n_nn": f"{rres['n_defined']:,}",
                "n_pos": f"{rres['n_pos']:,}",
                "present": _pp(_pct(rres["n_defined"], n_sl) if n_sl else float("nan")),
                "CV": "LOW_POWER" if rres["low_power"] else _f(rres["cv"]),
            }
        )
    prose = (
        f"Y3 Q6 DSO lag1 {_f(_cv(store[('all', 'e_dso_proxy_lag1')]))} "
        f"short {_f(_cv(store[('short_<12_sofar', 'e_dso_proxy_lag1')]))}. "
        f"After-month7 leftover-days {_f(_cv(store[('after_month7', 'leftover days')]))}."
    )
    print(prose)
    return {
        "rows": rows,
        "all_lag1": _cv(store[("all", "e_dso_proxy_lag1")]),
        "short_lag": _cv(store[("short_<12_sofar", "e_dso_proxy_lag1")]),
        "after_left": _cv(store[("after_month7", "leftover days")]),
        "prose": prose,
    }


def pass_lag_leftover(tr: pd.DataFrame) -> dict:
    dso1 = tr["e_dso_proxy_lag1"]
    days = tr["c_n_days_with_tx"]
    drop = pd.to_numeric(dso1, errors="coerce").where(
        pd.to_numeric(dso1, errors="coerce").abs() <= WINSOR
    )
    resid, _ = ols_resid(dso1, days)
    resid_drop, _ = ols_resid(drop, days)
    resid_now, _ = ols_resid(tr["e_dso_proxy"], days)
    rows = []
    store = {}
    for ycol in (Y3, Y7):
        lab = tr[ycol].notna()
        for name, s in (
            ("lag1 raw", dso1),
            ("lag1 leftover days", resid),
            ("lag1 leftover days drop>24", resid_drop),
            ("now leftover days", resid_now),
        ):
            res = signed_oof_auroc(tr[ycol], s, tr["fold"], lab)
            store[(ycol, name)] = res
            rows.append(_auc_row(ycol, name, res, None))
    prose = (
        f"Y3 lag1 leftover after days {_f(_cv(store[(Y3, 'lag1 leftover days')]))} "
        f"drop>24 {_f(_cv(store[(Y3, 'lag1 leftover days drop>24')]))} "
        f"(now leftover {_f(_cv(store[(Y3, 'now leftover days')]))})."
    )
    print(prose)
    return {
        "rows": rows,
        "y3_lag": _cv(store[(Y3, "lag1 leftover days")]),
        "y3_drop": _cv(store[(Y3, "lag1 leftover days drop>24")]),
        "y3_now": _cv(store[(Y3, "now leftover days")]),
        "prose": prose,
    }


def pass_delay_tail(tr: pd.DataFrame) -> dict:
    """Is Y7 leftover-after-delay the |DSO|>24 tail? Clip24 leftover was 0.404."""
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    delay = pd.to_numeric(tr["e_delay_coll"], errors="coerce")
    drop24 = dso.where(dso.abs() <= WINSOR)
    clip24 = dso.clip(upper=WINSOR)
    clip12 = dso.clip(upper=12.0)
    resid, info = ols_resid(dso, delay)
    resid_drop, info_d = ols_resid(drop24, delay)
    resid_c24, info_c = ols_resid(clip24, delay)
    resid_c12, _ = ols_resid(clip12, delay)
    resid_iss, _ = ols_resid(dso, tr["e_ar_issued_lag1"], delay)
    rows = []
    store = {}
    lab = tr[Y7].notna()
    for name, s in (
        ("leftover delay (raw)", resid),
        ("leftover delay drop |DSO|>24", resid_drop),
        ("leftover delay clip24", resid_c24),
        ("leftover delay clip12", resid_c12),
        ("leftover delay+issued", resid_iss),
        ("DSO raw on delay-overlap", dso),
        ("delay raw on overlap", delay),
    ):
        res = signed_oof_auroc(tr[Y7], s, tr["fold"], lab)
        store[name] = res
        rows.append(_auc_row(Y7, name, res, None))
    after_raw = _cv(store["leftover delay (raw)"])
    after_drop = _cv(store["leftover delay drop |DSO|>24"])
    after_c24 = _cv(store["leftover delay clip24"])
    tail = bool(
        np.isfinite(after_raw)
        and after_raw >= CHANCE
        and (
            (np.isfinite(after_drop) and after_drop < CHANCE)
            or (np.isfinite(after_c24) and after_c24 < CHANCE)
        )
    )
    prose = (
        f"Y7 leftover after delay raw {_f(after_raw)} drop>|24| {_f(after_drop)} "
        f"clip24 {_f(after_c24)} clip12 {_f(_cv(store['leftover delay clip12']))} "
        f"R² raw={_f(info['r2'])} drop={_f(info_d['r2'])} clip={_f(info_c['r2'])}. "
        f"{'0.561 leftover-after-delay is a |DSO|>24 tail (clip/drop dies)' if tail else 'leftover-after-delay survives clip/drop'}."
    )
    print(prose)
    return {
        "rows": rows,
        "after_raw": after_raw,
        "after_drop": after_drop,
        "after_c24": after_c24,
        "after_c12": _cv(store["leftover delay clip12"]),
        "after_both": _cv(store["leftover delay+issued"]),
        "tail": tail,
        "r2": info["r2"],
        "prose": prose,
    }


def pass_y3_days_size(tr: pd.DataFrame) -> dict:
    """Why leftover-after-days+size 0.618 while leftover-after-days 0.474?"""
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    drop24 = dso.where(dso.abs() <= WINSOR)
    resid_d, info_d = ols_resid(dso, days)
    resid_s, info_s = ols_resid(dso, size)
    resid_b, info_b = ols_resid(dso, days, size)
    resid_drop_b, _ = ols_resid(drop24, days, size)
    resid_drop_d, _ = ols_resid(drop24, days)
    lab = tr[Y3].notna()
    ok = lab & dso.notna() & days.notna() & size.notna()
    rows = []
    store = {}
    for name, s, mask in (
        ("leftover days all", resid_d, lab),
        ("leftover size all", resid_s, lab),
        ("leftover days+size all", resid_b, lab),
        ("leftover days+size drop>24", resid_drop_b, lab),
        ("leftover days drop>24", resid_drop_d, lab),
        ("days same-n three", days, ok),
        ("size same-n three", size, ok),
        ("DSO same-n three", dso, ok),
        ("leftover days same-n three", resid_d, ok),
        ("leftover days+size same-n three", resid_b, ok),
    ):
        res = signed_oof_auroc(tr[Y3], s, tr["fold"], mask)
        store[name] = res
        rows.append(_auc_row(Y3, name, res, None))
    prose = (
        f"Y3 leftover days {_f(_cv(store['leftover days all']))} "
        f"size {_f(_cv(store['leftover size all']))} "
        f"days+size {_f(_cv(store['leftover days+size all']))} "
        f"days+size drop>24 {_f(_cv(store['leftover days+size drop>24']))} "
        f"R² days={_f(info_d['r2'])} size={_f(info_s['r2'])} both={_f(info_b['r2'])}. "
        f"Same-n days {_f(_cv(store['days same-n three']))} leftover-days "
        f"{_f(_cv(store['leftover days same-n three']))}."
    )
    print(prose)
    return {
        "rows": rows,
        "days": _cv(store["leftover days all"]),
        "size": _cv(store["leftover size all"]),
        "both": _cv(store["leftover days+size all"]),
        "both_drop": _cv(store["leftover days+size drop>24"]),
        "prose": prose,
    }


def pass_lag1_equals_days(tr: pd.DataFrame) -> dict:
    """Y3 lag1 leftover-days printed 0.711 — is leftover a days clone?"""
    dso1 = pd.to_numeric(tr["e_dso_proxy_lag1"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    resid, info = ols_resid(dso1, days)
    rho_res_days, n = spearman_n(resid, days)
    lab = tr[Y3].notna()
    ok = lab & dso1.notna() & days.notna()
    rows = []
    store = {}
    for name, s in (
        ("lag1 raw", dso1),
        ("days same-n lag1", days),
        ("lag1 leftover days", resid),
        ("|leftover|", resid.abs()),
        ("days on leftover-defined", days),
    ):
        res = signed_oof_auroc(tr[Y3], s, tr["fold"], ok)
        store[name] = res
        rows.append(_auc_row(Y3, name, res, None))
    rho_raw, _ = spearman_n(dso1, days)
    clone = bool(np.isfinite(rho_raw) and abs(rho_raw) >= 0.80)
    false_clone = bool(
        np.isfinite(rho_res_days)
        and abs(rho_res_days) >= 0.80
        and (not np.isfinite(rho_raw) or abs(rho_raw) < 0.30)
    )
    prose = (
        f"Y3 lag1 leftover vs days ρ={_f(rho_res_days)} n={n} R²={_f(info['r2'])} "
        f"(raw lag1 vs days {_f(rho_raw)}"
        f"{'; FALSE clone — unclipped residual rank artifact' if false_clone else (' CLONE of days' if clone else '; not a days clone')}"
        f"). Same-n leftover {_f(_cv(store['lag1 leftover days']))} "
        f"days {_f(_cv(store['days same-n lag1']))} lag1 raw {_f(_cv(store['lag1 raw']))}."
    )
    print(prose)
    return {
        "rows": rows,
        "rho": rho_res_days,
        "r2": info["r2"],
        "clone": clone,
        "false_clone": false_clone,
        "rho_raw": rho_raw,
        "leftover": _cv(store["lag1 leftover days"]),
        "days": _cv(store["days same-n lag1"]),
        "prose": prose,
    }


def pass_fat_issued(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    pos = iss[iss > 0]
    p10 = float(pos.quantile(0.10)) if len(pos) else float("nan")
    fat = iss > p10
    resid_y3, _ = ols_resid(tr["e_dso_proxy"], tr["c_n_days_with_tx"])
    resid_y7, _ = ols_resid(tr["e_dso_proxy"], tr["e_ar_issued_lag1"])
    rows = []
    store = {}
    for ycol, feat, s in (
        (Y3, "fat issued DSO", tr["e_dso_proxy"]),
        (Y3, "fat leftover days", resid_y3),
        (Y3, "fat days", tr["c_n_days_with_tx"]),
        (Y7, "fat issued DSO", tr["e_dso_proxy"]),
        (Y7, "fat leftover iss", resid_y7),
        (Y7, "fat issued_lag1", tr["e_ar_issued_lag1"]),
    ):
        lab = fat & tr[ycol].notna()
        res = signed_oof_auroc(tr[ycol], s, tr["fold"], lab)
        store[(ycol, feat)] = res
        rows.append(_auc_row(ycol, feat, res, None))
    prose = (
        f"Fat issued>p10={_f(p10, 0)}: Y3 DSO {_f(_cv(store[(Y3, 'fat issued DSO')]))} "
        f"leftover-days {_f(_cv(store[(Y3, 'fat leftover days')]))} vs days "
        f"{_f(_cv(store[(Y3, 'fat days')]))}. Y7 leftover-iss {_f(_cv(store[(Y7, 'fat leftover iss')]))}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3": _cv(store[(Y3, "fat leftover days")]),
        "y3_days": _cv(store[(Y3, "fat days")]),
        "y7": _cv(store[(Y7, "fat leftover iss")]),
        "p10": p10,
        "prose": prose,
    }


def pass_fold_leftover(tr: pd.DataFrame) -> dict:
    resid_iss, _ = ols_resid(tr["e_dso_proxy"], tr["e_ar_issued_lag1"])
    resid_delay, _ = ols_resid(tr["e_dso_proxy"], tr["e_delay_coll"])
    resid_days, _ = ols_resid(tr["e_dso_proxy"], tr["c_n_days_with_tx"])
    lab7 = tr[Y7].notna()
    lab3 = tr[Y3].notna()
    rows = []
    store = {}
    for name, s, lab, ycol in (
        ("e_dso_proxy", tr["e_dso_proxy"], lab7, Y7),
        ("e_ar_issued_lag1", tr["e_ar_issued_lag1"], lab7, Y7),
        ("leftover iss", resid_iss, lab7, Y7),
        ("leftover delay", resid_delay, lab7, Y7),
        ("Y3 leftover days", resid_days, lab3, Y3),
        ("Y3 days", tr["c_n_days_with_tx"], lab3, Y3),
    ):
        res = signed_oof_auroc(tr[ycol], s, tr["fold"], lab)
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
        f"Y7 fold-wise leftover-iss {_f(_cv(store['leftover iss']))} "
        f"folds {fold_bits(store['leftover iss'])} fold4 {_f(fold_k(store['leftover iss'], 4))} "
        f"vs issued fold4 {_f(fold_k(store['e_ar_issued_lag1'], 4))}."
    )
    print(prose)
    return {
        "rows": rows,
        "iss4": fold_k(store["leftover iss"], 4),
        "dso4": fold_k(store["e_dso_proxy"], 4),
        "prose": prose,
    }


def pass_chronic(tr: pd.DataFrame) -> dict:
    """Chronic 12 Y2 names — does Y3 DSO flip? Diagnostic only."""
    y2 = pd.to_numeric(tr[Y2], errors="coerce")
    co_rate = (
        tr.assign(y2=y2)
        .groupby("company_id")["y2"]
        .mean()
    )
    chronic = set(co_rate[co_rate >= 0.5].index.astype(str))
    # night cards used 12 names; keep a rate floor but report count
    drop = ~tr["company_id"].isin(chronic)
    lab = tr[Y3].notna()
    all_dso = signed_oof_auroc(tr[Y3], tr["e_dso_proxy"], tr["fold"], lab)
    drop_dso = signed_oof_auroc(tr[Y3], tr["e_dso_proxy"], tr["fold"], lab & drop)
    all_days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], lab)
    drop_days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], lab & drop)
    rows = [
        {"slice": "Y3 all", "DSO": _f(_cv(all_dso)), "days": _f(_cv(all_days))},
        {"slice": f"Y3 drop-chronic n_co={len(chronic)}", "DSO": _f(_cv(drop_dso)), "days": _f(_cv(drop_days))},
    ]
    prose = (
        f"Chronic Y2 (rate≥0.5) companies: {len(chronic)}. "
        f"Y3 DSO {_f(_cv(all_dso))} → drop {_f(_cv(drop_dso))}. "
        f"Days {_f(_cv(all_days))} → {_f(_cv(drop_days))}. "
        f"{'does not flip' if np.isfinite(_cv(all_dso)) and np.isfinite(_cv(drop_dso)) and abs(_cv(all_dso)-_cv(drop_dso))<0.02 else 'moves'}."
    )
    print(prose)
    return {
        "rows": rows,
        "n_chronic": len(chronic),
        "all": _cv(all_dso),
        "drop": _cv(drop_dso),
        "prose": prose,
    }


def pass_logdso(tr: pd.DataFrame) -> dict:
    """In-memory log1p(|DSO|) leftover — does the honest leftover live without the tail?"""
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    logx = np.log1p(dso.clip(lower=0))
    resid_d, info_d = ols_resid(logx, tr["c_n_days_with_tx"])
    resid_i, info_i = ols_resid(logx, tr["e_ar_issued_lag1"])
    resid_c, _ = ols_resid(logx, tr["e_delay_coll"])
    rows = []
    store = {}
    for ycol, name, s in (
        (Y3, "log1p DSO", logx),
        (Y3, "log1p leftover days", resid_d),
        (Y3, "clip24 leftover days", ols_resid(tr["e_dso_clip24"], tr["c_n_days_with_tx"])[0]),
        (Y7, "log1p DSO", logx),
        (Y7, "log1p leftover iss", resid_i),
        (Y7, "log1p leftover delay", resid_c),
        (Y7, "clip24 leftover iss", ols_resid(tr["e_dso_clip24"], tr["e_ar_issued_lag1"])[0]),
    ):
        res = signed_oof_auroc(tr[ycol], s, tr["fold"], tr[ycol].notna())
        store[(ycol, name)] = res
        rows.append(_auc_row(ycol, name, res, None))
    prose = (
        f"log1p(DSO) Y3 {_f(_cv(store[(Y3, 'log1p DSO')]))} leftover-days "
        f"{_f(_cv(store[(Y3, 'log1p leftover days')]))} R²={_f(info_d['r2'])}. "
        f"Y7 log1p leftover-iss {_f(_cv(store[(Y7, 'log1p leftover iss')]))} "
        f"leftover-delay {_f(_cv(store[(Y7, 'log1p leftover delay')]))} R²iss={_f(info_i['r2'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3": _cv(store[(Y3, "log1p leftover days")]),
        "y3_raw": _cv(store[(Y3, "log1p DSO")]),
        "y7_iss": _cv(store[(Y7, "log1p leftover iss")]),
        "y7_delay": _cv(store[(Y7, "log1p leftover delay")]),
        "prose": prose,
    }


def pass_open_vs_dso(tr: pd.DataFrame) -> dict:
    """Is DSO just AR open? Twin gate vs open / leftover of open after issued."""
    rho_open, n_open = spearman_n(tr["e_dso_proxy"], tr["e_ar_open"])
    rho_iss, n_iss = spearman_n(tr["e_dso_proxy"], tr["e_ar_issued"])
    resid_open, info = ols_resid(tr["e_ar_open"], tr["e_ar_issued"])
    lab = tr[Y7].notna()
    rows = []
    store = {}
    for name, s in (
        ("e_dso_proxy", tr["e_dso_proxy"]),
        ("e_ar_open", tr["e_ar_open"]),
        ("e_ar_issued", tr["e_ar_issued"]),
        ("open leftover issued", resid_open),
    ):
        res = signed_oof_auroc(tr[Y7], s, tr["fold"], lab)
        store[name] = res
        rows.append(_auc_row(Y7, name, res, None))
    twin = bool(np.isfinite(rho_open) and abs(rho_open) >= TWIN_RHO)
    prose = (
        f"DSO vs open ρ={_f(rho_open)} n={n_open} "
        f"({'TWIN' if twin else 'not a twin — locked 0.600'}). "
        f"vs issued {_f(rho_iss)}. Y7 open {_f(_cv(store['e_ar_open']))} "
        f"open-leftover-issued {_f(_cv(store['open leftover issued']))} "
        f"DSO {_f(_cv(store['e_dso_proxy']))} R²={_f(info['r2'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "rho_open": rho_open,
        "twin": twin,
        "open": _cv(store["e_ar_open"]),
        "open_left": _cv(store["open leftover issued"]),
        "prose": prose,
    }


def pass_lag1_spearman(tr: pd.DataFrame) -> dict:
    """Explain ρ leftover-vs-days = −0.984: is lag1 DSO a days twin, or a tail rank?"""
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    dso1 = pd.to_numeric(tr["e_dso_proxy_lag1"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    clip1 = dso1.clip(upper=WINSOR)
    pairs = [
        ("DSO now vs days", dso, days),
        ("DSO lag1 vs days", dso1, days),
        ("clip24 lag1 vs days", clip1, days),
        ("DSO lag1 vs DSO now", dso1, dso),
        ("clip24 lag1 vs clip24 now", clip1, dso.clip(upper=WINSOR)),
    ]
    rows = []
    rhos = {}
    for name, a, b in pairs:
        rho, n = spearman_n(a, b)
        rhos[name] = rho
        rows.append({"pair": name, "ρ": _f(rho), "n": f"{n:,}", "twin?": "TWIN" if np.isfinite(rho) and abs(rho) >= TWIN_RHO else ""})
    resid, info = ols_resid(clip1, days)
    lab = tr[Y3].notna()
    res_left = signed_oof_auroc(tr[Y3], resid, tr["fold"], lab)
    res_raw = signed_oof_auroc(tr[Y3], clip1, tr["fold"], lab)
    prose = (
        f"Spearman DSO-now vs days {_f(rhos['DSO now vs days'])} "
        f"lag1 vs days {_f(rhos['DSO lag1 vs days'])} "
        f"clip24-lag1 vs days {_f(rhos['clip24 lag1 vs days'])}. "
        f"Y3 clip24-lag1 leftover-days {_f(_cv(res_left))} raw clip24-lag1 {_f(_cv(res_raw))} "
        f"R²={_f(info['r2'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "now_days": rhos["DSO now vs days"],
        "lag1_days": rhos["DSO lag1 vs days"],
        "clip_days": rhos["clip24 lag1 vs days"],
        "y3_left": _cv(res_left),
        "y3_raw": _cv(res_raw),
        "prose": prose,
    }


def pass_y3_iss_tail(tr: pd.DataFrame) -> dict:
    """Y3 leftover after issued_lag1 was 0.560 — tail or real?"""
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    drop24 = dso.where(dso.abs() <= WINSOR)
    resid, _ = ols_resid(dso, tr["e_ar_issued_lag1"])
    resid_d, _ = ols_resid(drop24, tr["e_ar_issued_lag1"])
    resid_c, _ = ols_resid(tr["e_dso_clip24"], tr["e_ar_issued_lag1"])
    lab = tr[Y3].notna()
    rows = []
    store = {}
    for name, s in (
        ("leftover iss raw", resid),
        ("leftover iss drop>24", resid_d),
        ("leftover iss clip24", resid_c),
        ("DSO drop>24 raw", drop24),
    ):
        res = signed_oof_auroc(tr[Y3], s, tr["fold"], lab)
        store[name] = res
        rows.append(_auc_row(Y3, name, res, None))
    tail = bool(
        np.isfinite(_cv(store["leftover iss raw"]))
        and _cv(store["leftover iss raw"]) >= CHANCE
        and np.isfinite(_cv(store["leftover iss drop>24"]))
        and _cv(store["leftover iss drop>24"]) < CHANCE
    )
    prose = (
        f"Y3 leftover after issued_lag1 raw {_f(_cv(store['leftover iss raw']))} "
        f"drop>24 {_f(_cv(store['leftover iss drop>24']))} "
        f"clip24 {_f(_cv(store['leftover iss clip24']))}. "
        f"{'0.560 is a tail' if tail else 'leftover-after-issued is not only a tail'}."
    )
    print(prose)
    return {
        "rows": rows,
        "raw": _cv(store["leftover iss raw"]),
        "drop": _cv(store["leftover iss drop>24"]),
        "clip": _cv(store["leftover iss clip24"]),
        "tail": tail,
        "prose": prose,
    }


def pass_extreme(tr: pd.DataFrame) -> dict:
    """Who owns the 2.19e6 max? Train only. Do not name holdout."""
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    opn = pd.to_numeric(tr["e_ar_open"], errors="coerce")
    finite = dso.notna()
    top = tr.loc[finite].assign(dso=dso[finite], iss=iss[finite], opn=opn[finite])
    top = top.sort_values("dso", ascending=False).head(8)
    rows = []
    for _, r in top.iterrows():
        rows.append(
            {
                "company": str(r["company_id"])[:12],
                "period": str(pd.Timestamp(r["period"]).date()),
                "DSO": _f(float(r["dso"]), 1),
                "AR open": _f(float(r["opn"]), 0),
                "AR issued": _f(float(r["iss"]), 1),
                "|DSO|>24": "yes" if abs(float(r["dso"])) > WINSOR else "",
            }
        )
    mx = float(dso.max()) if finite.any() else float("nan")
    n_gt1e3 = int((dso[finite] > 1000).sum())
    n_gt1e5 = int((dso[finite] > 1e5).sum())
    prose = (
        f"Max DSO={_f(mx, 1)} on train. n>1000={n_gt1e3} n>1e5={n_gt1e5}. "
        f"Top rows are tiny-issued / fat-open (table). Do not rewrite the store."
    )
    print(prose)
    return {"rows": rows, "max": mx, "n_gt1e3": n_gt1e3, "n_gt1e5": n_gt1e5, "prose": prose}


def pass_demean_leftover(tr: pd.DataFrame) -> dict:
    de = company_demean(tr["e_dso_proxy"], tr["company_id"])
    mu = company_mean(tr["e_dso_proxy"], tr["company_id"])
    resid_de, _ = ols_resid(de, tr["e_ar_issued_lag1"])
    resid_mu, _ = ols_resid(mu, tr["e_ar_issued_lag1"])
    resid_de_d, _ = ols_resid(de, tr["e_delay_coll"])
    lab = tr[Y7].notna()
    rows = []
    store = {}
    for name, s in (
        ("demean leftover iss", resid_de),
        ("mean leftover iss", resid_mu),
        ("demean leftover delay", resid_de_d),
        ("demean raw", de),
        ("company-mean raw", mu),
    ):
        res = signed_oof_auroc(tr[Y7], s, tr["fold"], lab)
        store[name] = res
        rows.append(_auc_row(Y7, name, res, None))
    prose = (
        f"Y7 demean leftover-iss {_f(_cv(store['demean leftover iss']))} "
        f"mean leftover-iss {_f(_cv(store['mean leftover iss']))} "
        f"demean leftover-delay {_f(_cv(store['demean leftover delay']))}."
    )
    print(prose)
    return {
        "rows": rows,
        "de_iss": _cv(store["demean leftover iss"]),
        "mu_iss": _cv(store["mean leftover iss"]),
        "de_delay": _cv(store["demean leftover delay"]),
        "prose": prose,
    }


def pass_y3_fold_left(tr: pd.DataFrame) -> dict:
    resid, _ = ols_resid(tr["e_dso_proxy"], tr["c_n_days_with_tx"])
    lab = tr[Y3].notna()
    rows = []
    store = {}
    for name, s in (
        ("e_dso_proxy", tr["e_dso_proxy"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("leftover days", resid),
        ("e_dso_clip24", tr["e_dso_clip24"]),
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
    return {
        "rows": rows,
        "left": _cv(store["leftover days"]),
        "days4": fold_k(store["c_n_days_with_tx"], 4),
        "left4": fold_k(store["leftover days"], 4),
        "prose": prose,
    }


def pass_q1_leftover(tr: pd.DataFrame) -> dict:
    """Leftover of DSO after issued / delay inside the short-DSO fifth."""
    lab = tr[Y7].notna()
    sl = tr.loc[lab & tr["e_dso_proxy"].notna()].copy()
    dso = pd.to_numeric(sl["e_dso_proxy"], errors="coerce").clip(upper=WINSOR)
    sl["dso_q"] = pd.qcut(dso.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    q1 = sl["dso_q"] == "Q1"
    mask = pd.Series(False, index=tr.index)
    mask.loc[sl.index[q1]] = True
    resid_i, _ = ols_resid(tr["e_dso_proxy"], tr["e_ar_issued_lag1"])
    resid_d, _ = ols_resid(tr["e_dso_proxy"], tr["e_delay_coll"])
    rows = []
    store = {}
    for name, s in (
        ("DSO Q1", tr["e_dso_proxy"]),
        ("issued Q1", tr["e_ar_issued_lag1"]),
        ("delay Q1", tr["e_delay_coll"]),
        ("leftover iss Q1", resid_i),
        ("leftover delay Q1", resid_d),
    ):
        res = signed_oof_auroc(tr[Y7], s, tr["fold"], mask)
        store[name] = res
        rows.append(_auc_row(Y7, name, res, None))
    prose = (
        f"Short-DSO Q1 leftover-iss {_f(_cv(store['leftover iss Q1']))} "
        f"leftover-delay {_f(_cv(store['leftover delay Q1']))} "
        f"vs issued {_f(_cv(store['issued Q1']))} DSO {_f(_cv(store['DSO Q1']))}."
    )
    print(prose)
    return {
        "rows": rows,
        "left_iss": _cv(store["leftover iss Q1"]),
        "left_delay": _cv(store["leftover delay Q1"]),
        "iss": _cv(store["issued Q1"]),
        "dso": _cv(store["DSO Q1"]),
        "prose": prose,
    }


def pass_after7_left(tr: pd.DataFrame) -> dict:
    """After month 7 only — delay is defined; leftover of DSO after delay / issued."""
    sl = ~tr["early6"]
    resid_d, _ = ols_resid(tr["e_dso_proxy"], tr["e_delay_coll"])
    resid_i, _ = ols_resid(tr["e_dso_proxy"], tr["e_ar_issued_lag1"])
    drop = pd.to_numeric(tr["e_dso_proxy"], errors="coerce").where(
        pd.to_numeric(tr["e_dso_proxy"], errors="coerce").abs() <= WINSOR
    )
    resid_dd, _ = ols_resid(drop, tr["e_delay_coll"])
    rows = []
    store = {}
    for ycol in (Y7, Y3):
        lab = sl & tr[ycol].notna()
        for name, s in (
            ("DSO after7", tr["e_dso_proxy"]),
            ("leftover delay after7", resid_d),
            ("leftover delay drop after7", resid_dd),
            ("leftover iss after7", resid_i),
        ):
            res = signed_oof_auroc(tr[ycol], s, tr["fold"], lab)
            store[(ycol, name)] = res
            rows.append(_auc_row(ycol, name, res, None))
    prose = (
        f"After month 7: Y7 leftover-delay {_f(_cv(store[(Y7, 'leftover delay after7')]))} "
        f"drop {_f(_cv(store[(Y7, 'leftover delay drop after7')]))} "
        f"leftover-iss {_f(_cv(store[(Y7, 'leftover iss after7')]))}. "
        f"Y3 leftover-iss {_f(_cv(store[(Y3, 'leftover iss after7')]))}."
    )
    print(prose)
    return {
        "rows": rows,
        "y7_delay": _cv(store[(Y7, "leftover delay after7")]),
        "y7_drop": _cv(store[(Y7, "leftover delay drop after7")]),
        "y7_iss": _cv(store[(Y7, "leftover iss after7")]),
        "prose": prose,
    }


def pass_holdout_tails(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    assert set(ho["company_id"]) <= load_holdout()
    dso = pd.to_numeric(ho["e_dso_proxy"], errors="coerce")
    finite = dso.dropna()
    rows = [
        {
            "slice": "holdout all",
            "n_cm": f"{len(ho):,}",
            "nn": f"{int(dso.notna().sum()):,}",
            "p50": _f(float(finite.median()) if len(finite) else float("nan"), 2),
            "p99": _f(float(finite.quantile(0.99)) if len(finite) else float("nan"), 1),
            "max": _f(float(finite.max()) if len(finite) else float("nan"), 1),
            "|DSO|>24": _pp(_pct(int((finite.abs() > WINSOR).sum()), len(finite))),
        }
    ]
    prose = (
        f"Holdout tails (coverage only): nn={int(dso.notna().sum()):,} "
        f"p50={_f(float(finite.median()) if len(finite) else float('nan'), 2)} "
        f"p99={_f(float(finite.quantile(0.99)) if len(finite) else float('nan'), 1)} "
        f"|DSO|>24 {_pp(_pct(int((finite.abs() > WINSOR).sum()), len(finite)))}. No AUROC."
    )
    print(prose)
    return {"rows": rows, "nn": int(dso.notna().sum()), "prose": prose}


def pass_long_leftover(tr: pd.DataFrame) -> dict:
    """Long so-far DSO was 0.607 — leftover after issued on long trails?"""
    resid, _ = ols_resid(tr["e_dso_proxy"], tr["e_ar_issued_lag1"])
    resid_d, _ = ols_resid(tr["e_dso_proxy"], tr["e_delay_coll"])
    drop = pd.to_numeric(tr["e_dso_proxy"], errors="coerce").where(
        pd.to_numeric(tr["e_dso_proxy"], errors="coerce").abs() <= WINSOR
    )
    resid_drop, _ = ols_resid(drop, tr["e_ar_issued_lag1"])
    slices = {
        "long_>=18_sofar": tr["so_far_class"] == "long_>=18",
        "short_<12_sofar": tr["so_far_class"] == "short_<12",
        "long_>=18_company": tr["co_class"] == "long_>=18",
        "short_<12_company": tr["co_class"] == "short_<12",
    }
    rows = []
    store = {}
    for sl_name, sl in slices.items():
        lab = sl & tr[Y7].notna()
        for name, s in (
            ("DSO", tr["e_dso_proxy"]),
            ("issued_lag1", tr["e_ar_issued_lag1"]),
            ("leftover iss", resid),
            ("leftover iss drop>24", resid_drop),
            ("leftover delay", resid_d),
        ):
            res = signed_oof_auroc(tr[Y7], s, tr["fold"], lab)
            store[(sl_name, name)] = res
            rows.append(_auc_row(Y7, f"{name} {sl_name}", res, None))
    long_left = _cv(store[("long_>=18_sofar", "leftover iss")])
    long_raw = _cv(store[("long_>=18_sofar", "DSO")])
    long_iss = _cv(store[("long_>=18_sofar", "issued_lag1")])
    prose = (
        f"Long so-far Y7 DSO {_f(long_raw)} leftover-iss {_f(long_left)} "
        f"issued {_f(long_iss)} drop>24 {_f(_cv(store[('long_>=18_sofar', 'leftover iss drop>24')]))}. "
        f"Short so-far leftover-iss {_f(_cv(store[('short_<12_sofar', 'leftover iss')]))}."
    )
    print(prose)
    return {
        "rows": rows,
        "long_raw": long_raw,
        "long_left": long_left,
        "long_iss": long_iss,
        "short_left": _cv(store[("short_<12_sofar", "leftover iss")]),
        "prose": prose,
    }


def pass_od30_leftover(tr: pd.DataFrame) -> dict:
    """Closest Spearman was ar_overdue_30 ρ=0.534 — leftover after it?"""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["e_ar_overdue_30"])
    resid_c, _ = ols_resid(tr["e_dso_clip24"], tr["e_ar_overdue_30"])
    rows = []
    store = {}
    for ycol in (Y7, Y3):
        lab = tr[ycol].notna()
        for name, s in (
            ("leftover od30", resid),
            ("clip24 leftover od30", resid_c),
            ("e_ar_overdue_30", tr["e_ar_overdue_30"]),
        ):
            res = signed_oof_auroc(tr[ycol], s, tr["fold"], lab)
            store[(ycol, name)] = res
            rows.append(_auc_row(ycol, name, res, None))
    prose = (
        f"Y7 leftover after ar_overdue_30 {_f(_cv(store[(Y7, 'leftover od30')]))} "
        f"clip24 {_f(_cv(store[(Y7, 'clip24 leftover od30')]))} "
        f"od30 {_f(_cv(store[(Y7, 'e_ar_overdue_30')]))} R²={_f(info['r2'])}. "
        f"Y3 leftover-od30 {_f(_cv(store[(Y3, 'leftover od30')]))}."
    )
    print(prose)
    return {
        "rows": rows,
        "y7": _cv(store[(Y7, "leftover od30")]),
        "y7_clip": _cv(store[(Y7, "clip24 leftover od30")]),
        "y3": _cv(store[(Y3, "leftover od30")]),
        "r2": info["r2"],
        "prose": prose,
    }


def pass_long_folds(tr: pd.DataFrame) -> dict:
    """Does long-sofar leftover-iss 0.613 hold fold-wise, or one fold?"""
    sl = tr["so_far_class"] == "long_>=18"
    resid, _ = ols_resid(tr["e_dso_proxy"], tr["e_ar_issued_lag1"])
    lab = sl & tr[Y7].notna()
    rows = []
    store = {}
    for name, s in (
        ("DSO long", tr["e_dso_proxy"]),
        ("issued long", tr["e_ar_issued_lag1"]),
        ("leftover iss long", resid),
        ("days long", tr["c_n_days_with_tx"]),
    ):
        res = signed_oof_auroc(tr[Y7], s, tr["fold"], lab)
        store[name] = res
        rows.append(
            {
                "feature": name,
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "folds": fold_bits(res) if not res["low_power"] else "—",
                "fold4": _f(fold_k(res, 4)),
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
            }
        )
    prose = (
        f"Long so-far fold-wise leftover-iss {_f(_cv(store['leftover iss long']))} "
        f"folds {fold_bits(store['leftover iss long'])} "
        f"n={store['leftover iss long']['n_defined']:,} pos={store['leftover iss long']['n_pos']:,} "
        f"vs issued {_f(_cv(store['issued long']))}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(store["leftover iss long"]),
        "iss": _cv(store["issued long"]),
        "n": store["leftover iss long"]["n_defined"],
        "n_pos": store["leftover iss long"]["n_pos"],
        "prose": prose,
    }


def pass_rolling(tr: pd.DataFrame) -> dict:
    """Last 9 origins: leftover slope on periods < t; AUROC on Y7 at t+3. Train only."""
    origins = rolling_origins(tr["period"], horizon=3, n_last=9)
    y = pd.to_numeric(tr[Y7], errors="coerce")
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    iss = pd.to_numeric(tr["e_ar_issued_lag1"], errors="coerce")
    rows = []
    aucs_raw = []
    aucs_left = []
    aucs_iss = []
    for o in origins:
        t = o["t"]
        pred = o["predict_period"]
        past = tr["period"] < t
        now = tr["period"] == t
        lab_past = past & y.notna()
        feat = tr.loc[
            now, ["company_id", "e_dso_proxy", "e_ar_issued_lag1"]
        ].copy()
        lab = tr.loc[tr["period"] == pred, ["company_id", Y7]].copy()
        m = feat.merge(lab, on="company_id", how="inner")
        y_now = pd.to_numeric(m[Y7], errors="coerce")
        dso_now = pd.to_numeric(m["e_dso_proxy"], errors="coerce")
        iss_now = pd.to_numeric(m["e_ar_issued_lag1"], errors="coerce")
        n_pos = int((y_now == 1).sum())
        n = int(y_now.notna().sum())
        if n_pos < MIN_POS:
            rows.append(
                {
                    "t": str(pd.Timestamp(t).date()),
                    "pred": str(pd.Timestamp(pred).date()),
                    "n": n,
                    "n_pos": n_pos,
                    "DSO": "LOW_POWER",
                    "issued": "LOW_POWER",
                    "leftover": "LOW_POWER",
                }
            )
            continue
        fit = past & dso.notna() & iss.notna()
        resid_now = pd.Series(np.nan, index=m.index, dtype=float)
        if int(fit.sum()) >= 25:
            Yf = dso[fit].to_numpy(dtype=float)
            Xf = np.column_stack(
                [np.ones(int(fit.sum())), iss[fit].to_numpy(dtype=float)]
            )
            beta, _, _, _ = np.linalg.lstsq(Xf, Yf, rcond=None)
            apply = dso_now.notna() & iss_now.notna()
            resid_now.loc[apply] = dso_now[apply] - (
                beta[0] + beta[1] * iss_now[apply]
            )
            resid_past = pd.Series(np.nan, index=tr.index, dtype=float)
            resid_past.loc[fit] = Yf - (Xf @ beta)
            sign_r = choose_sign(y[lab_past], resid_past[lab_past])
        else:
            sign_r = 1
        sign_d = choose_sign(y[lab_past], dso[lab_past])
        raw = auroc(y_now, sign_d * dso_now)
        sign_i = choose_sign(y[lab_past], iss[lab_past])
        iss_auc = auroc(y_now, sign_i * iss_now)
        left = auroc(y_now, sign_r * resid_now)
        aucs_raw.append(raw)
        aucs_iss.append(iss_auc)
        aucs_left.append(left)
        rows.append(
            {
                "t": str(pd.Timestamp(t).date()),
                "pred": str(pd.Timestamp(pred).date()),
                "n": n,
                "n_pos": n_pos,
                "DSO": _f(raw),
                "issued": _f(iss_auc),
                "leftover": _f(left),
            }
        )
    mean_raw = float(np.nanmean(aucs_raw)) if aucs_raw else float("nan")
    mean_iss = float(np.nanmean(aucs_iss)) if aucs_iss else float("nan")
    mean_left = float(np.nanmean(aucs_left)) if aucs_left else float("nan")
    prose = (
        f"Rolling 9 origins (Y7 at t+3, leftover slope on < t): mean DSO {_f(mean_raw)} "
        f"issued {_f(mean_iss)} leftover-iss {_f(mean_left)}. "
        f"{'leftover dies on the origin path' if not np.isfinite(mean_left) or mean_left < CHANCE else 'leftover lives on origins — still do not KEEP'}."
    )
    print(prose)
    return {
        "rows": rows,
        "raw": mean_raw,
        "iss": mean_iss,
        "left": mean_left,
        "prose": prose,
    }


def pass_rolling_y3(tr: pd.DataFrame) -> dict:
    """Last 9 origins: leftover after days on < t; AUROC on Y3 at t+3. Train only."""
    origins = rolling_origins(tr["period"], horizon=3, n_last=9)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    rows = []
    aucs_raw = []
    aucs_left = []
    aucs_days = []
    for o in origins:
        t = o["t"]
        pred = o["predict_period"]
        past = tr["period"] < t
        now = tr["period"] == t
        lab_past = past & y.notna()
        feat = tr.loc[now, ["company_id", "e_dso_proxy", "c_n_days_with_tx"]].copy()
        lab = tr.loc[tr["period"] == pred, ["company_id", Y3]].copy()
        m = feat.merge(lab, on="company_id", how="inner")
        y_now = pd.to_numeric(m[Y3], errors="coerce")
        dso_now = pd.to_numeric(m["e_dso_proxy"], errors="coerce")
        days_now = pd.to_numeric(m["c_n_days_with_tx"], errors="coerce")
        n_pos = int((y_now == 1).sum())
        n = int(y_now.notna().sum())
        if n_pos < MIN_POS:
            rows.append(
                {
                    "t": str(pd.Timestamp(t).date()),
                    "pred": str(pd.Timestamp(pred).date()),
                    "n": n,
                    "n_pos": n_pos,
                    "DSO": "LOW_POWER",
                    "days": "LOW_POWER",
                    "leftover": "LOW_POWER",
                }
            )
            continue
        fit = past & dso.notna() & days.notna()
        resid_now = pd.Series(np.nan, index=m.index, dtype=float)
        if int(fit.sum()) >= 25:
            Yf = dso[fit].to_numpy(dtype=float)
            Xf = np.column_stack(
                [np.ones(int(fit.sum())), days[fit].to_numpy(dtype=float)]
            )
            beta, _, _, _ = np.linalg.lstsq(Xf, Yf, rcond=None)
            apply = dso_now.notna() & days_now.notna()
            resid_now.loc[apply] = dso_now[apply] - (
                beta[0] + beta[1] * days_now[apply]
            )
            resid_past = pd.Series(np.nan, index=tr.index, dtype=float)
            resid_past.loc[fit] = Yf - (Xf @ beta)
            sign_r = choose_sign(y[lab_past], resid_past[lab_past])
        else:
            sign_r = 1
        sign_d = choose_sign(y[lab_past], dso[lab_past])
        raw = auroc(y_now, sign_d * dso_now)
        sign_i = choose_sign(y[lab_past], days[lab_past])
        days_auc = auroc(y_now, sign_i * days_now)
        left = auroc(y_now, sign_r * resid_now)
        aucs_raw.append(raw)
        aucs_days.append(days_auc)
        aucs_left.append(left)
        rows.append(
            {
                "t": str(pd.Timestamp(t).date()),
                "pred": str(pd.Timestamp(pred).date()),
                "n": n,
                "n_pos": n_pos,
                "DSO": _f(raw),
                "days": _f(days_auc),
                "leftover": _f(left),
            }
        )
    mean_raw = float(np.nanmean(aucs_raw)) if aucs_raw else float("nan")
    mean_days = float(np.nanmean(aucs_days)) if aucs_days else float("nan")
    mean_left = float(np.nanmean(aucs_left)) if aucs_left else float("nan")
    prose = (
        f"Rolling Y3 at t+3 leftover-days: mean DSO {_f(mean_raw)} days {_f(mean_days)} "
        f"leftover {_f(mean_left)}. "
        f"{'leftover dies on the origin path' if not np.isfinite(mean_left) or mean_left < CHANCE else 'leftover lives on origins — still do not KEEP as X'}."
    )
    print(prose)
    return {
        "rows": rows,
        "raw": mean_raw,
        "days": mean_days,
        "left": mean_left,
        "prose": prose,
    }


def pass_gt24_persist(tr: pd.DataFrame) -> dict:
    """How sticky is the |DSO|>24 tail? Train only."""
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    gt = dso.abs() > WINSOR
    g = tr.assign(gt24=gt.astype(float)).groupby("company_id")["gt24"]
    n_co = int((dso.notna().groupby(tr["company_id"]).sum() > 0).sum())
    n_once = int((g.sum() >= 1).sum())
    n_twice = int((g.sum() >= 2).sum())
    n_half = 0
    for cid, sl in tr.loc[dso.notna()].groupby("company_id"):
        nn = int(sl["e_dso_proxy"].notna().sum())
        if nn >= 4 and int((pd.to_numeric(sl["e_dso_proxy"], errors="coerce").abs() > WINSOR).sum()) >= max(2, nn // 2):
            n_half += 1
    share_once = n_once / n_co if n_co else float("nan")
    share_twice = n_twice / n_co if n_co else float("nan")
    rows = [
        {"slice": "companies with any DSO", "n": n_co, "share": _pp(1.0)},
        {"slice": "|DSO|>24 at least once", "n": n_once, "share": _pp(share_once)},
        {"slice": "|DSO|>24 at least twice", "n": n_twice, "share": _pp(share_twice)},
        {"slice": "|DSO|>24 on ≥ half of nn months (nn≥4)", "n": n_half, "share": _pp(n_half / n_co if n_co else float("nan"))},
    ]
    prose = (
        f"|DSO|>24 persist: {n_once}/{n_co} companies at least once ({_pp(share_once)}), "
        f"{n_twice} twice ({_pp(share_twice)}), {n_half} chronic-half. "
        f"{'tail is company-sticky' if share_twice >= 0.20 else 'tail is mostly one-off, not a chronic object'}."
    )
    print(prose)
    return {
        "rows": rows,
        "n_once": n_once,
        "n_twice": n_twice,
        "n_half": n_half,
        "share_once": share_once,
        "share_twice": share_twice,
        "prose": prose,
    }


def _roll_pairs(tr: pd.DataFrame, ycol: str, xcols: tuple[str, ...], horizon: int = 3):
    """Stack (feature at t, label at t+h) per origin. Train only."""
    origins = rolling_origins(tr["period"], horizon=horizon, n_last=9)
    chunks = []
    for o in origins:
        t = o["t"]
        pred = o["predict_period"]
        feat = tr.loc[tr["period"] == t, ["company_id", *xcols]].copy()
        lab = tr.loc[tr["period"] == pred, ["company_id", ycol]].copy()
        m = feat.merge(lab, on="company_id", how="inner")
        if m.empty:
            continue
        m = m.copy()
        m["origin"] = pd.Timestamp(t)
        m["pred"] = pd.Timestamp(pred)
        chunks.append(m)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def pass_y3_pool(tr: pd.DataFrame) -> dict:
    """Pool low-power Y3 origins: leftover after days on stacked t → t+3. Train only."""
    stacked = _roll_pairs(tr, Y3, ("e_dso_proxy", "c_n_days_with_tx", "fold"))
    y = pd.to_numeric(stacked[Y3], errors="coerce") if not stacked.empty else pd.Series(dtype=float)
    dso = pd.to_numeric(stacked["e_dso_proxy"], errors="coerce") if not stacked.empty else pd.Series(dtype=float)
    days = pd.to_numeric(stacked["c_n_days_with_tx"], errors="coerce") if not stacked.empty else pd.Series(dtype=float)
    n = int(y.notna().sum()) if not stacked.empty else 0
    n_pos = int((y == 1).sum()) if not stacked.empty else 0
    n_orig = int(stacked["origin"].nunique()) if not stacked.empty else 0
    if stacked.empty or n_pos < MIN_POS:
        prose = (
            f"Pooled Y3 t+3 leftover: n={n} pos={n_pos} origins={n_orig} LOW_POWER "
            f"(per-origin pos 32–34)."
        )
        print(prose)
        return {
            "rows": [{"slice": "pooled Y3 t+3", "n": n, "n_pos": n_pos, "DSO": "LOW_POWER", "days": "LOW_POWER", "leftover": "LOW_POWER"}],
            "raw": float("nan"),
            "days": float("nan"),
            "left": float("nan"),
            "n": n,
            "n_pos": n_pos,
            "prose": prose,
        }
    past = tr["period"] < stacked["origin"].min()
    fit = past & tr["e_dso_proxy"].notna() & tr["c_n_days_with_tx"].notna()
    resid = pd.Series(np.nan, index=stacked.index, dtype=float)
    if int(fit.sum()) >= 25:
        Yf = pd.to_numeric(tr.loc[fit, "e_dso_proxy"], errors="coerce").to_numpy(dtype=float)
        Xf = np.column_stack(
            [
                np.ones(int(fit.sum())),
                pd.to_numeric(tr.loc[fit, "c_n_days_with_tx"], errors="coerce").to_numpy(dtype=float),
            ]
        )
        beta, _, _, _ = np.linalg.lstsq(Xf, Yf, rcond=None)
        apply = dso.notna() & days.notna()
        resid.loc[apply] = dso[apply] - (beta[0] + beta[1] * days[apply])
        resid_past = pd.Series(np.nan, index=tr.index, dtype=float)
        resid_past.loc[fit] = Yf - (Xf @ beta)
        lab_past = past & tr[Y3].notna()
        sign_r = choose_sign(tr.loc[lab_past, Y3], resid_past[lab_past])
        sign_d = choose_sign(tr.loc[lab_past, Y3], tr.loc[lab_past, "e_dso_proxy"])
        sign_i = choose_sign(tr.loc[lab_past, Y3], tr.loc[lab_past, "c_n_days_with_tx"])
    else:
        sign_r = sign_d = sign_i = 1
    raw = auroc(y, sign_d * dso)
    days_auc = auroc(y, sign_i * days)
    left = auroc(y, sign_r * resid)
    folds = pd.to_numeric(stacked["fold"], errors="coerce") if "fold" in stacked else pd.Series(0, index=stacked.index)
    oof = signed_oof_auroc(y, resid, folds, y.notna())
    rows = [
        {"slice": "pooled Y3 t+3", "n": n, "n_pos": n_pos, "DSO": _f(raw), "days": _f(days_auc), "leftover": _f(left)},
        {
            "slice": "pooled Y3 t+3 group-fold leftover",
            "n": n,
            "n_pos": n_pos,
            "DSO": _f(_cv(oof)),
            "days": "—",
            "leftover": _f(_cv(oof)),
        },
    ]
    prose = (
        f"Pooled Y3 t+3 (origins={n_orig} n={n} pos={n_pos}): DSO {_f(raw)} days {_f(days_auc)} "
        f"leftover {_f(left)} group-fold leftover {_f(_cv(oof))}. "
        f"{'leftover dies when pooled' if not np.isfinite(left) or left < CHANCE else 'pooled leftover lives — still do not KEEP'}."
    )
    print(prose)
    return {
        "rows": rows,
        "raw": raw,
        "days": days_auc,
        "left": left,
        "oof": _cv(oof),
        "n": n,
        "n_pos": n_pos,
        "prose": prose,
    }


def pass_y3_pool_robust(tr: pd.DataFrame) -> dict:
    """Is pooled Y3 t+3 leftover 0.728 a tail / size / LOW_POWER fold?"""
    stacked = _roll_pairs(tr, Y3, ("e_dso_proxy", "c_n_days_with_tx", "log_in3", "fold"))
    if stacked.empty:
        prose = "Pooled Y3 robust: empty."
        print(prose)
        return {
            "rows": [],
            "left": float("nan"),
            "drop": float("nan"),
            "clip": float("nan"),
            "dso": float("nan"),
            "days": float("nan"),
            "size": float("nan"),
            "n_pos": 0,
            "prose": prose,
        }
    y = pd.to_numeric(stacked[Y3], errors="coerce")
    dso = pd.to_numeric(stacked["e_dso_proxy"], errors="coerce")
    days = pd.to_numeric(stacked["c_n_days_with_tx"], errors="coerce")
    size = pd.to_numeric(stacked["log_in3"], errors="coerce")
    clip = dso.clip(upper=WINSOR, lower=-WINSOR)
    body = dso.abs() <= WINSOR
    past = tr["period"] < stacked["origin"].min()
    fit = past & tr["e_dso_proxy"].notna() & tr["c_n_days_with_tx"].notna()
    resid = pd.Series(np.nan, index=stacked.index, dtype=float)
    if int(fit.sum()) >= 25:
        Yf = pd.to_numeric(tr.loc[fit, "e_dso_proxy"], errors="coerce").to_numpy(dtype=float)
        Xf = np.column_stack(
            [
                np.ones(int(fit.sum())),
                pd.to_numeric(tr.loc[fit, "c_n_days_with_tx"], errors="coerce").to_numpy(dtype=float),
            ]
        )
        beta, _, _, _ = np.linalg.lstsq(Xf, Yf, rcond=None)
        apply = dso.notna() & days.notna()
        resid.loc[apply] = dso[apply] - (beta[0] + beta[1] * days[apply])
        resid_past = pd.Series(np.nan, index=tr.index, dtype=float)
        resid_past.loc[fit] = Yf - (Xf @ beta)
        lab_past = past & tr[Y3].notna()
        sign_r = choose_sign(tr.loc[lab_past, Y3], resid_past[lab_past])
        sign_d = choose_sign(tr.loc[lab_past, Y3], tr.loc[lab_past, "e_dso_proxy"])
        sign_i = choose_sign(tr.loc[lab_past, Y3], tr.loc[lab_past, "c_n_days_with_tx"])
        sign_s = choose_sign(tr.loc[lab_past, Y3], tr.loc[lab_past, "log_in3"])
    else:
        sign_r = sign_d = sign_i = sign_s = 1
    raw = auroc(y, sign_d * dso)
    days_auc = auroc(y, sign_i * days)
    size_auc = auroc(y, sign_s * size)
    left = auroc(y, sign_r * resid)
    left_drop = auroc(y[body], sign_r * resid[body])
    left_clip = auroc(y, sign_r * (clip - (beta[1] * days if int(fit.sum()) >= 25 else 0)))
    clip_raw = auroc(y, sign_d * clip)
    n = int(y.notna().sum())
    n_pos = int((y == 1).sum())
    n_body = int((body & y.notna()).sum())
    n_pos_body = int((body & (y == 1)).sum())
    folds = pd.to_numeric(stacked["fold"], errors="coerce")
    fold_pos = []
    for k in range(N_FOLDS):
        fold_pos.append(int(((folds == k) & (y == 1)).sum()))
    rows = [
        {"slice": "pooled Y3 t+3 DSO", "n": n, "n_pos": n_pos, "AUROC": _f(raw)},
        {"slice": "pooled Y3 t+3 days", "n": n, "n_pos": n_pos, "AUROC": _f(days_auc)},
        {"slice": "pooled Y3 t+3 size", "n": n, "n_pos": n_pos, "AUROC": _f(size_auc)},
        {"slice": "pooled leftover-days", "n": n, "n_pos": n_pos, "AUROC": _f(left)},
        {"slice": "pooled leftover drop>|24|", "n": n_body, "n_pos": n_pos_body, "AUROC": _f(left_drop)},
        {"slice": "pooled clip24", "n": n, "n_pos": n_pos, "AUROC": _f(clip_raw)},
        {"slice": "fold pos 0-4", "n": n, "n_pos": n_pos, "AUROC": " ".join(str(x) for x in fold_pos)},
    ]
    tail = (not np.isfinite(left_drop) or left_drop < CHANCE) and np.isfinite(left) and left >= CHANCE
    prose = (
        f"Pooled Y3 robust: DSO {_f(raw)} days {_f(days_auc)} size {_f(size_auc)} leftover {_f(left)} "
        f"drop>|24| {_f(left_drop)} clip24 {_f(clip_raw)} fold-pos {fold_pos}. "
        f"{'0.728 is a |DSO|>24 tail' if tail else 'not only a tail / still LOW_POWER folds — do not KEEP'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": left,
        "drop": left_drop,
        "clip": clip_raw,
        "dso": raw,
        "days": days_auc,
        "size": size_auc,
        "n_pos": n_pos,
        "tail": tail,
        "prose": prose,
    }


def pass_month_tail(tr: pd.DataFrame) -> dict:
    """Calendar share of |DSO|>24 — is the tail a month spike? Train only."""
    rows = []
    for p, sl in tr.groupby(tr["period"]):
        x = pd.to_numeric(sl["e_dso_proxy"], errors="coerce")
        nn = int(x.notna().sum())
        n_gt = int((x.abs() > WINSOR).sum())
        rows.append(
            {
                "period": str(pd.Timestamp(p).date()),
                "nn": nn,
                "|DSO|>24": n_gt,
                "share": _pp(n_gt / nn if nn else float("nan")),
                "p50": _f(float(x.median()) if nn else float("nan"), 2),
                "p99": _f(float(x.quantile(0.99)) if nn else float("nan"), 1),
            }
        )
    rows = sorted(rows, key=lambda r: r["period"])
    shares = [(r["|DSO|>24"] / r["nn"]) if r["nn"] else float("nan") for r in rows]
    finite = [s for s in shares if np.isfinite(s)]
    mx = max(finite) if finite else float("nan")
    mn = min(finite) if finite else float("nan")
    spike = max(rows, key=lambda r: r["|DSO|>24"] / r["nn"] if r["nn"] else -1)
    prose = (
        f"|DSO|>24 by month: min {_pp(mn)} max {_pp(mx)} on {spike['period']} "
        f"({spike['|DSO|>24']}/{spike['nn']}). "
        f"{'one month spike' if np.isfinite(mx) and np.isfinite(mn) and mx >= 2 * max(mn, 0.01) else 'tail is not a single-month spike'}."
    )
    print(prose)
    return {
        "rows": rows,
        "min": mn,
        "max": mx,
        "spike": spike["period"],
        "prose": prose,
    }


def pass_drop_aug(tr: pd.DataFrame) -> dict:
    """Leftover after dropping the 2026-08 |DSO|>24 spike month. Train only."""
    spike = pd.Timestamp("2026-08-01")
    keep = tr["period"] != spike
    resid_iss, _ = ols_resid(tr["e_dso_proxy"].where(keep), tr["e_ar_issued_lag1"].where(keep))
    resid_days, _ = ols_resid(tr["e_dso_proxy"].where(keep), tr["c_n_days_with_tx"].where(keep))
    resid_delay, _ = ols_resid(tr["e_dso_proxy"].where(keep), tr["e_delay_coll"].where(keep))
    rows = []
    store = {}
    for ycol, name, s in (
        (Y7, "Y7 leftover-iss drop-Aug", resid_iss),
        (Y7, "Y7 leftover-delay drop-Aug", resid_delay),
        (Y7, "Y7 DSO drop-Aug", tr["e_dso_proxy"]),
        (Y7, "Y7 issued drop-Aug", tr["e_ar_issued_lag1"]),
        (Y3, "Y3 leftover-days drop-Aug", resid_days),
        (Y3, "Y3 DSO drop-Aug", tr["e_dso_proxy"]),
        (Y3, "Y3 days drop-Aug", tr["c_n_days_with_tx"]),
    ):
        mask = keep & tr[ycol].notna()
        res = signed_oof_auroc(tr[ycol], s, tr["fold"], mask)
        store[name] = res
        rows.append(_auc_row(ycol, name, res, None))
    prose = (
        f"Drop 2026-08 spike: Y7 leftover-iss {_f(_cv(store['Y7 leftover-iss drop-Aug']))} "
        f"leftover-delay {_f(_cv(store['Y7 leftover-delay drop-Aug']))} issued {_f(_cv(store['Y7 issued drop-Aug']))}. "
        f"Y3 leftover-days {_f(_cv(store['Y3 leftover-days drop-Aug']))} days {_f(_cv(store['Y3 days drop-Aug']))}. "
        f"Spike month is last grid month — labeled leftover barely moves."
    )
    print(prose)
    return {
        "rows": rows,
        "y7_iss": _cv(store["Y7 leftover-iss drop-Aug"]),
        "y7_delay": _cv(store["Y7 leftover-delay drop-Aug"]),
        "y3_days": _cv(store["Y3 leftover-days drop-Aug"]),
        "prose": prose,
    }


def pass_y6_left(tr: pd.DataFrame) -> dict:
    """Unused leftover of DSO on Y6 (activity). Not a new Y. Train only."""
    resid_iss, info_i = ols_resid(tr["e_dso_proxy"], tr["e_ar_issued_lag1"])
    resid_days, info_d = ols_resid(tr["e_dso_proxy"], tr["c_n_days_with_tx"])
    rows = []
    store = {}
    for name, s in (
        ("Y6 DSO", tr["e_dso_proxy"]),
        ("Y6 leftover-iss", resid_iss),
        ("Y6 leftover-days", resid_days),
        ("Y6 issued_lag1", tr["e_ar_issued_lag1"]),
        ("Y6 days", tr["c_n_days_with_tx"]),
        ("Y6 size", tr["log_in3"]),
        ("Y6 clip24", tr["e_dso_clip24"]),
    ):
        res = signed_oof_auroc(tr[Y6Z], s, tr["fold"], tr[Y6Z].notna())
        store[name] = res
        rows.append(_auc_row(Y6Z, name, res, None))
    prose = (
        f"Y6 leftover (not a new Y): DSO {_f(_cv(store['Y6 DSO']))} leftover-iss "
        f"{_f(_cv(store['Y6 leftover-iss']))} leftover-days {_f(_cv(store['Y6 leftover-days']))} "
        f"issued {_f(_cv(store['Y6 issued_lag1']))} days {_f(_cv(store['Y6 days']))} "
        f"size {_f(_cv(store['Y6 size']))}. R²iss={_f(info_i['r2'])} R²days={_f(info_d['r2'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "dso": _cv(store["Y6 DSO"]),
        "left_iss": _cv(store["Y6 leftover-iss"]),
        "left_days": _cv(store["Y6 leftover-days"]),
        "iss": _cv(store["Y6 issued_lag1"]),
        "prose": prose,
    }


def pass_labeled_slope(tr: pd.DataFrame) -> dict:
    """Leftover slope fit only on labeled rows (Aug-2026 unlabeled tail cannot pull beta)."""
    rows = []
    store = {}
    infos = {}
    for ycol, bar, bar_name, tag in (
        (Y3, "c_n_days_with_tx", "days", "Y3 leftover-days labeled-slope"),
        (Y3, "log_in3", "size", "Y3 leftover-size labeled-slope"),
        (Y7, "e_ar_issued_lag1", "issued_lag1", "Y7 leftover-iss labeled-slope"),
        (Y7, "e_delay_coll", "delay", "Y7 leftover-delay labeled-slope"),
        (Y6Z, "e_ar_issued_lag1", "issued_lag1", "Y6 leftover-iss labeled-slope"),
    ):
        lab = tr[ycol].notna()
        resid, info = ols_resid(tr["e_dso_proxy"].where(lab), tr[bar].where(lab))
        infos[tag] = info
        res = signed_oof_auroc(tr[ycol], resid, tr["fold"], lab)
        store[tag] = res
        rows.append(_auc_row(ycol, tag, res, None))
        bar_res = signed_oof_auroc(tr[ycol], tr[bar], tr["fold"], lab)
        store[f"{tag} bar"] = bar_res
        rows.append(_auc_row(ycol, f"{bar_name} labeled", bar_res, None))
    y3 = _cv(store["Y3 leftover-days labeled-slope"])
    y7 = _cv(store["Y7 leftover-iss labeled-slope"])
    y7d = _cv(store["Y7 leftover-delay labeled-slope"])
    prose = (
        f"Labeled-slope leftover: Y3-days {_f(y3)} (days {_f(_cv(store['Y3 leftover-days labeled-slope bar']))}) "
        f"Y7-iss {_f(y7)} (issued {_f(_cv(store['Y7 leftover-iss labeled-slope bar']))}) "
        f"Y7-delay {_f(y7d)}. "
        f"{'Y3 leftover lives on labeled slope' if np.isfinite(y3) and y3 >= CHANCE else 'Y3 leftover dies on labeled slope'} "
        f"/ {'Y7 leftover lives' if np.isfinite(y7) and y7 >= CHANCE else 'Y7 leftover dies'}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3": y3,
        "y7": y7,
        "y7_delay": y7d,
        "prose": prose,
    }


def pass_rank_left(tr: pd.DataFrame) -> dict:
    """Rank(DSO) leftover — kills the 2e6 blow-up without rewriting the store."""
    rank = pd.to_numeric(tr["e_dso_proxy"], errors="coerce").rank(method="average")
    resid_days, info_d = ols_resid(rank, tr["c_n_days_with_tx"])
    resid_iss, info_i = ols_resid(rank, tr["e_ar_issued_lag1"])
    resid_delay, info_b = ols_resid(rank, tr["e_delay_coll"])
    rows = []
    store = {}
    for ycol, name, s in (
        (Y3, "Y3 rank(DSO)", rank),
        (Y3, "Y3 leftover rank-days", resid_days),
        (Y3, "Y3 days", tr["c_n_days_with_tx"]),
        (Y7, "Y7 rank(DSO)", rank),
        (Y7, "Y7 leftover rank-iss", resid_iss),
        (Y7, "Y7 leftover rank-delay", resid_delay),
        (Y7, "Y7 issued_lag1", tr["e_ar_issued_lag1"]),
    ):
        res = signed_oof_auroc(tr[ycol], s, tr["fold"], tr[ycol].notna())
        store[name] = res
        rows.append(_auc_row(ycol, name, res, None))
    prose = (
        f"Rank(DSO) leftover: Y3 {_f(_cv(store['Y3 leftover rank-days']))} vs days {_f(_cv(store['Y3 days']))} "
        f"raw-rank {_f(_cv(store['Y3 rank(DSO)']))}. "
        f"Y7 leftover-iss {_f(_cv(store['Y7 leftover rank-iss']))} leftover-delay "
        f"{_f(_cv(store['Y7 leftover rank-delay']))} issued {_f(_cv(store['Y7 issued_lag1']))}. "
        f"R²days={_f(info_d['r2'])} R²iss={_f(info_i['r2'])} R²delay={_f(info_b['r2'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3": _cv(store["Y3 leftover rank-days"]),
        "y3_raw": _cv(store["Y3 rank(DSO)"]),
        "y7": _cv(store["Y7 leftover rank-iss"]),
        "y7_delay": _cv(store["Y7 leftover rank-delay"]),
        "prose": prose,
    }


def pass_labeled_clone(tr: pd.DataFrame) -> dict:
    """Is labeled-slope leftover 0.697 just days again (lag1-clone pattern)?"""
    lab = tr[Y3].notna()
    resid, info = ols_resid(tr["e_dso_proxy"].where(lab), tr["c_n_days_with_tx"].where(lab))
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    rho_res, n_res = spearman_n(resid, days)
    rho_raw, n_raw = spearman_n(dso.where(lab), days.where(lab))
    clip = dso.clip(upper=WINSOR, lower=-WINSOR)
    resid_c, info_c = ols_resid(clip.where(lab), days.where(lab))
    rho_clip, _ = spearman_n(resid_c, days)
    res_left = signed_oof_auroc(tr[Y3], resid, tr["fold"], lab)
    res_days = signed_oof_auroc(tr[Y3], days, tr["fold"], lab)
    res_clip = signed_oof_auroc(tr[Y3], resid_c, tr["fold"], lab)
    clone = np.isfinite(rho_res) and abs(rho_res) >= 0.80
    false_clone = clone and np.isfinite(rho_raw) and abs(rho_raw) < 0.30
    rows = [
        {"slice": "resid vs days ρ", "n": n_res, "value": _f(rho_res)},
        {"slice": "raw DSO vs days ρ (labeled)", "n": n_raw, "value": _f(rho_raw)},
        {"slice": "clip24-resid vs days ρ", "n": n_res, "value": _f(rho_clip)},
        {"slice": "leftover AUROC", "n": res_left["n_defined"], "value": _f(_cv(res_left))},
        {"slice": "days AUROC", "n": res_days["n_defined"], "value": _f(_cv(res_days))},
        {"slice": "clip leftover AUROC", "n": res_clip["n_defined"], "value": _f(_cv(res_clip))},
        {"slice": "R² labeled", "n": info["n"], "value": _f(info["r2"])},
        {"slice": "R² clip labeled", "n": info_c["n"], "value": _f(info_c["r2"])},
    ]
    prose = (
        f"Labeled leftover vs days ρ={_f(rho_res)} n={n_res} (raw DSO-days {_f(rho_raw)}). "
        f"Leftover {_f(_cv(res_left))} days {_f(_cv(res_days))} clip-leftover {_f(_cv(res_clip))}. "
        f"{'FALSE clone (tail rank)' if false_clone else ('CLONE of days' if clone else 'not a clone')}."
    )
    print(prose)
    return {
        "rows": rows,
        "rho": rho_res,
        "rho_raw": rho_raw,
        "left": _cv(res_left),
        "days": _cv(res_days),
        "clip": _cv(res_clip),
        "clone": clone,
        "false_clone": false_clone,
        "prose": prose,
    }


def pass_interact(tr: pd.DataFrame) -> dict:
    """DSO × log1p(issued) leftover after issued — mix term, not a new stem."""
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    iss = pd.to_numeric(tr["e_ar_issued_lag1"], errors="coerce")
    mix = dso * np.log1p(iss.clip(lower=0))
    resid_dso, info_d = ols_resid(dso, iss)
    resid_mix, info_m = ols_resid(mix, iss)
    rows = []
    store = {}
    for ycol, name, s in (
        (Y7, "Y7 leftover DSO after iss", resid_dso),
        (Y7, "Y7 leftover mix after iss", resid_mix),
        (Y7, "Y7 mix raw", mix),
        (Y7, "Y7 issued", iss),
        (Y3, "Y3 leftover DSO after days", ols_resid(dso, tr["c_n_days_with_tx"])[0]),
        (Y3, "Y3 leftover mix after days", ols_resid(mix, tr["c_n_days_with_tx"])[0]),
        (Y3, "Y3 days", tr["c_n_days_with_tx"]),
    ):
        res = signed_oof_auroc(tr[ycol], s, tr["fold"], tr[ycol].notna())
        store[name] = res
        rows.append(_auc_row(ycol, name, res, None))
    prose = (
        f"Mix DSO×log1p(iss) leftover-iss {_f(_cv(store['Y7 leftover mix after iss']))} "
        f"vs DSO leftover {_f(_cv(store['Y7 leftover DSO after iss']))} issued {_f(_cv(store['Y7 issued']))}. "
        f"Y3 mix leftover-days {_f(_cv(store['Y3 leftover mix after days']))}. "
        f"R²dso={_f(info_d['r2'])} R²mix={_f(info_m['r2'])}. Do not grow TURNOVER."
    )
    print(prose)
    return {
        "rows": rows,
        "y7_mix": _cv(store["Y7 leftover mix after iss"]),
        "y7_dso": _cv(store["Y7 leftover DSO after iss"]),
        "y3_mix": _cv(store["Y3 leftover mix after days"]),
        "prose": prose,
    }


def pass_holdout_month(panel: pd.DataFrame) -> dict:
    """Holdout coverage of DSO / |DSO|>24 by month. No AUROC."""
    ho = panel[panel["split"] == "holdout"].copy()
    hold = set(load_holdout())
    if not set(ho["company_id"]).issubset(hold):
        raise RuntimeError("holdout month table leaked train companies")
    rows = []
    for p, sl in ho.groupby("period"):
        x = pd.to_numeric(sl["e_dso_proxy"], errors="coerce")
        nn = int(x.notna().sum())
        n = int(len(sl))
        n_gt = int((x.abs() > WINSOR).sum())
        rows.append(
            {
                "period": str(pd.Timestamp(p).date()),
                "n_cm": n,
                "nn": nn,
                "cov": _pp(nn / n if n else float("nan")),
                "|DSO|>24": n_gt,
                "share": _pp(n_gt / nn if nn else float("nan")),
            }
        )
    rows = sorted(rows, key=lambda r: r["period"])
    nn = int(pd.to_numeric(ho["e_dso_proxy"], errors="coerce").notna().sum())
    prose = (
        f"Holdout DSO by month (coverage only, no AUROC): nn={nn} / {len(ho)} CM. "
        f"Last-month spike is train-side 2026-08; holdout table is coverage."
    )
    print(prose)
    return {"rows": rows, "nn": nn, "prose": prose}


def pass_mix_clone(tr: pd.DataFrame) -> dict:
    """Y3 mix leftover 0.692 — days clone?"""
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    iss = pd.to_numeric(tr["e_ar_issued_lag1"], errors="coerce")
    mix = dso * np.log1p(iss.clip(lower=0))
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    resid, info = ols_resid(mix, days)
    rho, n = spearman_n(resid, days)
    rho_raw, n_raw = spearman_n(mix, days)
    left = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    days_auc = signed_oof_auroc(tr[Y3], days, tr["fold"], tr[Y3].notna())
    clone = np.isfinite(rho) and abs(rho) >= 0.80
    false_clone = clone and np.isfinite(rho_raw) and abs(rho_raw) < 0.30
    rows = [
        {"slice": "mix-resid vs days ρ", "n": n, "value": _f(rho)},
        {"slice": "mix vs days ρ", "n": n_raw, "value": _f(rho_raw)},
        {"slice": "mix leftover", "n": left["n_defined"], "value": _f(_cv(left))},
        {"slice": "days", "n": days_auc["n_defined"], "value": _f(_cv(days_auc))},
        {"slice": "R²", "n": info["n"], "value": _f(info["r2"])},
    ]
    prose = (
        f"Mix leftover vs days ρ={_f(rho)} (mix-days {_f(rho_raw)}). "
        f"Leftover {_f(_cv(left))} days {_f(_cv(days_auc))}. "
        f"{'FALSE clone (tail rank)' if false_clone else ('CLONE' if clone else 'not a clone')}."
    )
    print(prose)
    return {
        "rows": rows,
        "rho": rho,
        "rho_raw": rho_raw,
        "left": _cv(left),
        "false_clone": false_clone,
        "prose": prose,
    }


def pass_pending_left(tr: pd.DataFrame) -> dict:
    """Leftover after pending / AP overdue — not twins (ρ 0.492 / 0.288)."""
    resid_p, info_p = ols_resid(tr["e_dso_proxy"], tr["e_pending_amt_share"])
    resid_ap, info_a = ols_resid(tr["e_dso_proxy"], tr["e_ap_overdue"])
    resid_pd, info_pd = ols_resid(
        tr["e_dso_proxy"], tr["e_pending_amt_share"], tr["c_n_days_with_tx"]
    )
    rows = []
    store = {}
    for ycol, name, s in (
        (Y7, "Y7 leftover pending", resid_p),
        (Y7, "Y7 leftover AP od", resid_ap),
        (Y7, "Y7 pending", tr["e_pending_amt_share"]),
        (Y3, "Y3 leftover pending", resid_p),
        (Y3, "Y3 leftover pending+days", resid_pd),
        (Y3, "Y3 leftover AP od", resid_ap),
        (Y3, "Y3 days", tr["c_n_days_with_tx"]),
        (Y3, "Y3 pending", tr["e_pending_amt_share"]),
    ):
        res = signed_oof_auroc(tr[ycol], s, tr["fold"], tr[ycol].notna())
        store[name] = res
        rows.append(_auc_row(ycol, name, res, None))
    prose = (
        f"Leftover after pending: Y7 {_f(_cv(store['Y7 leftover pending']))} "
        f"Y3 {_f(_cv(store['Y3 leftover pending']))} pending+days {_f(_cv(store['Y3 leftover pending+days']))}. "
        f"After AP overdue Y7 {_f(_cv(store['Y7 leftover AP od']))} Y3 {_f(_cv(store['Y3 leftover AP od']))}. "
        f"R²pend={_f(info_p['r2'])} R²ap={_f(info_a['r2'])} R²pd={_f(info_pd['r2'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "y7_p": _cv(store["Y7 leftover pending"]),
        "y3_p": _cv(store["Y3 leftover pending"]),
        "y3_pd": _cv(store["Y3 leftover pending+days"]),
        "y7_ap": _cv(store["Y7 leftover AP od"]),
        "prose": prose,
    }


def pass_more_bars(tr: pd.DataFrame) -> dict:
    """Leftover after delay_paid / zero_in / f_ds_r — unused E/C/F bars."""
    resid_paid, i1 = ols_resid(tr["e_dso_proxy"], tr["e_delay_paid"])
    resid_z, i2 = ols_resid(tr["e_dso_proxy"], tr["c_zero_in_month"])
    resid_f, i3 = ols_resid(tr["e_dso_proxy"], tr["f_ds_r"])
    rows = []
    store = {}
    for ycol, name, s in (
        (Y7, "Y7 leftover delay_paid", resid_paid),
        (Y7, "Y7 leftover zero_in", resid_z),
        (Y7, "Y7 leftover f_ds_r", resid_f),
        (Y7, "Y7 delay_paid", tr["e_delay_paid"]),
        (Y3, "Y3 leftover delay_paid", resid_paid),
        (Y3, "Y3 leftover zero_in", resid_z),
        (Y3, "Y3 leftover f_ds_r", resid_f),
        (Y3, "Y3 leftover days (honest)", ols_resid(tr["e_dso_proxy"], tr["c_n_days_with_tx"])[0]),
    ):
        res = signed_oof_auroc(tr[ycol], s, tr["fold"], tr[ycol].notna())
        store[name] = res
        rows.append(_auc_row(ycol, name, res, None))
    prose = (
        f"More bars leftover: delay_paid Y7 {_f(_cv(store['Y7 leftover delay_paid']))} "
        f"Y3 {_f(_cv(store['Y3 leftover delay_paid']))}; "
        f"zero_in Y7 {_f(_cv(store['Y7 leftover zero_in']))} Y3 {_f(_cv(store['Y3 leftover zero_in']))}; "
        f"f_ds_r Y7 {_f(_cv(store['Y7 leftover f_ds_r']))} Y3 {_f(_cv(store['Y3 leftover f_ds_r']))}. "
        f"R²paid={_f(i1['r2'])} R²z={_f(i2['r2'])} R²f={_f(i3['r2'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "y7_paid": _cv(store["Y7 leftover delay_paid"]),
        "y3_paid": _cv(store["Y3 leftover delay_paid"]),
        "y7_z": _cv(store["Y7 leftover zero_in"]),
        "y3_z": _cv(store["Y3 leftover zero_in"]),
        "y7_f": _cv(store["Y7 leftover f_ds_r"]),
        "y3_f": _cv(store["Y3 leftover f_ds_r"]),
        "prose": prose,
    }


def pass_zero_clone(tr: pd.DataFrame) -> dict:
    """Y3 leftover after zero_in 0.574 — days clone?"""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["c_zero_in_month"])
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    z = pd.to_numeric(tr["c_zero_in_month"], errors="coerce")
    rho, n = spearman_n(resid, days)
    rho_z, _ = spearman_n(resid, z)
    left = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    days_auc = signed_oof_auroc(tr[Y3], days, tr["fold"], tr[Y3].notna())
    clone = np.isfinite(rho) and abs(rho) >= 0.80
    rows = [
        {"slice": "zero-resid vs days ρ", "n": n, "value": _f(rho)},
        {"slice": "zero-resid vs zero_in ρ", "n": n, "value": _f(rho_z)},
        {"slice": "leftover", "n": left["n_defined"], "value": _f(_cv(left))},
        {"slice": "days", "n": days_auc["n_defined"], "value": _f(_cv(days_auc))},
        {"slice": "R²", "n": info["n"], "value": _f(info["r2"])},
    ]
    prose = (
        f"Zero_in leftover vs days ρ={_f(rho)} vs zero_in {_f(rho_z)}. "
        f"Leftover {_f(_cv(left))} days {_f(_cv(days_auc))}. "
        f"{'CLONE of days' if clone else 'not a days clone — still dies vs days / no KEEP'}."
    )
    print(prose)
    return {
        "rows": rows,
        "rho": rho,
        "left": _cv(left),
        "clone": clone,
        "prose": prose,
    }


def pass_zero_days(tr: pd.DataFrame) -> dict:
    """Leftover after zero_in+days — 0.574 must die on the honest Y3 bar."""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["c_zero_in_month"], tr["c_n_days_with_tx"])
    res = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna())
    z = signed_oof_auroc(tr[Y3], tr["c_zero_in_month"], tr["fold"], tr[Y3].notna())
    rows = [
        _auc_row(Y3, "leftover zero_in+days", res, None),
        _auc_row(Y3, "days", days, None),
        _auc_row(Y3, "zero_in", z, None),
    ]
    prose = (
        f"Leftover after zero_in+days {_f(_cv(res))} vs days {_f(_cv(days))} "
        f"zero_in {_f(_cv(z))} R²={_f(info['r2'])}. "
        f"{'dies on the honest bar' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — still loses to days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "days": _cv(days),
        "prose": prose,
    }


def pass_zd_clone(tr: pd.DataFrame) -> dict:
    """Is leftover after zero_in+days 0.624 a days clone?"""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["c_zero_in_month"], tr["c_n_days_with_tx"])
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    rho, n = spearman_n(resid, days)
    left = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    clone = np.isfinite(rho) and abs(rho) >= 0.80
    rows = [
        {"slice": "resid vs days ρ", "n": n, "value": _f(rho)},
        {"slice": "leftover", "n": left["n_defined"], "value": _f(_cv(left))},
        {"slice": "R²", "n": info["n"], "value": _f(info["r2"])},
    ]
    prose = (
        f"zero_in+days leftover vs days ρ={_f(rho)}. leftover {_f(_cv(left))}. "
        f"{'FALSE clone (tail rank)' if clone else 'not a clone — still loses to days'}."
    )
    print(prose)
    return {"rows": rows, "rho": rho, "left": _cv(left), "clone": clone, "prose": prose}


def pass_iss_days(tr: pd.DataFrame) -> dict:
    """Leftover after issued_lag1+days on Y3 — joint bar, not KEEP if it dies or loses days."""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["e_ar_issued_lag1"], tr["c_n_days_with_tx"])
    res = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna())
    iss = signed_oof_auroc(tr[Y3], tr["e_ar_issued_lag1"], tr["fold"], tr[Y3].notna())
    rho, n = spearman_n(resid, tr["c_n_days_with_tx"])
    rows = [
        _auc_row(Y3, "leftover iss+days", res, None),
        _auc_row(Y3, "days", days, None),
        _auc_row(Y3, "issued_lag1", iss, None),
        {"slice": "resid vs days ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Leftover after issued+days {_f(_cv(res))} vs days {_f(_cv(days))} "
        f"issued {_f(_cv(iss))} ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — still loses to days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "days": _cv(days),
        "rho": rho,
        "prose": prose,
    }


def pass_clip_iss_days(tr: pd.DataFrame) -> dict:
    """Clip24 leftover after issued+days — 0.691 should die if it is the unclipped clone."""
    resid, info = ols_resid(
        tr["e_dso_clip24"], tr["e_ar_issued_lag1"], tr["c_n_days_with_tx"]
    )
    res = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna())
    rho, n = spearman_n(resid, tr["c_n_days_with_tx"])
    rows = [
        _auc_row(Y3, "clip leftover iss+days", res, None),
        _auc_row(Y3, "days", days, None),
        {"slice": "resid vs days ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Clip24 leftover after issued+days {_f(_cv(res))} vs days {_f(_cv(days))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'0.691 dies under clip' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'clip leftover still lives — still loses to days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "days": _cv(days),
        "rho": rho,
        "prose": prose,
    }


def pass_fds_days(tr: pd.DataFrame) -> dict:
    """Leftover after f_ds_r+days. Y3 leftover after f_ds_r was 0.566."""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["f_ds_r"], tr["c_n_days_with_tx"])
    res = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna())
    rho, n = spearman_n(resid, tr["c_n_days_with_tx"])
    rows = [
        _auc_row(Y3, "leftover f_ds_r+days", res, None),
        _auc_row(Y3, "days", days, None),
        {"slice": "resid vs days ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Leftover after f_ds_r+days {_f(_cv(res))} vs days {_f(_cv(days))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — still loses to days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "days": _cv(days),
        "rho": rho,
        "prose": prose,
    }


def pass_paid_days(tr: pd.DataFrame) -> dict:
    """Leftover after delay_paid+days. Y3 leftover after delay_paid was 0.376."""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["e_delay_paid"], tr["c_n_days_with_tx"])
    res = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna())
    rho, n = spearman_n(resid, tr["c_n_days_with_tx"])
    rows = [
        _auc_row(Y3, "leftover delay_paid+days", res, None),
        _auc_row(Y3, "days", days, None),
        {"slice": "resid vs days ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Leftover after delay_paid+days {_f(_cv(res))} vs days {_f(_cv(days))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — still loses to days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "days": _cv(days),
        "rho": rho,
        "prose": prose,
    }


def pass_ap_days(tr: pd.DataFrame) -> dict:
    """Leftover after AP overdue+days. Y3 leftover after AP was 0.547."""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["e_ap_overdue"], tr["c_n_days_with_tx"])
    res = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna())
    rho, n = spearman_n(resid, tr["c_n_days_with_tx"])
    rows = [
        _auc_row(Y3, "leftover AP+days", res, None),
        _auc_row(Y3, "days", days, None),
        {"slice": "resid vs days ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Leftover after AP overdue+days {_f(_cv(res))} vs days {_f(_cv(days))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — still loses to days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "days": _cv(days),
        "rho": rho,
        "prose": prose,
    }


def pass_dpo_days(tr: pd.DataFrame) -> dict:
    """Leftover after DPO+days. DPO already DROP from 44; is unused DSO just AP/DPO?"""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["e_dpo_proxy"], tr["c_n_days_with_tx"])
    res = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna())
    rho, n = spearman_n(resid, tr["c_n_days_with_tx"])
    rows = [
        _auc_row(Y3, "leftover DPO+days", res, None),
        _auc_row(Y3, "days", days, None),
        {"slice": "resid vs days ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Leftover after DPO+days {_f(_cv(res))} vs days {_f(_cv(days))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — still loses to days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "days": _cv(days),
        "rho": rho,
        "prose": prose,
    }


def pass_od_days(tr: pd.DataFrame) -> dict:
    """Leftover after AR overdue+days. Y3 leftover after overdue was 0.374."""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["e_ar_overdue"], tr["c_n_days_with_tx"])
    res = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna())
    rho, n = spearman_n(resid, tr["c_n_days_with_tx"])
    rows = [
        _auc_row(Y3, "leftover overdue+days", res, None),
        _auc_row(Y3, "days", days, None),
        {"slice": "resid vs days ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Leftover after AR overdue+days {_f(_cv(res))} vs days {_f(_cv(days))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — still loses to days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "days": _cv(days),
        "rho": rho,
        "prose": prose,
    }


def pass_clip_paid_days(tr: pd.DataFrame) -> dict:
    """Clip24 leftover after delay_paid+days. Raw leftover 0.624 — tail?"""
    resid, info = ols_resid(tr["e_dso_clip24"], tr["e_delay_paid"], tr["c_n_days_with_tx"])
    res = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna())
    rho, n = spearman_n(resid, tr["c_n_days_with_tx"])
    rows = [
        _auc_row(Y3, "clip leftover delay_paid+days", res, None),
        _auc_row(Y3, "days", days, None),
        {"slice": "resid vs days ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Clip24 leftover after delay_paid+days {_f(_cv(res))} vs days {_f(_cv(days))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — still loses to days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "days": _cv(days),
        "rho": rho,
        "prose": prose,
    }


def pass_clip_dpo_days(tr: pd.DataFrame) -> dict:
    """Clip24 leftover after DPO+days. Raw leftover 0.647 ρ=0.807 days-clone?"""
    resid, info = ols_resid(tr["e_dso_clip24"], tr["e_dpo_proxy"], tr["c_n_days_with_tx"])
    res = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna())
    rho, n = spearman_n(resid, tr["c_n_days_with_tx"])
    raw_rho, _ = spearman_n(tr["e_dso_proxy"], tr["c_n_days_with_tx"])
    clone = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    rows = [
        _auc_row(Y3, "clip leftover DPO+days", res, None),
        _auc_row(Y3, "days", days, None),
        {"slice": "resid vs days ρ", "n": n, "value": _f(rho)},
        {"slice": "raw DSO vs days ρ", "n": n, "value": _f(raw_rho)},
    ]
    prose = (
        f"Clip24 leftover after DPO+days {_f(_cv(res))} vs days {_f(_cv(days))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])} raw DSO-days {_f(raw_rho)}. "
        f"{'FALSE clone' if clone else 'not a days clone'} — "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'still loses to days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "days": _cv(days),
        "rho": rho,
        "clone": clone,
        "prose": prose,
    }


def pass_open_days(tr: pd.DataFrame) -> dict:
    """Leftover after AR open+days. DSO vs open ρ=0.600 — leftover of the stock?"""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["e_ar_open"], tr["c_n_days_with_tx"])
    res = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna())
    rho, n = spearman_n(resid, tr["c_n_days_with_tx"])
    rows = [
        _auc_row(Y3, "leftover open+days", res, None),
        _auc_row(Y3, "days", days, None),
        {"slice": "resid vs days ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Leftover after AR open+days {_f(_cv(res))} vs days {_f(_cv(days))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — still loses to days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "days": _cv(days),
        "rho": rho,
        "prose": prose,
    }


def pass_delay_days(tr: pd.DataFrame) -> dict:
    """Leftover after delay_coll+days on Y3. Honest leftover after days already dies."""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["e_delay_coll"], tr["c_n_days_with_tx"])
    res = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna())
    rho, n = spearman_n(resid, tr["c_n_days_with_tx"])
    rows = [
        _auc_row(Y3, "leftover delay+days", res, None),
        _auc_row(Y3, "days", days, None),
        {"slice": "resid vs days ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Leftover after delay_coll+days {_f(_cv(res))} vs days {_f(_cv(days))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — still loses to days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "days": _cv(days),
        "rho": rho,
        "prose": prose,
    }


def pass_clip_delay_days(tr: pd.DataFrame) -> dict:
    """Clip24 leftover after delay_coll+days. Raw leftover 0.647 ρ=0.878 days-clone."""
    resid, info = ols_resid(tr["e_dso_clip24"], tr["e_delay_coll"], tr["c_n_days_with_tx"])
    res = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna())
    rho, n = spearman_n(resid, tr["c_n_days_with_tx"])
    clone = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    rows = [
        _auc_row(Y3, "clip leftover delay+days", res, None),
        _auc_row(Y3, "days", days, None),
        {"slice": "resid vs days ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Clip24 leftover after delay_coll+days {_f(_cv(res))} vs days {_f(_cv(days))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'FALSE clone' if clone else 'not a days clone'} — "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'still loses to days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "days": _cv(days),
        "rho": rho,
        "clone": clone,
        "prose": prose,
    }


def pass_holdout_size(panel: pd.DataFrame) -> dict:
    """Holdout DSO coverage by SIZE tercile. Coverage only — no AUROC."""
    ho = panel[panel["split"] == "holdout"].copy()
    assert set(ho["company_id"]) <= load_holdout()
    size = pd.to_numeric(ho["log_in3"], errors="coerce")
    dso = pd.to_numeric(ho["e_dso_proxy"], errors="coerce")
    defined = size.notna()
    terc = pd.Series(np.nan, index=ho.index, dtype=object)
    terc.loc[defined] = pd.qcut(size[defined].rank(method="first"), 3, labels=["T1", "T2", "T3"])
    rows = []
    store = {}
    for label in ("T1", "T2", "T3"):
        m = terc == label
        nn = int(dso[m].notna().sum())
        n = int(m.sum())
        gt = int((dso[m].abs() > WINSOR).sum())
        cov = _pct(nn, n)
        store[label] = cov
        rows.append(
            {
                "slice": f"holdout {label}",
                "n": n,
                "nn": nn,
                "cov": _pp(cov),
                "|DSO|>24": _pp(_pct(gt, nn)),
            }
        )
    prose = (
        f"Holdout SIZE tercile coverage only: T1 {_pp(store['T1'])} "
        f"T2 {_pp(store['T2'])} T3 {_pp(store['T3'])}. No AUROC."
    )
    print(prose)
    return {
        "rows": rows,
        "t1": store["T1"],
        "t2": store["T2"],
        "t3": store["T3"],
        "prose": prose,
    }


def pass_y7_open_iss(tr: pd.DataFrame) -> dict:
    """Y7 leftover after AR open+issued_lag1. Open leftover-issued was 0.596."""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["e_ar_open"], tr["e_ar_issued_lag1"])
    res = signed_oof_auroc(tr[Y7], resid, tr["fold"], tr[Y7].notna())
    iss = signed_oof_auroc(tr[Y7], tr["e_ar_issued_lag1"], tr["fold"], tr[Y7].notna())
    rho, n = spearman_n(resid, tr["e_ar_issued_lag1"])
    rows = [
        _auc_row(Y7, "leftover open+issued", res, None),
        _auc_row(Y7, "issued_lag1", iss, None),
        {"slice": "resid vs issued ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Y7 leftover after open+issued {_f(_cv(res))} vs issued {_f(_cv(iss))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — do not grow TURNOVER'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "iss": _cv(iss),
        "rho": rho,
        "prose": prose,
    }


def pass_pend_iss(tr: pd.DataFrame) -> dict:
    """Y7 leftover after pending+issued. Y7 leftover after pending was 0.431."""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["e_pending_amt_share"], tr["e_ar_issued_lag1"])
    res = signed_oof_auroc(tr[Y7], resid, tr["fold"], tr[Y7].notna())
    iss = signed_oof_auroc(tr[Y7], tr["e_ar_issued_lag1"], tr["fold"], tr[Y7].notna())
    rho, n = spearman_n(resid, tr["e_ar_issued_lag1"])
    rows = [
        _auc_row(Y7, "leftover pending+issued", res, None),
        _auc_row(Y7, "issued_lag1", iss, None),
        {"slice": "resid vs issued ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Y7 leftover after pending+issued {_f(_cv(res))} vs issued {_f(_cv(iss))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — do not grow TURNOVER'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "iss": _cv(iss),
        "rho": rho,
        "prose": prose,
    }


def pass_ap_iss(tr: pd.DataFrame) -> dict:
    """Y7 leftover after AP overdue+issued. Y7 leftover after AP was 0.449."""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["e_ap_overdue"], tr["e_ar_issued_lag1"])
    res = signed_oof_auroc(tr[Y7], resid, tr["fold"], tr[Y7].notna())
    iss = signed_oof_auroc(tr[Y7], tr["e_ar_issued_lag1"], tr["fold"], tr[Y7].notna())
    rho, n = spearman_n(resid, tr["e_ar_issued_lag1"])
    rows = [
        _auc_row(Y7, "leftover AP+issued", res, None),
        _auc_row(Y7, "issued_lag1", iss, None),
        {"slice": "resid vs issued ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Y7 leftover after AP+issued {_f(_cv(res))} vs issued {_f(_cv(iss))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — do not grow TURNOVER'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "iss": _cv(iss),
        "rho": rho,
        "prose": prose,
    }


def pass_clip_fds_days(tr: pd.DataFrame) -> dict:
    """Clip24 leftover after f_ds_r+days. Raw leftover 0.675 ρ=-0.937 days-clone."""
    resid, info = ols_resid(tr["e_dso_clip24"], tr["f_ds_r"], tr["c_n_days_with_tx"])
    res = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna())
    rho, n = spearman_n(resid, tr["c_n_days_with_tx"])
    clone = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    rows = [
        _auc_row(Y3, "clip leftover f_ds_r+days", res, None),
        _auc_row(Y3, "days", days, None),
        {"slice": "resid vs days ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Clip24 leftover after f_ds_r+days {_f(_cv(res))} vs days {_f(_cv(days))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'FALSE clone' if clone else 'not a days clone'} — "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'still loses to days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "days": _cv(days),
        "rho": rho,
        "clone": clone,
        "prose": prose,
    }


def pass_dpo_iss(tr: pd.DataFrame) -> dict:
    """Y7 leftover after DPO+issued. DPO already DROP; leftover after DPO was 0.473."""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["e_dpo_proxy"], tr["e_ar_issued_lag1"])
    res = signed_oof_auroc(tr[Y7], resid, tr["fold"], tr[Y7].notna())
    iss = signed_oof_auroc(tr[Y7], tr["e_ar_issued_lag1"], tr["fold"], tr[Y7].notna())
    rho, n = spearman_n(resid, tr["e_ar_issued_lag1"])
    rows = [
        _auc_row(Y7, "leftover DPO+issued", res, None),
        _auc_row(Y7, "issued_lag1", iss, None),
        {"slice": "resid vs issued ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Y7 leftover after DPO+issued {_f(_cv(res))} vs issued {_f(_cv(iss))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — do not grow TURNOVER'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "iss": _cv(iss),
        "rho": rho,
        "prose": prose,
    }


def pass_clip_zero_days(tr: pd.DataFrame) -> dict:
    """Clip24 leftover after zero_in+days. Raw leftover 0.624 ρ=0.746."""
    resid, info = ols_resid(tr["e_dso_clip24"], tr["c_zero_in_month"], tr["c_n_days_with_tx"])
    res = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna())
    rho, n = spearman_n(resid, tr["c_n_days_with_tx"])
    clone = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    rows = [
        _auc_row(Y3, "clip leftover zero_in+days", res, None),
        _auc_row(Y3, "days", days, None),
        {"slice": "resid vs days ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Clip24 leftover after zero_in+days {_f(_cv(res))} vs days {_f(_cv(days))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'FALSE clone' if clone else 'not a days clone'} — "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'still loses to days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "days": _cv(days),
        "rho": rho,
        "clone": clone,
        "prose": prose,
    }


def pass_od_iss(tr: pd.DataFrame) -> dict:
    """Y7 leftover after AR overdue+issued. Leftover after overdue was 0.524."""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["e_ar_overdue"], tr["e_ar_issued_lag1"])
    res = signed_oof_auroc(tr[Y7], resid, tr["fold"], tr[Y7].notna())
    iss = signed_oof_auroc(tr[Y7], tr["e_ar_issued_lag1"], tr["fold"], tr[Y7].notna())
    rho, n = spearman_n(resid, tr["e_ar_issued_lag1"])
    rows = [
        _auc_row(Y7, "leftover overdue+issued", res, None),
        _auc_row(Y7, "issued_lag1", iss, None),
        {"slice": "resid vs issued ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Y7 leftover after overdue+issued {_f(_cv(res))} vs issued {_f(_cv(iss))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — do not grow TURNOVER'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "iss": _cv(iss),
        "rho": rho,
        "prose": prose,
    }


def pass_paid_iss(tr: pd.DataFrame) -> dict:
    """Y7 leftover after delay_paid+issued. Y7 leftover after delay_paid was 0.471."""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["e_delay_paid"], tr["e_ar_issued_lag1"])
    res = signed_oof_auroc(tr[Y7], resid, tr["fold"], tr[Y7].notna())
    iss = signed_oof_auroc(tr[Y7], tr["e_ar_issued_lag1"], tr["fold"], tr[Y7].notna())
    rho, n = spearman_n(resid, tr["e_ar_issued_lag1"])
    rows = [
        _auc_row(Y7, "leftover delay_paid+issued", res, None),
        _auc_row(Y7, "issued_lag1", iss, None),
        {"slice": "resid vs issued ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Y7 leftover after delay_paid+issued {_f(_cv(res))} vs issued {_f(_cv(iss))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — do not grow TURNOVER'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "iss": _cv(iss),
        "rho": rho,
        "prose": prose,
    }


def pass_size_iss(tr: pd.DataFrame) -> dict:
    """Y7 leftover after size+issued. Leftover after size was 0.462."""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["log_in3"], tr["e_ar_issued_lag1"])
    res = signed_oof_auroc(tr[Y7], resid, tr["fold"], tr[Y7].notna())
    iss = signed_oof_auroc(tr[Y7], tr["e_ar_issued_lag1"], tr["fold"], tr[Y7].notna())
    rho, n = spearman_n(resid, tr["e_ar_issued_lag1"])
    rows = [
        _auc_row(Y7, "leftover size+issued", res, None),
        _auc_row(Y7, "issued_lag1", iss, None),
        {"slice": "resid vs issued ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Y7 leftover after size+issued {_f(_cv(res))} vs issued {_f(_cv(iss))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — do not grow TURNOVER'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "iss": _cv(iss),
        "rho": rho,
        "prose": prose,
    }


def pass_fds_iss(tr: pd.DataFrame) -> dict:
    """Y7 leftover after f_ds_r+issued. Y7 leftover after f_ds_r was 0.424."""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["f_ds_r"], tr["e_ar_issued_lag1"])
    res = signed_oof_auroc(tr[Y7], resid, tr["fold"], tr[Y7].notna())
    iss = signed_oof_auroc(tr[Y7], tr["e_ar_issued_lag1"], tr["fold"], tr[Y7].notna())
    rho, n = spearman_n(resid, tr["e_ar_issued_lag1"])
    rows = [
        _auc_row(Y7, "leftover f_ds_r+issued", res, None),
        _auc_row(Y7, "issued_lag1", iss, None),
        {"slice": "resid vs issued ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Y7 leftover after f_ds_r+issued {_f(_cv(res))} vs issued {_f(_cv(iss))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — do not grow TURNOVER'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "iss": _cv(iss),
        "rho": rho,
        "prose": prose,
    }


def pass_apiss_days(tr: pd.DataFrame) -> dict:
    """Leftover after AP issued+days. AP issued is the AP flow twin of AR issued."""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["e_ap_issued"], tr["c_n_days_with_tx"])
    res = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna())
    rho, n = spearman_n(resid, tr["c_n_days_with_tx"])
    rows = [
        _auc_row(Y3, "leftover AP issued+days", res, None),
        _auc_row(Y3, "days", days, None),
        {"slice": "resid vs days ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Leftover after AP issued+days {_f(_cv(res))} vs days {_f(_cv(days))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — still loses to days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "days": _cv(days),
        "rho": rho,
        "prose": prose,
    }


def pass_od30_iss(tr: pd.DataFrame) -> dict:
    """Y7 leftover after overdue_30+issued. Leftover after od30 was 0.477."""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["e_ar_overdue_30"], tr["e_ar_issued_lag1"])
    res = signed_oof_auroc(tr[Y7], resid, tr["fold"], tr[Y7].notna())
    iss = signed_oof_auroc(tr[Y7], tr["e_ar_issued_lag1"], tr["fold"], tr[Y7].notna())
    rho, n = spearman_n(resid, tr["e_ar_issued_lag1"])
    rows = [
        _auc_row(Y7, "leftover od30+issued", res, None),
        _auc_row(Y7, "issued_lag1", iss, None),
        {"slice": "resid vs issued ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Y7 leftover after od30+issued {_f(_cv(res))} vs issued {_f(_cv(iss))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — do not grow TURNOVER'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "iss": _cv(iss),
        "rho": rho,
        "prose": prose,
    }


def pass_apiss_iss(tr: pd.DataFrame) -> dict:
    """Y7 leftover after AP issued+issued. AP flow is the payable twin of AR issued."""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["e_ap_issued"], tr["e_ar_issued_lag1"])
    res = signed_oof_auroc(tr[Y7], resid, tr["fold"], tr[Y7].notna())
    iss = signed_oof_auroc(tr[Y7], tr["e_ar_issued_lag1"], tr["fold"], tr[Y7].notna())
    rho, n = spearman_n(resid, tr["e_ar_issued_lag1"])
    rows = [
        _auc_row(Y7, "leftover AP issued+issued", res, None),
        _auc_row(Y7, "issued_lag1", iss, None),
        {"slice": "resid vs issued ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Y7 leftover after AP issued+issued {_f(_cv(res))} vs issued {_f(_cv(iss))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — do not grow TURNOVER'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "iss": _cv(iss),
        "rho": rho,
        "prose": prose,
    }


def pass_zero_iss(tr: pd.DataFrame) -> dict:
    """Y7 leftover after zero_in+issued. Y7 leftover after zero_in was 0.392."""
    resid, info = ols_resid(tr["e_dso_proxy"], tr["c_zero_in_month"], tr["e_ar_issued_lag1"])
    res = signed_oof_auroc(tr[Y7], resid, tr["fold"], tr[Y7].notna())
    iss = signed_oof_auroc(tr[Y7], tr["e_ar_issued_lag1"], tr["fold"], tr[Y7].notna())
    rho, n = spearman_n(resid, tr["e_ar_issued_lag1"])
    rows = [
        _auc_row(Y7, "leftover zero_in+issued", res, None),
        _auc_row(Y7, "issued_lag1", iss, None),
        {"slice": "resid vs issued ρ", "n": n, "value": _f(rho)},
    ]
    prose = (
        f"Y7 leftover after zero_in+issued {_f(_cv(res))} vs issued {_f(_cv(iss))} "
        f"ρ={_f(rho)} R²={_f(info['r2'])}. "
        f"{'dies' if not np.isfinite(_cv(res)) or _cv(res) < CHANCE else 'lives — do not grow TURNOVER'}."
    )
    print(prose)
    return {
        "rows": rows,
        "left": _cv(res),
        "iss": _cv(iss),
        "rho": rho,
        "prose": prose,
    }


def pass_rolling_delay(tr: pd.DataFrame) -> dict:
    """Y7 at t+3 leftover after delay_coll (is SHAP #1 delay on the origin path?)."""
    origins = rolling_origins(tr["period"], horizon=3, n_last=9)
    y = pd.to_numeric(tr[Y7], errors="coerce")
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    delay = pd.to_numeric(tr["e_delay_coll"], errors="coerce")
    rows = []
    aucs_raw = []
    aucs_left = []
    aucs_delay = []
    for o in origins:
        t = o["t"]
        pred = o["predict_period"]
        past = tr["period"] < t
        now = tr["period"] == t
        lab_past = past & y.notna()
        feat = tr.loc[now, ["company_id", "e_dso_proxy", "e_delay_coll"]].copy()
        lab = tr.loc[tr["period"] == pred, ["company_id", Y7]].copy()
        m = feat.merge(lab, on="company_id", how="inner")
        y_now = pd.to_numeric(m[Y7], errors="coerce")
        dso_now = pd.to_numeric(m["e_dso_proxy"], errors="coerce")
        delay_now = pd.to_numeric(m["e_delay_coll"], errors="coerce")
        n_pos = int((y_now == 1).sum())
        n = int(y_now.notna().sum())
        if n_pos < MIN_POS:
            rows.append(
                {
                    "t": str(pd.Timestamp(t).date()),
                    "pred": str(pd.Timestamp(pred).date()),
                    "n": n,
                    "n_pos": n_pos,
                    "DSO": "LOW_POWER",
                    "delay": "LOW_POWER",
                    "leftover": "LOW_POWER",
                }
            )
            continue
        fit = past & dso.notna() & delay.notna()
        resid_now = pd.Series(np.nan, index=m.index, dtype=float)
        if int(fit.sum()) >= 25:
            Yf = dso[fit].to_numpy(dtype=float)
            Xf = np.column_stack(
                [np.ones(int(fit.sum())), delay[fit].to_numpy(dtype=float)]
            )
            beta, _, _, _ = np.linalg.lstsq(Xf, Yf, rcond=None)
            apply = dso_now.notna() & delay_now.notna()
            resid_now.loc[apply] = dso_now[apply] - (
                beta[0] + beta[1] * delay_now[apply]
            )
            resid_past = pd.Series(np.nan, index=tr.index, dtype=float)
            resid_past.loc[fit] = Yf - (Xf @ beta)
            sign_r = choose_sign(y[lab_past], resid_past[lab_past])
        else:
            sign_r = 1
        sign_d = choose_sign(y[lab_past], dso[lab_past])
        raw = auroc(y_now, sign_d * dso_now)
        sign_i = choose_sign(y[lab_past], delay[lab_past])
        delay_auc = auroc(y_now, sign_i * delay_now)
        left = auroc(y_now, sign_r * resid_now)
        aucs_raw.append(raw)
        aucs_delay.append(delay_auc)
        aucs_left.append(left)
        rows.append(
            {
                "t": str(pd.Timestamp(t).date()),
                "pred": str(pd.Timestamp(pred).date()),
                "n": n,
                "n_pos": n_pos,
                "DSO": _f(raw),
                "delay": _f(delay_auc),
                "leftover": _f(left),
            }
        )
    mean_raw = float(np.nanmean(aucs_raw)) if aucs_raw else float("nan")
    mean_delay = float(np.nanmean(aucs_delay)) if aucs_delay else float("nan")
    mean_left = float(np.nanmean(aucs_left)) if aucs_left else float("nan")
    prose = (
        f"Rolling Y7 t+3 leftover-delay: mean DSO {_f(mean_raw)} delay {_f(mean_delay)} "
        f"leftover {_f(mean_left)}. "
        f"{'leftover dies on the origin path' if not np.isfinite(mean_left) or mean_left < CHANCE else 'leftover lives on origins — still a tail, do not KEEP'}."
    )
    print(prose)
    return {
        "rows": rows,
        "raw": mean_raw,
        "delay": mean_delay,
        "left": mean_left,
        "prose": prose,
    }


def pass_clean_double(tr: pd.DataFrame) -> dict:
    """Leftover after the honest bar on |DSO|≤24 AND issued>p10. Train only."""
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    p10 = float(iss[iss.notna() & (iss > 0)].quantile(0.10)) if iss.notna().any() else float("nan")
    clean = dso.notna() & (dso.abs() <= WINSOR) & iss.notna() & (iss > p10)
    rows = []
    store = {}
    resid_iss, info_i = ols_resid(tr["e_dso_proxy"], tr["e_ar_issued_lag1"])
    resid_days, info_d = ols_resid(tr["e_dso_proxy"], tr["c_n_days_with_tx"])
    for ycol, name, s in (
        (Y7, "Y7 leftover-iss clean", resid_iss),
        (Y7, "Y7 issued clean", tr["e_ar_issued_lag1"]),
        (Y7, "Y7 DSO clean", tr["e_dso_proxy"]),
        (Y3, "Y3 leftover-days clean", resid_days),
        (Y3, "Y3 days clean", tr["c_n_days_with_tx"]),
        (Y3, "Y3 DSO clean", tr["e_dso_proxy"]),
    ):
        mask = clean & tr[ycol].notna()
        res = signed_oof_auroc(tr[ycol], s, tr["fold"], mask)
        store[name] = res
        rows.append(_auc_row(ycol, name, res, None))
    prose = (
        f"Clean |DSO|≤24 & issued>p10={_f(p10, 0)}: Y7 leftover-iss {_f(_cv(store['Y7 leftover-iss clean']))} "
        f"issued {_f(_cv(store['Y7 issued clean']))} DSO {_f(_cv(store['Y7 DSO clean']))}. "
        f"Y3 leftover-days {_f(_cv(store['Y3 leftover-days clean']))} days {_f(_cv(store['Y3 days clean']))}. "
        f"R²iss={_f(info_i['r2'])} R²days={_f(info_d['r2'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "y7_left": _cv(store["Y7 leftover-iss clean"]),
        "y7_iss": _cv(store["Y7 issued clean"]),
        "y3_left": _cv(store["Y3 leftover-days clean"]),
        "y3_days": _cv(store["Y3 days clean"]),
        "p10": p10,
        "prose": prose,
    }


def pass_clean_refit(tr: pd.DataFrame) -> dict:
    """Refit leftover slope on the clean body only — 0.654 may be a tail-slope artifact."""
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    p10 = float(iss[iss.notna() & (iss > 0)].quantile(0.10)) if iss.notna().any() else float("nan")
    clean = dso.notna() & (dso.abs() <= WINSOR) & iss.notna() & (iss > p10)
    resid_iss, info_i = ols_resid(
        tr["e_dso_proxy"].where(clean), tr["e_ar_issued_lag1"].where(clean)
    )
    resid_days, info_d = ols_resid(
        tr["e_dso_proxy"].where(clean), tr["c_n_days_with_tx"].where(clean)
    )
    resid_both, info_b = ols_resid(
        tr["e_dso_proxy"].where(clean),
        tr["c_n_days_with_tx"].where(clean),
        tr["log_in3"].where(clean),
    )
    rows = []
    store = {}
    for ycol, name, s in (
        (Y7, "Y7 leftover-iss refit", resid_iss),
        (Y7, "Y7 issued clean", tr["e_ar_issued_lag1"]),
        (Y7, "Y7 DSO clean", tr["e_dso_proxy"]),
        (Y7, "Y7 size clean", tr["log_in3"]),
        (Y3, "Y3 leftover-days refit", resid_days),
        (Y3, "Y3 leftover days+size refit", resid_both),
        (Y3, "Y3 days clean", tr["c_n_days_with_tx"]),
        (Y3, "Y3 DSO clean", tr["e_dso_proxy"]),
        (Y3, "Y3 size clean", tr["log_in3"]),
    ):
        mask = clean & tr[ycol].notna()
        res = signed_oof_auroc(tr[ycol], s, tr["fold"], mask)
        store[name] = res
        rows.append(_auc_row(ycol, name, res, None))
    y3_left = _cv(store["Y3 leftover-days refit"])
    y3_days = _cv(store["Y3 days clean"])
    y3_dso = _cv(store["Y3 DSO clean"])
    y3_size = _cv(store["Y3 size clean"])
    y7_left = _cv(store["Y7 leftover-iss refit"])
    artifact = np.isfinite(y3_left) and y3_left >= CHANCE and (
        not np.isfinite(y3_dso) or y3_dso < y3_days or y3_dso < (y3_size + KEEP_DELTA)
    )
    prose = (
        f"Clean-refit leftover: Y3 leftover-days {_f(y3_left)} days {_f(y3_days)} "
        f"DSO {_f(y3_dso)} size {_f(y3_size)} days+size leftover {_f(_cv(store['Y3 leftover days+size refit']))}. "
        f"Y7 leftover-iss {_f(y7_left)} issued {_f(_cv(store['Y7 issued clean']))} "
        f"DSO {_f(_cv(store['Y7 DSO clean']))}. "
        f"R²days={_f(info_d['r2'])} R²iss={_f(info_i['r2'])} R²both={_f(info_b['r2'])}. "
        f"{'0.654 was a full-slope artifact or still loses to days/size' if artifact or (np.isfinite(y3_left) and y3_left < CHANCE) else 'clean-refit leftover lives — still must beat size'}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3_left": y3_left,
        "y3_days": y3_days,
        "y3_dso": y3_dso,
        "y3_size": y3_size,
        "y7_left": y7_left,
        "artifact": artifact,
        "prose": prose,
    }


def decide(p1, p3, p4, p5, p6, p8, p10, p11, p22=None) -> dict:
    twin = bool(p3["gate_twins"])
    size = bool(p3["size_flag"])
    beat_size_y3 = bool(
        np.isfinite(p4["dso_y3"])
        and np.isfinite(p4["size_y3"])
        and (p4["dso_y3"] - p4["size_y3"]) >= KEEP_DELTA
    )
    beat_size_y7 = bool(
        np.isfinite(p4["dso_y7"])
        and np.isfinite(p4["size_y7"])
        and (p4["dso_y7"] - p4["size_y7"]) >= KEEP_DELTA
    )
    if p5["y3_lives"] and beat_size_y3 and not size and not twin:
        y3_x = "KEEP"
    elif p6["tail_artifact"]:
        y3_x = "CLOSE / DROP from the 44"
    elif not p5["y3_lives"] or (
        np.isfinite(p4["dso_y3"]) and np.isfinite(p4["days_y3"]) and p4["dso_y3"] < p4["days_y3"]
    ):
        y3_x = "CLOSE / DROP from the 44"
    else:
        y3_x = "CLOSE"
    y7_x = "KEEP" if (p5["y7_lives"] and beat_size_y7 and not size and not twin) else "CLOSE"
    if p8["empty_early"] or (np.isfinite(p8["short_lag"]) and p8["short_lag"] < CHANCE) or not np.isfinite(p8["short_lag"]):
        q6 = "CLOSE"
    elif np.isfinite(p8["short_lag"]) and p8["short_lag"] >= CHANCE:
        q6 = "KEEP"
    else:
        q6 = "CLOSE"
    if p8["exists_early"]:
        q6_note = (
            f"exists early (unlike delay; early6 Y7 finite {_pp(p8['early_cov'])}) "
            f"but Y7 lag1 short {_f(p8['short_lag'])} "
            f"{'dies' if np.isfinite(p8['short_lag']) and p8['short_lag'] < CHANCE else 'lives'} "
            f"— stock/flow is not a TURNOVER lead"
        )
    else:
        q6_note = "empty-on-short or lag dies"
    drop44 = y3_x.startswith("CLOSE") and y7_x != "KEEP"
    clip_changes = bool(
        np.isfinite(p5["clip_days"])
        and np.isfinite(p5["after_days"])
        and abs(p5["clip_days"] - p5["after_days"]) >= 0.02
    ) or bool(
        np.isfinite(p5["clip_iss"])
        and np.isfinite(p5["after_iss"])
        and abs(p5["clip_iss"] - p5["after_iss"]) >= 0.02
    )
    weaker_twin = ""
    if "e_dpo_proxy" in p3["twins"]:
        weaker_twin = "TWIN of DPO — DPO already DROPPED; drop DSO too"
    if "e_delay_coll" in p3["twins"]:
        weaker_twin = "TWIN of delay_coll — DROP weaker"
    if "e_ar_issued" in p3["twins"] or "e_ar_issued_lag1" in p3["twins"]:
        weaker_twin = "TWIN of issued — DROP DSO (issued owns TURNOVER)"
    if p5["delay_is_shap"] or p11["shap_is_delay"]:
        shap_delay = "just delay"
    elif p22 is not None and p22.get("tail"):
        shap_delay = "not a delay twin; leftover-after-delay is a |DSO|>24 tail"
    else:
        shap_delay = "not just delay"
    tail_note = (
        f"Y3 leftover-after-days {_f(p5['after_days'])} is a |DSO|>24 tail "
        f"(drop-tail {_f(p6['after_drop'])})."
        if p6["tail_artifact"]
        else f"Y3 leftover-after-days {_f(p5['after_days'])} "
        f"{'is not only the |DSO|>24 tail' if np.isfinite(p6['after_drop']) else ''} "
        f"(drop-tail {_f(p6['after_drop'])})."
    )
    return {
        "y3": y3_x,
        "y7": y7_x,
        "q6": q6,
        "q6_note": q6_note,
        "park_y": "PARK",
        "drop44": drop44,
        "twin": twin,
        "size": size,
        "beat_size_y3": beat_size_y3,
        "beat_size_y7": beat_size_y7,
        "weaker_twin": weaker_twin,
        "clip_changes": clip_changes,
        "shap_delay": shap_delay,
        "turnover": "CLOSE",
        "headline": (
            f"DSO vs DPO ρ={_f(p3['dpo'])} "
            f"({'TWIN' if 'e_dpo_proxy' in p3['twins'] else 'not a twin'}). "
            f"vs delay_coll {_f(p3['delay'])} (not a twin). "
            f"Y7 leftover after issued_lag1 {_f(p5['after_iss'])} — {y7_x}. "
            f"Y3 leftover after days {_f(p5['after_days'])} vs days {_f(p4['days_y3'])} — {y3_x}. "
            f"{tail_note} "
            f"Y7 leftover after delay {_f(p5['after_delay'])} ({shap_delay}). "
            f"Short-DSO Q1 univariate {_f(p10['q1_dso'])} (B_shallow OOF 0.410 locked). "
            f"Fold 4 DSO {_f(p10['dso4'])} vs issued {_f(p10['iss4'])}. "
            f"SIZE ρ={_f(p3['size'])}. Q6 {q6}. "
            f"{'DROP e_dso_proxy from the 44' if drop44 else 'keep-list stay pending leftover'}. "
            f"Do not put DSO back on TURNOVER 0.720."
        ),
    }


def make_png(tr: pd.DataFrame) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    sl = tr.loc[tr["e_dso_proxy"].notna()].copy()
    if sl.empty:
        return False
    dso = pd.to_numeric(sl["e_dso_proxy"], errors="coerce")
    iss = pd.to_numeric(sl["e_ar_issued"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    both = sl["e_delay_coll"].notna()
    ax.scatter(
        pd.to_numeric(sl.loc[both, "e_delay_coll"], errors="coerce").clip(-10, 80),
        dso[both].clip(upper=40),
        s=6,
        alpha=0.25,
        c="#3d5a80",
    )
    ax.axhline(WINSOR, color="#ee6c4d", ls="--", lw=1, label="winsor 24")
    ax.set_xlabel("e_delay_coll (clip viz)")
    ax.set_ylabel("e_dso_proxy (clip 40 for viz)")
    ax.set_title("DSO vs delay_coll (train finite both)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    finite = dso.dropna()
    ax.hist(np.log10(finite.clip(lower=1e-4)), bins=40, color="#98c1d9", edgecolor="#293241")
    ax.axvline(np.log10(WINSOR), color="#ee6c4d", ls="--", lw=1)
    ax.set_xlabel("log10 e_dso_proxy")
    ax.set_ylabel("train CM")
    ax.set_title(f"DSO tail — |x|>24 = {_pp(_pct(int((finite.abs()>WINSOR).sum()), len(finite)))}")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def write_md(ctx: dict) -> None:
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
    p6, p7, p8, p9 = ctx["p6"], ctx["p7"], ctx["p8"], ctx["p9"]
    p10, p11 = ctx["p10"], ctx["p11"]
    d = ctx["decision"]
    lines = [
        "# Q4/Q5 unused leftover of `e_dso_proxy` on the 44",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        f"`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        f"Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        f"No new GBM. No `build_targets`. Do not invent `y_dso`. "
        f"Night Y7 quote stays **TURNOVER 0.720 / B_shallow 0.712**. Even card **drops DSO**. "
        f"Do **not** put DSO back on TURNOVER. Do not change 0.720. "
        f"Night Y3 stays **0.762 / 0.752**. Days **0.711**. Size **0.617**. "
        f"Y7 never D. Y5 never E. Y3 never B. Dark 470 stay NaN not 0. "
        f"DPO just DROPPED from the 44. Delay leftover KEEP after DSO (0.581). "
        f"Winsorise-at-24 is model-layer only.",
        "",
        "`e_dso_proxy` = AR open / this-period AR issued (months of billings outstanding). "
        "Feature report: 40.6% cov, acf1 0.25, ICC 0.63; means unusable. "
        "DSO is already CLOSE as a TURNOVER stem. This card is the unused leftover "
        "after days (Y3 X) and after issued_lag1 (Y7), plus leftover after delay_coll.",
        "",
        "## Headline",
        "",
        d["headline"],
        "",
        "## Brief questions",
        "",
        _md_table(
            [
                {
                    "#": "1",
                    "question": "Who is healthy?",
                    "what this cut says": "**PARK** as a health Y. Do not invent `y_dso`. Dark 470 = NaN, not 0.",
                },
                {"#": "2", "question": "Who is improving?", "what this cut says": "Not this clock."},
                {
                    "#": "3",
                    "question": "Who is turning?",
                    "what this cut says": f"Q6 {d['q6']} — {d['q6_note']}.",
                },
                {
                    "#": "4",
                    "question": "Dip vs fall?",
                    "what this cut says": (
                        f"Y7 leftover after issued_lag1 {d['y7']} ({_f(p5['after_iss'])}). "
                        f"DSO is CLOSE as a TURNOVER stem — do not grow 0.720. "
                        f"Fold 4 issued {_f(p10['iss4'])} vs DSO {_f(p10['dso4'])}."
                    ),
                },
                {
                    "#": "5",
                    "question": "Why did it change?",
                    "what this cut says": (
                        f"Y3 leftover after days {d['y3']} ({_f(p5['after_days'])} vs days {_f(p4['days_y3'])}). "
                        f"SHAP #1 is {d['shap_delay']} (leftover after delay {_f(p5['after_delay'])})."
                    ),
                },
                {
                    "#": "6",
                    "question": "Months earlier?",
                    "what this cut says": (
                        f"lag1 {_f(p8['all_lag1'])} lag3 {_f(p8['all_lag3'])} "
                        f"short lag1 {_f(p8['short_lag'])}. Early6 finite {_pp(p8['early_cov'])}."
                    ),
                },
            ]
        ),
        "",
        "## PARK / CLOSE / KEEP",
        "",
        _md_table(
            [
                {
                    "object": "DSO as Y7 leftover after issued_lag1",
                    "decision": f"**{d['y7']}**",
                    "why": f"after issued_lag1 {_f(p5['after_iss'])}; twin={d['twin']}; SIZE={d['size']}",
                },
                {
                    "object": "DSO as Y7 TURNOVER stem / put DSO back",
                    "decision": "**CLOSE**",
                    "why": "even card drops DSO; TURNOVER 0.720 / fold 4 0.680 is issued; do not change 0.720",
                },
                {
                    "object": "DSO as Y3 X / the 44",
                    "decision": f"**{d['y3']}**",
                    "why": f"leftover after days {_f(p5['after_days'])} vs days {_f(p4['days_y3'])} size {_f(p4['size_y3'])}; drop-tail {_f(p6['after_drop'])}",
                },
                {
                    "object": "DSO leftover after delay_coll (SHAP #1?)",
                    "decision": f"**{d['shap_delay']}**",
                    "why": f"leftover after delay {_f(p5['after_delay'])} same-n {_f(p11['dso_after'])}; delay leftover KEEP {_f(p11['delay_after'])}",
                },
                {
                    "object": "DSO as a health Y",
                    "decision": f"**{d['park_y']}**",
                    "why": "do not invent `y_dso`",
                },
                {
                    "object": "Q6 DSO lag1/lag3 on short books",
                    "decision": f"**{d['q6']}**",
                    "why": d["q6_note"],
                },
                {
                    "object": "winsorise-at-24 in the store",
                    "decision": "**CLOSE**",
                    "why": "model-layer only; clip leftover change=" + ("yes" if d["clip_changes"] else "no"),
                },
                {
                    "object": "e_dso_proxy on the keep-list 44",
                    "decision": "**DROP from the 44**" if d["drop44"] else "**stay pending leftover**",
                    "why": f"Y3 {d['y3']}; Y7 {d['y7']}",
                },
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
        f"Y3 {p1['y3_n']:,} pos {p1['y3_pos']:,}.",
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
        "Do not put DSO back on TURNOVER.",
        "",
        _md_table(p4["rows"]),
        "",
        "## 5. Honest leftover",
        "",
        p5["prose"],
        "",
        "Leftover <0.55 dies. Y3 honest bar = days. Y7 honest bar = issued_lag1. "
        "Do not grow TURNOVER even if leftover lives.",
        "",
        _md_table(p5["rows"]),
        "",
        "## 6. |DSO|>24 drop-tail leftover (DPO pattern)",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        "## 7. Twin screen",
        "",
        f"TWIN |ρ|≥0.80 vs DPO / delay_coll / ar_overdue / issued: "
        f"{', '.join(p3['twins']) if p3['twins'] else 'none'}. "
        f"{d['weaker_twin'] or 'No twin drop.'}",
        "",
        "## 8. Winsorise-at-24 (in-memory only)",
        "",
        f"Clip24 single Y3 {_f(p4['clip_y3'])} Y7 {_f(p4['clip_y7'])}. "
        f"Clip leftover Y3-days {_f(p5['clip_days'])} (raw leftover {_f(p5['after_days'])}) "
        f"Y7-iss {_f(p5['clip_iss'])} (raw {_f(p5['after_iss'])}). "
        f"{'Clip moves leftover ≥0.02 — still do not write the clip to parquet.' if d['clip_changes'] else 'Clip does not move leftover by ≥0.02. Do not rewrite the store.'}",
        "",
        "## 9. SIZE terciles and invoice-book-only (drop 470)",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 10. Q6 — lag1/lag3; empty-on-short; exists early like DPO?",
        "",
        p8["prose"],
        "",
        _md_table(p8["rows"]),
        "",
        "## 11. ICC / company-demean",
        "",
        p9["prose"],
        "",
        _md_table(p9["rows"]),
        "",
        "## 12. Fold 4 / short-DSO quintile",
        "",
        p10["prose"],
        "",
        _md_table(p10["rows"]),
        "",
        _md_table(p10["qrows"]),
        "",
        "## 13. vs delay leftover KEEP — leftover of DSO after delay_coll",
        "",
        p11["prose"],
        "",
        _md_table(p11["rows"]),
        "",
        "## Extra — holdout coverage only (no AUROC)",
        "",
        ctx["ph"]["prose"],
        "",
        _md_table(ctx["ph"]["rows"]),
        "",
        "## Extra — same-n leftover after issued_lag1",
        "",
        ctx["ps"]["prose"],
        "",
        _md_table(ctx["ps"]["rows"]),
        "",
        "## Extra — quintiles of DSO vs Y3 / Y7",
        "",
        ctx["pq"]["prose"],
        "",
        _md_table(ctx["pq"]["rows"]),
        "",
        "## Extra — tiny-issued blow-up",
        "",
        ctx["pt"]["prose"],
        "",
        _md_table(ctx["pt"]["rows"]),
        "",
        "## Extra — Y3 same-n leftover vs days",
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
        "## Extra — Y3 Q6 lag / early leftover",
        "",
        ctx["p17"]["prose"],
        "",
        _md_table(ctx["p17"]["rows"]),
        "",
        "## Extra — Y3/Y7 lag1 leftover after days",
        "",
        ctx["p21"]["prose"],
        "",
        _md_table(ctx["p21"]["rows"]),
        "",
        "## Extra — Y7 leftover-after-delay is a tail?",
        "",
        ctx["p22"]["prose"],
        "",
        _md_table(ctx["p22"]["rows"]),
        "",
        "## Extra — Y3 leftover days vs days+size",
        "",
        ctx["p23"]["prose"],
        "",
        _md_table(ctx["p23"]["rows"]),
        "",
        "## Extra — Y3 lag1 leftover vs days clone",
        "",
        ctx["p24"]["prose"],
        "",
        _md_table(ctx["p24"]["rows"]),
        "",
        "## Extra — fat issued (issued > p10) leftover",
        "",
        ctx["p25"]["prose"],
        "",
        _md_table(ctx["p25"]["rows"]),
        "",
        "## Extra — fold-wise leftover",
        "",
        ctx["p26"]["prose"],
        "",
        _md_table(ctx["p26"]["rows"]),
        "",
        "## Extra — chronic Y2 (Y3 only)",
        "",
        ctx["p27"]["prose"],
        "",
        _md_table(ctx["p27"]["rows"]),
        "",
        "## Extra — log1p(DSO) leftover (in-memory)",
        "",
        ctx["p28"]["prose"],
        "",
        _md_table(ctx["p28"]["rows"]),
        "",
        "## Extra — DSO vs AR open",
        "",
        ctx["p29"]["prose"],
        "",
        _md_table(ctx["p29"]["rows"]),
        "",
        "## Extra — lag1 Spearman vs days (clone check)",
        "",
        ctx["p30"]["prose"],
        "",
        _md_table(ctx["p30"]["rows"]),
        "",
        "## Extra — Y3 leftover after issued_lag1 tail",
        "",
        ctx["p31"]["prose"],
        "",
        _md_table(ctx["p31"]["rows"]),
        "",
        "## Extra — extreme DSO rows (train)",
        "",
        ctx["p32"]["prose"],
        "",
        _md_table(ctx["p32"]["rows"]),
        "",
        "## Extra — demean leftover after issued / delay",
        "",
        ctx["p33"]["prose"],
        "",
        _md_table(ctx["p33"]["rows"]),
        "",
        "## Extra — Y3 fold-wise leftover after days",
        "",
        ctx["p34"]["prose"],
        "",
        _md_table(ctx["p34"]["rows"]),
        "",
        "## Extra — short-DSO Q1 leftover",
        "",
        ctx["p35"]["prose"],
        "",
        _md_table(ctx["p35"]["rows"]),
        "",
        "## Extra — after month 7 leftover",
        "",
        ctx["p36"]["prose"],
        "",
        _md_table(ctx["p36"]["rows"]),
        "",
        "## Extra — holdout tails (coverage only)",
        "",
        ctx["p37"]["prose"],
        "",
        _md_table(ctx["p37"]["rows"]),
        "",
        "## Extra — long-trail leftover after issued",
        "",
        ctx["p38"]["prose"],
        "",
        _md_table(ctx["p38"]["rows"]),
        "",
        "## Extra — leftover after ar_overdue_30",
        "",
        ctx["p39"]["prose"],
        "",
        _md_table(ctx["p39"]["rows"]),
        "",
        "## Extra — long-trail leftover fold-wise",
        "",
        ctx["p40"]["prose"],
        "",
        _md_table(ctx["p40"]["rows"]),
        "",
        "## Extra — rolling-origin leftover (Y7 at t+3)",
        "",
        ctx["p41"]["prose"],
        "",
        _md_table(ctx["p41"]["rows"]),
        "",
        "## Extra — rolling-origin leftover (Y3 at t+3)",
        "",
        ctx["p42"]["prose"],
        "",
        _md_table(ctx["p42"]["rows"]),
        "",
        "## Extra — |DSO|>24 persistence",
        "",
        ctx["p43"]["prose"],
        "",
        _md_table(ctx["p43"]["rows"]),
        "",
        "## Extra — pooled Y3 t+3 leftover (per-origin LOW_POWER)",
        "",
        ctx["p44"]["prose"],
        "",
        _md_table(ctx["p44"]["rows"]),
        "",
        "## Extra — rolling leftover after delay_coll (Y7 at t+3)",
        "",
        ctx["p45"]["prose"],
        "",
        _md_table(ctx["p45"]["rows"]),
        "",
        "## Extra — leftover on |DSO|≤24 and fat issued",
        "",
        ctx["p46"]["prose"],
        "",
        _md_table(ctx["p46"]["rows"]),
        "",
        "## Extra — clean-refit leftover (slope on the body only)",
        "",
        ctx["p47"]["prose"],
        "",
        _md_table(ctx["p47"]["rows"]),
        "",
        "## Extra — pooled Y3 t+3 robust (tail / size / folds)",
        "",
        ctx["p48"]["prose"],
        "",
        _md_table(ctx["p48"]["rows"]),
        "",
        "## Extra — |DSO|>24 by calendar month",
        "",
        ctx["p49"]["prose"],
        "",
        _md_table(ctx["p49"]["rows"]),
        "",
        "## Extra — leftover after dropping 2026-08 spike",
        "",
        ctx["p50"]["prose"],
        "",
        _md_table(ctx["p50"]["rows"]),
        "",
        "## Extra — Y6 leftover (not a new Y)",
        "",
        ctx["p51"]["prose"],
        "",
        _md_table(ctx["p51"]["rows"]),
        "",
        "## Extra — leftover slope fit on labeled rows only",
        "",
        ctx["p52"]["prose"],
        "",
        _md_table(ctx["p52"]["rows"]),
        "",
        "## Extra — rank(DSO) leftover (in-memory)",
        "",
        ctx["p53"]["prose"],
        "",
        _md_table(ctx["p53"]["rows"]),
        "",
        "## Extra — labeled-slope leftover vs days clone",
        "",
        ctx["p54"]["prose"],
        "",
        _md_table(ctx["p54"]["rows"]),
        "",
        "## Extra — DSO × log1p(issued) leftover",
        "",
        ctx["p55"]["prose"],
        "",
        _md_table(ctx["p55"]["rows"]),
        "",
        "## Extra — holdout DSO by month (coverage only)",
        "",
        ctx["p56"]["prose"],
        "",
        _md_table(ctx["p56"]["rows"]),
        "",
        "## Extra — mix leftover vs days clone",
        "",
        ctx["p57"]["prose"],
        "",
        _md_table(ctx["p57"]["rows"]),
        "",
        "## Extra — leftover after pending / AP overdue",
        "",
        ctx["p58"]["prose"],
        "",
        _md_table(ctx["p58"]["rows"]),
        "",
        "## Extra — leftover after delay_paid / zero_in / f_ds_r",
        "",
        ctx["p59"]["prose"],
        "",
        _md_table(ctx["p59"]["rows"]),
        "",
        "## Extra — zero_in leftover vs days clone",
        "",
        ctx["p60"]["prose"],
        "",
        _md_table(ctx["p60"]["rows"]),
        "",
        "## Extra — leftover after zero_in+days",
        "",
        ctx["p61"]["prose"],
        "",
        _md_table(ctx["p61"]["rows"]),
        "",
        "## Extra — zero_in+days leftover vs days clone",
        "",
        ctx["p62"]["prose"],
        "",
        _md_table(ctx["p62"]["rows"]),
        "",
        "## Extra — leftover after issued+days",
        "",
        ctx["p63"]["prose"],
        "",
        _md_table(ctx["p63"]["rows"]),
        "",
        "## Extra — clip24 leftover after issued+days",
        "",
        ctx["p64"]["prose"],
        "",
        _md_table(ctx["p64"]["rows"]),
        "",
        "## Extra — leftover after f_ds_r+days",
        "",
        ctx["p65"]["prose"],
        "",
        _md_table(ctx["p65"]["rows"]),
        "",
        "## Extra — leftover after delay_paid+days",
        "",
        ctx["p66"]["prose"],
        "",
        _md_table(ctx["p66"]["rows"]),
        "",
        "## Extra — leftover after AP overdue+days",
        "",
        ctx["p67"]["prose"],
        "",
        _md_table(ctx["p67"]["rows"]),
        "",
        "## Extra — leftover after DPO+days",
        "",
        ctx["p68"]["prose"],
        "",
        _md_table(ctx["p68"]["rows"]),
        "",
        "## Extra — leftover after AR overdue+days",
        "",
        ctx["p69"]["prose"],
        "",
        _md_table(ctx["p69"]["rows"]),
        "",
        "## Extra — clip leftover after delay_paid+days",
        "",
        ctx["p70"]["prose"],
        "",
        _md_table(ctx["p70"]["rows"]),
        "",
        "## Extra — clip leftover after DPO+days",
        "",
        ctx["p71"]["prose"],
        "",
        _md_table(ctx["p71"]["rows"]),
        "",
        "## Extra — leftover after AR open+days",
        "",
        ctx["p72"]["prose"],
        "",
        _md_table(ctx["p72"]["rows"]),
        "",
        "## Extra — leftover after delay_coll+days",
        "",
        ctx["p73"]["prose"],
        "",
        _md_table(ctx["p73"]["rows"]),
        "",
        "## Extra — clip leftover after delay_coll+days",
        "",
        ctx["p74"]["prose"],
        "",
        _md_table(ctx["p74"]["rows"]),
        "",
        "## Extra — holdout SIZE tercile coverage",
        "",
        ctx["p75"]["prose"],
        "",
        _md_table(ctx["p75"]["rows"]),
        "",
        "## Extra — Y7 leftover after open+issued",
        "",
        ctx["p76"]["prose"],
        "",
        _md_table(ctx["p76"]["rows"]),
        "",
        "## Extra — Y7 leftover after pending+issued",
        "",
        ctx["p77"]["prose"],
        "",
        _md_table(ctx["p77"]["rows"]),
        "",
        "## Extra — Y7 leftover after AP+issued",
        "",
        ctx["p78"]["prose"],
        "",
        _md_table(ctx["p78"]["rows"]),
        "",
        "## Extra — clip leftover after f_ds_r+days",
        "",
        ctx["p79"]["prose"],
        "",
        _md_table(ctx["p79"]["rows"]),
        "",
        "## Extra — Y7 leftover after DPO+issued",
        "",
        ctx["p80"]["prose"],
        "",
        _md_table(ctx["p80"]["rows"]),
        "",
        "## Extra — clip leftover after zero_in+days",
        "",
        ctx["p81"]["prose"],
        "",
        _md_table(ctx["p81"]["rows"]),
        "",
        "## Extra — Y7 leftover after overdue+issued",
        "",
        ctx["p82"]["prose"],
        "",
        _md_table(ctx["p82"]["rows"]),
        "",
        "## Extra — Y7 leftover after delay_paid+issued",
        "",
        ctx["p83"]["prose"],
        "",
        _md_table(ctx["p83"]["rows"]),
        "",
        "## Extra — Y7 leftover after size+issued",
        "",
        ctx["p84"]["prose"],
        "",
        _md_table(ctx["p84"]["rows"]),
        "",
        "## Extra — Y7 leftover after f_ds_r+issued",
        "",
        ctx["p85"]["prose"],
        "",
        _md_table(ctx["p85"]["rows"]),
        "",
        "## Extra — leftover after AP issued+days",
        "",
        ctx["p86"]["prose"],
        "",
        _md_table(ctx["p86"]["rows"]),
        "",
        "## Extra — Y7 leftover after od30+issued",
        "",
        ctx["p87"]["prose"],
        "",
        _md_table(ctx["p87"]["rows"]),
        "",
        "## Extra — Y7 leftover after AP issued+issued",
        "",
        ctx["p88"]["prose"],
        "",
        _md_table(ctx["p88"]["rows"]),
        "",
        "## Extra — Y7 leftover after zero_in+issued",
        "",
        ctx["p89"]["prose"],
        "",
        _md_table(ctx["p89"]["rows"]),
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
            f"|DSO|>24 tail, SIZE/book, Q6, ICC, fold 4 / short-Q1, delay leftover, "
            f"holdout, same-n, quintiles, tiny-issued, Y3 same-n, fold4 groups, clip ICC, "
            f"Y3 Q6, lag leftover, delay-tail, days+size, lag1-clone, fat issued, "
            f"fold leftover, chronic, log1p, open, lag1-spearman, Y3-iss-tail, "
            f"extreme rows, demean leftover, Y3 folds, Q1 leftover, after7, holdout tails, "
            f"long leftover, od30, long folds, rolling Y7/Y3, persist, pool, delay-roll, "
            f"clean/refit, pool-robust, month-tail, drop-Aug, Y6, labeled-slope, rank, "
            f"labeled-clone, mix, holdout-month, mix-clone.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    p1, p3, p4, p5, p8, d = ctx["p1"], ctx["p3"], ctx["p4"], ctx["p5"], ctx["p8"], ctx["decision"]
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
            "metric": "auroc_e_dso_proxy",
            "value": p4["dso_y7"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"iss={p4['iss_y7']:.4f} leftover_iss={p5['after_iss']:.4f} leftover_delay={p5['after_delay']:.4f} y7={d['y7']}",
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
            "metric": "auroc_e_dso_proxy",
            "value": p4["dso_y3"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={p4['days_y3']:.4f} leftover_days={p5['after_days']:.4f} y3={d['y3']}",
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
            "metric": "rho_dso_vs_dpo",
            "value": p3["dpo"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"delay={p3['delay']:.4f} od={p3['od']:.4f} issued={p3['iss']:.4f} size={p3['size']:.4f}",
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
            "metric": "auroc_dso_resid_issued_lag1",
            "value": p5["after_iss"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"after_delay={p5['after_delay']:.4f} lives={p5['y7_lives']} drop44={d['drop44']}",
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
            "metric": "auroc_dso_resid_days",
            "value": p5["after_days"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"clip_days={p5['clip_days']:.4f} drop24={ctx['p6']['after_drop']:.4f} tail={ctx['p6']['tail_artifact']}",
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
            "metric": "auroc_dso_resid_delay_coll",
            "value": p5["after_delay"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"shap_delay={d['shap_delay']} samen={ctx['p11']['dso_after']:.4f} delay_keep={ctx['p11']['delay_after']:.4f}",
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
            "metric": "auroc_e_dso_proxy_fold4",
            "value": ctx["p10"]["dso4"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"iss4={ctx['p10']['iss4']:.4f} q1_dso={ctx['p10']['q1_dso']:.4f} q1_n={ctx['p10']['q1_n']} issued_owns={ctx['p10']['issued_owns']}",
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
            "metric": "auroc_e_dso_proxy_lag1_short",
            "value": p8["short_lag"],
            "coverage": f"{p8['early_cov']:.4f}" if np.isfinite(p8["early_cov"]) else "",
            "notes": f"q6={d['q6']} early_cov={p8['early_cov']:.4f} exists_early={p8['exists_early']}",
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
            "metric": "dso_share_gt24",
            "value": p1["gt24_share"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"p99={p1['p99']:.4f} max={p1['max']:.4f} acf1={p1['acf1']:.4f} dark_nan={p1['dark_nan']} vs_dpo_gt24={p1['dpo_gt24']:.4f}",
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
            "metric": "icc_e_dso_proxy",
            "value": ctx["p9"]["icc"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"acf1={ctx['p9']['acf1']:.4f} clip_icc={ctx['p20']['icc_clip']:.4f} BETWEEN={ctx['p9']['style']}",
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
            "metric": "auroc_dso_resid_delay_drop24",
            "value": ctx["p22"]["after_drop"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"raw={ctx['p22']['after_raw']:.4f} clip24={ctx['p22']['after_c24']:.4f} tail={ctx['p22']['tail']}",
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
            "metric": "auroc_dso_resid_days_size",
            "value": ctx["p23"]["both"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={ctx['p23']['days']:.4f} both_drop={ctx['p23']['both_drop']:.4f} lag1_clone={ctx['p24']['clone']}",
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
            "metric": "auroc_log1p_dso_resid_days",
            "value": ctx["p28"]["y3"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"log_raw={ctx['p28']['y3_raw']:.4f} y7_iss={ctx['p28']['y7_iss']:.4f} y7_delay={ctx['p28']['y7_delay']:.4f}",
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
            "metric": "rho_dso_lag1_vs_days",
            "value": ctx["p30"]["lag1_days"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"now_days={ctx['p30']['now_days']:.4f} clip_days={ctx['p30']['clip_days']:.4f} clip_left={ctx['p30']['y3_left']:.4f}",
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
            "metric": "auroc_dso_resid_iss_shortq1",
            "value": ctx["p35"]["left_iss"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"q1_dso={ctx['p35']['dso']:.4f} q1_iss={ctx['p35']['iss']:.4f} q1_delay_left={ctx['p35']['left_delay']:.4f}",
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
            "metric": "auroc_dso_resid_iss_long18",
            "value": ctx["p38"]["long_left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"long_raw={ctx['p38']['long_raw']:.4f} long_iss={ctx['p38']['long_iss']:.4f} short_left={ctx['p38']['short_left']:.4f} slice_only_not_KEEP",
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
            "metric": "auroc_dso_resid_iss_roll_t3",
            "value": ctx["p41"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"raw={ctx['p41']['raw']:.4f} iss={ctx['p41']['iss']:.4f} y3_left={ctx['p42']['left']:.4f} persist_twice={ctx['p43']['n_twice']}",
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
            "metric": "auroc_dso_resid_days_roll_pool",
            "value": ctx["p44"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={ctx['p44']['days']:.4f} n={ctx['p44']['n']} pos={ctx['p44']['n_pos']} delay_roll={ctx['p45']['left']:.4f} clean_iss={ctx['p46']['y7_left']:.4f}",
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
            "metric": "auroc_dso_resid_days_clean_refit",
            "value": ctx["p47"]["y3_left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"dso={ctx['p47']['y3_dso']:.4f} days={ctx['p47']['y3_days']:.4f} size={ctx['p47']['y3_size']:.4f} y7_left={ctx['p47']['y7_left']:.4f} artifact={ctx['p47']['artifact']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y3,
            "model": MODEL,
            "split": "train",
            "metric": "auroc_dso_resid_days_roll_drop24",
            "value": ctx["p48"]["drop"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"left={ctx['p48']['left']:.4f} dso={ctx['p48']['dso']:.4f} days={ctx['p48']['days']:.4f} size={ctx['p48']['size']:.4f} tail={ctx['p48']['tail']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y6Z,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_dso_resid_iss_y6",
            "value": ctx["p51"]["left_iss"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"dso={ctx['p51']['dso']:.4f} iss={ctx['p51']['iss']:.4f} drop_aug_iss={ctx['p50']['y7_iss']:.4f} not_a_new_y",
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
            "metric": "auroc_dso_resid_days_labeled_slope",
            "value": ctx["p52"]["y3"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"y7_iss={ctx['p52']['y7']:.4f} y7_delay={ctx['p52']['y7_delay']:.4f}",
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
            "metric": "auroc_rank_dso_resid_days",
            "value": ctx["p53"]["y3"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"raw_rank={ctx['p53']['y3_raw']:.4f} y7_iss={ctx['p53']['y7']:.4f} y7_delay={ctx['p53']['y7_delay']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y3,
            "model": MODEL,
            "split": "train",
            "metric": "rho_dso_labeled_resid_vs_days",
            "value": ctx["p54"]["rho"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"raw={ctx['p54']['rho_raw']:.4f} left={ctx['p54']['left']:.4f} clip={ctx['p54']['clip']:.4f} false_clone={ctx['p54']['false_clone']} mix={ctx['p55']['y7_mix']:.4f}",
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
            "metric": "auroc_dso_resid_pending",
            "value": ctx["p58"]["y7_p"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"y3={ctx['p58']['y3_p']:.4f} y3_pd={ctx['p58']['y3_pd']:.4f} y7_ap={ctx['p58']['y7_ap']:.4f} mix_clone={ctx['p57']['false_clone']}",
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
            "metric": "auroc_dso_resid_zero_in",
            "value": ctx["p60"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"rho_vs_days={ctx['p60']['rho']:.4f} clone={ctx['p60']['clone']} y7_paid={ctx['p59']['y7_paid']:.4f}",
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
            "metric": "auroc_dso_resid_zero_days",
            "value": ctx["p61"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={ctx['p61']['days']:.4f} rho={ctx['p62']['rho']:.4f} clone={ctx['p62']['clone']}",
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
            "metric": "auroc_clip_dso_resid_iss_days",
            "value": ctx["p64"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"iss_days={ctx['p63']['left']:.4f} rho_raw={ctx['p63']['rho']:.4f} rho_clip={ctx['p64']['rho']:.4f}",
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
            "metric": "auroc_dso_resid_fds_days",
            "value": ctx["p65"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={ctx['p65']['days']:.4f} rho={ctx['p65']['rho']:.4f}",
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
            "metric": "auroc_dso_resid_paid_days",
            "value": ctx["p66"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={ctx['p66']['days']:.4f} rho={ctx['p66']['rho']:.4f}",
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
            "metric": "auroc_dso_resid_ap_days",
            "value": ctx["p67"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={ctx['p67']['days']:.4f} rho={ctx['p67']['rho']:.4f}",
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
            "metric": "auroc_dso_resid_dpo_days",
            "value": ctx["p68"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={ctx['p68']['days']:.4f} rho={ctx['p68']['rho']:.4f}",
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
            "metric": "auroc_dso_resid_od_days",
            "value": ctx["p69"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={ctx['p69']['days']:.4f} rho={ctx['p69']['rho']:.4f}",
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
            "metric": "auroc_clip_dso_resid_paid_days",
            "value": ctx["p70"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={ctx['p70']['days']:.4f} rho={ctx['p70']['rho']:.4f}",
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
            "metric": "auroc_clip_dso_resid_dpo_days",
            "value": ctx["p71"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={ctx['p71']['days']:.4f} rho={ctx['p71']['rho']:.4f} clone={ctx['p71']['clone']}",
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
            "metric": "auroc_dso_resid_open_days",
            "value": ctx["p72"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={ctx['p72']['days']:.4f} rho={ctx['p72']['rho']:.4f}",
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
            "metric": "auroc_dso_resid_delay_days",
            "value": ctx["p73"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={ctx['p73']['days']:.4f} rho={ctx['p73']['rho']:.4f}",
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
            "metric": "auroc_clip_dso_resid_delay_days",
            "value": ctx["p74"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={ctx['p74']['days']:.4f} rho={ctx['p74']['rho']:.4f} clone={ctx['p74']['clone']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": "holdout_cov",
            "model": MODEL,
            "split": "holdout",
            "metric": "cov_dso_holdout_size_t1",
            "value": ctx["p75"]["t1"],
            "coverage": f"{ctx['p75']['t1']:.4f}",
            "notes": f"t2={ctx['p75']['t2']:.4f} t3={ctx['p75']['t3']:.4f}",
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
            "metric": "auroc_dso_resid_open_iss",
            "value": ctx["p76"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"iss={ctx['p76']['iss']:.4f} rho={ctx['p76']['rho']:.4f}",
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
            "metric": "auroc_dso_resid_pend_iss",
            "value": ctx["p77"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"iss={ctx['p77']['iss']:.4f} rho={ctx['p77']['rho']:.4f}",
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
            "metric": "auroc_dso_resid_ap_iss",
            "value": ctx["p78"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"iss={ctx['p78']['iss']:.4f} rho={ctx['p78']['rho']:.4f}",
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
            "metric": "auroc_clip_dso_resid_fds_days",
            "value": ctx["p79"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={ctx['p79']['days']:.4f} rho={ctx['p79']['rho']:.4f} clone={ctx['p79']['clone']}",
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
            "metric": "auroc_dso_resid_dpo_iss",
            "value": ctx["p80"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"iss={ctx['p80']['iss']:.4f} rho={ctx['p80']['rho']:.4f}",
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
            "metric": "auroc_clip_dso_resid_zero_days",
            "value": ctx["p81"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={ctx['p81']['days']:.4f} rho={ctx['p81']['rho']:.4f} clone={ctx['p81']['clone']}",
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
            "metric": "auroc_dso_resid_od_iss",
            "value": ctx["p82"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"iss={ctx['p82']['iss']:.4f} rho={ctx['p82']['rho']:.4f}",
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
            "metric": "auroc_dso_resid_paid_iss",
            "value": ctx["p83"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"iss={ctx['p83']['iss']:.4f} rho={ctx['p83']['rho']:.4f}",
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
            "metric": "auroc_dso_resid_size_iss",
            "value": ctx["p84"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"iss={ctx['p84']['iss']:.4f} rho={ctx['p84']['rho']:.4f}",
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
            "metric": "auroc_dso_resid_fds_iss",
            "value": ctx["p85"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"iss={ctx['p85']['iss']:.4f} rho={ctx['p85']['rho']:.4f}",
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
            "metric": "auroc_dso_resid_apiss_days",
            "value": ctx["p86"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={ctx['p86']['days']:.4f} rho={ctx['p86']['rho']:.4f}",
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
            "metric": "auroc_dso_resid_od30_iss",
            "value": ctx["p87"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"iss={ctx['p87']['iss']:.4f} rho={ctx['p87']['rho']:.4f}",
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
            "metric": "auroc_dso_resid_apiss_iss",
            "value": ctx["p88"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"iss={ctx['p88']['iss']:.4f} rho={ctx['p88']['rho']:.4f}",
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
            "metric": "auroc_dso_resid_zero_iss",
            "value": ctx["p89"]["left"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"iss={ctx['p89']['iss']:.4f} rho={ctx['p89']['rho']:.4f}",
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
    p3, p4, p5, p8 = ctx["p3"], ctx["p4"], ctx["p5"], ctx["p8"]
    text = (
        f"# Wave 4 — DSO leftover\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/dso_qa.py`\n"
        f"- `analysis/outputs/dso_qa.md`\n"
        f"- `analysis/outputs/dso_leftover.png`\n"
        f"- append-only `analysis/experiments/registry.csv` "
        f"(skip key agent+x_families+y+model+split+metric)\n"
        f"- this note\n\n"
        f"Did not touch `invoices.py`, `dpo_qa.py`, `delay_qa.py`, `gbm_y7_core.py`, "
        f"`product/`, parquet / duckdb, `build_targets`, parent journal, LIVE, canvas, "
        f"TURNOVER, or the 15-col card. Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. "
        f"Do not put DSO back. Night Y3 stays **0.762 / 0.752**. Days 0.711 untouched.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| DSO as Y7 leftover after issued_lag1 | **{d['y7']}** |\n"
        f"| DSO as TURNOVER stem / put DSO back | **CLOSE** (do not grow 0.720) |\n"
        f"| DSO as Y3 X / the 44 | **{d['y3']}** |\n"
        f"| DSO leftover after delay_coll | **{d['shap_delay']}** |\n"
        f"| DSO as a health Y | **PARK** |\n"
        f"| Q6 DSO on short books | **{d['q6']}** |\n"
        f"| e_dso_proxy on the 44 | **{'DROP' if d['drop44'] else 'pending'}** |\n\n"
        f"ρ DSO vs DPO {_f(p3['dpo'])} vs delay {_f(p3['delay'])} vs size {_f(p3['size'])}. "
        f"Y7 leftover after issued_lag1 {_f(p5['after_iss'])} / delay {_f(p5['after_delay'])}. "
        f"Y3 DSO {_f(p4['dso_y3'])} leftover-days {_f(p5['after_days'])} vs days {_f(p4['days_y3'])} "
        f"(drop>|24| leftover {_f(ctx['p6']['after_drop'])} — "
        f"{'tail' if ctx['p6']['tail_artifact'] else 'not only tail'}). "
        f"Q6 short lag1 {_f(p8['short_lag'])} early cov {_pp(p8['early_cov'])}. "
        f"Fold 4 DSO {_f(ctx['p10']['dso4'])} vs issued {_f(ctx['p10']['iss4'])}. "
        f"Short-DSO Q1 univariate {_f(ctx['p10']['q1_dso'])} (B_shallow OOF 0.410 locked). "
        f"Y7 leftover-after-delay {_f(p5['after_delay'])} "
        f"{'is a |DSO|>24 tail' if ctx['p22']['tail'] else 'survives clip'} "
        f"(drop {_f(ctx['p22']['after_drop'])} clip24 {_f(ctx['p22']['after_c24'])}). "
        f"log1p leftover Y3 {_f(ctx['p28']['y3'])} Y7-iss {_f(ctx['p28']['y7_iss'])}. "
        f"lag1 vs days {_f(ctx['p30']['lag1_days'])} "
        f"(unclipped leftover-vs-days ρ was a false clone). "
        f"Long so-far leftover-iss {_f(ctx['p38']['long_left'])} vs issued {_f(ctx['p38']['long_iss'])} "
        f"— slice only, does not flip DROP. "
        f"Rolling Y7 t+3 leftover {_f(ctx['p41']['left'])} issued {_f(ctx['p41']['iss'])}. "
        f"Rolling Y3 t+3 leftover {_f(ctx['p42']['left'])} days {_f(ctx['p42']['days'])}. "
        f"|DSO|>24 persist twice {ctx['p43']['n_twice']} chronic-half {ctx['p43']['n_half']}. "
        f"Pooled Y3 t+3 leftover {_f(ctx['p44']['left'])} days {_f(ctx['p44']['days'])}. "
        f"Rolling leftover-delay {_f(ctx['p45']['left'])}. "
        f"Clean leftover-iss {_f(ctx['p46']['y7_left'])}. "
        f"Clean-refit leftover-days {_f(ctx['p47']['y3_left'])} vs days {_f(ctx['p47']['y3_days'])} "
        f"size {_f(ctx['p47']['y3_size'])}. "
        f"Pooled Y3 leftover {_f(ctx['p48']['left'])} drop>|24| {_f(ctx['p48']['drop'])} "
        f"{'is a tail' if ctx['p48']['tail'] else 'not KEEP'}. "
        f"|DSO|>24 month min {_pp(ctx['p49']['min'])} max {_pp(ctx['p49']['max'])} "
        f"spike {ctx['p49'].get('spike', '?')}. "
        f"Drop-Aug leftover-iss {_f(ctx['p50']['y7_iss'])} leftover-delay {_f(ctx['p50']['y7_delay'])}. "
        f"Y6 leftover-iss {_f(ctx['p51']['left_iss'])} (not a new Y). "
        f"Labeled-slope leftover Y3 {_f(ctx['p52']['y3'])} Y7-iss {_f(ctx['p52']['y7'])}. "
        f"Rank leftover-days {_f(ctx['p53']['y3'])} leftover-iss {_f(ctx['p53']['y7'])}. "
        f"Issued+days leftover {_f(ctx['p63']['left'])} ρ={_f(ctx['p63']['rho'])}. "
        f"Clip leftover issued+days {_f(ctx['p64']['left'])} ρ={_f(ctx['p64']['rho'])}. "
        f"Labeled leftover-vs-days ρ={_f(ctx['p54']['rho'])} "
        f"{'FALSE clone' if ctx['p54']['false_clone'] else ('CLONE' if ctx['p54']['clone'] else 'not a clone')}. "
        f"Mix leftover-iss {_f(ctx['p55']['y7_mix'])}.\n\n"
        f"## What failed / next\n\n"
        + "\n".join(f"- {x}" for x in ctx["failed"])
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.parent.mkdir(parents=True, exist_ok=True)
    WAVE_NOTE.write_text(text, encoding="utf-8")
    print(f"wrote {WAVE_NOTE}")


def run() -> dict:
    t0 = time.time()
    print(f"dso_qa start seed={FOLD_SEED} holdout=72 agent={AGENT} wave={WRITE_WAVE}")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(
        panel,
        ["e_dso_proxy", "e_dpo_proxy", "e_ar_issued", "e_delay_coll"],
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
        print("reconstruct DSO from raw SQL")
        recon = reconstruct_dso(con)
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
    print("pass 6 |DSO|>24 tail")
    p6 = pass6_tail(tr)
    print("pass 7 slices")
    p7 = pass7_slices(tr, book)
    print("pass 8 Q6")
    p8 = pass8_q6(tr)
    print("pass 9 ICC")
    p9 = pass9_icc(tr)
    print("pass 10 fold 4 / short-Q1")
    p10 = pass10_fold4(tr, p4)
    print("pass 11 delay leftover")
    p11 = pass11_delay(tr, p5)
    print("extra holdout")
    ph = pass_holdout(panel, book)
    print("extra same-n")
    ps = pass_samen(tr)
    print("extra quintiles")
    pq = pass_quintiles(tr)
    print("extra tiny-issued")
    pt = pass_tiny(tr)
    print("extra Y3 same-n")
    p18 = pass_y3_samen(tr)
    print("extra fold4 groups")
    p19 = pass_fold4_groups(tr)
    print("extra clip ICC")
    p20 = pass_clip_icc(tr)
    print("extra Y3 Q6")
    p17 = pass_y3_q6(tr)
    print("extra lag leftover")
    p21 = pass_lag_leftover(tr)
    print("extra delay-tail leftover")
    p22 = pass_delay_tail(tr)
    print("extra Y3 days+size")
    p23 = pass_y3_days_size(tr)
    print("extra lag1 vs days clone")
    p24 = pass_lag1_equals_days(tr)
    print("extra fat issued")
    p25 = pass_fat_issued(tr)
    print("extra fold leftover")
    p26 = pass_fold_leftover(tr)
    print("extra chronic")
    p27 = pass_chronic(tr)
    print("extra log1p leftover")
    p28 = pass_logdso(tr)
    print("extra open vs DSO")
    p29 = pass_open_vs_dso(tr)
    print("extra lag1 spearman")
    p30 = pass_lag1_spearman(tr)
    print("extra Y3 issued tail")
    p31 = pass_y3_iss_tail(tr)
    print("extra extreme rows")
    p32 = pass_extreme(tr)
    print("extra demean leftover")
    p33 = pass_demean_leftover(tr)
    print("extra Y3 fold leftover")
    p34 = pass_y3_fold_left(tr)
    print("extra Q1 leftover")
    p35 = pass_q1_leftover(tr)
    print("extra after month 7 leftover")
    p36 = pass_after7_left(tr)
    print("extra holdout tails")
    p37 = pass_holdout_tails(panel)
    print("extra long-trail leftover")
    p38 = pass_long_leftover(tr)
    print("extra overdue_30 leftover")
    p39 = pass_od30_leftover(tr)
    print("extra long leftover folds")
    p40 = pass_long_folds(tr)
    print("extra rolling-origin leftover")
    p41 = pass_rolling(tr)
    print("extra rolling Y3 leftover")
    p42 = pass_rolling_y3(tr)
    print("extra |DSO|>24 persist")
    p43 = pass_gt24_persist(tr)
    print("extra pooled Y3 t+3 leftover")
    p44 = pass_y3_pool(tr)
    print("extra rolling leftover-delay")
    p45 = pass_rolling_delay(tr)
    print("extra clean |DSO|<=24 leftover")
    p46 = pass_clean_double(tr)
    print("extra clean-refit leftover")
    p47 = pass_clean_refit(tr)
    print("extra pooled Y3 robust")
    p48 = pass_y3_pool_robust(tr)
    print("extra |DSO|>24 by month")
    p49 = pass_month_tail(tr)
    print("extra drop Aug-2026 spike leftover")
    p50 = pass_drop_aug(tr)
    print("extra Y6 leftover")
    p51 = pass_y6_left(tr)
    print("extra labeled-slope leftover")
    p52 = pass_labeled_slope(tr)
    print("extra rank(DSO) leftover")
    p53 = pass_rank_left(tr)
    print("extra labeled leftover clone")
    p54 = pass_labeled_clone(tr)
    print("extra mix leftover")
    p55 = pass_interact(tr)
    print("extra holdout month coverage")
    p56 = pass_holdout_month(panel)
    print("extra mix leftover clone")
    p57 = pass_mix_clone(tr)
    print("extra leftover after pending / AP")
    p58 = pass_pending_left(tr)
    print("extra leftover after more bars")
    p59 = pass_more_bars(tr)
    print("extra zero_in leftover clone")
    p60 = pass_zero_clone(tr)
    print("extra leftover zero_in+days")
    p61 = pass_zero_days(tr)
    print("extra zero_in+days clone")
    p62 = pass_zd_clone(tr)
    print("extra leftover issued+days")
    p63 = pass_iss_days(tr)
    print("extra clip leftover issued+days")
    p64 = pass_clip_iss_days(tr)
    print("extra leftover f_ds_r+days")
    p65 = pass_fds_days(tr)
    print("extra leftover delay_paid+days")
    p66 = pass_paid_days(tr)
    print("extra leftover AP overdue+days")
    p67 = pass_ap_days(tr)
    print("extra leftover DPO+days")
    p68 = pass_dpo_days(tr)
    print("extra leftover AR overdue+days")
    p69 = pass_od_days(tr)
    print("extra clip leftover delay_paid+days")
    p70 = pass_clip_paid_days(tr)
    print("extra clip leftover DPO+days")
    p71 = pass_clip_dpo_days(tr)
    print("extra leftover AR open+days")
    p72 = pass_open_days(tr)
    print("extra leftover delay_coll+days")
    p73 = pass_delay_days(tr)
    print("extra clip leftover delay_coll+days")
    p74 = pass_clip_delay_days(tr)
    print("extra holdout SIZE coverage")
    p75 = pass_holdout_size(panel)
    print("extra leftover open+issued Y7")
    p76 = pass_y7_open_iss(tr)
    print("extra leftover pending+issued Y7")
    p77 = pass_pend_iss(tr)
    print("extra leftover AP+issued Y7")
    p78 = pass_ap_iss(tr)
    print("extra clip leftover f_ds_r+days")
    p79 = pass_clip_fds_days(tr)
    print("extra leftover DPO+issued Y7")
    p80 = pass_dpo_iss(tr)
    print("extra clip leftover zero_in+days")
    p81 = pass_clip_zero_days(tr)
    print("extra leftover overdue+issued Y7")
    p82 = pass_od_iss(tr)
    print("extra leftover delay_paid+issued Y7")
    p83 = pass_paid_iss(tr)
    print("extra leftover size+issued Y7")
    p84 = pass_size_iss(tr)
    print("extra leftover f_ds_r+issued Y7")
    p85 = pass_fds_iss(tr)
    print("extra leftover AP issued+days")
    p86 = pass_apiss_days(tr)
    print("extra leftover od30+issued Y7")
    p87 = pass_od30_iss(tr)
    print("extra leftover AP issued+issued Y7")
    p88 = pass_apiss_iss(tr)
    print("extra leftover zero_in+issued Y7")
    p89 = pass_zero_iss(tr)
    png = make_png(tr)
    decision = decide(p1, p3, p4, p5, p6, p8, p10, p11, p22)
    failed = []
    if not p5["y3_lives"]:
        failed.append(f"Y3 leftover after days {_f(p5['after_days'])} — {decision['y3']}")
    if not p5["y7_lives"]:
        failed.append(f"Y7 leftover dies after issued_lag1 {_f(p5['after_iss'])}")
    if decision["twin"]:
        failed.append(f"twin screen: {', '.join(p3['twins'])} {decision['weaker_twin']}")
    if decision["drop44"]:
        failed.append("DROP e_dso_proxy from the 44")
    failed.append(f"Q6 {decision['q6']} — {decision['q6_note']}")
    failed.append("do not put DSO back on TURNOVER; night quote stays 0.720 / 0.712")
    if p10["issued_owns"]:
        failed.append(
            f"CONFIRM issued owns fold 4 ({_f(p10['iss4'])} vs DSO {_f(p10['dso4'])})"
        )
    if p10["q1_hole"]:
        failed.append(
            f"CONFIRM short-DSO Q1 hole univariate {_f(p10['q1_dso'])} "
            f"(B_shallow OOF 0.410 locked; n={p10['q1_n']})"
        )
    if p6["tail_artifact"]:
        failed.append(
            f"Y3 leftover after days is a |DSO|>24 tail "
            f"(drop>24 {_f(p6['after_drop'])} clip12 {_f(p6['after_c12'])})"
        )
    else:
        failed.append(
            f"Y3 leftover after days drop-tail {_f(p6['after_drop'])} "
            f"(raw {_f(p6['after_raw'])}) — not only a tail / still loses to days"
        )
    failed.append(
        f"SHAP #1 {decision['shap_delay']}: leftover after delay {_f(p5['after_delay'])} "
        f"same-n {_f(p11['dso_after'])}; delay leftover KEEP {_f(p11['delay_after'])}"
    )
    failed.append(f"Y3 Q6 lag1 {_f(p17['all_lag1'])} short {_f(p17['short_lag'])}")
    failed.append(f"Y3 lag1 leftover-days {_f(p21['y3_lag'])} drop>24 {_f(p21['y3_drop'])}")
    failed.append(
        f"clip24 ICC {_f(p20['icc_clip'])} — do not write clip to store"
    )
    if p22["tail"]:
        failed.append(
            f"Y7 leftover-after-delay {_f(p22['after_raw'])} is a |DSO|>24 tail "
            f"(drop {_f(p22['after_drop'])} clip24 {_f(p22['after_c24'])})"
        )
    else:
        failed.append(
            f"Y7 leftover-after-delay survives clip/drop {_f(p22['after_c24'])} / {_f(p22['after_drop'])}"
        )
    failed.append(
        f"Y3 leftover days+size {_f(p23['both'])} drop>24 {_f(p23['both_drop'])} "
        f"(days-only leftover {_f(p23['days'])})"
    )
    failed.append(
        f"Y3 lag1 leftover vs days ρ={_f(p24['rho'])} raw-lag1-days {_f(p24.get('rho_raw', float('nan')))} "
        f"{'FALSE clone (tail rank)' if p24.get('false_clone') else ('CLONE' if p24['clone'] else 'not a clone')} "
        f"leftover {_f(p24['leftover'])}"
    )
    failed.append(f"fat-issued leftover Y3 {_f(p25['y3'])} vs days {_f(p25['y3_days'])}")
    failed.append(f"chronic-{p27['n_chronic']} Y3 DSO {_f(p27['all'])} → {_f(p27['drop'])}")
    failed.append(
        f"log1p leftover Y3-days {_f(p28['y3'])} Y7-iss {_f(p28['y7_iss'])} "
        f"Y7-delay {_f(p28['y7_delay'])}"
    )
    failed.append(
        f"DSO vs open ρ={_f(p29['rho_open'])} Y7 open {_f(p29['open'])} "
        f"open-leftover-iss {_f(p29['open_left'])}"
    )
    failed.append(
        f"lag1 vs days ρ={_f(p30['lag1_days'])} clip24-lag1 vs days {_f(p30['clip_days'])} "
        f"clip24-lag1 leftover {_f(p30['y3_left'])}"
    )
    failed.append(
        f"Y3 leftover-after-issued {_f(p31['raw'])} drop>24 {_f(p31['drop'])} "
        f"{'tail' if p31['tail'] else 'not only tail'}"
    )
    failed.append(
        f"max DSO {_f(p32['max'], 1)} n>1000={p32['n_gt1e3']} n>1e5={p32['n_gt1e5']}"
    )
    failed.append(
        f"demean leftover-iss {_f(p33['de_iss'])} mean leftover-iss {_f(p33['mu_iss'])}"
    )
    failed.append(f"Y3 fold leftover-days {_f(p34['left'])} fold4 {_f(p34['left4'])}")
    failed.append(
        f"Q1 leftover-iss {_f(p35['left_iss'])} leftover-delay {_f(p35['left_delay'])} "
        f"issued {_f(p35['iss'])}"
    )
    failed.append(
        f"after7 leftover-delay {_f(p36['y7_delay'])} drop {_f(p36['y7_drop'])} "
        f"leftover-iss {_f(p36['y7_iss'])}"
    )
    failed.append(
        f"long so-far DSO {_f(p38['long_raw'])} leftover-iss {_f(p38['long_left'])} "
        f"issued {_f(p38['long_iss'])} short leftover-iss {_f(p38['short_left'])}"
    )
    failed.append(
        f"leftover after od30 Y7 {_f(p39['y7'])} clip {_f(p39['y7_clip'])} Y3 {_f(p39['y3'])}"
    )
    failed.append(
        f"long leftover fold-wise {_f(p40['left'])} n={p40['n']:,} pos={p40['n_pos']:,} "
        f"vs issued {_f(p40['iss'])} — slice only"
    )
    failed.append(
        f"rolling Y7 t+3 leftover-iss {_f(p41['left'])} DSO {_f(p41['raw'])} "
        f"issued {_f(p41['iss'])}"
    )
    failed.append(
        f"rolling Y3 t+3 leftover-days {_f(p42['left'])} DSO {_f(p42['raw'])} "
        f"days {_f(p42['days'])}"
    )
    failed.append(
        f"|DSO|>24 persist once {p43['n_once']} twice {p43['n_twice']} "
        f"chronic-half {p43['n_half']}"
    )
    failed.append(
        f"pooled Y3 t+3 leftover {_f(p44['left'])} days {_f(p44['days'])} "
        f"n={p44['n']} pos={p44['n_pos']}"
    )
    failed.append(
        f"rolling Y7 t+3 leftover-delay {_f(p45['left'])} delay {_f(p45['delay'])} "
        f"DSO {_f(p45['raw'])}"
    )
    failed.append(
        f"clean |DSO|≤24 leftover-iss {_f(p46['y7_left'])} leftover-days {_f(p46['y3_left'])} "
        f"vs issued {_f(p46['y7_iss'])} days {_f(p46['y3_days'])}"
    )
    failed.append(
        f"clean-refit leftover-days {_f(p47['y3_left'])} DSO {_f(p47['y3_dso'])} "
        f"days {_f(p47['y3_days'])} size {_f(p47['y3_size'])} leftover-iss {_f(p47['y7_left'])}"
    )
    failed.append(
        f"pooled Y3 robust leftover {_f(p48['left'])} drop>|24| {_f(p48['drop'])} "
        f"DSO {_f(p48['dso'])} days {_f(p48['days'])} size {_f(p48['size'])} "
        f"{'tail' if p48['tail'] else 'not KEEP'}"
    )
    failed.append(
        f"|DSO|>24 by month min {_pp(p49['min'])} max {_pp(p49['max'])} spike {p49.get('spike', '?')}"
    )
    failed.append(
        f"drop-Aug leftover-iss {_f(p50['y7_iss'])} leftover-delay {_f(p50['y7_delay'])} "
        f"leftover-days {_f(p50['y3_days'])}"
    )
    failed.append(
        f"Y6 leftover-iss {_f(p51['left_iss'])} DSO {_f(p51['dso'])} issued {_f(p51['iss'])} "
        f"— not a new Y"
    )
    failed.append(
        f"labeled-slope leftover Y3-days {_f(p52['y3'])} Y7-iss {_f(p52['y7'])} "
        f"Y7-delay {_f(p52['y7_delay'])}"
    )
    failed.append(
        f"rank(DSO) leftover-days {_f(p53['y3'])} leftover-iss {_f(p53['y7'])} "
        f"leftover-delay {_f(p53['y7_delay'])} raw-rank Y3 {_f(p53['y3_raw'])}"
    )
    failed.append(
        f"labeled leftover vs days ρ={_f(p54['rho'])} raw {_f(p54['rho_raw'])} "
        f"{'FALSE clone' if p54['false_clone'] else ('CLONE' if p54['clone'] else 'not a clone')} "
        f"leftover {_f(p54['left'])} clip {_f(p54['clip'])}"
    )
    failed.append(
        f"mix leftover-iss {_f(p55['y7_mix'])} DSO leftover {_f(p55['y7_dso'])} "
        f"Y3 mix leftover-days {_f(p55['y3_mix'])}"
    )
    failed.append(f"holdout month coverage nn={p56['nn']} (no AUROC)")
    failed.append(
        f"mix leftover vs days ρ={_f(p57['rho'])} leftover {_f(p57['left'])} "
        f"{'FALSE clone' if p57['false_clone'] else 'not a clone'}"
    )
    failed.append(
        f"leftover pending Y7 {_f(p58['y7_p'])} Y3 {_f(p58['y3_p'])} "
        f"pending+days {_f(p58['y3_pd'])} AP Y7 {_f(p58['y7_ap'])}"
    )
    failed.append(
        f"leftover delay_paid Y7 {_f(p59['y7_paid'])} Y3 {_f(p59['y3_paid'])} "
        f"zero_in Y7 {_f(p59['y7_z'])} f_ds_r Y7 {_f(p59['y7_f'])}"
    )
    failed.append(
        f"zero_in leftover vs days ρ={_f(p60['rho'])} leftover {_f(p60['left'])} "
        f"{'CLONE' if p60['clone'] else 'not a clone'}"
    )
    failed.append(
        f"leftover zero_in+days {_f(p61['left'])} vs days {_f(p61['days'])}"
    )
    failed.append(
        f"zero_in+days leftover vs days ρ={_f(p62['rho'])} leftover {_f(p62['left'])} "
        f"{'FALSE clone' if p62['clone'] else 'not a clone'}"
    )
    failed.append(
        f"leftover issued+days {_f(p63['left'])} vs days {_f(p63['days'])} ρ={_f(p63['rho'])}"
    )
    failed.append(
        f"clip leftover issued+days {_f(p64['left'])} ρ={_f(p64['rho'])} vs days {_f(p64['days'])}"
    )
    failed.append(
        f"leftover f_ds_r+days {_f(p65['left'])} ρ={_f(p65['rho'])} vs days {_f(p65['days'])}"
    )
    failed.append(
        f"leftover delay_paid+days {_f(p66['left'])} ρ={_f(p66['rho'])} vs days {_f(p66['days'])}"
    )
    failed.append(
        f"leftover AP+days {_f(p67['left'])} ρ={_f(p67['rho'])} vs days {_f(p67['days'])}"
    )
    failed.append(
        f"leftover DPO+days {_f(p68['left'])} ρ={_f(p68['rho'])} vs days {_f(p68['days'])}"
    )
    failed.append(
        f"leftover overdue+days {_f(p69['left'])} ρ={_f(p69['rho'])} vs days {_f(p69['days'])}"
    )
    failed.append(
        f"clip leftover delay_paid+days {_f(p70['left'])} ρ={_f(p70['rho'])} vs days {_f(p70['days'])}"
    )
    failed.append(
        f"clip leftover DPO+days {_f(p71['left'])} ρ={_f(p71['rho'])} "
        f"{'FALSE clone' if p71['clone'] else 'not a clone'} vs days {_f(p71['days'])}"
    )
    failed.append(
        f"leftover open+days {_f(p72['left'])} ρ={_f(p72['rho'])} vs days {_f(p72['days'])}"
    )
    failed.append(
        f"leftover delay+days {_f(p73['left'])} ρ={_f(p73['rho'])} vs days {_f(p73['days'])}"
    )
    failed.append(
        f"clip leftover delay+days {_f(p74['left'])} ρ={_f(p74['rho'])} "
        f"{'FALSE clone' if p74['clone'] else 'not a clone'} vs days {_f(p74['days'])}"
    )
    failed.append(
        f"holdout SIZE cov T1 {_pp(p75['t1'])} T2 {_pp(p75['t2'])} T3 {_pp(p75['t3'])}"
    )
    failed.append(
        f"leftover open+issued Y7 {_f(p76['left'])} ρ={_f(p76['rho'])} vs issued {_f(p76['iss'])}"
    )
    failed.append(
        f"leftover pending+issued Y7 {_f(p77['left'])} ρ={_f(p77['rho'])} vs issued {_f(p77['iss'])}"
    )
    failed.append(
        f"leftover AP+issued Y7 {_f(p78['left'])} ρ={_f(p78['rho'])} vs issued {_f(p78['iss'])}"
    )
    failed.append(
        f"clip leftover f_ds_r+days {_f(p79['left'])} ρ={_f(p79['rho'])} "
        f"{'FALSE clone' if p79['clone'] else 'not a clone'} vs days {_f(p79['days'])}"
    )
    failed.append(
        f"leftover DPO+issued Y7 {_f(p80['left'])} ρ={_f(p80['rho'])} vs issued {_f(p80['iss'])}"
    )
    failed.append(
        f"clip leftover zero_in+days {_f(p81['left'])} ρ={_f(p81['rho'])} "
        f"{'FALSE clone' if p81['clone'] else 'not a clone'} vs days {_f(p81['days'])}"
    )
    failed.append(
        f"leftover overdue+issued Y7 {_f(p82['left'])} ρ={_f(p82['rho'])} vs issued {_f(p82['iss'])}"
    )
    failed.append(
        f"leftover delay_paid+issued Y7 {_f(p83['left'])} ρ={_f(p83['rho'])} vs issued {_f(p83['iss'])}"
    )
    failed.append(
        f"leftover size+issued Y7 {_f(p84['left'])} ρ={_f(p84['rho'])} vs issued {_f(p84['iss'])}"
    )
    failed.append(
        f"leftover f_ds_r+issued Y7 {_f(p85['left'])} ρ={_f(p85['rho'])} vs issued {_f(p85['iss'])}"
    )
    failed.append(
        f"leftover AP issued+days {_f(p86['left'])} ρ={_f(p86['rho'])} vs days {_f(p86['days'])}"
    )
    failed.append(
        f"leftover od30+issued Y7 {_f(p87['left'])} ρ={_f(p87['rho'])} vs issued {_f(p87['iss'])}"
    )
    failed.append(
        f"leftover AP issued+issued Y7 {_f(p88['left'])} ρ={_f(p88['rho'])} vs issued {_f(p88['iss'])}"
    )
    failed.append(
        f"leftover zero_in+issued Y7 {_f(p89['left'])} ρ={_f(p89['rho'])} vs issued {_f(p89['iss'])}"
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
        "ph": ph,
        "ps": ps,
        "pq": pq,
        "pt": pt,
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
        "p27": p27,
        "p28": p28,
        "p29": p29,
        "p30": p30,
        "p31": p31,
        "p32": p32,
        "p33": p33,
        "p34": p34,
        "p35": p35,
        "p36": p36,
        "p37": p37,
        "p38": p38,
        "p39": p39,
        "p40": p40,
        "p41": p41,
        "p42": p42,
        "p43": p43,
        "p44": p44,
        "p45": p45,
        "p46": p46,
        "p47": p47,
        "p48": p48,
        "p49": p49,
        "p50": p50,
        "p51": p51,
        "p52": p52,
        "p53": p53,
        "p54": p54,
        "p55": p55,
        "p56": p56,
        "p57": p57,
        "p58": p58,
        "p59": p59,
        "p60": p60,
        "p61": p61,
        "p62": p62,
        "p63": p63,
        "p64": p64,
        "p65": p65,
        "p66": p66,
        "p67": p67,
        "p68": p68,
        "p69": p69,
        "p70": p70,
        "p71": p71,
        "p72": p72,
        "p73": p73,
        "p74": p74,
        "p75": p75,
        "p76": p76,
        "p77": p77,
        "p78": p78,
        "p79": p79,
        "p80": p80,
        "p81": p81,
        "p82": p82,
        "p83": p83,
        "p84": p84,
        "p85": p85,
        "p86": p86,
        "p87": p87,
        "p88": p88,
        "p89": p89,
        "decision": decision,
        "failed": failed,
        "elapsed_s": elapsed,
        "png": png,
    }
    write_md(ctx)
    append_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    print(
        f"DONE elapsed={elapsed:.0f}s y3={decision['y3']} y7={decision['y7']} "
        f"drop44={decision['drop44']}"
    )
    print(decision["headline"])
    return ctx


if __name__ == "__main__":
    run()
