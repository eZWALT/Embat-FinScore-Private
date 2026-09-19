"""Unused leftover of ``e_ar_issued`` after ``c_n_days_with_tx`` as Y3 X.

NORTH_STAR: ``e_ar_issued`` = this-period AR issuance volume (abs amount,
amount>0 = AR). Feature report: 64.1% cov, acf1 0.14, ICC 0.94 BETWEEN /
LOW_PERSIST; company-median VIF collinear with size.

TURNOVER already KEEPs ``e_ar_issued_lag1`` (night Y7 0.720 / B_shallow
0.712; Q6 KEEP 0.626 vs 0.630). DSO just DROPPED from the 44. Do **not**
grow TURNOVER. Do **not** put contemporaneous issued on the 15-col Y3
card (already off — raw-level park). Night Y3 stays 0.762 / 0.752. Days
0.711. Size 0.617. Y7 never D. Y5 never E. Y3 never B. Dark 470 stay
NaN not 0. CN leftover KEEP 0.597 after issued_lag1 — do not overwrite
credit_note_qa.

KEEP-as-X: beat size ≥0.02 AND leftover after days AND not SIZE
(|ρ| vs log1p(a_in3) ≥0.50) AND not a twin (|ρ|≥0.80 vs days / a_in3 /
issued_lag1). Leftover <0.55 dies. Rank leftover is honest; OLS can
fake a days leak.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.issued_qa

Owned: analysis/evaluate/issued_qa.py, analysis/outputs/issued_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_issued.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "issued_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "issued_leftover.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_issued.md"
AGENT = "e8b2c0d4"
WAVE = "4"
ROUND = "R4"
MODEL = "issued_qa"
X_FAM = "E"

Y3 = "y3_recover_cash_6m"
Y5 = "y5_ap_od30_ownp80"
Y7 = "y7_top1_lost"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
ISSUED_Y3_QUOTE = 0.687
ISSUED_LAG1_Y3_QUOTE = 0.671
ISSUED_Y7_QUOTE = 0.663
ISSUED_LAG1_Y7_QUOTE = 0.630
Q6_LAG1_QUOTE = 0.626
ISSUED_F4_QUOTE = 0.647
CN_LEFTOVER_QUOTE = 0.597
TURNOVER_QUOTE = 0.720
B_SHALLOW_QUOTE = 0.712
NIGHT_Y3 = 0.762
NIGHT_Y3_CORE = 0.752
COV_QUOTE = 0.641
ICC_QUOTE = 0.94
ACF1_QUOTE = 0.14
TWIN_RHO = 0.80
KEEP_DELTA = 0.02
CHANCE = 0.55
SIZE_RHO = 0.50
ICC_STYLE = 0.85
LOW_PERSIST = 0.25
MIN_POS = 50
MIN_ACF_PAIRS = 4
WRITE_WAVE = True

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "a_n_tx",
    "c_n_days_with_tx",
    "e_ar_issued",
    "e_credit_note_ratio",
    "e_dso_proxy",
    "e_pending_amt_share",
)

Y_KEEP = (Y3, Y5, Y7)


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
        if int((va & (y == 1)).sum()) == 0 or int((va & (y == 0)).sum()) == 0:
            fold_rows.append(
                {
                    "fold": k,
                    "auroc": float("nan"),
                    "sign": 0,
                    "n_va": int(va.sum()),
                    "n_pos": int((va & (y == 1)).sum()),
                }
            )
            continue
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


def leftover_diag(y, x, controls, folds, mask) -> dict:
    resid, info = ols_resid(x, *controls)
    rec = signed_oof_auroc(y, resid, folds, mask)
    rho_c = spearman(resid, controls[0]) if controls else float("nan")
    xr = pd.to_numeric(x, errors="coerce").rank(method="average")
    cr = [pd.to_numeric(c, errors="coerce").rank(method="average") for c in controls]
    rresid, _ = ols_resid(xr, *cr)
    rrec = signed_oof_auroc(y, rresid, folds, mask)
    fake = bool(np.isfinite(rho_c) and abs(rho_c) >= TWIN_RHO)
    rank_cv = _cv(rrec)
    ols_cv = _cv(rec)
    honest_dies = bool(fake or (np.isfinite(rank_cv) and rank_cv < CHANCE))
    return {
        "ols": ols_cv,
        "rank": rank_cv,
        "rho_ctrl": rho_c,
        "r2": info["r2"],
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
        "fake": fake,
        "honest_dies": honest_dies,
        "folds": fold_bits(rec),
        "rank_folds": fold_bits(rrec),
        "resid": resid,
        "rresid": rresid,
        "info": info,
        "rec": rec,
        "rrec": rrec,
    }


def _cv(res: dict) -> float:
    return float("nan") if res.get("low_power") else float(res.get("cv", float("nan")))


def _auc_row(y: str, feat: str, res: dict) -> dict:
    return {
        "y": y,
        "feature": feat,
        "n": f"{res['n_defined']:,}",
        "n_pos": f"{res['n_pos']:,}",
        "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
        "sd": _f(res["sd"]) if not res["low_power"] else "—",
        "sign": res["train_sign"] if not res["low_power"] else "—",
        "folds": fold_bits(res) if not res["low_power"] else "—",
    }


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
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        g = s.groupby(cid, sort=False)
        for k in lags:
            extra[f"{c}_lag{k}"] = g.shift(k)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    return out


def size_terciles(tr: pd.DataFrame) -> pd.Series:
    med = tr.groupby("company_id")["log_in3"].median()
    cuts = med.quantile([1 / 3, 2 / 3])
    lab = pd.Series("T2", index=med.index)
    lab[med <= cuts.iloc[0]] = "T1"
    lab[med > cuts.iloc[1]] = "T3"
    return tr["company_id"].map(lab)


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
    iss = pd.to_numeric(panel["e_ar_issued"], errors="coerce")
    panel["log1p_issued"] = np.log1p(iss.clip(lower=0))
    days = pd.to_numeric(panel["c_n_days_with_tx"], errors="coerce")
    panel["issued_per_day"] = iss / days.where(days > 0)
    panel["first_month"] = pd.to_datetime(panel["first_month"])
    panel["months_so_far"] = (
        (panel["period"].dt.year - panel["first_month"].dt.year) * 12
        + (panel["period"].dt.month - panel["first_month"].dt.month)
        + 1
    )
    grid_n = panel.groupby("company_id")["period"].size().rename("n_grid_months")
    panel = panel.merge(grid_n, on="company_id", how="left")
    panel["so_far_class"] = panel["months_so_far"].map(_trail_class)
    panel["early6"] = panel["period"] < DELAY_MASK_BEFORE
    panel = add_panel_lags(
        panel,
        ["e_ar_issued", "c_n_days_with_tx", "log1p_issued"],
        (1, 3),
    )
    leak7 = leakage_check(["e_ar_issued", "e_ar_issued_lag1"], Y7, forbidden_prefixes=["d"])
    leak3 = leakage_check(["e_ar_issued"], Y3, forbidden_prefixes=["b"])
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


def pass1_cov(tr: pd.DataFrame, book: set[str]) -> dict:
    dark = ~tr["company_id"].isin(book)
    erp = tr["company_id"].isin(book)
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    n_cm = int(len(tr))
    n_nn = int(iss.notna().sum())
    dark_nn = int(iss[dark].notna().sum())
    dark_zero = int((iss[dark] == 0).sum())
    dark_pos = int((iss[dark] > 0).sum())
    erp_nn = int(iss[erp].notna().sum())
    erp_zero = int((iss[erp] == 0).sum())
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    n_erp_co = int(tr.loc[erp, "company_id"].nunique())
    rho_size, n_size = spearman_n(iss, tr["log_in3"])
    size_flag = bool(np.isfinite(rho_size) and abs(rho_size) >= SIZE_RHO)
    finite = iss[iss.notna()]
    p50 = float(finite.median()) if n_nn else float("nan")
    p99 = float(finite.quantile(0.99)) if n_nn else float("nan")
    mx = float(finite.max()) if n_nn else float("nan")
    eq0 = _pct(int((finite == 0).sum()), n_nn)
    acf1 = median_acf(iss, tr["company_id"], 1)
    icc = icc_anova(iss, tr["company_id"])
    rows = [
        {
            "slice": "all train",
            "n_cm": f"{n_cm:,}",
            "nn": f"{n_nn:,}",
            "cov": _pp(_pct(n_nn, n_cm)),
            "eq0": _pp(eq0),
            "p50": _f(p50, 0),
            "p99": _f(p99, 0),
        },
        {
            "slice": "ever-ERP",
            "n_cm": f"{int(erp.sum()):,}",
            "nn": f"{erp_nn:,}",
            "cov": _pp(_pct(erp_nn, int(erp.sum()))),
            "eq0": _pp(_pct(erp_zero, erp_nn)),
            "p50": "—",
            "p99": "—",
        },
        {
            "slice": "dark 470",
            "n_cm": f"{int(dark.sum()):,}",
            "nn": f"{dark_nn:,}",
            "cov": _pp(_pct(dark_nn, int(dark.sum()))),
            "eq0": _pp(_pct(dark_zero, dark_nn) if dark_nn else float("nan")),
            "p50": "—",
            "p99": "—",
        },
    ]
    prose = (
        f"Train issued nn={n_nn:,} cov={_pp(_pct(n_nn, n_cm))} "
        f"(feature report {COV_QUOTE:.1%}). Dark never-ERP {n_dark_co} "
        f"(want 470): nn={dark_nn} zero={dark_zero} pos={dark_pos} "
        f"{'CONFIRM NaN not 0' if dark_nn == 0 and dark_zero == 0 else ('BOOK stub' if dark_nn else 'check')}. "
        f"Ever-ERP {n_erp_co} nn={erp_nn:,} of which zero={erp_zero:,}. "
        f"p50={_f(p50, 0)} p99={_f(p99, 0)} max={_f(mx, 0)}. "
        f"ρ vs log1p(a_in3)={_f(rho_size)} n={n_size} "
        f"{'SIZE' if size_flag else 'not SIZE'}. "
        f"acf1={_f(acf1)} (quote {ACF1_QUOTE:.2f}) ICC={_f(icc['icc'])} "
        f"(quote {ICC_QUOTE:.2f}) k={icc['k']}."
    )
    print(prose)
    return {
        "rows": rows,
        "cov": _pct(n_nn, n_cm),
        "nn": n_nn,
        "dark_nn": dark_nn,
        "dark_zero": dark_zero,
        "dark_pos": dark_pos,
        "n_dark_co": n_dark_co,
        "n_erp_co": n_erp_co,
        "rho_size": rho_size,
        "size_flag": size_flag,
        "p50": p50,
        "p99": p99,
        "acf1": acf1,
        "icc": icc["icc"],
        "prose": prose,
    }


def pass2_twins(tr: pd.DataFrame) -> dict:
    iss = tr["e_ar_issued"]
    pairs = [
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("a_in3", tr["a_in3"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("a_n_tx", tr["a_n_tx"]),
        ("e_ar_issued_lag1", tr["e_ar_issued_lag1"]),
        ("e_credit_note_ratio", tr["e_credit_note_ratio"]),
        ("e_dso_proxy", tr["e_dso_proxy"]),
        ("e_pending_amt_share", tr["e_pending_amt_share"]),
    ]
    rows = []
    rhos = {}
    twins = []
    for name, s in pairs:
        rho, n = spearman_n(iss, s)
        rhos[name] = rho
        twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
        if twin:
            twins.append(name)
        rows.append(
            {
                "vs": name,
                "ρ": _f(rho),
                "n": f"{n:,}",
                "twin?": "TWIN" if twin else "",
            }
        )
    size_flag = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    gate = [t for t in twins if t in ("c_n_days_with_tx", "a_in3", "e_ar_issued_lag1")]
    prose = (
        f"issued vs days ρ={_f(rhos['c_n_days_with_tx'])} "
        f"({'TWIN' if 'c_n_days_with_tx' in twins else 'not a twin'}). "
        f"vs a_in3 {_f(rhos['a_in3'])} vs log1p(a_in3) {_f(rhos['log1p(a_in3)'])} "
        f"({'SIZE' if size_flag else 'not SIZE'}). "
        f"vs a_n_tx {_f(rhos['a_n_tx'])} vs issued_lag1 {_f(rhos['e_ar_issued_lag1'])} "
        f"vs CN {_f(rhos['e_credit_note_ratio'])} vs DSO {_f(rhos['e_dso_proxy'])} "
        f"vs pending {_f(rhos['e_pending_amt_share'])}. "
        f"TWIN |ρ|≥0.80: {', '.join(twins) if twins else 'none'}."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "twins": twins,
        "gate_twins": gate,
        "size_flag": size_flag,
        "rho_days": rhos["c_n_days_with_tx"],
        "rho_size": rhos["log1p(a_in3)"],
        "rho_lag1": rhos["e_ar_issued_lag1"],
        "rho_n_tx": rhos["a_n_tx"],
        "rho_cn": rhos["e_credit_note_ratio"],
        "rho_dso": rhos["e_dso_proxy"],
        "prose": prose,
    }


def pass3_singles(tr: pd.DataFrame) -> dict:
    feats = {
        "e_ar_issued": tr["e_ar_issued"],
        "e_ar_issued_lag1": tr["e_ar_issued_lag1"],
        "log1p_issued": tr["log1p_issued"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "log1p_a_in3": tr["log_in3"],
        "a_n_tx": tr["a_n_tx"],
        "e_credit_note_ratio": tr["e_credit_note_ratio"],
        "e_dso_proxy": tr["e_dso_proxy"],
    }
    rows = []
    store = {}
    for ycol in (Y3, Y7):
        for name, s in feats.items():
            res = signed_oof_auroc(tr[ycol], s, tr["fold"], tr[ycol].notna())
            store[(ycol, name)] = res
            rows.append(_auc_row(ycol, name, res))
            print(f"{ycol} {name}: {_f(_cv(res))} n={res['n_defined']:,} pos={res['n_pos']:,}")
    y3 = _cv(store[(Y3, "e_ar_issued")])
    y3_lag = _cv(store[(Y3, "e_ar_issued_lag1")])
    days = _cv(store[(Y3, "c_n_days_with_tx")])
    size = _cv(store[(Y3, "log1p_a_in3")])
    y7 = _cv(store[(Y7, "e_ar_issued")])
    y7_lag = _cv(store[(Y7, "e_ar_issued_lag1")])
    f4 = fold_k(store[(Y7, "e_ar_issued")], 4)
    days_ok = bool(np.isfinite(days) and abs(days - DAYS_BENCH) < 0.005)
    size_ok = bool(np.isfinite(size) and abs(size - SIZE_QUOTE) < 0.005)
    y3_ok = bool(np.isfinite(y3) and abs(y3 - ISSUED_Y3_QUOTE) < 0.005)
    y7_lag_ok = bool(np.isfinite(y7_lag) and abs(y7_lag - ISSUED_LAG1_Y7_QUOTE) < 0.005)
    beat = float(y3 - size) if np.isfinite(y3) and np.isfinite(size) else float("nan")
    prose = (
        f"Y3 issued {_f(y3)} "
        f"({'CONFIRM' if y3_ok else 'off'} {ISSUED_Y3_QUOTE:.3f}) "
        f"issued_lag1 {_f(y3_lag)} vs days {_f(days)} "
        f"({'CONFIRM' if days_ok else 'off'} {DAYS_BENCH:.3f}) "
        f"vs size {_f(size)} ({'CONFIRM' if size_ok else 'off'} {SIZE_QUOTE:.3f}). "
        f"Y7 issued {_f(y7)} (quote {ISSUED_Y7_QUOTE:.3f}) issued_lag1 {_f(y7_lag)} "
        f"({'CONFIRM' if y7_lag_ok else 'off'} {ISSUED_LAG1_Y7_QUOTE:.3f}). "
        f"Fold 4 issued {_f(f4)} (quote {ISSUED_F4_QUOTE:.3f}). "
        f"Beat size {_f(beat)}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "y3": y3,
        "y3_lag": y3_lag,
        "days": days,
        "size": size,
        "y7": y7,
        "y7_lag": y7_lag,
        "f4": f4,
        "beat_size": beat,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "y3_ok": y3_ok,
        "y7_lag_ok": y7_lag_ok,
        "prose": prose,
    }


def pass4_leftover(tr: pd.DataFrame) -> dict:
    """Honest leftover after days (Y3). Inverse: days leftover after issued."""
    d_days = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    d_inv = leftover_diag(
        tr[Y3], tr["c_n_days_with_tx"], (tr["e_ar_issued"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "cut": "issued leftover after days (Y3)",
            "rank": _f(d_days["rank"]),
            "OLS": _f(d_days["ols"]),
            "ρ(resid,ctrl)": _f(d_days["rho_ctrl"]),
            "R²": _f(d_days["r2"]),
            "fake?": "FALSE clone" if d_days["fake"] else "",
        },
        {
            "cut": "days leftover after issued (Y3)",
            "rank": _f(d_inv["rank"]),
            "OLS": _f(d_inv["ols"]),
            "ρ(resid,ctrl)": _f(d_inv["rho_ctrl"]),
            "R²": _f(d_inv["r2"]),
            "fake?": "FALSE clone" if d_inv["fake"] else "",
        },
    ]
    inv_lives = bool(np.isfinite(d_inv["rank"]) and d_inv["rank"] >= CHANCE and not d_inv["fake"])
    prose = (
        f"Y3 leftover after days rank {_f(d_days['rank'])} OLS {_f(d_days['ols'])} "
        f"ρ(resid,days)={_f(d_days['rho_ctrl'])} R²={_f(d_days['r2'])} "
        f"{'FALSE clone' if d_days['fake'] else 'not a days clone'} — "
        f"{'dies' if d_days['honest_dies'] else 'lives'}. "
        f"Inverse: days leftover after issued rank {_f(d_inv['rank'])} OLS {_f(d_inv['ols'])} "
        f"{'survives — keep the 0.711 bar' if inv_lives else 'also dies'}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3_rank": d_days["rank"],
        "y3_ols": d_days["ols"],
        "y3_rho": d_days["rho_ctrl"],
        "y3_fake": d_days["fake"],
        "y3_dies": d_days["honest_dies"],
        "inv_rank": d_inv["rank"],
        "inv_ols": d_inv["ols"],
        "inv_lives": inv_lives,
        "d_days": d_days,
        "d_inv": d_inv,
        "prose": prose,
    }


def pass5_more_bars(tr: pd.DataFrame) -> dict:
    """Leftover after size; after issued_lag1 (is now vs lag a twin?)."""
    d_size = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["log_in3"],), tr["fold"], tr[Y3].notna()
    )
    d_lag = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["e_ar_issued_lag1"],), tr["fold"], tr[Y3].notna()
    )
    d_both = leftover_diag(
        tr[Y3],
        tr["e_ar_issued"],
        (tr["c_n_days_with_tx"], tr["log_in3"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "after size",
            "rank": _f(d_size["rank"]),
            "OLS": _f(d_size["ols"]),
            "ρ(resid,ctrl)": _f(d_size["rho_ctrl"]),
            "dies?": "dies" if d_size["honest_dies"] else "lives",
        },
        {
            "bar": "after issued_lag1",
            "rank": _f(d_lag["rank"]),
            "OLS": _f(d_lag["ols"]),
            "ρ(resid,ctrl)": _f(d_lag["rho_ctrl"]),
            "dies?": "dies" if d_lag["honest_dies"] else "lives",
        },
        {
            "bar": "after days+size",
            "rank": _f(d_both["rank"]),
            "OLS": _f(d_both["ols"]),
            "ρ(resid,ctrl)": _f(d_both["rho_ctrl"]),
            "dies?": "dies" if d_both["honest_dies"] else "lives",
        },
    ]
    prose = (
        f"Y3 leftover after size rank {_f(d_size['rank'])} OLS {_f(d_size['ols'])}. "
        f"After issued_lag1 rank {_f(d_lag['rank'])} OLS {_f(d_lag['ols'])} "
        f"ρ(resid,lag1)={_f(d_lag['rho_ctrl'])} "
        f"{'(now vs lag is leftover-live)' if not d_lag['honest_dies'] else '(now vs lag leftover dies)'}. "
        f"After days+size rank {_f(d_both['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "size_rank": d_size["rank"],
        "lag_rank": d_lag["rank"],
        "lag_ols": d_lag["ols"],
        "lag_dies": d_lag["honest_dies"],
        "both_rank": d_both["rank"],
        "both_dies": d_both["honest_dies"],
        "prose": prose,
    }


def pass6_terciles(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    rows = []
    store = {}
    for label, mask in (
        ("T1", terc == "T1"),
        ("T2", terc == "T2"),
        ("T3", terc == "T3"),
        ("T2+T3", terc.isin(["T2", "T3"])),
        ("all", pd.Series(True, index=tr.index)),
    ):
        d = leftover_diag(
            tr[Y3],
            tr["e_ar_issued"],
            (tr["c_n_days_with_tx"],),
            tr["fold"],
            mask & tr[Y3].notna(),
        )
        store[label] = d
        raw = signed_oof_auroc(tr[Y3], tr["e_ar_issued"], tr["fold"], mask & tr[Y3].notna())
        rows.append(
            {
                "slice": label,
                "raw issued": _f(_cv(raw)),
                "leftover rank": _f(d["rank"]),
                "leftover OLS": _f(d["ols"]),
                "n": f"{d['n']:,}",
                "dies?": "dies" if d["honest_dies"] else "lives",
            }
        )
    dies_inside = bool(store["T1"]["honest_dies"] and store["T2+T3"]["honest_dies"])
    prose = (
        f"SIZE terciles leftover after days: T1 rank {_f(store['T1']['rank'])} "
        f"T2+T3 {_f(store['T2+T3']['rank'])} T3 {_f(store['T3']['rank'])}. "
        f"{'dies inside T1 and T2+T3' if dies_inside else 'lives in a tercile'}."
    )
    print(prose)
    return {
        "rows": rows,
        "t1": store["T1"]["rank"],
        "t23": store["T2+T3"]["rank"],
        "t3": store["T3"]["rank"],
        "dies_inside": dies_inside,
        "prose": prose,
    }


def pass7_q6(tr: pd.DataFrame) -> dict:
    """lag1 already KEEP. Contemporaneous leftover after days_lag1 / issued_lag1 on short books?"""
    short = tr["so_far_class"] == "short_<12"
    rows = []
    store = {}
    for label, mask in (
        ("all", tr[Y3].notna()),
        ("short_<12", short & tr[Y3].notna()),
        ("long_>=18", (tr["so_far_class"] == "long_>=18") & tr[Y3].notna()),
    ):
        for name, s in (
            ("issued now", tr["e_ar_issued"]),
            ("issued_lag1", tr["e_ar_issued_lag1"]),
            ("days", tr["c_n_days_with_tx"]),
            ("days_lag1", tr["c_n_days_with_tx_lag1"]),
        ):
            res = signed_oof_auroc(tr[Y3], s, tr["fold"], mask)
            store[(label, name)] = res
            rows.append(_auc_row(Y3, f"{label} {name}", res))
        d_days_l1 = leftover_diag(
            tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx_lag1"],), tr["fold"], mask
        )
        d_iss_l1 = leftover_diag(
            tr[Y3], tr["e_ar_issued"], (tr["e_ar_issued_lag1"],), tr["fold"], mask
        )
        store[(label, "left_days_lag1")] = d_days_l1
        store[(label, "left_iss_lag1")] = d_iss_l1
        rows.append(
            {
                "y": Y3,
                "feature": f"{label} leftover after days_lag1",
                "n": f"{d_days_l1['n']:,}",
                "n_pos": f"{d_days_l1['n_pos']:,}",
                "CV": _f(d_days_l1["rank"]),
                "sd": "rank",
                "sign": "—",
                "folds": d_days_l1["rank_folds"],
            }
        )
        rows.append(
            {
                "y": Y3,
                "feature": f"{label} leftover after issued_lag1",
                "n": f"{d_iss_l1['n']:,}",
                "n_pos": f"{d_iss_l1['n_pos']:,}",
                "CV": _f(d_iss_l1["rank"]),
                "sd": "rank",
                "sign": "—",
                "folds": d_iss_l1["rank_folds"],
            }
        )
    y7_lag_short = signed_oof_auroc(
        tr[Y7], tr["e_ar_issued_lag1"], tr["fold"], short & tr[Y7].notna()
    )
    short_left = store[("short_<12", "left_days_lag1")]
    short_iss = store[("short_<12", "left_iss_lag1")]
    q6_lag = _cv(y7_lag_short)
    q6_ok = bool(np.isfinite(q6_lag) and abs(q6_lag - Q6_LAG1_QUOTE) < 0.015)
    short_dies = bool(short_left["honest_dies"] and short_iss["honest_dies"])
    q6 = "CLOSE" if short_dies or short_left["fake"] or short_iss["fake"] else "KEEP-check"
    prose = (
        f"Q6 Y7 issued_lag1 on short so-far {_f(q6_lag)} "
        f"({'CONFIRM' if q6_ok else 'off'} {Q6_LAG1_QUOTE:.3f} vs night 0.630). "
        f"Y3 contemporaneous leftover after days_lag1 on short rank "
        f"{_f(short_left['rank'])}; after issued_lag1 {_f(short_iss['rank'])}. "
        f"{'dies on short books' if short_dies or short_left['fake'] else 'lives on short — still do not grow TURNOVER'}. "
        f"{'days_lag1 leftover is a FALSE clone' if short_left['fake'] else ''}. "
        f"Q6 contemporaneous {q6} (lag1 KEEP locked). "
    )
    print(prose)
    return {
        "rows": rows,
        "q6": q6,
        "q6_lag": q6_lag,
        "q6_ok": q6_ok,
        "short_days_l1": short_left["rank"],
        "short_iss_l1": short_iss["rank"],
        "short_dies": short_dies,
        "prose": prose,
    }


def pass8_y7_addon(tr: pd.DataFrame) -> dict:
    """Y7 leftover after issued_lag1 of contemporaneous issued — CLOSE as TURNOVER add-on."""
    d = leftover_diag(
        tr[Y7], tr["e_ar_issued"], (tr["e_ar_issued_lag1"],), tr["fold"], tr[Y7].notna()
    )
    d_days = leftover_diag(
        tr[Y7], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y7].notna()
    )
    raw = signed_oof_auroc(tr[Y7], tr["e_ar_issued"], tr["fold"], tr[Y7].notna())
    lag = signed_oof_auroc(tr[Y7], tr["e_ar_issued_lag1"], tr["fold"], tr[Y7].notna())
    rows = [
        _auc_row(Y7, "issued now", raw),
        _auc_row(Y7, "issued_lag1", lag),
        {
            "y": Y7,
            "feature": "leftover after issued_lag1 (rank)",
            "n": f"{d['n']:,}",
            "n_pos": f"{d['n_pos']:,}",
            "CV": _f(d["rank"]),
            "sd": _f(d["ols"]),
            "sign": "rank/OLS",
            "folds": d["rank_folds"],
        },
        {
            "y": Y7,
            "feature": "leftover after days (rank)",
            "n": f"{d_days['n']:,}",
            "n_pos": f"{d_days['n_pos']:,}",
            "CV": _f(d_days["rank"]),
            "sd": _f(d_days["ols"]),
            "sign": "rank/OLS",
            "folds": d_days["rank_folds"],
        },
    ]
    addon_close = bool(d["honest_dies"] or (np.isfinite(d["rank"]) and d["rank"] < 0.60))
    prose = (
        f"Y7 contemporaneous leftover after issued_lag1 rank {_f(d['rank'])} "
        f"OLS {_f(d['ols'])} ρ={_f(d['rho_ctrl'])} R²={_f(d['r2'])}. "
        f"Raw issued {_f(_cv(raw))} vs lag1 {_f(_cv(lag))}. "
        f"{'CLOSE as TURNOVER add-on — do not grow 0.720' if addon_close else 'leftover lives — still do not grow TURNOVER'}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "ols": d["ols"],
        "dies": d["honest_dies"],
        "addon_close": addon_close,
        "days_rank": d_days["rank"],
        "prose": prose,
    }


def pass9_cn(tr: pd.DataFrame) -> dict:
    """CN leftover after issued_lag1 already KEEP 0.597 — confirm, do not overwrite."""
    d = leftover_diag(
        tr[Y7],
        tr["e_credit_note_ratio"],
        (tr["e_ar_issued_lag1"],),
        tr["fold"],
        tr[Y7].notna(),
    )
    raw = signed_oof_auroc(tr[Y7], tr["e_credit_note_ratio"], tr["fold"], tr[Y7].notna())
    ok = bool(np.isfinite(d["rank"]) and abs(d["rank"] - CN_LEFTOVER_QUOTE) < 0.03) or (
        np.isfinite(d["ols"]) and abs(d["ols"] - CN_LEFTOVER_QUOTE) < 0.03
    )
    rows = [
        _auc_row(Y7, "CN raw", raw),
        {
            "y": Y7,
            "feature": "CN leftover after issued_lag1 rank",
            "n": f"{d['n']:,}",
            "n_pos": f"{d['n_pos']:,}",
            "CV": _f(d["rank"]),
            "sd": _f(d["ols"]),
            "sign": "rank/OLS",
            "folds": d["rank_folds"],
        },
    ]
    prose = (
        f"CN leftover after issued_lag1 rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"(locked KEEP {CN_LEFTOVER_QUOTE:.3f}"
        f"{' CONFIRM' if ok else ' — this-run rank/OLS; do not overwrite credit_note_qa'}). "
        "This lane does not rewrite credit_note_qa."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "ols": d["ols"],
        "ok": ok,
        "prose": prose,
    }


def pass10_fold4(tr: pd.DataFrame) -> dict:
    """Fold 4: issued already owns it — leftover after issued_lag1 must not reopen TURNOVER."""
    raw = signed_oof_auroc(tr[Y7], tr["e_ar_issued"], tr["fold"], tr[Y7].notna())
    lag = signed_oof_auroc(tr[Y7], tr["e_ar_issued_lag1"], tr["fold"], tr[Y7].notna())
    d = leftover_diag(
        tr[Y7], tr["e_ar_issued"], (tr["e_ar_issued_lag1"],), tr["fold"], tr[Y7].notna()
    )
    f4_iss = fold_k(raw, 4)
    f4_lag = fold_k(lag, 4)
    f4_left = None
    for r, bit in zip(d["rrec"].get("folds", []), d["rank_folds"].split()):
        if int(r["fold"]) == 4:
            f4_left = float(r["auroc"]) if np.isfinite(r["auroc"]) else float("nan")
    owns = bool(np.isfinite(f4_iss) and abs(f4_iss - ISSUED_F4_QUOTE) < 0.015)
    rows = [
        {
            "stem": "issued now",
            "CV": _f(_cv(raw)),
            "fold4": _f(f4_iss),
            "folds": fold_bits(raw),
        },
        {
            "stem": "issued_lag1",
            "CV": _f(_cv(lag)),
            "fold4": _f(f4_lag),
            "folds": fold_bits(lag),
        },
        {
            "stem": "now leftover after lag1 (rank)",
            "CV": _f(d["rank"]),
            "fold4": _f(f4_left),
            "folds": d["rank_folds"],
        },
    ]
    prose = (
        f"Fold 4 Y7 issued {_f(f4_iss)} (quote {ISSUED_F4_QUOTE:.3f}"
        f"{' CONFIRM' if owns else ''}) vs issued_lag1 {_f(f4_lag)}. "
        f"Contemporaneous leftover-after-lag1 fold 4 {_f(f4_left)}. "
        "Do not reopen TURNOVER 0.720."
    )
    print(prose)
    return {
        "rows": rows,
        "f4_iss": f4_iss,
        "f4_lag": f4_lag,
        "f4_left": f4_left,
        "owns": owns,
        "prose": prose,
    }


def pass_holdout(panel: pd.DataFrame, book: set[str]) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    assert set(ho["company_id"]) <= load_holdout()
    iss = pd.to_numeric(ho["e_ar_issued"], errors="coerce")
    dark = ~ho["company_id"].isin(book)
    y7 = pd.to_numeric(ho[Y7], errors="coerce")
    rows = [
        {
            "col": "e_ar_issued",
            "n_cm": f"{len(ho):,}",
            "nn": f"{int(iss.notna().sum()):,}",
            "cov": _pp(_pct(int(iss.notna().sum()), len(ho))),
            "dark nn": int(iss[dark].notna().sum()),
            "Y7 pos": int((y7 == 1).sum()),
        }
    ]
    prose = (
        f"Holdout 72 coverage only: {len(ho):,} CM / {ho['company_id'].nunique()} companies. "
        f"issued cov {_pp(_pct(int(iss.notna().sum()), len(ho)))}; "
        f"dark nn={int(iss[dark].notna().sum())}. Y7 pos={int((y7==1).sum())}. No AUROC."
    )
    print(prose)
    return {"rows": rows, "cov": _pct(int(iss.notna().sum()), len(ho)), "prose": prose}


def pass_samen(tr: pd.DataFrame) -> dict:
    ok = tr[Y3].notna() & tr["e_ar_issued"].notna() & tr["e_ar_issued_lag1"].notna()
    now = signed_oof_auroc(tr[Y3], tr["e_ar_issued"], tr["fold"], ok)
    lag = signed_oof_auroc(tr[Y3], tr["e_ar_issued_lag1"], tr["fold"], ok)
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], ok)
    d = leftover_diag(tr[Y3], tr["e_ar_issued"], (tr["e_ar_issued_lag1"],), tr["fold"], ok)
    rows = [
        _auc_row(Y3, "issued same-n", now),
        _auc_row(Y3, "issued_lag1 same-n", lag),
        _auc_row(Y3, "days same-n", days),
    ]
    prose = (
        f"Same-n issued vs lag1: now {_f(_cv(now))} lag1 {_f(_cv(lag))} days {_f(_cv(days))} "
        f"leftover-after-lag1 rank {_f(d['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "now": _cv(now),
        "lag": _cv(lag),
        "days": _cv(days),
        "left": d["rank"],
        "prose": prose,
    }


def pass_log1p(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["log1p_issued"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    raw = signed_oof_auroc(tr[Y3], tr["log1p_issued"], tr["fold"], tr[Y3].notna())
    rows = [
        _auc_row(Y3, "log1p(issued)", raw),
        {
            "y": Y3,
            "feature": "log1p leftover after days (rank)",
            "n": f"{d['n']:,}",
            "n_pos": f"{d['n_pos']:,}",
            "CV": _f(d["rank"]),
            "sd": _f(d["ols"]),
            "sign": "rank/OLS",
            "folds": d["rank_folds"],
        },
    ]
    prose = (
        f"log1p(issued) Y3 {_f(_cv(raw))} leftover-after-days rank {_f(d['rank'])} "
        f"OLS {_f(d['ols'])} ρ={_f(d['rho_ctrl'])} "
        f"{'dies' if d['honest_dies'] else 'lives — still loses if below days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "raw": _cv(raw),
        "rank": d["rank"],
        "ols": d["ols"],
        "dies": d["honest_dies"],
        "prose": prose,
    }


def pass_intensity(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["issued_per_day"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    raw = signed_oof_auroc(tr[Y3], tr["issued_per_day"], tr["fold"], tr[Y3].notna())
    rows = [
        _auc_row(Y3, "issued/days", raw),
        {
            "y": Y3,
            "feature": "intensity leftover after days (rank)",
            "n": f"{d['n']:,}",
            "n_pos": f"{d['n_pos']:,}",
            "CV": _f(d["rank"]),
            "sd": _f(d["ols"]),
            "sign": "rank/OLS",
            "folds": d["rank_folds"],
        },
    ]
    prose = (
        f"issued/days intensity Y3 {_f(_cv(raw))} leftover-after-days rank {_f(d['rank'])} "
        f"OLS {_f(d['ols'])} {'dies' if d['honest_dies'] else 'lives'}."
    )
    print(prose)
    return {
        "rows": rows,
        "raw": _cv(raw),
        "rank": d["rank"],
        "dies": d["honest_dies"],
        "prose": prose,
    }


def pass_fat(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    p90 = float(iss[iss > 0].quantile(0.90)) if int((iss > 0).sum()) else float("nan")
    fat = iss > p90
    rest = iss.notna() & ~fat
    rows = []
    store = {}
    for label, mask in (("fat p90+", fat), ("rest", rest), ("all", iss.notna())):
        d = leftover_diag(
            tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], mask & tr[Y3].notna()
        )
        raw = signed_oof_auroc(tr[Y3], tr["e_ar_issued"], tr["fold"], mask & tr[Y3].notna())
        store[label] = d
        rows.append(
            {
                "slice": label,
                "raw": _f(_cv(raw)),
                "leftover rank": _f(d["rank"]),
                "n": f"{d['n']:,}",
                "dies?": "dies" if d["honest_dies"] else "lives",
            }
        )
    prose = (
        f"Fat-issued p90={_f(p90, 0)}: leftover-days rank {_f(store['fat p90+']['rank'])} "
        f"vs rest {_f(store['rest']['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "p90": p90,
        "fat": store["fat p90+"]["rank"],
        "rest": store["rest"]["rank"],
        "prose": prose,
    }


def pass_dark_erp(tr: pd.DataFrame, book: set[str]) -> dict:
    erp = tr["company_id"].isin(book)
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], erp & tr[Y3].notna()
    )
    raw = signed_oof_auroc(tr[Y3], tr["e_ar_issued"], tr["fold"], erp & tr[Y3].notna())
    rows = [
        _auc_row(Y3, "issued book-only", raw),
        {
            "y": Y3,
            "feature": "book leftover after days (rank)",
            "n": f"{d['n']:,}",
            "n_pos": f"{d['n_pos']:,}",
            "CV": _f(d["rank"]),
            "sd": _f(d["ols"]),
            "sign": "rank/OLS",
            "folds": d["rank_folds"],
        },
    ]
    prose = (
        f"Book-only (drop 470) Y3 issued {_f(_cv(raw))} leftover-days rank {_f(d['rank'])}."
    )
    print(prose)
    return {"rows": rows, "raw": _cv(raw), "rank": d["rank"], "prose": prose}


def pass_dark_stub(tr: pd.DataFrame, book: set[str]) -> dict:
    """The one dark BOOK stub — document, do not rewrite invoices.py."""
    dark = ~tr["company_id"].isin(book)
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    hit = tr.loc[dark & iss.notna(), ["company_id", "period", "e_ar_issued", "group_id"]]
    rows = []
    for _, r in hit.iterrows():
        rows.append(
            {
                "company_id": r["company_id"],
                "period": str(pd.to_datetime(r["period"]).date()),
                "e_ar_issued": _f(r["e_ar_issued"], 2),
                "group_id": r["group_id"],
            }
        )
    prose = (
        f"Dark BOOK stub rows={len(hit)} "
        f"{'(expected 1)' if len(hit) == 1 else '(not 1 — document)'}. "
        "Do not rewrite invoices.py."
    )
    print(prose)
    return {"rows": rows, "n": int(len(hit)), "prose": prose}


def pass_log1p_both(tr: pd.DataFrame) -> dict:
    """log1p leftover after days was 0.608 ρ=-0.038 — leftover after days+size?"""
    d = leftover_diag(
        tr[Y3],
        tr["log1p_issued"],
        (tr["c_n_days_with_tx"], tr["log_in3"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    d_size = leftover_diag(
        tr[Y3], tr["log1p_issued"], (tr["log_in3"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "bar": "log1p after days+size",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        },
        {
            "bar": "log1p after size",
            "rank": _f(d_size["rank"]),
            "OLS": _f(d_size["ols"]),
            "ρ": _f(d_size["rho_ctrl"]),
            "dies?": "dies" if d_size["honest_dies"] else "lives",
        },
    ]
    prose = (
        f"log1p leftover after days+size rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"ρ={_f(d['rho_ctrl'])} {'dies' if d['honest_dies'] else 'lives — still off the card'}. "
        f"After size {_f(d_size['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "both": d["rank"],
        "size": d_size["rank"],
        "dies": d["honest_dies"],
        "prose": prose,
    }


def pass_fold_left(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna())
    rows = []
    for r, rr in zip(d["rrec"].get("folds", []), days.get("folds", [])):
        rows.append(
            {
                "fold": r["fold"],
                "leftover rank": _f(r["auroc"]),
                "days": _f(rr["auroc"]),
                "n_pos": r["n_pos"],
            }
        )
    prose = (
        f"Y3 fold-wise leftover-days rank {_f(d['rank'])} folds {d['rank_folds']} "
        f"vs days {fold_bits(days)}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "folds": d["rank_folds"], "prose": prose}


def pass_demean(tr: pd.DataFrame) -> dict:
    dso = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    demean = dso - dso.groupby(tr["company_id"]).transform("mean")
    mean = dso.groupby(tr["company_id"]).transform("mean")
    raw = signed_oof_auroc(tr[Y3], dso, tr["fold"], tr[Y3].notna())
    d_raw = leftover_diag(
        tr[Y3], dso, (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    d_dm = leftover_diag(
        tr[Y3], demean, (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    d_mn = leftover_diag(
        tr[Y3], mean, (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        _auc_row(Y3, "issued raw", raw),
        {
            "y": Y3,
            "feature": "demean leftover-days rank",
            "n": f"{d_dm['n']:,}",
            "n_pos": f"{d_dm['n_pos']:,}",
            "CV": _f(d_dm["rank"]),
            "sd": _f(d_dm["ols"]),
            "sign": "rank/OLS",
            "folds": d_dm["rank_folds"],
        },
        {
            "y": Y3,
            "feature": "company-mean leftover-days rank",
            "n": f"{d_mn['n']:,}",
            "n_pos": f"{d_mn['n_pos']:,}",
            "CV": _f(d_mn["rank"]),
            "sd": _f(d_mn["ols"]),
            "sign": "rank/OLS",
            "folds": d_mn["rank_folds"],
        },
    ]
    prose = (
        f"Demean leftover-days rank {_f(d_dm['rank'])} mean leftover {_f(d_mn['rank'])} "
        f"raw leftover {_f(d_raw['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "demean": d_dm["rank"],
        "mean": d_mn["rank"],
        "raw": d_raw["rank"],
        "prose": prose,
    }


def pass_samen_days(tr: pd.DataFrame) -> dict:
    ok = tr[Y3].notna() & tr["e_ar_issued"].notna() & tr["c_n_days_with_tx"].notna()
    d = leftover_diag(tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], ok)
    now = signed_oof_auroc(tr[Y3], tr["e_ar_issued"], tr["fold"], ok)
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], ok)
    rows = [
        _auc_row(Y3, "issued same-n days", now),
        _auc_row(Y3, "days same-n", days),
        {
            "y": Y3,
            "feature": "leftover same-n rank",
            "n": f"{d['n']:,}",
            "n_pos": f"{d['n_pos']:,}",
            "CV": _f(d["rank"]),
            "sd": _f(d["ols"]),
            "sign": f"fake={d['fake']}",
            "folds": d["rank_folds"],
        },
    ]
    prose = (
        f"Same-n leftover after days rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"ρ={_f(d['rho_ctrl'])} fake={d['fake']} issued {_f(_cv(now))} days {_f(_cv(days))}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "ols": d["ols"],
        "fake": d["fake"],
        "days": _cv(days),
        "prose": prose,
    }


def pass_holdout_size(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    assert set(ho["company_id"]) <= load_holdout()
    terc = size_terciles(ho)
    iss = pd.to_numeric(ho["e_ar_issued"], errors="coerce")
    rows = []
    store = {}
    for label in ("T1", "T2", "T3"):
        m = terc == label
        nn = int(iss[m].notna().sum())
        n = int(m.sum())
        cov = _pct(nn, n)
        store[label] = cov
        rows.append({"slice": f"holdout {label}", "n": n, "nn": nn, "cov": _pp(cov)})
    prose = (
        f"Holdout SIZE tercile coverage only: T1 {_pp(store['T1'])} "
        f"T2 {_pp(store['T2'])} T3 {_pp(store['T3'])}. No AUROC."
    )
    print(prose)
    return {"rows": rows, "t1": store["T1"], "t2": store["T2"], "t3": store["T3"], "prose": prose}


def pass_t23_fake(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    d = leftover_diag(
        tr[Y3],
        tr["e_ar_issued"],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        (terc.isin(["T2", "T3"])) & tr[Y3].notna(),
    )
    rows = [
        {
            "slice": "T2+T3 leftover after days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"T2+T3 leftover after days rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"ρ={_f(d['rho_ctrl'])} {'FALSE clone' if d['fake'] else 'not a clone'}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "fake": d["fake"],
        "dies": d["honest_dies"],
        "prose": prose,
    }


def pass_pos_only(tr: pd.DataFrame) -> dict:
    """Leftover after days on issued>0 only (zeros are ERP fill, not dark)."""
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    pos = iss > 0
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], pos & tr[Y3].notna()
    )
    raw = signed_oof_auroc(tr[Y3], tr["e_ar_issued"], tr["fold"], pos & tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], pos & tr[Y3].notna())
    rows = [
        _auc_row(Y3, "issued>0", raw),
        _auc_row(Y3, "days on issued>0", days),
        {
            "y": Y3,
            "feature": "issued>0 leftover-days rank",
            "n": f"{d['n']:,}",
            "n_pos": f"{d['n_pos']:,}",
            "CV": _f(d["rank"]),
            "sd": _f(d["ols"]),
            "sign": f"fake={d['fake']}",
            "folds": d["rank_folds"],
        },
    ]
    prose = (
        f"issued>0 leftover after days rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"ρ={_f(d['rho_ctrl'])} fake={d['fake']} raw {_f(_cv(raw))} days {_f(_cv(days))}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "fake": d["fake"],
        "raw": _cv(raw),
        "days": _cv(days),
        "prose": prose,
    }


def pass_log1p_lag(tr: pd.DataFrame) -> dict:
    """Is log1p(issued) a twin of issued_lag1? Leftover after lag1."""
    rho, n = spearman_n(tr["log1p_issued"], tr["e_ar_issued_lag1"])
    twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    d = leftover_diag(
        tr[Y3], tr["log1p_issued"], (tr["e_ar_issued_lag1"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {"slice": "log1p vs issued_lag1 ρ", "n": n, "value": _f(rho), "twin?": "TWIN" if twin else ""},
        {
            "slice": "log1p leftover after lag1 rank",
            "n": d["n"],
            "value": _f(d["rank"]),
            "twin?": "FALSE clone" if d["fake"] else "",
        },
    ]
    prose = (
        f"log1p vs issued_lag1 ρ={_f(rho)} n={n} {'TWIN' if twin else 'not a twin'}. "
        f"leftover after lag1 rank {_f(d['rank'])} OLS {_f(d['ols'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "rho": rho,
        "twin": twin,
        "rank": d["rank"],
        "prose": prose,
    }


def pass_short_fake(tr: pd.DataFrame) -> dict:
    """Short-book leftover after days_lag1 0.585 — fake days leak?"""
    short = (tr["so_far_class"] == "short_<12") & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx_lag1"],), tr["fold"], short
    )
    d_now = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], short
    )
    rows = [
        {
            "bar": "short leftover after days_lag1",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "bar": "short leftover after days now",
            "rank": _f(d_now["rank"]),
            "OLS": _f(d_now["ols"]),
            "ρ": _f(d_now["rho_ctrl"]),
            "fake?": "FALSE clone" if d_now["fake"] else "",
        },
    ]
    prose = (
        f"Short leftover after days_lag1 rank {_f(d['rank'])} ρ={_f(d['rho_ctrl'])} "
        f"{'FALSE clone' if d['fake'] else 'not a clone'}. "
        f"After days now {_f(d_now['rank'])} ρ={_f(d_now['rho_ctrl'])} "
        f"{'FALSE clone' if d_now['fake'] else 'not a clone'}."
    )
    print(prose)
    return {
        "rows": rows,
        "lag_rank": d["rank"],
        "lag_fake": d["fake"],
        "now_rank": d_now["rank"],
        "now_fake": d_now["fake"],
        "prose": prose,
    }


def pass_after_ntx(tr: pd.DataFrame) -> dict:
    """Leftover after a_n_tx (volume/count bar)."""
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["a_n_tx"],), tr["fold"], tr[Y3].notna()
    )
    d_both = leftover_diag(
        tr[Y3],
        tr["e_ar_issued"],
        (tr["a_n_tx"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "after a_n_tx",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        },
        {
            "bar": "after n_tx+days",
            "rank": _f(d_both["rank"]),
            "OLS": _f(d_both["ols"]),
            "ρ": _f(d_both["rho_ctrl"]),
            "dies?": "dies" if d_both["honest_dies"] else "lives",
        },
    ]
    prose = (
        f"Leftover after a_n_tx rank {_f(d['rank'])} ρ={_f(d['rho_ctrl'])}. "
        f"After n_tx+days {_f(d_both['rank'])} {'dies' if d_both['honest_dies'] else 'lives'}."
    )
    print(prose)
    return {
        "rows": rows,
        "ntx": d["rank"],
        "both": d_both["rank"],
        "both_dies": d_both["honest_dies"],
        "prose": prose,
    }


def pass_ntx_days_fake(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3],
        tr["e_ar_issued"],
        (tr["a_n_tx"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rho_days = spearman(d["resid"], tr["c_n_days_with_tx"])
    rows = [
        {
            "cut": "leftover n_tx+days vs days ρ",
            "rank": _f(d["rank"]),
            "ρ(resid,days)": _f(rho_days),
            "fake?": "FALSE clone" if (np.isfinite(rho_days) and abs(rho_days) >= TWIN_RHO) else "",
        }
    ]
    fake = bool(np.isfinite(rho_days) and abs(rho_days) >= TWIN_RHO)
    prose = (
        f"n_tx+days leftover rank {_f(d['rank'])} ρ(resid,days)={_f(rho_days)} "
        f"{'FALSE clone of days' if fake else 'not a days clone'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "rho": rho_days, "fake": fake, "prose": prose}


def pass_early6(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    early = tr["early6"]
    rows = [
        {
            "slice": "early6",
            "nn": int(iss[early].notna().sum()),
            "n": int(early.sum()),
            "cov": _pp(_pct(int(iss[early].notna().sum()), int(early.sum()))),
            "eq0": _pp(_pct(int((iss[early] == 0).sum()), int(iss[early].notna().sum()))),
        },
        {
            "slice": "after month 7",
            "nn": int(iss[~early].notna().sum()),
            "n": int((~early).sum()),
            "cov": _pp(_pct(int(iss[~early].notna().sum()), int((~early).sum()))),
            "eq0": _pp(_pct(int((iss[~early] == 0).sum()), int(iss[~early].notna().sum()))),
        },
    ]
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], (~early) & tr[Y3].notna()
    )
    prose = (
        f"Early6 issued cov {rows[0]['cov']} after7 {rows[1]['cov']}. "
        f"After-month7 leftover-days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {
        "rows": rows,
        "early_cov": _pct(int(iss[early].notna().sum()), int(early.sum())),
        "late_rank": d["rank"],
        "late_fake": d["fake"],
        "prose": prose,
    }


def pass_zero_flag(tr: pd.DataFrame) -> dict:
    """Binary issued==0 leftover after days — is the leftover just the zero fill?"""
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    z = (iss == 0).astype(float)
    z = z.where(iss.notna())
    raw = signed_oof_auroc(tr[Y3], z, tr["fold"], tr[Y3].notna())
    d = leftover_diag(tr[Y3], z, (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna())
    rows = [
        _auc_row(Y3, "issued==0 flag", raw),
        {
            "y": Y3,
            "feature": "zero-flag leftover-days rank",
            "n": f"{d['n']:,}",
            "n_pos": f"{d['n_pos']:,}",
            "CV": _f(d["rank"]),
            "sd": _f(d["ols"]),
            "sign": f"fake={d['fake']}",
            "folds": d["rank_folds"],
        },
    ]
    prose = (
        f"issued==0 flag Y3 {_f(_cv(raw))} leftover-days rank {_f(d['rank'])} "
        f"fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "raw": _cv(raw), "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_y7_days_left(tr: pd.DataFrame) -> dict:
    """Y7 leftover of contemporaneous issued after days (not the TURNOVER bar)."""
    d = leftover_diag(
        tr[Y7], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y7].notna()
    )
    rows = [
        {
            "cut": "Y7 leftover after days rank",
            "value": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"Y7 leftover after days rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"ρ={_f(d['rho_ctrl'])} {'FALSE clone' if d['fake'] else 'not a clone'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_log1p_pos(tr: pd.DataFrame) -> dict:
    """log1p leftover after days on issued>0 — does 0.608 survive without zeros?"""
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    pos = iss > 0
    d = leftover_diag(
        tr[Y3], tr["log1p_issued"], (tr["c_n_days_with_tx"],), tr["fold"], pos & tr[Y3].notna()
    )
    raw = signed_oof_auroc(tr[Y3], tr["log1p_issued"], tr["fold"], pos & tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], pos & tr[Y3].notna())
    rows = [
        _auc_row(Y3, "log1p issued>0", raw),
        _auc_row(Y3, "days issued>0", days),
        {
            "y": Y3,
            "feature": "log1p>0 leftover-days rank",
            "n": f"{d['n']:,}",
            "n_pos": f"{d['n_pos']:,}",
            "CV": _f(d["rank"]),
            "sd": _f(d["ols"]),
            "sign": f"fake={d['fake']}",
            "folds": d["rank_folds"],
        },
    ]
    prose = (
        f"log1p issued>0 leftover after days rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"ρ={_f(d['rho_ctrl'])} fake={d['fake']} raw {_f(_cv(raw))} days {_f(_cv(days))}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "fake": d["fake"],
        "raw": _cv(raw),
        "days": _cv(days),
        "prose": prose,
    }


def pass_days_lag1_both(tr: pd.DataFrame) -> dict:
    """Leftover after days + issued_lag1 — unused after both honest bars."""
    d = leftover_diag(
        tr[Y3],
        tr["e_ar_issued"],
        (tr["c_n_days_with_tx"], tr["e_ar_issued_lag1"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "after days+issued_lag1",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Leftover after days+issued_lag1 rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"ρ={_f(d['rho_ctrl'])} {'dies' if d['honest_dies'] else 'lives — still off the card'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "dies": d["honest_dies"], "fake": d["fake"], "prose": prose}


def pass_cn_now(tr: pd.DataFrame) -> dict:
    """CN leftover after contemporaneous issued — KEEP is vs lag1, not now."""
    d = leftover_diag(
        tr[Y7],
        tr["e_credit_note_ratio"],
        (tr["e_ar_issued"],),
        tr["fold"],
        tr[Y7].notna(),
    )
    rows = [
        {
            "cut": "CN leftover after issued now",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
        }
    ]
    prose = (
        f"CN leftover after contemporaneous issued rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"(lag1 KEEP 0.597 stays; do not overwrite credit_note_qa)."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "ols": d["ols"], "prose": prose}


def pass_holdout_early(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    assert set(ho["company_id"]) <= load_holdout()
    iss = pd.to_numeric(ho["e_ar_issued"], errors="coerce")
    early = ho["early6"]
    rows = [
        {
            "slice": "holdout early6",
            "nn": int(iss[early].notna().sum()),
            "n": int(early.sum()),
            "cov": _pp(_pct(int(iss[early].notna().sum()), int(early.sum()))),
        },
        {
            "slice": "holdout after7",
            "nn": int(iss[~early].notna().sum()),
            "n": int((~early).sum()),
            "cov": _pp(_pct(int(iss[~early].notna().sum()), int((~early).sum()))),
        },
    ]
    prose = (
        f"Holdout early6 issued cov {rows[0]['cov']} after7 {rows[1]['cov']}. No AUROC."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_after_pending(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["e_pending_amt_share"],), tr["fold"], tr[Y3].notna()
    )
    both = leftover_diag(
        tr[Y3],
        tr["e_ar_issued"],
        (tr["e_pending_amt_share"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "after pending",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "bar": "after pending+days",
            "rank": _f(both["rank"]),
            "OLS": _f(both["ols"]),
            "ρ": _f(both["rho_ctrl"]),
            "fake?": "FALSE clone" if both["fake"] else "",
        },
    ]
    prose = (
        f"Leftover after pending rank {_f(d['rank'])} ρ={_f(d['rho_ctrl'])}. "
        f"After pending+days {_f(both['rank'])} fake={both['fake']}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "both": both["rank"],
        "fake": both["fake"],
        "prose": prose,
    }


def pass_fold4_days(tr: pd.DataFrame) -> dict:
    """Fold 4 leftover after days — issued already owns fold 4 vs DSO."""
    m = (tr["fold"] == 4) & tr[Y3].notna()
    y3 = leftover_diag(tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m)
    m7 = (tr["fold"] == 4) & tr[Y7].notna()
    y7 = leftover_diag(tr[Y7], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m7)
    raw = signed_oof_auroc(tr[Y3], tr["e_ar_issued"], tr["fold"], m)
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], m)
    rows = [
        _auc_row(Y3, "issued fold4", raw),
        _auc_row(Y3, "days fold4", days),
        {
            "y": Y3,
            "feature": "fold4 leftover-days rank",
            "n": f"{y3['n']:,}",
            "n_pos": f"{y3['n_pos']:,}",
            "CV": _f(y3["rank"]),
            "sd": _f(y3["ols"]),
            "sign": f"fake={y3['fake']}",
            "folds": y3["rank_folds"],
        },
        {
            "y": Y7,
            "feature": "fold4 leftover-days rank",
            "n": f"{y7['n']:,}",
            "n_pos": f"{y7['n_pos']:,}",
            "CV": _f(y7["rank"]),
            "sd": _f(y7["ols"]),
            "sign": f"fake={y7['fake']}",
            "folds": y7["rank_folds"],
        },
    ]
    prose = (
        f"Fold 4 Y3 leftover after days rank {_f(y3['rank'])} fake={y3['fake']} "
        f"issued {_f(_cv(raw))} days {_f(_cv(days))}. Y7 leftover {_f(y7['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3": y3["rank"],
        "y7": y7["rank"],
        "fake": y3["fake"],
        "prose": prose,
    }


def pass_lag1_after_now(tr: pd.DataFrame) -> dict:
    """Inverse: does lag1 leftover-live after contemporaneous? KEEP is the lag."""
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued_lag1"], (tr["e_ar_issued"],), tr["fold"], tr[Y3].notna()
    )
    y7 = leftover_diag(
        tr[Y7], tr["e_ar_issued_lag1"], (tr["e_ar_issued"],), tr["fold"], tr[Y7].notna()
    )
    rows = [
        {
            "y": Y3,
            "cut": "lag1 leftover after now",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        },
        {
            "y": Y7,
            "cut": "lag1 leftover after now",
            "rank": _f(y7["rank"]),
            "OLS": _f(y7["ols"]),
            "ρ": _f(y7["rho_ctrl"]),
            "dies?": "dies" if y7["honest_dies"] else "lives",
        },
    ]
    prose = (
        f"lag1 leftover after now Y3 rank {_f(d['rank'])} Y7 {_f(y7['rank'])}. "
        "TURNOVER KEEP is the lag; contemporaneous does not steal it."
    )
    print(prose)
    return {
        "rows": rows,
        "y3": d["rank"],
        "y7": y7["rank"],
        "y3_dies": d["honest_dies"],
        "prose": prose,
    }


def pass_after_cn(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["e_credit_note_ratio"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "bar": "after CN",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"Y3 leftover after CN rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"ρ={_f(d['rho_ctrl'])}. CN KEEP vs lag1 is a different bar."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_issued_over_in3(tr: pd.DataFrame) -> dict:
    """issued / a_in3 intensity — SIZE-scaled volume leftover."""
    inn = pd.to_numeric(tr["a_in3"], errors="coerce")
    ratio = pd.to_numeric(tr["e_ar_issued"], errors="coerce") / inn.where(inn > 0)
    d = leftover_diag(tr[Y3], ratio, (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna())
    raw = signed_oof_auroc(tr[Y3], ratio, tr["fold"], tr[Y3].notna())
    rows = [
        _auc_row(Y3, "issued / a_in3", raw),
        {
            "y": Y3,
            "feature": "issued/in3 leftover-days rank",
            "n": f"{d['n']:,}",
            "n_pos": f"{d['n_pos']:,}",
            "CV": _f(d["rank"]),
            "sd": _f(d["ols"]),
            "sign": f"fake={d['fake']}",
            "folds": d["rank_folds"],
        },
    ]
    prose = (
        f"issued/a_in3 Y3 {_f(_cv(raw))} leftover-days rank {_f(d['rank'])} "
        f"fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "raw": _cv(raw), "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_trail_terciles(tr: pd.DataFrame) -> dict:
    """Leftover after days inside months-so-far terciles (Q6-adjacent)."""
    so = pd.to_numeric(tr["months_so_far"], errors="coerce")
    cuts = so.quantile([1 / 3, 2 / 3])
    t1 = so <= cuts.iloc[0]
    t3 = so > cuts.iloc[1]
    t2 = ~t1 & ~t3
    rows = []
    ranks = {}
    for name, mask in (("T1 short", t1), ("T2 mid", t2), ("T3 long", t3)):
        d = leftover_diag(
            tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], mask & tr[Y3].notna()
        )
        ranks[name] = d["rank"]
        rows.append(
            {
                "slice": name,
                "rank": _f(d["rank"]),
                "OLS": _f(d["ols"]),
                "ρ": _f(d["rho_ctrl"]),
                "fake?": "FALSE clone" if d["fake"] else "",
                "n": f"{d['n']:,}",
            }
        )
    prose = (
        f"Trail terciles leftover-days T1 {_f(ranks['T1 short'])} "
        f"T2 {_f(ranks['T2 mid'])} T3 {_f(ranks['T3 long'])}."
    )
    print(prose)
    return {"rows": rows, "t1": ranks["T1 short"], "t3": ranks["T3 long"], "prose": prose}


def pass_log1p_days_lag1(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3],
        tr["log1p_issued"],
        (tr["c_n_days_with_tx"], tr["e_ar_issued_lag1"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "log1p after days+lag1",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"log1p leftover after days+lag1 rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"{'dies' if d['honest_dies'] else 'lives — still off the card'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "dies": d["honest_dies"], "prose": prose}


def pass_train_size_cov(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    rows = []
    for lab in ("T1", "T2", "T3"):
        m = terc == lab
        rows.append(
            {
                "SIZE": lab,
                "nn": int(iss[m].notna().sum()),
                "n": int(m.sum()),
                "cov": _pp(_pct(int(iss[m].notna().sum()), int(m.sum()))),
                "p50": _f(iss[m].median(), 0) if iss[m].notna().any() else "—",
            }
        )
    prose = (
        f"Train SIZE issued cov T1 {rows[0]['cov']} T2 {rows[1]['cov']} "
        f"T3 {rows[2]['cov']}."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_company_median_rho(tr: pd.DataFrame) -> dict:
    """Company-median VIF/size flag — row ρ was 0.463; what is the company median?"""
    g = tr.groupby("company_id")[["e_ar_issued", "c_n_days_with_tx", "a_in3", "log_in3"]].median()
    rows = [
        {"pair": "issued vs days (co-median)", "ρ": _f(spearman(g["e_ar_issued"], g["c_n_days_with_tx"])), "n": f"{g['e_ar_issued'].notna().sum():,}"},
        {"pair": "issued vs a_in3 (co-median)", "ρ": _f(spearman(g["e_ar_issued"], g["a_in3"])), "n": f"{g['e_ar_issued'].notna().sum():,}"},
        {"pair": "issued vs log1p(a_in3) (co-median)", "ρ": _f(spearman(g["e_ar_issued"], g["log_in3"])), "n": f"{g['e_ar_issued'].notna().sum():,}"},
    ]
    rho_size = spearman(g["e_ar_issued"], g["log_in3"])
    prose = (
        f"Company-median ρ issued vs days {_f(spearman(g['e_ar_issued'], g['c_n_days_with_tx']))} "
        f"vs log1p(a_in3) {_f(rho_size)} {'SIZE' if abs(rho_size) >= SIZE_RHO else 'not SIZE'} "
        "(feature report flagged company-median VIF / size)."
    )
    print(prose)
    return {"rows": rows, "rho_size": rho_size, "size_flag": abs(rho_size) >= SIZE_RHO, "prose": prose}


def pass_lag3(tr: pd.DataFrame) -> dict:
    days_l3 = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx_lag3"],), tr["fold"], tr[Y3].notna()
    )
    iss_l3 = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["e_ar_issued_lag3"],), tr["fold"], tr[Y3].notna()
    )
    raw_l3 = signed_oof_auroc(tr[Y3], tr["e_ar_issued_lag3"], tr["fold"], tr[Y3].notna())
    rows = [
        _auc_row(Y3, "issued_lag3", raw_l3),
        {
            "bar": "issued leftover after days_lag3",
            "rank": _f(days_l3["rank"]),
            "OLS": _f(days_l3["ols"]),
            "ρ": _f(days_l3["rho_ctrl"]),
            "fake?": "FALSE clone" if days_l3["fake"] else "",
        },
        {
            "bar": "issued leftover after issued_lag3",
            "rank": _f(iss_l3["rank"]),
            "OLS": _f(iss_l3["ols"]),
            "ρ": _f(iss_l3["rho_ctrl"]),
            "fake?": "FALSE clone" if iss_l3["fake"] else "",
        },
    ]
    prose = (
        f"issued_lag3 Y3 {_f(_cv(raw_l3))}. Leftover after days_lag3 {_f(days_l3['rank'])} "
        f"fake={days_l3['fake']}. After issued_lag3 {_f(iss_l3['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "days_l3": days_l3["rank"],
        "iss_l3": iss_l3["rank"],
        "raw_l3": _cv(raw_l3),
        "prose": prose,
    }


def pass_y5_cov(tr: pd.DataFrame) -> dict:
    """Y5 never E — coverage only, no AUROC as X."""
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    y5 = tr[Y5].notna()
    rows = [
        {
            "slice": "train Y5-defined",
            "nn": int(iss[y5].notna().sum()),
            "n": int(y5.sum()),
            "cov": _pp(_pct(int(iss[y5].notna().sum()), int(y5.sum()))),
        }
    ]
    prose = (
        f"Y5-defined issued cov {rows[0]['cov']}. Y5 never E — no AUROC, no card."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_holdout_trail(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    assert set(ho["company_id"]) <= load_holdout()
    iss = pd.to_numeric(ho["e_ar_issued"], errors="coerce")
    so = pd.to_numeric(ho["months_so_far"], errors="coerce")
    cuts = so.quantile([1 / 3, 2 / 3])
    rows = []
    for name, mask in (
        ("T1 short", so <= cuts.iloc[0]),
        ("T2 mid", (so > cuts.iloc[0]) & (so <= cuts.iloc[1])),
        ("T3 long", so > cuts.iloc[1]),
    ):
        rows.append(
            {
                "slice": name,
                "nn": int(iss[mask].notna().sum()),
                "n": int(mask.sum()),
                "cov": _pp(_pct(int(iss[mask].notna().sum()), int(mask.sum()))),
            }
        )
    prose = (
        f"Holdout trail issued cov T1 {rows[0]['cov']} T2 {rows[1]['cov']} "
        f"T3 {rows[2]['cov']}. No AUROC."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_fold_lag1(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["e_ar_issued_lag1"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "cut": "leftover after lag1 fold-wise",
            "rank": _f(d["rank"]),
            "folds": d["rank_folds"],
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
        }
    ]
    prose = (
        f"Leftover after lag1 fold-wise rank {_f(d['rank'])} folds {d['rank_folds']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "folds": d["rank_folds"], "prose": prose}


def pass_y7_days_lag1(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y7],
        tr["e_ar_issued"],
        (tr["c_n_days_with_tx"], tr["e_ar_issued_lag1"]),
        tr["fold"],
        tr[Y7].notna(),
    )
    rows = [
        {
            "bar": "Y7 leftover after days+lag1",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Y7 leftover after days+lag1 rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"{'dies' if d['honest_dies'] else 'lives — still CLOSE TURNOVER add-on'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "dies": d["honest_dies"], "prose": prose}


def pass_acf_size(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    rows = []
    acfs = {}
    for lab in ("T1", "T2", "T3"):
        m = terc == lab
        acf = median_acf(tr.loc[m, "e_ar_issued"], tr.loc[m, "company_id"], 1)
        acfs[lab] = acf
        rows.append({"SIZE": lab, "acf1": _f(acf), "nn": f"{int(tr.loc[m, 'e_ar_issued'].notna().sum()):,}"})
    prose = (
        f"issued acf1 by SIZE T1 {_f(acfs['T1'])} T2 {_f(acfs['T2'])} T3 {_f(acfs['T3'])} "
        f"(panel 0.143 LOW_PERSIST)."
    )
    print(prose)
    return {"rows": rows, "t1": acfs["T1"], "t3": acfs["T3"], "prose": prose}


def pass_zero_share(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    rows = []
    for lab in ("T1", "T2", "T3"):
        m = (terc == lab) & iss.notna()
        z = int((iss[m] == 0).sum())
        n = int(m.sum())
        rows.append({"SIZE": lab, "zeros": z, "nn": n, "zero_share": _pp(_pct(z, n))})
    z_all = int((iss == 0).sum())
    nn = int(iss.notna().sum())
    prose = (
        f"issued==0 share train {_pp(_pct(z_all, nn))} T1 {rows[0]['zero_share']} "
        f"T2 {rows[1]['zero_share']} T3 {rows[2]['zero_share']} "
        "(zeros inflated leftover-after-days; issued>0 leftover 0.412)."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_inv_days_log1p(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["c_n_days_with_tx"], (tr["log1p_issued"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "bar": "days leftover after log1p(issued)",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Days leftover after log1p(issued) rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"{'dies' if d['honest_dies'] else 'lives — keep the 0.711 bar'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "dies": d["honest_dies"], "prose": prose}


def pass_size_trail_grid(tr: pd.DataFrame) -> dict:
    """Leftover after days inside SIZE × trail cells."""
    terc = size_terciles(tr)
    so = pd.to_numeric(tr["months_so_far"], errors="coerce")
    cuts = so.quantile([1 / 3, 2 / 3])
    trail = pd.Series("T2", index=tr.index)
    trail[so <= cuts.iloc[0]] = "T1"
    trail[so > cuts.iloc[1]] = "T3"
    rows = []
    ranks = {}
    for sl in ("T1", "T2", "T3"):
        for tl in ("T1", "T2", "T3"):
            m = (terc == sl) & (trail == tl) & tr[Y3].notna()
            d = leftover_diag(
                tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
            )
            key = f"{sl}×{tl}"
            ranks[key] = d["rank"]
            rows.append(
                {
                    "SIZE×trail": key,
                    "rank": _f(d["rank"]),
                    "OLS": _f(d["ols"]),
                    "fake?": "FALSE clone" if d["fake"] else "",
                    "n": f"{d['n']:,}",
                    "n_pos": f"{d['n_pos']:,}",
                }
            )
    prose = (
        f"SIZE×trail leftover-days T1×T1 {_f(ranks['T1×T1'])} "
        f"T3×T3 {_f(ranks['T3×T3'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "t1t1": ranks["T1×T1"],
        "t3t3": ranks["T3×T3"],
        "prose": prose,
    }


def pass_triple(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3],
        tr["e_ar_issued"],
        (tr["c_n_days_with_tx"], tr["log_in3"], tr["e_ar_issued_lag1"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "after days+size+lag1",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Leftover after days+size+lag1 rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"{'dies' if d['honest_dies'] else 'lives — still off the card'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "dies": d["honest_dies"], "prose": prose}


def pass_q6_triple(tr: pd.DataFrame) -> dict:
    short = tr["so_far_class"] == "short_<12"
    d = leftover_diag(
        tr[Y3],
        tr["e_ar_issued"],
        (tr["c_n_days_with_tx_lag1"], tr["e_ar_issued_lag1"]),
        tr["fold"],
        short & tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "Q6 leftover after days_lag1+issued_lag1",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Q6 leftover after days_lag1+issued_lag1 rank {_f(d['rank'])} "
        f"{'dies' if d['honest_dies'] else 'lives — Q6 contemporaneous still CLOSE'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "dies": d["honest_dies"], "prose": prose}


def pass_after_dso(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["e_dso_proxy"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "bar": "after DSO (already DROPPED)",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"Leftover after DSO rank {_f(d['rank'])} ρ={_f(d['rho_ctrl'])}. "
        "DSO stays DROPPED; do not put DSO back on TURNOVER."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "prose": prose}


def pass_holdout_zero(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    assert set(ho["company_id"]) <= load_holdout()
    iss = pd.to_numeric(ho["e_ar_issued"], errors="coerce")
    nn = int(iss.notna().sum())
    z = int((iss == 0).sum())
    rows = [
        {
            "slice": "holdout issued==0",
            "zeros": z,
            "nn": nn,
            "zero_share": _pp(_pct(z, nn)),
        }
    ]
    prose = f"Holdout issued==0 share {_pp(_pct(z, nn))} of nn={nn:,}. No AUROC."
    print(prose)
    return {"rows": rows, "zero_share": _pct(z, nn), "prose": prose}


def pass_fold1(tr: pd.DataFrame) -> dict:
    """Fold 1 leftover-days was 0.723 — isolate vs days 0.665."""
    raw = signed_oof_auroc(tr[Y3], tr["e_ar_issued"], tr["fold"], tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna())
    rows = [
        {
            "fold": "1",
            "issued": _f(fold_k(raw, 1)),
            "days": _f(fold_k(days, 1)),
            "issued−days": _f(fold_k(raw, 1) - fold_k(days, 1)),
        }
    ]
    prose = (
        f"Fold 1 issued {_f(fold_k(raw, 1))} days {_f(fold_k(days, 1))} "
        f"(leftover-days fold 1 0.723 is OLS-inflated; days still wins the fold)."
    )
    print(prose)
    return {
        "rows": rows,
        "iss": fold_k(raw, 1),
        "days": fold_k(days, 1),
        "prose": prose,
    }


def pass_resid_icc(tr: pd.DataFrame) -> dict:
    """ICC of rank-residual after days — is leftover BETWEEN like raw ICC 0.94?"""
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    icc = icc_anova(d["rresid"], tr["company_id"])
    rows = [
        {
            "object": "rank-resid after days",
            "ICC": _f(icc.get("icc", float("nan"))),
            "k": f"{icc.get('k', 0):,}",
        }
    ]
    prose = (
        f"Rank-resid after days ICC {_f(icc.get('icc', float('nan')))} "
        f"(raw issued ICC 0.942 BETWEEN). Leftover is still a company-level smear."
    )
    print(prose)
    return {"rows": rows, "icc": icc.get("icc", float("nan")), "prose": prose}


def pass_calendar(tr: pd.DataFrame) -> dict:
    yr = tr["period"].dt.year
    rows = []
    ranks = {}
    for y in sorted(yr.dropna().unique()):
        m = (yr == y) & tr[Y3].notna()
        d = leftover_diag(
            tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
        )
        ranks[int(y)] = d["rank"]
        rows.append(
            {
                "year": int(y),
                "rank": _f(d["rank"]),
                "OLS": _f(d["ols"]),
                "fake?": "FALSE clone" if d["fake"] else "",
                "n": f"{d['n']:,}",
                "n_pos": f"{d['n_pos']:,}",
            }
        )
    prose = (
        "Calendar leftover-days "
        + " ".join(f"{y}={_f(ranks[y])}" for y in ranks)
        + "."
    )
    print(prose)
    return {"rows": rows, "ranks": ranks, "prose": prose}


def pass_rho_sofar(tr: pd.DataFrame) -> dict:
    rho, n = spearman_n(tr["e_ar_issued"], tr["months_so_far"])
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["months_so_far"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {"pair": "issued vs months_so_far", "ρ": _f(rho), "n": f"{n:,}"},
        {
            "bar": "leftover after months_so_far",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
        },
    ]
    prose = (
        f"issued vs months_so_far ρ={_f(rho)} n={n:,}. "
        f"Leftover after trail rank {_f(d['rank'])}."
    )
    print(prose)
    return {"rows": rows, "rho": rho, "rank": d["rank"], "prose": prose}


def pass_rho_by_size(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    rows = []
    rhos = {}
    for lab in ("T1", "T2", "T3"):
        m = terc == lab
        rho, n = spearman_n(tr.loc[m, "e_ar_issued"], tr.loc[m, "c_n_days_with_tx"])
        rs, ns = spearman_n(tr.loc[m, "e_ar_issued"], tr.loc[m, "log_in3"])
        rhos[lab] = rho
        rows.append(
            {
                "SIZE": lab,
                "ρ vs days": _f(rho),
                "ρ vs log1p(a_in3)": _f(rs),
                "n": f"{n:,}",
                "SIZE?": "SIZE" if abs(rs) >= SIZE_RHO else "",
            }
        )
    prose = (
        f"issued vs days ρ T1 {_f(rhos['T1'])} T2 {_f(rhos['T2'])} T3 {_f(rhos['T3'])}."
    )
    print(prose)
    return {"rows": rows, "t1": rhos["T1"], "t3": rhos["T3"], "prose": prose}


def pass_inv_ntx(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["a_n_tx"], (tr["e_ar_issued"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "bar": "n_tx leftover after issued",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"n_tx leftover after issued rank {_f(d['rank'])} "
        f"{'dies' if d['honest_dies'] else 'lives — n_tx/days stay the volume bar'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "dies": d["honest_dies"], "prose": prose}


def pass_after_intensity(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["issued_per_day"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "bar": "after issued/days intensity",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"Leftover after intensity rank {_f(d['rank'])} ρ={_f(d['rho_ctrl'])} "
        f"fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_y7_after_cn(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y7], tr["e_ar_issued"], (tr["e_credit_note_ratio"],), tr["fold"], tr[Y7].notna()
    )
    rows = [
        {
            "bar": "Y7 leftover after CN",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
        }
    ]
    prose = (
        f"Y7 leftover after CN rank {_f(d['rank'])} OLS {_f(d['ols'])}. "
        "CN KEEP vs lag1 stays; do not overwrite credit_note_qa."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "prose": prose}


def pass_t2_only(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    d = leftover_diag(
        tr[Y3],
        tr["e_ar_issued"],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        (terc == "T2") & tr[Y3].notna(),
    )
    rows = [
        {
            "slice": "SIZE T2 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"SIZE T2 leftover after days rank {_f(d['rank'])} fake={d['fake']} "
        "(T3 was LOW_POWER in pass 6)."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_month(tr: pd.DataFrame) -> dict:
    mo = tr["period"].dt.month
    rows = []
    ranks = {}
    for m in range(1, 13):
        mask = (mo == m) & tr[Y3].notna()
        d = leftover_diag(
            tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], mask
        )
        ranks[m] = d["rank"]
        rows.append(
            {
                "month": m,
                "rank": _f(d["rank"]),
                "OLS": _f(d["ols"]),
                "fake?": "FALSE clone" if d["fake"] else "",
                "n": f"{d['n']:,}",
                "n_pos": f"{d['n_pos']:,}",
            }
        )
    finite = [ranks[m] for m in ranks if np.isfinite(ranks[m])]
    prose = (
        "Calendar-month leftover-days "
        + " ".join(f"{m}={_f(ranks[m])}" for m in ranks)
        + (f" median {_f(float(np.median(finite)))}" if finite else "")
        + "."
    )
    print(prose)
    return {
        "rows": rows,
        "median": float(np.median(finite)) if finite else float("nan"),
        "prose": prose,
    }


def pass_lag1_rho_size(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    rows = []
    rhos = {}
    for lab in ("T1", "T2", "T3"):
        m = terc == lab
        rho, n = spearman_n(tr.loc[m, "e_ar_issued"], tr.loc[m, "e_ar_issued_lag1"])
        rhos[lab] = rho
        rows.append(
            {
                "SIZE": lab,
                "ρ now vs lag1": _f(rho),
                "n": f"{n:,}",
                "twin?": "TWIN" if abs(rho) >= TWIN_RHO else "",
            }
        )
    prose = (
        f"now vs lag1 ρ T1 {_f(rhos['T1'])} T2 {_f(rhos['T2'])} T3 {_f(rhos['T3'])} "
        "(panel 0.814 TWIN)."
    )
    print(prose)
    return {"rows": rows, "t1": rhos["T1"], "t3": rhos["T3"], "prose": prose}


def pass_days_after_ntx(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["c_n_days_with_tx"], (tr["a_n_tx"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "bar": "days leftover after n_tx",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Days leftover after n_tx rank {_f(d['rank'])} "
        f"{'dies' if d['honest_dies'] else 'lives — days is not just n_tx'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "dies": d["honest_dies"], "prose": prose}


def pass_company_mean(tr: pd.DataFrame) -> dict:
    """Company-mean issued leftover after company-mean days — BETWEEN smear."""
    g = (
        tr.loc[tr[Y3].notna()]
        .groupby("company_id")
        .agg(
            issued=("e_ar_issued", "mean"),
            days=("c_n_days_with_tx", "mean"),
            y3=(Y3, "max"),
            fold=("fold", "first"),
        )
    )
    raw = signed_oof_auroc(g["y3"], g["issued"], g["fold"], g["y3"].notna())
    days = signed_oof_auroc(g["y3"], g["days"], g["fold"], g["y3"].notna())
    d = leftover_diag(g["y3"], g["issued"], (g["days"],), g["fold"], g["y3"].notna())
    rows = [
        _auc_row(Y3, "co-mean issued", raw),
        _auc_row(Y3, "co-mean days", days),
        {
            "y": Y3,
            "feature": "co-mean leftover-days rank",
            "n": f"{d['n']:,}",
            "n_pos": f"{d['n_pos']:,}",
            "CV": _f(d["rank"]),
            "sd": _f(d["ols"]),
            "sign": f"fake={d['fake']}",
            "folds": d["rank_folds"],
        },
    ]
    prose = (
        f"Company-mean issued {_f(_cv(raw))} days {_f(_cv(days))} "
        f"leftover-days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {
        "rows": rows,
        "raw": _cv(raw),
        "days": _cv(days),
        "rank": d["rank"],
        "fake": d["fake"],
        "prose": prose,
    }


def pass_holdout_year(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    assert set(ho["company_id"]) <= load_holdout()
    iss = pd.to_numeric(ho["e_ar_issued"], errors="coerce")
    yr = ho["period"].dt.year
    rows = []
    for y in sorted(yr.dropna().unique()):
        m = yr == y
        rows.append(
            {
                "year": int(y),
                "nn": int(iss[m].notna().sum()),
                "n": int(m.sum()),
                "cov": _pp(_pct(int(iss[m].notna().sum()), int(m.sum()))),
            }
        )
    prose = "Holdout year issued cov " + " ".join(
        f"{r['year']}={r['cov']}" for r in rows
    ) + ". No AUROC."
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_days_lag1_cn(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3],
        tr["e_ar_issued"],
        (tr["c_n_days_with_tx"], tr["e_ar_issued_lag1"], tr["e_credit_note_ratio"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "after days+lag1+CN",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Leftover after days+lag1+CN rank {_f(d['rank'])} "
        f"{'dies' if d['honest_dies'] else 'lives — still off the card'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "dies": d["honest_dies"], "prose": prose}


def pass_quarter(tr: pd.DataFrame) -> dict:
    q = tr["period"].dt.quarter
    rows = []
    ranks = {}
    for k in (1, 2, 3, 4):
        m = (q == k) & tr[Y3].notna()
        d = leftover_diag(
            tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
        )
        ranks[k] = d["rank"]
        rows.append(
            {
                "Q": k,
                "rank": _f(d["rank"]),
                "OLS": _f(d["ols"]),
                "fake?": "FALSE clone" if d["fake"] else "",
                "n": f"{d['n']:,}",
                "n_pos": f"{d['n_pos']:,}",
            }
        )
    prose = "Quarter leftover-days " + " ".join(f"Q{k}={_f(ranks[k])}" for k in ranks) + "."
    print(prose)
    return {"rows": rows, "q1": ranks[1], "q3": ranks[3], "prose": prose}


def pass_lag1_after_days(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued_lag1"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    y7 = leftover_diag(
        tr[Y7], tr["e_ar_issued_lag1"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y7].notna()
    )
    rows = [
        {
            "y": Y3,
            "bar": "lag1 leftover after days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "y": Y7,
            "bar": "lag1 leftover after days",
            "rank": _f(y7["rank"]),
            "OLS": _f(y7["ols"]),
            "ρ": _f(y7["rho_ctrl"]),
            "fake?": "FALSE clone" if y7["fake"] else "",
        },
    ]
    prose = (
        f"lag1 leftover after days Y3 {_f(d['rank'])} fake={d['fake']} "
        f"Y7 {_f(y7['rank'])} fake={y7['fake']}. TURNOVER KEEP is Y7, not this Y3 leftover."
    )
    print(prose)
    return {"rows": rows, "y3": d["rank"], "y7": y7["rank"], "y3_fake": d["fake"], "prose": prose}


def pass_y7_after_size(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y7], tr["e_ar_issued"], (tr["log_in3"],), tr["fold"], tr[Y7].notna()
    )
    rows = [
        {
            "bar": "Y7 leftover after size",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Y7 leftover after size rank {_f(d['rank'])} OLS {_f(d['ols'])}. "
        "Do not grow TURNOVER."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "prose": prose}


def pass_pos_lag1_rho(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    m = iss > 0
    rho, n = spearman_n(tr.loc[m, "e_ar_issued"], tr.loc[m, "e_ar_issued_lag1"])
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["e_ar_issued_lag1"],), tr["fold"], m & tr[Y3].notna()
    )
    rows = [
        {"pair": "issued>0 vs lag1", "ρ": _f(rho), "n": f"{n:,}", "twin?": "TWIN" if abs(rho) >= TWIN_RHO else ""},
        {
            "bar": "issued>0 leftover after lag1",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
        },
    ]
    prose = (
        f"issued>0 vs lag1 ρ={_f(rho)} n={n:,} "
        f"{'TWIN' if abs(rho) >= TWIN_RHO else 'not a twin'}. "
        f"Leftover after lag1 {_f(d['rank'])}."
    )
    print(prose)
    return {"rows": rows, "rho": rho, "rank": d["rank"], "prose": prose}


def pass_split_gap(panel: pd.DataFrame) -> dict:
    tr = panel[panel["split"] == "train"]
    ho = panel[panel["split"] == "holdout"]
    assert set(ho["company_id"]) <= load_holdout()
    iss_tr = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    iss_ho = pd.to_numeric(ho["e_ar_issued"], errors="coerce")
    rows = [
        {
            "split": "train",
            "cov": _pp(_pct(int(iss_tr.notna().sum()), len(tr))),
            "p50": _f(iss_tr.median(), 0),
            "zero_share": _pp(_pct(int((iss_tr == 0).sum()), int(iss_tr.notna().sum()))),
        },
        {
            "split": "holdout",
            "cov": _pp(_pct(int(iss_ho.notna().sum()), len(ho))),
            "p50": _f(iss_ho.median(), 0),
            "zero_share": _pp(_pct(int((iss_ho == 0).sum()), int(iss_ho.notna().sum()))),
        },
    ]
    prose = (
        f"Train vs holdout issued cov {rows[0]['cov']} vs {rows[1]['cov']}; "
        f"p50 {rows[0]['p50']} vs {rows[1]['p50']}. No AUROC."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_t1_zero(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    flag = (iss == 0).astype(float)
    d = leftover_diag(
        tr[Y3],
        flag,
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        (terc == "T1") & iss.notna() & tr[Y3].notna(),
    )
    rows = [
        {
            "slice": "T1 issued==0 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"T1 issued==0 leftover after days rank {_f(d['rank'])} fake={d['fake']} "
        "(T1 zero share 55.9%)."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_both_pos(tr: pd.DataFrame) -> dict:
    """Same-n leftover when both now and lag1 are strictly positive."""
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    lag = pd.to_numeric(tr["e_ar_issued_lag1"], errors="coerce")
    m = (iss > 0) & (lag > 0)
    rho, n = spearman_n(iss[m], lag[m])
    raw = signed_oof_auroc(tr[Y3], tr["e_ar_issued"], tr["fold"], m & tr[Y3].notna())
    lag_auc = signed_oof_auroc(tr[Y3], tr["e_ar_issued_lag1"], tr["fold"], m & tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], m & tr[Y3].notna())
    d_days = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m & tr[Y3].notna()
    )
    d_lag = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["e_ar_issued_lag1"],), tr["fold"], m & tr[Y3].notna()
    )
    rows = [
        {"pair": "both>0 now vs lag1", "ρ": _f(rho), "n": f"{n:,}", "twin?": "TWIN" if abs(rho) >= TWIN_RHO else ""},
        _auc_row(Y3, "both>0 issued", raw),
        _auc_row(Y3, "both>0 lag1", lag_auc),
        _auc_row(Y3, "both>0 days", days),
        {
            "y": Y3,
            "feature": "both>0 leftover-days rank",
            "n": f"{d_days['n']:,}",
            "n_pos": f"{d_days['n_pos']:,}",
            "CV": _f(d_days["rank"]),
            "sd": _f(d_days["ols"]),
            "sign": f"fake={d_days['fake']}",
            "folds": d_days["rank_folds"],
        },
        {
            "y": Y3,
            "feature": "both>0 leftover-lag1 rank",
            "n": f"{d_lag['n']:,}",
            "n_pos": f"{d_lag['n_pos']:,}",
            "CV": _f(d_lag["rank"]),
            "sd": _f(d_lag["ols"]),
            "sign": f"fake={d_lag['fake']}",
            "folds": d_lag["rank_folds"],
        },
    ]
    prose = (
        f"both>0 now vs lag1 ρ={_f(rho)} issued {_f(_cv(raw))} lag1 {_f(_cv(lag_auc))} "
        f"days {_f(_cv(days))}. leftover-days {_f(d_days['rank'])} leftover-lag1 {_f(d_lag['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "rho": rho,
        "days_rank": d_days["rank"],
        "lag_rank": d_lag["rank"],
        "fake": d_days["fake"],
        "prose": prose,
    }


def pass_days_ntx_lag1(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3],
        tr["e_ar_issued"],
        (tr["c_n_days_with_tx"], tr["a_n_tx"], tr["e_ar_issued_lag1"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "after days+n_tx+lag1",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Leftover after days+n_tx+lag1 rank {_f(d['rank'])} "
        f"{'dies' if d['honest_dies'] else 'lives — still off the card'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "dies": d["honest_dies"], "prose": prose}


def pass_size_after_issued(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["log_in3"], (tr["e_ar_issued"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "bar": "size leftover after issued",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Size leftover after issued rank {_f(d['rank'])} "
        f"{'dies' if d['honest_dies'] else 'lives — size is not issued'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "dies": d["honest_dies"], "prose": prose}


def pass_days_lag1_all(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx_lag1"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "bar": "leftover after days_lag1 (all books)",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"Leftover after days_lag1 on all books rank {_f(d['rank'])} "
        f"ρ={_f(d['rho_ctrl'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_fold_after_size(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["log_in3"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "cut": "leftover after size fold-wise",
            "rank": _f(d["rank"]),
            "folds": d["rank_folds"],
            "OLS": _f(d["ols"]),
        }
    ]
    prose = f"Leftover after size fold-wise rank {_f(d['rank'])} folds {d['rank_folds']}."
    print(prose)
    return {"rows": rows, "rank": d["rank"], "folds": d["rank_folds"], "prose": prose}


def pass_y7_both_pos(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    lag = pd.to_numeric(tr["e_ar_issued_lag1"], errors="coerce")
    m = (iss > 0) & (lag > 0)
    raw = signed_oof_auroc(tr[Y7], tr["e_ar_issued"], tr["fold"], m & tr[Y7].notna())
    lag_auc = signed_oof_auroc(tr[Y7], tr["e_ar_issued_lag1"], tr["fold"], m & tr[Y7].notna())
    d = leftover_diag(
        tr[Y7], tr["e_ar_issued"], (tr["e_ar_issued_lag1"],), tr["fold"], m & tr[Y7].notna()
    )
    rows = [
        _auc_row(Y7, "both>0 issued", raw),
        _auc_row(Y7, "both>0 lag1", lag_auc),
        {
            "y": Y7,
            "feature": "both>0 leftover-lag1 rank",
            "n": f"{d['n']:,}",
            "n_pos": f"{d['n_pos']:,}",
            "CV": _f(d["rank"]),
            "sd": _f(d["ols"]),
            "sign": f"fake={d['fake']}",
            "folds": d["rank_folds"],
        },
    ]
    prose = (
        f"Y7 both>0 issued {_f(_cv(raw))} lag1 {_f(_cv(lag_auc))} "
        f"leftover-lag1 {_f(d['rank'])}. CLOSE TURNOVER add-on."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "raw": _cv(raw), "lag": _cv(lag_auc), "prose": prose}


def pass_pending_cn(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3],
        tr["e_ar_issued"],
        (tr["e_pending_amt_share"], tr["e_credit_note_ratio"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "after pending+CN",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Leftover after pending+CN rank {_f(d['rank'])} "
        f"{'dies' if d['honest_dies'] else 'lives — still off the card'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "prose": prose}


def pass_holdout_both(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    assert set(ho["company_id"]) <= load_holdout()
    iss = pd.to_numeric(ho["e_ar_issued"], errors="coerce")
    lag = pd.to_numeric(ho["e_ar_issued_lag1"], errors="coerce")
    both = (iss > 0) & (lag > 0)
    rows = [
        {
            "slice": "holdout both>0",
            "nn": int(both.sum()),
            "n": len(ho),
            "share": _pp(_pct(int(both.sum()), len(ho))),
        }
    ]
    prose = f"Holdout both>0 share {rows[0]['share']} nn={int(both.sum()):,}. No AUROC."
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_inv_days_both(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    lag = pd.to_numeric(tr["e_ar_issued_lag1"], errors="coerce")
    m = (iss > 0) & (lag > 0)
    d = leftover_diag(
        tr[Y3], tr["c_n_days_with_tx"], (tr["e_ar_issued"],), tr["fold"], m & tr[Y3].notna()
    )
    rows = [
        {
            "bar": "days leftover after issued on both>0",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Days leftover after issued on both>0 rank {_f(d['rank'])} "
        f"{'dies' if d['honest_dies'] else 'lives — keep the 0.711 bar'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "dies": d["honest_dies"], "prose": prose}


def pass_fold_after_ntx(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["a_n_tx"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "cut": "leftover after n_tx fold-wise",
            "rank": _f(d["rank"]),
            "folds": d["rank_folds"],
            "OLS": _f(d["ols"]),
        }
    ]
    prose = f"Leftover after n_tx fold-wise rank {_f(d['rank'])} folds {d['rank_folds']}."
    print(prose)
    return {"rows": rows, "rank": d["rank"], "folds": d["rank_folds"], "prose": prose}


def pass_group_left(tr: pd.DataFrame) -> dict:
    """Leftover after days inside groups with enough Y3 pos — BETWEEN smear check."""
    rows = []
    ranks = []
    for gid, sl in tr.groupby("group_id"):
        n_pos = int((sl[Y3] == 1).sum())
        if n_pos < MIN_POS:
            continue
        d = leftover_diag(
            sl[Y3], sl["e_ar_issued"], (sl["c_n_days_with_tx"],), sl["fold"], sl[Y3].notna()
        )
        ranks.append(d["rank"])
        rows.append(
            {
                "group": str(gid)[:12],
                "rank": _f(d["rank"]),
                "fake?": "FALSE clone" if d["fake"] else "",
                "n": f"{d['n']:,}",
                "n_pos": f"{d['n_pos']:,}",
            }
        )
    finite = [r for r in ranks if np.isfinite(r)]
    med = float(np.median(finite)) if finite else float("nan")
    prose = (
        f"Group leftover-days n_groups={len(rows)} median {_f(med)} "
        f"(panel leftover 0.608)."
    )
    print(prose)
    return {"rows": rows[:20], "n": len(rows), "median": med, "prose": prose}


def pass_cohort(tr: pd.DataFrame) -> dict:
    yr = pd.to_datetime(tr["first_month"]).dt.year
    rows = []
    ranks = {}
    for y in sorted(yr.dropna().unique()):
        m = (yr == y) & tr[Y3].notna()
        d = leftover_diag(
            tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
        )
        ranks[int(y)] = d["rank"]
        rows.append(
            {
                "first_year": int(y),
                "rank": _f(d["rank"]),
                "OLS": _f(d["ols"]),
                "fake?": "FALSE clone" if d["fake"] else "",
                "n": f"{d['n']:,}",
                "n_pos": f"{d['n_pos']:,}",
            }
        )
    prose = "First-month cohort leftover-days " + " ".join(
        f"{y}={_f(ranks[y])}" for y in ranks
    ) + "."
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_y7_after_ntx(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y7], tr["e_ar_issued"], (tr["a_n_tx"],), tr["fold"], tr[Y7].notna()
    )
    rows = [
        {
            "bar": "Y7 leftover after n_tx",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"Y7 leftover after n_tx rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"fake={d['fake']}. Do not grow TURNOVER."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_resid_acf(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    acf = median_acf(d["rresid"], tr["company_id"], 1)
    rows = [{"object": "rank-resid after days", "acf1": _f(acf)}]
    prose = (
        f"Rank-resid after days acf1 {_f(acf)} (raw issued acf1 0.143 LOW_PERSIST)."
    )
    print(prose)
    return {"rows": rows, "acf": acf, "prose": prose}


def pass_holdout_size_trail(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    assert set(ho["company_id"]) <= load_holdout()
    terc = size_terciles(ho)
    so = pd.to_numeric(ho["months_so_far"], errors="coerce")
    cuts = so.quantile([1 / 3, 2 / 3])
    trail = pd.Series("T2", index=ho.index)
    trail[so <= cuts.iloc[0]] = "T1"
    trail[so > cuts.iloc[1]] = "T3"
    iss = pd.to_numeric(ho["e_ar_issued"], errors="coerce")
    rows = []
    for sl in ("T1", "T2", "T3"):
        for tl in ("T1", "T2", "T3"):
            m = (terc == sl) & (trail == tl)
            rows.append(
                {
                    "SIZE×trail": f"{sl}×{tl}",
                    "nn": int(iss[m].notna().sum()),
                    "n": int(m.sum()),
                    "cov": _pp(_pct(int(iss[m].notna().sum()), int(m.sum()))),
                }
            )
    prose = "Holdout SIZE×trail issued coverage only. No AUROC."
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_intensity_both(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    lag = pd.to_numeric(tr["e_ar_issued_lag1"], errors="coerce")
    m = (iss > 0) & (lag > 0)
    d = leftover_diag(
        tr[Y3], tr["issued_per_day"], (tr["c_n_days_with_tx"],), tr["fold"], m & tr[Y3].notna()
    )
    raw = signed_oof_auroc(tr[Y3], tr["issued_per_day"], tr["fold"], m & tr[Y3].notna())
    rows = [
        _auc_row(Y3, "both>0 intensity", raw),
        {
            "y": Y3,
            "feature": "both>0 intensity leftover-days",
            "n": f"{d['n']:,}",
            "n_pos": f"{d['n_pos']:,}",
            "CV": _f(d["rank"]),
            "sd": _f(d["ols"]),
            "sign": f"fake={d['fake']}",
            "folds": d["rank_folds"],
        },
    ]
    prose = (
        f"both>0 intensity Y3 {_f(_cv(raw))} leftover-days {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_cohort_2025(tr: pd.DataFrame) -> dict:
    """2025 first-month leftover 0.685 — is it a days clone?"""
    yr = pd.to_datetime(tr["first_month"]).dt.year
    m = (yr == 2025) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    raw = signed_oof_auroc(tr[Y3], tr["e_ar_issued"], tr["fold"], m)
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], m)
    rows = [
        _auc_row(Y3, "2025-cohort issued", raw),
        _auc_row(Y3, "2025-cohort days", days),
        {
            "y": Y3,
            "feature": "2025-cohort leftover-days rank",
            "n": f"{d['n']:,}",
            "n_pos": f"{d['n_pos']:,}",
            "CV": _f(d["rank"]),
            "sd": _f(d["ols"]),
            "sign": f"fake={d['fake']}",
            "folds": d["rank_folds"],
        },
    ]
    prose = (
        f"2025-cohort leftover-days {_f(d['rank'])} fake={d['fake']} "
        f"issued {_f(_cv(raw))} days {_f(_cv(days))}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "fake": d["fake"],
        "raw": _cv(raw),
        "days": _cv(days),
        "prose": prose,
    }


def pass_kitchen(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3],
        tr["e_ar_issued"],
        (
            tr["c_n_days_with_tx"],
            tr["a_n_tx"],
            tr["log_in3"],
            tr["e_ar_issued_lag1"],
            tr["e_credit_note_ratio"],
        ),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "after days+n_tx+size+lag1+CN",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Kitchen-sink leftover rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"{'dies' if d['honest_dies'] else 'lives — still off the card'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "dies": d["honest_dies"], "prose": prose}


def pass_inv_intensity(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["issued_per_day"], (tr["e_ar_issued"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "bar": "intensity leftover after issued",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Intensity leftover after issued rank {_f(d['rank'])} "
        f"{'dies' if d['honest_dies'] else 'lives — still not a card stem'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "prose": prose}


def pass_rho_dso_size(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    rows = []
    for lab in ("T1", "T2", "T3"):
        m = terc == lab
        rho, n = spearman_n(tr.loc[m, "e_ar_issued"], tr.loc[m, "e_dso_proxy"])
        rows.append({"SIZE": lab, "ρ vs DSO": _f(rho), "n": f"{n:,}"})
    prose = (
        f"issued vs DSO ρ T1 {rows[0]['ρ vs DSO']} T2 {rows[1]['ρ vs DSO']} "
        f"T3 {rows[2]['ρ vs DSO']}. DSO stays DROPPED."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_cohort_2024(tr: pd.DataFrame) -> dict:
    yr = pd.to_datetime(tr["first_month"]).dt.year
    m = (yr == 2024) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    raw = signed_oof_auroc(tr[Y3], tr["e_ar_issued"], tr["fold"], m)
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], m)
    rows = [
        _auc_row(Y3, "2024-cohort issued", raw),
        _auc_row(Y3, "2024-cohort days", days),
        {
            "y": Y3,
            "feature": "2024-cohort leftover-days rank",
            "n": f"{d['n']:,}",
            "n_pos": f"{d['n_pos']:,}",
            "CV": _f(d["rank"]),
            "sd": _f(d["ols"]),
            "sign": f"fake={d['fake']}",
            "folds": d["rank_folds"],
        },
    ]
    prose = (
        f"2024-cohort leftover-days {_f(d['rank'])} fake={d['fake']} "
        f"issued {_f(_cv(raw))} days {_f(_cv(days))}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_y7_after_pending(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y7], tr["e_ar_issued"], (tr["e_pending_amt_share"],), tr["fold"], tr[Y7].notna()
    )
    rows = [
        {
            "bar": "Y7 leftover after pending",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"Y7 leftover after pending rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"fake={d['fake']}. Do not grow TURNOVER."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_rho_cn_size(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    rows = []
    for lab in ("T1", "T2", "T3"):
        m = terc == lab
        rho, n = spearman_n(tr.loc[m, "e_ar_issued"], tr.loc[m, "e_credit_note_ratio"])
        rows.append({"SIZE": lab, "ρ vs CN": _f(rho), "n": f"{n:,}"})
    prose = (
        f"issued vs CN ρ T1 {rows[0]['ρ vs CN']} T2 {rows[1]['ρ vs CN']} "
        f"T3 {rows[2]['ρ vs CN']}."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_days_dso(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3],
        tr["e_ar_issued"],
        (tr["c_n_days_with_tx"], tr["e_dso_proxy"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "after days+DSO",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"Leftover after days+DSO rank {_f(d['rank'])} fake={d['fake']}. "
        "DSO stays DROPPED."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_log1p_kitchen(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3],
        tr["log1p_issued"],
        (tr["c_n_days_with_tx"], tr["a_n_tx"], tr["e_ar_issued_lag1"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "log1p after days+n_tx+lag1",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"log1p leftover after days+n_tx+lag1 rank {_f(d['rank'])} "
        f"{'dies' if d['honest_dies'] else 'lives — still off the card'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "prose": prose}


def pass_erp_zeros(tr: pd.DataFrame, book: set[str]) -> dict:
    """Ever-ERP issued==0 leftover — fill artifact, not KEEP."""
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    m = tr["company_id"].isin(book) & (iss == 0)
    d = leftover_diag(
        tr[Y3], iss, (tr["c_n_days_with_tx"],), tr["fold"], m & tr[Y3].notna()
    )
    raw = signed_oof_auroc(tr[Y3], (iss == 0).astype(float), tr["fold"], tr["company_id"].isin(book) & iss.notna() & tr[Y3].notna())
    rows = [
        _auc_row(Y3, "ERP issued==0 flag", raw),
        {
            "y": Y3,
            "feature": "ERP-zero leftover-days rank",
            "n": f"{d['n']:,}",
            "n_pos": f"{d['n_pos']:,}",
            "CV": _f(d["rank"]),
            "sd": _f(d["ols"]),
            "sign": f"fake={d['fake']}",
            "folds": d["rank_folds"],
        },
    ]
    prose = (
        f"Ever-ERP issued==0 leftover-days {_f(d['rank'])} fake={d['fake']} "
        f"flag {_f(_cv(raw))}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_mid_trail(tr: pd.DataFrame) -> dict:
    m = (tr["so_far_class"] == "mid_12_17") & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "mid_12_17 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"mid_12_17 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_y7_days_size(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y7],
        tr["e_ar_issued"],
        (tr["c_n_days_with_tx"], tr["log_in3"]),
        tr["fold"],
        tr[Y7].notna(),
    )
    rows = [
        {
            "bar": "Y7 leftover after days+size",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Y7 leftover after days+size rank {_f(d['rank'])} "
        f"{'dies' if d['honest_dies'] else 'lives — still CLOSE TURNOVER add-on'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "prose": prose}


def pass_rho_pending_size(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    rows = []
    for lab in ("T1", "T2", "T3"):
        m = terc == lab
        rho, n = spearman_n(tr.loc[m, "e_ar_issued"], tr.loc[m, "e_pending_amt_share"])
        rows.append({"SIZE": lab, "ρ vs pending": _f(rho), "n": f"{n:,}"})
    prose = (
        f"issued vs pending ρ T1 {rows[0]['ρ vs pending']} T2 {rows[1]['ρ vs pending']} "
        f"T3 {rows[2]['ρ vs pending']}."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_long_trail(tr: pd.DataFrame) -> dict:
    m = (tr["so_far_class"] == "long_>=18") & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "long_>=18 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"long_>=18 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_y7_triple(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y7],
        tr["e_ar_issued"],
        (tr["c_n_days_with_tx"], tr["e_ar_issued_lag1"], tr["log_in3"]),
        tr["fold"],
        tr[Y7].notna(),
    )
    rows = [
        {
            "bar": "Y7 leftover after days+lag1+size",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Y7 leftover after days+lag1+size rank {_f(d['rank'])} "
        f"{'dies' if d['honest_dies'] else 'lives — still CLOSE TURNOVER add-on'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "prose": prose}


def pass_rho_ntx_size(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    rows = []
    for lab in ("T1", "T2", "T3"):
        m = terc == lab
        rho, n = spearman_n(tr.loc[m, "e_ar_issued"], tr.loc[m, "a_n_tx"])
        rows.append({"SIZE": lab, "ρ vs n_tx": _f(rho), "n": f"{n:,}"})
    prose = (
        f"issued vs n_tx ρ T1 {rows[0]['ρ vs n_tx']} T2 {rows[1]['ρ vs n_tx']} "
        f"T3 {rows[2]['ρ vs n_tx']}."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_acf_split(tr: pd.DataFrame) -> dict:
    """Leftover after days on high- vs low-acf1 companies."""
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    acf_co = []
    for cid, sl in tr.groupby("company_id"):
        a = median_acf(sl["e_ar_issued"], sl["company_id"], 1)
        acf_co.append((cid, a))
    adf = pd.DataFrame(acf_co, columns=["company_id", "acf1"])
    med = adf["acf1"].median()
    high = set(adf.loc[adf["acf1"] >= med, "company_id"])
    low = set(adf.loc[adf["acf1"] < med, "company_id"])
    rows = []
    ranks = {}
    for name, ids in (("high-acf1", high), ("low-acf1", low)):
        m = tr["company_id"].isin(ids) & tr[Y3].notna()
        d = leftover_diag(
            tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
        )
        ranks[name] = d["rank"]
        rows.append(
            {
                "slice": name,
                "rank": _f(d["rank"]),
                "OLS": _f(d["ols"]),
                "fake?": "FALSE clone" if d["fake"] else "",
                "n": f"{d['n']:,}",
            }
        )
    prose = (
        f"Leftover-days high-acf1 {_f(ranks['high-acf1'])} "
        f"low-acf1 {_f(ranks['low-acf1'])} (median acf1 {_f(med)})."
    )
    print(prose)
    return {
        "rows": rows,
        "high": ranks["high-acf1"],
        "low": ranks["low-acf1"],
        "prose": prose,
    }


def pass_y7_pending_cn(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y7],
        tr["e_ar_issued"],
        (tr["e_pending_amt_share"], tr["e_credit_note_ratio"]),
        tr["fold"],
        tr[Y7].notna(),
    )
    rows = [
        {
            "bar": "Y7 leftover after pending+CN",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Y7 leftover after pending+CN rank {_f(d['rank'])} "
        f"{'dies' if d['honest_dies'] else 'lives — still CLOSE TURNOVER add-on'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "prose": prose}


def pass_p99_gap(panel: pd.DataFrame) -> dict:
    tr = panel[panel["split"] == "train"]
    ho = panel[panel["split"] == "holdout"]
    assert set(ho["company_id"]) <= load_holdout()
    iss_tr = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    iss_ho = pd.to_numeric(ho["e_ar_issued"], errors="coerce")
    rows = [
        {
            "split": "train",
            "p90": _f(iss_tr.quantile(0.90), 0),
            "p99": _f(iss_tr.quantile(0.99), 0),
            "max": _f(iss_tr.max(), 0),
        },
        {
            "split": "holdout",
            "p90": _f(iss_ho.quantile(0.90), 0),
            "p99": _f(iss_ho.quantile(0.99), 0),
            "max": _f(iss_ho.max(), 0),
        },
    ]
    prose = (
        f"issued p99 train {rows[0]['p99']} holdout {rows[1]['p99']}. No AUROC."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_long_book(tr: pd.DataFrame) -> dict:
    m = (tr["n_grid_months"] >= 18) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "n_grid>=18 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"n_grid>=18 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_y7_after_dso(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y7], tr["e_ar_issued"], (tr["e_dso_proxy"],), tr["fold"], tr[Y7].notna()
    )
    rows = [
        {
            "bar": "Y7 leftover after DSO",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"Y7 leftover after DSO rank {_f(d['rank'])} fake={d['fake']}. "
        "DSO stays DROPPED; do not grow TURNOVER."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "prose": prose}


def pass_days_pending_lag1(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3],
        tr["e_ar_issued"],
        (tr["c_n_days_with_tx"], tr["e_pending_amt_share"], tr["e_ar_issued_lag1"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "after days+pending+lag1",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Leftover after days+pending+lag1 rank {_f(d['rank'])} "
        f"{'dies' if d['honest_dies'] else 'lives — still off the card'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "prose": prose}


def pass_short_grid(tr: pd.DataFrame) -> dict:
    m = (tr["n_grid_months"] < 12) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "n_grid<12 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"n_grid<12 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_days_cn_lag1(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3],
        tr["e_ar_issued"],
        (tr["c_n_days_with_tx"], tr["e_credit_note_ratio"], tr["e_ar_issued_lag1"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "after days+CN+lag1",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Leftover after days+CN+lag1 rank {_f(d['rank'])} "
        f"{'dies' if d['honest_dies'] else 'lives — still off the card'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "prose": prose}


def pass_y7_days_pending(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y7],
        tr["e_ar_issued"],
        (tr["c_n_days_with_tx"], tr["e_pending_amt_share"]),
        tr["fold"],
        tr[Y7].notna(),
    )
    rows = [
        {
            "bar": "Y7 leftover after days+pending",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"Y7 leftover after days+pending rank {_f(d['rank'])} fake={d['fake']}. "
        "Do not grow TURNOVER."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_days_ntx_size(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3],
        tr["e_ar_issued"],
        (tr["c_n_days_with_tx"], tr["a_n_tx"], tr["log_in3"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "after days+n_tx+size",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Leftover after days+n_tx+size rank {_f(d['rank'])} "
        f"{'dies' if d['honest_dies'] else 'lives — still off the card'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "prose": prose}


def pass_inv_days_both_bars(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3],
        tr["c_n_days_with_tx"],
        (tr["e_ar_issued"], tr["a_n_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "days leftover after issued+n_tx",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Days leftover after issued+n_tx rank {_f(d['rank'])} "
        f"{'dies' if d['honest_dies'] else 'lives — keep the 0.711 bar'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "dies": d["honest_dies"], "prose": prose}


def pass_t3_only(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    d = leftover_diag(
        tr[Y3],
        tr["e_ar_issued"],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        (terc == "T3") & tr[Y3].notna(),
    )
    rows = [
        {
            "slice": "SIZE T3 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
            "n_pos": f"{d['n_pos']:,}",
        }
    ]
    prose = (
        f"SIZE T3 leftover after days rank {_f(d['rank'])} fake={d['fake']} "
        "(pass 6 was LOW_POWER)."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_issued_median_split(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    med = iss.median()
    rows = []
    ranks = {}
    for name, mask in (("issued>p50", iss > med), ("issued<=p50", iss <= med)):
        d = leftover_diag(
            tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], mask & tr[Y3].notna()
        )
        ranks[name] = d["rank"]
        rows.append(
            {
                "slice": name,
                "rank": _f(d["rank"]),
                "OLS": _f(d["ols"]),
                "fake?": "FALSE clone" if d["fake"] else "",
                "n": f"{d['n']:,}",
            }
        )
    prose = (
        f"Leftover-days issued>p50 {_f(ranks['issued>p50'])} "
        f"issued<=p50 {_f(ranks['issued<=p50'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "hi": ranks["issued>p50"],
        "lo": ranks["issued<=p50"],
        "prose": prose,
    }


def pass_y7_days_cn(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y7],
        tr["e_ar_issued"],
        (tr["c_n_days_with_tx"], tr["e_credit_note_ratio"]),
        tr["fold"],
        tr[Y7].notna(),
    )
    rows = [
        {
            "bar": "Y7 leftover after days+CN",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"Y7 leftover after days+CN rank {_f(d['rank'])} fake={d['fake']}. "
        "Do not grow TURNOVER."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_issued_bands(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    p50 = iss.quantile(0.50)
    p90 = iss.quantile(0.90)
    rows = []
    ranks = {}
    for name, mask in (
        ("p50-p90", (iss > p50) & (iss <= p90)),
        ("p90+", iss > p90),
    ):
        d = leftover_diag(
            tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], mask & tr[Y3].notna()
        )
        ranks[name] = d["rank"]
        rows.append(
            {
                "slice": name,
                "rank": _f(d["rank"]),
                "OLS": _f(d["ols"]),
                "fake?": "FALSE clone" if d["fake"] else "",
                "n": f"{d['n']:,}",
                "n_pos": f"{d['n_pos']:,}",
            }
        )
    prose = (
        f"Leftover-days p50-p90 {_f(ranks['p50-p90'])} p90+ {_f(ranks['p90+'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "mid": ranks["p50-p90"],
        "fat": ranks["p90+"],
        "prose": prose,
    }


def pass_inv_days_size(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3],
        tr["c_n_days_with_tx"],
        (tr["e_ar_issued"], tr["log_in3"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "days leftover after issued+size",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Days leftover after issued+size rank {_f(d['rank'])} "
        f"{'dies' if d['honest_dies'] else 'lives — keep the 0.711 bar'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "dies": d["honest_dies"], "prose": prose}


def pass_issued_quartiles(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    p25 = iss.quantile(0.25)
    p75 = iss.quantile(0.75)
    rows = []
    ranks = {}
    for name, mask in (("<=p25", iss <= p25), (">p75", iss > p75)):
        d = leftover_diag(
            tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], mask & tr[Y3].notna()
        )
        ranks[name] = d["rank"]
        rows.append(
            {
                "slice": name,
                "rank": _f(d["rank"]),
                "OLS": _f(d["ols"]),
                "fake?": "FALSE clone" if d["fake"] else "",
                "n": f"{d['n']:,}",
            }
        )
    prose = (
        f"Leftover-days <=p25 {_f(ranks['<=p25'])} >p75 {_f(ranks['>p75'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "lo": ranks["<=p25"],
        "hi": ranks[">p75"],
        "prose": prose,
    }


def pass_y7_days_ntx(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y7],
        tr["e_ar_issued"],
        (tr["c_n_days_with_tx"], tr["a_n_tx"]),
        tr["fold"],
        tr[Y7].notna(),
    )
    rows = [
        {
            "bar": "Y7 leftover after days+n_tx",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"Y7 leftover after days+n_tx rank {_f(d['rank'])} fake={d['fake']}. "
        "Do not grow TURNOVER."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_issued_iqr(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    p25 = iss.quantile(0.25)
    p75 = iss.quantile(0.75)
    m = (iss > p25) & (iss <= p75)
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m & tr[Y3].notna()
    )
    rows = [
        {
            "slice": "p25-p75 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"p25-p75 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_inv_days_lag1(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3],
        tr["c_n_days_with_tx"],
        (tr["e_ar_issued"], tr["e_ar_issued_lag1"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "days leftover after issued+lag1",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "dies?": "dies" if d["honest_dies"] else "lives",
        }
    ]
    prose = (
        f"Days leftover after issued+lag1 rank {_f(d['rank'])} "
        f"{'dies' if d['honest_dies'] else 'lives — keep the 0.711 bar'}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "dies": d["honest_dies"], "prose": prose}


def pass_ntx_pos(tr: pd.DataFrame) -> dict:
    ntx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    m = (ntx > 0) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "n_tx>0 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"n_tx>0 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_winsor(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    p10 = iss.quantile(0.10)
    p90 = iss.quantile(0.90)
    m = (iss >= p10) & (iss <= p90)
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m & tr[Y3].notna()
    )
    rows = [
        {
            "slice": "p10-p90 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"p10-p90 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_days_pos(tr: pd.DataFrame) -> dict:
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    m = (days > 0) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "days>0 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"days>0 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_lag1_pos(tr: pd.DataFrame) -> dict:
    lag = pd.to_numeric(tr["e_ar_issued_lag1"], errors="coerce")
    m = (lag > 0) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "lag1>0 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"lag1>0 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_onset(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    lag = pd.to_numeric(tr["e_ar_issued_lag1"], errors="coerce")
    m = (iss > 0) & (lag == 0) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "onset issued>0 lag1==0 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"onset (issued>0 lag1==0) leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_stop(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    lag = pd.to_numeric(tr["e_ar_issued_lag1"], errors="coerce")
    m = (iss == 0) & (lag > 0) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "stop issued==0 lag1>0 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"stop (issued==0 lag1>0) leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_iss_days_pos(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    m = (iss > 0) & (days > 0) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "issued>0 days>0 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"issued>0 days>0 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_iss_pos_t1(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    m = (iss > 0) & (terc == "T1") & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "issued>0 SIZE T1 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"issued>0 SIZE T1 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_iss_pos_t2(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    m = (iss > 0) & (terc == "T2") & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "issued>0 SIZE T2 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"issued>0 SIZE T2 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_iss_pos_mid(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    m = (iss > 0) & (tr["so_far_class"] == "mid_12_17") & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "issued>0 mid_12_17 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"issued>0 mid_12_17 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_iss_pos_t23(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    m = (iss > 0) & terc.isin(["T2", "T3"]) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "issued>0 T2+T3 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"issued>0 T2+T3 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_either_pos(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    lag = pd.to_numeric(tr["e_ar_issued_lag1"], errors="coerce")
    m = ((iss > 0) | (lag > 0)) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "issued>0 or lag1>0 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"issued>0 or lag1>0 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_complete(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    ain = pd.to_numeric(tr["a_in3"], errors="coerce")
    m = iss.notna() & days.notna() & ain.notna() & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "complete issued/days/a_in3 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"complete-case leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_days_hi(tr: pd.DataFrame) -> dict:
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    med = days.median()
    m = (days > med) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": f"days>p50({med:.0f}) leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"days>p50 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_days_lo(tr: pd.DataFrame) -> dict:
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    med = days.median()
    m = (days <= med) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": f"days<=p50({med:.0f}) leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"days<=p50 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_ntx_hi(tr: pd.DataFrame) -> dict:
    ntx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    med = ntx.median()
    m = (ntx > med) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": f"n_tx>p50({med:.0f}) leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"n_tx>p50 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_ntx_lo(tr: pd.DataFrame) -> dict:
    ntx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    med = ntx.median()
    m = (ntx <= med) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": f"n_tx<=p50({med:.0f}) leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"n_tx<=p50 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_in3_hi(tr: pd.DataFrame) -> dict:
    ain = pd.to_numeric(tr["a_in3"], errors="coerce")
    med = ain.median()
    m = (ain > med) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "a_in3>p50 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"a_in3>p50 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_cn_nn(tr: pd.DataFrame) -> dict:
    cn = pd.to_numeric(tr["e_credit_note_ratio"], errors="coerce")
    m = cn.notna() & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "CN-defined leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"CN-defined leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_pending_nn(tr: pd.DataFrame) -> dict:
    pend = pd.to_numeric(tr["e_pending_amt_share"], errors="coerce")
    m = pend.notna() & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "pending-defined leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"pending-defined leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_in3_lo(tr: pd.DataFrame) -> dict:
    ain = pd.to_numeric(tr["a_in3"], errors="coerce")
    med = ain.median()
    m = (ain <= med) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "a_in3<=p50 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"a_in3<=p50 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_inv_days_size_lag1(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3],
        tr["c_n_days_with_tx"],
        (tr["e_ar_issued"], tr["log_in3"], tr["e_ar_issued_lag1"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "slice": "days leftover after issued+size+lag1",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"Days leftover after issued+size+lag1 rank {_f(d['rank'])} lives — keep the 0.711 bar."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_pending_cn_days(tr: pd.DataFrame) -> dict:
    pend = pd.to_numeric(tr["e_pending_amt_share"], errors="coerce")
    cn = pd.to_numeric(tr["e_credit_note_ratio"], errors="coerce")
    m = pend.notna() & cn.notna() & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "pending+CN leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"pending+CN leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_ntx_nn(tr: pd.DataFrame) -> dict:
    ntx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    m = ntx.notna() & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "n_tx-defined leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"n_tx-defined leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_iss_days_nn(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    m = iss.notna() & days.notna() & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "issued+days defined leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"issued+days defined leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_dso_nn(tr: pd.DataFrame) -> dict:
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    m = dso.notna() & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "DSO-defined leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"DSO-defined leftover after days rank {_f(d['rank'])} fake={d['fake']}. "
        "DSO stays DROPPED; do not put DSO back on TURNOVER."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_pend_days_pos(tr: pd.DataFrame) -> dict:
    pend = pd.to_numeric(tr["e_pending_amt_share"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    m = pend.notna() & (days > 0) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "pending-defined days>0 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"pending-defined days>0 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_log1p_complete(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    ain = pd.to_numeric(tr["a_in3"], errors="coerce")
    m = iss.notna() & days.notna() & ain.notna() & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["log1p_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "log1p complete-case leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"log1p complete-case leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_dso_iss_pos(tr: pd.DataFrame) -> dict:
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    m = dso.notna() & (iss > 0) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "DSO-defined issued>0 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"DSO-defined issued>0 leftover after days rank {_f(d['rank'])} fake={d['fake']}. "
        "DSO stays DROPPED."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_pend_iss_pos(tr: pd.DataFrame) -> dict:
    pend = pd.to_numeric(tr["e_pending_amt_share"], errors="coerce")
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    m = pend.notna() & (iss > 0) & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "pending-defined issued>0 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"pending-defined issued>0 leftover after days rank {_f(d['rank'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def pass_log1p_dso(tr: pd.DataFrame) -> dict:
    dso = pd.to_numeric(tr["e_dso_proxy"], errors="coerce")
    m = dso.notna() & tr[Y3].notna()
    d = leftover_diag(
        tr[Y3], tr["log1p_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    rows = [
        {
            "slice": "log1p DSO-defined leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"log1p DSO-defined leftover after days rank {_f(d['rank'])} fake={d['fake']}. "
        "DSO stays DROPPED."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "prose": prose}


def decide(p1, p2, p3, p4, p5, p6, p7, p8, p10) -> dict:
    twin = bool(p2["gate_twins"])
    size = bool(p1["size_flag"] or p2["size_flag"])
    leftover = p4["y3_rank"]
    leftover_dies = bool(p4["y3_dies"])
    beat = bool(np.isfinite(p3["beat_size"]) and p3["beat_size"] >= KEEP_DELTA)
    keep_x = bool(beat and (not leftover_dies) and (not size) and (not twin))
    if keep_x:
        card = "KEEP as unused leftover — still off the 15-col card"
        tag = "KEEP"
        why = (
            f"leftover after days rank {_f(leftover)} beats chance and size "
            f"({_f(p3['size'])}) by {_f(p3['beat_size'])}; not SIZE; not a twin. "
            "Do not put issued on the 15-col card yourself."
        )
    elif leftover_dies or twin:
        card = "CLOSE as unused leftover / KEEP off the 15-col card"
        tag = "CLOSE"
        why = (
            f"Y3 leftover after days rank {_f(leftover)} OLS {_f(p4['y3_ols'])} "
            f"{'dies' if leftover_dies else 'thin'}; twin={twin} SIZE={size}. "
            f"Inverse days-after-issued {_f(p4['inv_rank'])}. "
            "Contemporaneous issued stays off the 15-col card. Do not grow TURNOVER."
        )
    elif size:
        card = "CLOSE as SIZE / KEEP off the 15-col card"
        tag = "CLOSE"
        why = (
            f"SIZE ρ={_f(p2['rho_size'])}; leftover after days {_f(leftover)}. "
            "Feature report VIF/size flag holds. Stay off the card."
        )
    else:
        card = "KEEP off the 15-col card / CLOSE as unused leftover"
        tag = "CLOSE"
        why = (
            f"Y3 {_f(p3['y3'])} vs days {_f(p3['days'])} leftover {_f(leftover)}. "
            "Does not clear KEEP-as-X. Stay off the card."
        )
    return {
        "card": card,
        "headline_tag": tag,
        "why": why,
        "keep_x": keep_x,
        "twin": twin,
        "size": size,
        "leftover_dies": leftover_dies,
        "q6": p7["q6"],
        "y7_addon": "CLOSE" if p8["addon_close"] else "lives — do not grow",
        "headline": (
            f"{tag} leftover-after-days rank {_f(leftover)} "
            f"(OLS {_f(p4['y3_ols'])}, fake={p4['y3_fake']}). "
            f"Y3 issued {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size'])}. "
            f"SIZE={size} twin={twin}. Inverse days-after-issued {_f(p4['inv_rank'])}. "
            f"Y7 leftover after lag1 {_f(p8['rank'])} — {('CLOSE' if p8['addon_close'] else 'lives')}. "
            f"Card: {card}. Q6 {p7['q6']}. "
            f"Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged."
        ),
    }


def brief_map(d: dict, p3: dict, p4: dict, p7: dict, p8: dict) -> list[dict]:
    return [
        {
            "#": "1",
            "question": "Who is healthy?",
            "what this cut says": (
                "issued is a volume stem (feature-report BETWEEN / size VIF), not a health Y. "
                "Do not invent `y_issued`."
            ),
        },
        {
            "#": "2",
            "question": "Who is improving?",
            "what this cut says": (
                f"Y3 leftover after days {_f(p4['y3_rank'])} — {d['headline_tag']}. "
                "The 45→65 engine is days 0.711, not contemporaneous issued."
            ),
        },
        {
            "#": "3",
            "question": "Who is turning?",
            "what this cut says": (
                f"Q6 issued_lag1 already KEEP {_f(p7['q6_lag'])}. "
                "Contemporaneous issued is not the lead."
            ),
        },
        {
            "#": "4",
            "question": "Dip vs fall?",
            "what this cut says": (
                f"Y7 leftover after issued_lag1 {_f(p8['rank'])} — {d['y7_addon']}. "
                "Do not grow TURNOVER 0.720."
            ),
        },
        {
            "#": "5",
            "question": "Why did it change?",
            "what this cut says": (
                f"Issued Y3 {_f(p3['y3'])} loses to days {_f(p3['days'])}. "
                "Unused leftover after days is the only why-question."
            ),
        },
        {
            "#": "6",
            "question": "Months earlier?",
            "what this cut says": (
                f"lag1 already KEEP. Contemporaneous leftover on short "
                f"{_f(p7['short_days_l1'])} / {_f(p7['short_iss_l1'])} — {p7['q6']}."
            ),
        },
    ]


def make_png(tr: pd.DataFrame, p4: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    both = iss.notna() & days.notna() & (iss > 0)
    sl = tr.loc[both]
    if len(sl) > 8000:
        sl = sl.sample(8000, random_state=FOLD_SEED)
    sc = ax.scatter(
        pd.to_numeric(sl["c_n_days_with_tx"], errors="coerce"),
        np.log1p(pd.to_numeric(sl["e_ar_issued"], errors="coerce").clip(lower=0)),
        c=sl["log_in3"],
        s=6,
        alpha=0.28,
        cmap="viridis",
        linewidths=0,
    )
    ax.set_xlabel("c_n_days_with_tx")
    ax.set_ylabel("log1p(e_ar_issued)")
    ax.set_title(f"Train: issued vs days (ρ={spearman(iss, days):.3f})")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="log1p(a_in3)")

    ax2 = axes[1]
    lab = tr[Y3].notna() & iss.notna() & days.notna()
    work = tr.loc[lab].copy()
    work["_n"] = iss[lab]
    work["_d"] = days[lab]
    work["_y"] = pd.to_numeric(work[Y3], errors="coerce")
    try:
        work["_qn"] = pd.qcut(work["_n"].rank(method="first"), 5, labels=False) + 1
        work["_qd"] = pd.qcut(work["_d"].rank(method="first"), 5, labels=False) + 1
    except ValueError:
        work["_qn"] = 1
        work["_qd"] = 1
    qs = [1, 2, 3, 4, 5]
    gn = work.groupby("_qn")["_y"].mean().reindex(qs)
    gd = work.groupby("_qd")["_y"].mean().reindex(qs)
    x = np.arange(5)
    ax2.bar(x - 0.18, 100.0 * gn.to_numpy(dtype=float), width=0.36, color="#1f4e79", label="issued")
    ax2.bar(x + 0.18, 100.0 * gd.to_numpy(dtype=float), width=0.36, color="#9e6b4a", label="days")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax2.set_ylabel("Y3 rate (%)")
    title = (
        f"Y3 rate by quintile; leftover rank {p4['y3_rank']:.3f}"
        if np.isfinite(p4["y3_rank"])
        else "Y3 rate"
    )
    ax2.set_title(title)
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def write_md(ctx: dict) -> None:
    d = ctx["decision"]
    p1, p2, p3, p4 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"]
    lines = [
        "# Unused leftover of `e_ar_issued` after `c_n_days_with_tx`",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        f"`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        f"Holdout 72 is coverage only. Seed {FOLD_SEED} group folds. No 0–100. "
        f"No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_issued`. "
        f"Night Y7 stays **TURNOVER {TURNOVER_QUOTE:.3f} / B_shallow {B_SHALLOW_QUOTE:.3f}**. "
        f"Do **not** grow TURNOVER. Do **not** put contemporaneous issued on the 15-col Y3 card. "
        f"Night Y3 stays **{NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f}**. Days **{DAYS_BENCH:.3f}**. "
        f"Size **{SIZE_QUOTE:.3f}**. Y7 never D. Y5 never E. Y3 never B. Dark 470 stay NaN not 0. "
        f"CN leftover KEEP {CN_LEFTOVER_QUOTE:.3f} after issued_lag1 — not overwritten.",
        "",
        "`e_ar_issued` = this-period AR issuance volume. Feature report: 64.1% cov, "
        "acf1 0.14, ICC 0.94 BETWEEN / LOW_PERSIST; company-median VIF with size. "
        "This card is unused leftover after days as Y3 X, and leftover of *now* after "
        "`e_ar_issued_lag1` as a TURNOVER add-on.",
        "",
        "## Headline",
        "",
        d["headline"],
        "",
        "## Brief map",
        "",
        _md_table(brief_map(d, p3, p4, ctx["p7"], ctx["p8"])),
        "",
        "## PARK / CLOSE / KEEP",
        "",
        _md_table(
            [
                {
                    "object": "issued leftover after days (Y3 X)",
                    "decision": "**" + d["headline_tag"] + "**",
                    "why": d["why"],
                },
                {
                    "object": "15-col Y3 card stem",
                    "decision": "**KEEP off the card**",
                    "why": "already off (raw-level park). Do not put issued on the card.",
                },
                {
                    "object": "TURNOVER add-on (now after lag1)",
                    "decision": "**" + d["y7_addon"] + "**",
                    "why": f"Y7 leftover after issued_lag1 {_f(ctx['p8']['rank'])}; do not grow 0.720",
                },
                {
                    "object": "issued_lag1 Q6 / TURNOVER stem",
                    "decision": "**KEEP (locked)**",
                    "why": f"short so-far {_f(ctx['p7']['q6_lag'])} vs 0.626 / 0.630",
                },
                {
                    "object": "health Y `y_issued`",
                    "decision": "**PARK**",
                    "why": "do not invent y_issued",
                },
            ]
        ),
        "",
        "## 1. Coverage / 470 vs ERP / SIZE ρ",
        "",
        p1["prose"],
        "",
        _md_table(p1["rows"]),
        "",
        "## 2. Spearman twins",
        "",
        p2["prose"],
        "",
        _md_table(p2["rows"]),
        "",
        "## 3. Single-feature group-fold",
        "",
        p3["prose"],
        "",
        _md_table(p3["rows"]),
        "",
        "## 4. Honest leftover after days (Y3) + inverse",
        "",
        p4["prose"],
        "",
        _md_table(p4["rows"]),
        "",
        "## 5. Leftover after size / issued_lag1",
        "",
        ctx["p5"]["prose"],
        "",
        _md_table(ctx["p5"]["rows"]),
        "",
        "## 6. SIZE terciles",
        "",
        ctx["p6"]["prose"],
        "",
        _md_table(ctx["p6"]["rows"]),
        "",
        "## 7. Q6 short books",
        "",
        ctx["p7"]["prose"],
        "",
        _md_table(ctx["p7"]["rows"]),
        "",
        "## 8. Y7 leftover after issued_lag1 (TURNOVER add-on)",
        "",
        ctx["p8"]["prose"],
        "",
        _md_table(ctx["p8"]["rows"]),
        "",
        "## 9. CN leftover confirm (do not overwrite)",
        "",
        ctx["p9"]["prose"],
        "",
        _md_table(ctx["p9"]["rows"]),
        "",
        "## 10. Fold 4 — do not reopen TURNOVER",
        "",
        ctx["p10"]["prose"],
        "",
        _md_table(ctx["p10"]["rows"]),
        "",
        "## Extra — holdout coverage",
        "",
        ctx["ph"]["prose"],
        "",
        _md_table(ctx["ph"]["rows"]),
        "",
        "## Extra — same-n issued vs issued_lag1",
        "",
        ctx["ps"]["prose"],
        "",
        _md_table(ctx["ps"]["rows"]),
        "",
        "## Extra — log1p(issued)",
        "",
        ctx["pl"]["prose"],
        "",
        _md_table(ctx["pl"]["rows"]),
        "",
        "## Extra — issued/days intensity",
        "",
        ctx["pi"]["prose"],
        "",
        _md_table(ctx["pi"]["rows"]),
        "",
        "## Extra — fat-issued tail",
        "",
        ctx["pf"]["prose"],
        "",
        _md_table(ctx["pf"]["rows"]),
        "",
        "## Extra — book-only (drop 470)",
        "",
        ctx["pd"]["prose"],
        "",
        _md_table(ctx["pd"]["rows"]),
        "",
        "## Extra — dark BOOK stub",
        "",
        ctx["pst"]["prose"],
        "",
        _md_table(ctx["pst"]["rows"]),
        "",
        "## Extra — log1p leftover after days+size",
        "",
        ctx["plb"]["prose"],
        "",
        _md_table(ctx["plb"]["rows"]),
        "",
        "## Extra — fold-wise leftover after days",
        "",
        ctx["pfl"]["prose"],
        "",
        _md_table(ctx["pfl"]["rows"]),
        "",
        "## Extra — demean leftover",
        "",
        ctx["pdm"]["prose"],
        "",
        _md_table(ctx["pdm"]["rows"]),
        "",
        "## Extra — same-n leftover after days",
        "",
        ctx["psd"]["prose"],
        "",
        _md_table(ctx["psd"]["rows"]),
        "",
        "## Extra — holdout SIZE coverage",
        "",
        ctx["phs"]["prose"],
        "",
        _md_table(ctx["phs"]["rows"]),
        "",
        "## Extra — T2+T3 leftover fake check",
        "",
        ctx["pt23"]["prose"],
        "",
        _md_table(ctx["pt23"]["rows"]),
        "",
        "## Extra — issued>0 leftover after days",
        "",
        ctx["ppos"]["prose"],
        "",
        _md_table(ctx["ppos"]["rows"]),
        "",
        "## Extra — log1p vs issued_lag1",
        "",
        ctx["pll"]["prose"],
        "",
        _md_table(ctx["pll"]["rows"]),
        "",
        "## Extra — short-book leftover fake",
        "",
        ctx["psf"]["prose"],
        "",
        _md_table(ctx["psf"]["rows"]),
        "",
        "## Extra — leftover after a_n_tx",
        "",
        ctx["pnt"]["prose"],
        "",
        _md_table(ctx["pnt"]["rows"]),
        "",
        "## Extra — Y7 leftover after days",
        "",
        ctx["py7d"]["prose"],
        "",
        _md_table(ctx["py7d"]["rows"]),
        "",
        "## Extra — n_tx+days leftover vs days clone",
        "",
        ctx["pnf"]["prose"],
        "",
        _md_table(ctx["pnf"]["rows"]),
        "",
        "## Extra — early6 coverage",
        "",
        ctx["pe6"]["prose"],
        "",
        _md_table(ctx["pe6"]["rows"]),
        "",
        "## Extra — issued==0 flag leftover",
        "",
        ctx["pz"]["prose"],
        "",
        _md_table(ctx["pz"]["rows"]),
        "",
        "## Extra — log1p leftover on issued>0",
        "",
        ctx["plp"]["prose"],
        "",
        _md_table(ctx["plp"]["rows"]),
        "",
        "## Extra — leftover after days+issued_lag1",
        "",
        ctx["pdl"]["prose"],
        "",
        _md_table(ctx["pdl"]["rows"]),
        "",
        "## Extra — CN leftover after issued now",
        "",
        ctx["pcn"]["prose"],
        "",
        _md_table(ctx["pcn"]["rows"]),
        "",
        "## Extra — holdout early6 coverage",
        "",
        ctx["phe"]["prose"],
        "",
        _md_table(ctx["phe"]["rows"]),
        "",
        "## Extra — leftover after pending",
        "",
        ctx["ppend"]["prose"],
        "",
        _md_table(ctx["ppend"]["rows"]),
        "",
        "## Extra — fold 4 leftover after days",
        "",
        ctx["pf4d"]["prose"],
        "",
        _md_table(ctx["pf4d"]["rows"]),
        "",
        "## Extra — lag1 leftover after contemporaneous",
        "",
        ctx["pln"]["prose"],
        "",
        _md_table(ctx["pln"]["rows"]),
        "",
        "## Extra — leftover after CN",
        "",
        ctx["pacn"]["prose"],
        "",
        _md_table(ctx["pacn"]["rows"]),
        "",
        "## Extra — issued / a_in3 leftover",
        "",
        ctx["pratio"]["prose"],
        "",
        _md_table(ctx["pratio"]["rows"]),
        "",
        "## Extra — trail terciles leftover after days",
        "",
        ctx["ptrail"]["prose"],
        "",
        _md_table(ctx["ptrail"]["rows"]),
        "",
        "## Extra — log1p leftover after days+lag1",
        "",
        ctx["pldl"]["prose"],
        "",
        _md_table(ctx["pldl"]["rows"]),
        "",
        "## Extra — train SIZE tercile coverage",
        "",
        ctx["ptsc"]["prose"],
        "",
        _md_table(ctx["ptsc"]["rows"]),
        "",
        "## Extra — company-median ρ (VIF / size flag)",
        "",
        ctx["pmed"]["prose"],
        "",
        _md_table(ctx["pmed"]["rows"]),
        "",
        "## Extra — lag3 leftover",
        "",
        ctx["pl3"]["prose"],
        "",
        _md_table(ctx["pl3"]["rows"]),
        "",
        "## Extra — Y5 coverage (never E)",
        "",
        ctx["py5"]["prose"],
        "",
        _md_table(ctx["py5"]["rows"]),
        "",
        "## Extra — holdout trail coverage",
        "",
        ctx["pht"]["prose"],
        "",
        _md_table(ctx["pht"]["rows"]),
        "",
        "## Extra — leftover after lag1 fold-wise",
        "",
        ctx["pfl1"]["prose"],
        "",
        _md_table(ctx["pfl1"]["rows"]),
        "",
        "## Extra — Y7 leftover after days+lag1",
        "",
        ctx["py7dl"]["prose"],
        "",
        _md_table(ctx["py7dl"]["rows"]),
        "",
        "## Extra — acf1 by SIZE",
        "",
        ctx["pacf"]["prose"],
        "",
        _md_table(ctx["pacf"]["rows"]),
        "",
        "## Extra — issued==0 share by SIZE",
        "",
        ctx["pzsh"]["prose"],
        "",
        _md_table(ctx["pzsh"]["rows"]),
        "",
        "## Extra — days leftover after log1p(issued)",
        "",
        ctx["pidl"]["prose"],
        "",
        _md_table(ctx["pidl"]["rows"]),
        "",
        "## Extra — SIZE × trail leftover grid",
        "",
        ctx["pgrid"]["prose"],
        "",
        _md_table(ctx["pgrid"]["rows"]),
        "",
        "## Extra — leftover after days+size+lag1",
        "",
        ctx["ptrip"]["prose"],
        "",
        _md_table(ctx["ptrip"]["rows"]),
        "",
        "## Extra — Q6 leftover after days_lag1+issued_lag1",
        "",
        ctx["pq6t"]["prose"],
        "",
        _md_table(ctx["pq6t"]["rows"]),
        "",
        "## Extra — leftover after DSO (already DROPPED)",
        "",
        ctx["pdso"]["prose"],
        "",
        _md_table(ctx["pdso"]["rows"]),
        "",
        "## Extra — holdout issued==0 share",
        "",
        ctx["phz"]["prose"],
        "",
        _md_table(ctx["phz"]["rows"]),
        "",
        "## Extra — fold 1 isolate",
        "",
        ctx["pf1"]["prose"],
        "",
        _md_table(ctx["pf1"]["rows"]),
        "",
        "## Extra — rank-resid ICC after days",
        "",
        ctx["pricc"]["prose"],
        "",
        _md_table(ctx["pricc"]["rows"]),
        "",
        "## Extra — calendar leftover after days",
        "",
        ctx["pcal"]["prose"],
        "",
        _md_table(ctx["pcal"]["rows"]),
        "",
        "## Extra — issued vs months_so_far",
        "",
        ctx["psof"]["prose"],
        "",
        _md_table(ctx["psof"]["rows"]),
        "",
        "## Extra — issued ρ vs days / size by SIZE tercile",
        "",
        ctx["prsz"]["prose"],
        "",
        _md_table(ctx["prsz"]["rows"]),
        "",
        "## Extra — n_tx leftover after issued",
        "",
        ctx["pintx"]["prose"],
        "",
        _md_table(ctx["pintx"]["rows"]),
        "",
        "## Extra — leftover after intensity",
        "",
        ctx["paim"]["prose"],
        "",
        _md_table(ctx["paim"]["rows"]),
        "",
        "## Extra — Y7 leftover after CN",
        "",
        ctx["py7cn"]["prose"],
        "",
        _md_table(ctx["py7cn"]["rows"]),
        "",
        "## Extra — SIZE T2 leftover after days",
        "",
        ctx["pt2"]["prose"],
        "",
        _md_table(ctx["pt2"]["rows"]),
        "",
        "## Extra — calendar-month leftover after days",
        "",
        ctx["pmo"]["prose"],
        "",
        _md_table(ctx["pmo"]["rows"]),
        "",
        "## Extra — now vs lag1 ρ by SIZE",
        "",
        ctx["plrs"]["prose"],
        "",
        _md_table(ctx["plrs"]["rows"]),
        "",
        "## Extra — days leftover after n_tx",
        "",
        ctx["pdnt"]["prose"],
        "",
        _md_table(ctx["pdnt"]["rows"]),
        "",
        "## Extra — company-mean leftover",
        "",
        ctx["pcm"]["prose"],
        "",
        _md_table(ctx["pcm"]["rows"]),
        "",
        "## Extra — holdout year coverage",
        "",
        ctx["phy"]["prose"],
        "",
        _md_table(ctx["phy"]["rows"]),
        "",
        "## Extra — leftover after days+lag1+CN",
        "",
        ctx["pdlc"]["prose"],
        "",
        _md_table(ctx["pdlc"]["rows"]),
        "",
        "## Extra — quarter leftover after days",
        "",
        ctx["pq"]["prose"],
        "",
        _md_table(ctx["pq"]["rows"]),
        "",
        "## Extra — lag1 leftover after days",
        "",
        ctx["plad"]["prose"],
        "",
        _md_table(ctx["plad"]["rows"]),
        "",
        "## Extra — Y7 leftover after size",
        "",
        ctx["py7s"]["prose"],
        "",
        _md_table(ctx["py7s"]["rows"]),
        "",
        "## Extra — issued>0 vs lag1",
        "",
        ctx["pplr"]["prose"],
        "",
        _md_table(ctx["pplr"]["rows"]),
        "",
        "## Extra — train vs holdout coverage gap",
        "",
        ctx["pgap"]["prose"],
        "",
        _md_table(ctx["pgap"]["rows"]),
        "",
        "## Extra — T1 issued==0 leftover",
        "",
        ctx["pt1z"]["prose"],
        "",
        _md_table(ctx["pt1z"]["rows"]),
        "",
        "## Extra — both>0 leftover",
        "",
        ctx["pbp"]["prose"],
        "",
        _md_table(ctx["pbp"]["rows"]),
        "",
        "## Extra — leftover after days+n_tx+lag1",
        "",
        ctx["pdnl"]["prose"],
        "",
        _md_table(ctx["pdnl"]["rows"]),
        "",
        "## Extra — size leftover after issued",
        "",
        ctx["psai"]["prose"],
        "",
        _md_table(ctx["psai"]["rows"]),
        "",
        "## Extra — leftover after days_lag1 (all books)",
        "",
        ctx["pdla"]["prose"],
        "",
        _md_table(ctx["pdla"]["rows"]),
        "",
        "## Extra — leftover after size fold-wise",
        "",
        ctx["pfsz"]["prose"],
        "",
        _md_table(ctx["pfsz"]["rows"]),
        "",
        "## Extra — Y7 both>0 leftover after lag1",
        "",
        ctx["py7b"]["prose"],
        "",
        _md_table(ctx["py7b"]["rows"]),
        "",
        "## Extra — leftover after pending+CN",
        "",
        ctx["ppcn"]["prose"],
        "",
        _md_table(ctx["ppcn"]["rows"]),
        "",
        "## Extra — holdout both>0 coverage",
        "",
        ctx["phb"]["prose"],
        "",
        _md_table(ctx["phb"]["rows"]),
        "",
        "## Extra — days leftover after issued on both>0",
        "",
        ctx["pidb"]["prose"],
        "",
        _md_table(ctx["pidb"]["rows"]),
        "",
        "## Extra — leftover after n_tx fold-wise",
        "",
        ctx["pfnt"]["prose"],
        "",
        _md_table(ctx["pfnt"]["rows"]),
        "",
        "## Extra — group leftover after days",
        "",
        ctx["pgrp"]["prose"],
        "",
        _md_table(ctx["pgrp"]["rows"]),
        "",
        "## Extra — first-month cohort leftover",
        "",
        ctx["pcoh"]["prose"],
        "",
        _md_table(ctx["pcoh"]["rows"]),
        "",
        "## Extra — Y7 leftover after n_tx",
        "",
        ctx["py7n"]["prose"],
        "",
        _md_table(ctx["py7n"]["rows"]),
        "",
        "## Extra — rank-resid acf1",
        "",
        ctx["prac"]["prose"],
        "",
        _md_table(ctx["prac"]["rows"]),
        "",
        "## Extra — holdout SIZE×trail coverage",
        "",
        ctx["phst"]["prose"],
        "",
        _md_table(ctx["phst"]["rows"]),
        "",
        "## Extra — both>0 intensity leftover",
        "",
        ctx["pib"]["prose"],
        "",
        _md_table(ctx["pib"]["rows"]),
        "",
        "## Extra — 2025 first-month cohort leftover",
        "",
        ctx["p25"]["prose"],
        "",
        _md_table(ctx["p25"]["rows"]),
        "",
        "## Extra — kitchen-sink leftover",
        "",
        ctx["pkit"]["prose"],
        "",
        _md_table(ctx["pkit"]["rows"]),
        "",
        "## Extra — intensity leftover after issued",
        "",
        ctx["pii"]["prose"],
        "",
        _md_table(ctx["pii"]["rows"]),
        "",
        "## Extra — issued vs DSO by SIZE",
        "",
        ctx["prds"]["prose"],
        "",
        _md_table(ctx["prds"]["rows"]),
        "",
        "## Extra — 2024 first-month cohort leftover",
        "",
        ctx["p24"]["prose"],
        "",
        _md_table(ctx["p24"]["rows"]),
        "",
        "## Extra — Y7 leftover after pending",
        "",
        ctx["py7p"]["prose"],
        "",
        _md_table(ctx["py7p"]["rows"]),
        "",
        "## Extra — issued vs CN by SIZE",
        "",
        ctx["prcn"]["prose"],
        "",
        _md_table(ctx["prcn"]["rows"]),
        "",
        "## Extra — leftover after days+DSO",
        "",
        ctx["pdd"]["prose"],
        "",
        _md_table(ctx["pdd"]["rows"]),
        "",
        "## Extra — log1p leftover after days+n_tx+lag1",
        "",
        ctx["plk"]["prose"],
        "",
        _md_table(ctx["plk"]["rows"]),
        "",
        "## Extra — ever-ERP issued==0 leftover",
        "",
        ctx["perz"]["prose"],
        "",
        _md_table(ctx["perz"]["rows"]),
        "",
        "## Extra — mid-trail leftover",
        "",
        ctx["pmid"]["prose"],
        "",
        _md_table(ctx["pmid"]["rows"]),
        "",
        "## Extra — Y7 leftover after days+size",
        "",
        ctx["py7ds"]["prose"],
        "",
        _md_table(ctx["py7ds"]["rows"]),
        "",
        "## Extra — issued vs pending by SIZE",
        "",
        ctx["prpd"]["prose"],
        "",
        _md_table(ctx["prpd"]["rows"]),
        "",
        "## Extra — long-trail leftover",
        "",
        ctx["plng"]["prose"],
        "",
        _md_table(ctx["plng"]["rows"]),
        "",
        "## Extra — Y7 leftover after days+lag1+size",
        "",
        ctx["py7t"]["prose"],
        "",
        _md_table(ctx["py7t"]["rows"]),
        "",
        "## Extra — issued vs n_tx by SIZE",
        "",
        ctx["prnt"]["prose"],
        "",
        _md_table(ctx["prnt"]["rows"]),
        "",
        "## Extra — leftover by company acf1",
        "",
        ctx["pacs"]["prose"],
        "",
        _md_table(ctx["pacs"]["rows"]),
        "",
        "## Extra — Y7 leftover after pending+CN",
        "",
        ctx["py7pc"]["prose"],
        "",
        _md_table(ctx["py7pc"]["rows"]),
        "",
        "## Extra — issued p99 train vs holdout",
        "",
        ctx["pp99"]["prose"],
        "",
        _md_table(ctx["pp99"]["rows"]),
        "",
        "## Extra — n_grid>=18 leftover",
        "",
        ctx["pngrid"]["prose"],
        "",
        _md_table(ctx["pngrid"]["rows"]),
        "",
        "## Extra — Y7 leftover after DSO",
        "",
        ctx["py7dso"]["prose"],
        "",
        _md_table(ctx["py7dso"]["rows"]),
        "",
        "## Extra — leftover after days+pending+lag1",
        "",
        ctx["pdpl"]["prose"],
        "",
        _md_table(ctx["pdpl"]["rows"]),
        "",
        "## Extra — n_grid<12 leftover",
        "",
        ctx["psg"]["prose"],
        "",
        _md_table(ctx["psg"]["rows"]),
        "",
        "## Extra — leftover after days+CN+lag1",
        "",
        ctx["pdcl"]["prose"],
        "",
        _md_table(ctx["pdcl"]["rows"]),
        "",
        "## Extra — Y7 leftover after days+pending",
        "",
        ctx["py7dp"]["prose"],
        "",
        _md_table(ctx["py7dp"]["rows"]),
        "",
        "## Extra — leftover after days+n_tx+size",
        "",
        ctx["pdns"]["prose"],
        "",
        _md_table(ctx["pdns"]["rows"]),
        "",
        "## Extra — days leftover after issued+n_tx",
        "",
        ctx["pidn"]["prose"],
        "",
        _md_table(ctx["pidn"]["rows"]),
        "",
        "## Extra — SIZE T3 leftover after days",
        "",
        ctx["pt3"]["prose"],
        "",
        _md_table(ctx["pt3"]["rows"]),
        "",
        "## Extra — leftover by issued median",
        "",
        ctx["pmeds"]["prose"],
        "",
        _md_table(ctx["pmeds"]["rows"]),
        "",
        "## Extra — Y7 leftover after days+CN",
        "",
        ctx["py7dc"]["prose"],
        "",
        _md_table(ctx["py7dc"]["rows"]),
        "",
        "## Extra — leftover by issued bands",
        "",
        ctx["pbnd"]["prose"],
        "",
        _md_table(ctx["pbnd"]["rows"]),
        "",
        "## Extra — days leftover after issued+size",
        "",
        ctx["pids"]["prose"],
        "",
        _md_table(ctx["pids"]["rows"]),
        "",
        "## Extra — leftover by issued quartiles",
        "",
        ctx["pqrt"]["prose"],
        "",
        _md_table(ctx["pqrt"]["rows"]),
        "",
        "## Extra — Y7 leftover after days+n_tx",
        "",
        ctx["py7dn"]["prose"],
        "",
        _md_table(ctx["py7dn"]["rows"]),
        "",
        "## Extra — leftover p25-p75",
        "",
        ctx["piqr"]["prose"],
        "",
        _md_table(ctx["piqr"]["rows"]),
        "",
        "## Extra — days leftover after issued+lag1",
        "",
        ctx["pidl2"]["prose"],
        "",
        _md_table(ctx["pidl2"]["rows"]),
        "",
        "## Extra — n_tx>0 leftover after days",
        "",
        ctx["pnp"]["prose"],
        "",
        _md_table(ctx["pnp"]["rows"]),
        "",
        "## Extra — p10-p90 leftover after days",
        "",
        ctx["pwins"]["prose"],
        "",
        _md_table(ctx["pwins"]["rows"]),
        "",
        "## Extra — days>0 leftover after days",
        "",
        ctx["pdpos"]["prose"],
        "",
        _md_table(ctx["pdpos"]["rows"]),
        "",
        "## Extra — lag1>0 leftover after days",
        "",
        ctx["plagp"]["prose"],
        "",
        _md_table(ctx["plagp"]["rows"]),
        "",
        "## Extra — onset leftover after days",
        "",
        ctx["ponset"]["prose"],
        "",
        _md_table(ctx["ponset"]["rows"]),
        "",
        "## Extra — stop leftover after days",
        "",
        ctx["pstop"]["prose"],
        "",
        _md_table(ctx["pstop"]["rows"]),
        "",
        "## Extra — issued>0 days>0 leftover after days",
        "",
        ctx["pidp"]["prose"],
        "",
        _md_table(ctx["pidp"]["rows"]),
        "",
        "## Extra — issued>0 SIZE T1 leftover after days",
        "",
        ctx["pit1"]["prose"],
        "",
        _md_table(ctx["pit1"]["rows"]),
        "",
        "## Extra — issued>0 SIZE T2 leftover after days",
        "",
        ctx["pit2"]["prose"],
        "",
        _md_table(ctx["pit2"]["rows"]),
        "",
        "## Extra — issued>0 mid_12_17 leftover after days",
        "",
        ctx["pimid"]["prose"],
        "",
        _md_table(ctx["pimid"]["rows"]),
        "",
        "## Extra — issued>0 T2+T3 leftover after days",
        "",
        ctx["pit23"]["prose"],
        "",
        _md_table(ctx["pit23"]["rows"]),
        "",
        "## Extra — issued>0 or lag1>0 leftover after days",
        "",
        ctx["peith"]["prose"],
        "",
        _md_table(ctx["peith"]["rows"]),
        "",
        "## Extra — complete-case leftover after days",
        "",
        ctx["pcc"]["prose"],
        "",
        _md_table(ctx["pcc"]["rows"]),
        "",
        "## Extra — days>p50 leftover after days",
        "",
        ctx["pdhi"]["prose"],
        "",
        _md_table(ctx["pdhi"]["rows"]),
        "",
        "## Extra — days<=p50 leftover after days",
        "",
        ctx["pdlo"]["prose"],
        "",
        _md_table(ctx["pdlo"]["rows"]),
        "",
        "## Extra — n_tx>p50 leftover after days",
        "",
        ctx["pnth"]["prose"],
        "",
        _md_table(ctx["pnth"]["rows"]),
        "",
        "## Extra — n_tx<=p50 leftover after days",
        "",
        ctx["pntl"]["prose"],
        "",
        _md_table(ctx["pntl"]["rows"]),
        "",
        "## Extra — a_in3>p50 leftover after days",
        "",
        ctx["pin3h"]["prose"],
        "",
        _md_table(ctx["pin3h"]["rows"]),
        "",
        "## Extra — CN-defined leftover after days",
        "",
        ctx["pcnn"]["prose"],
        "",
        _md_table(ctx["pcnn"]["rows"]),
        "",
        "## Extra — pending-defined leftover after days",
        "",
        ctx["ppnn"]["prose"],
        "",
        _md_table(ctx["ppnn"]["rows"]),
        "",
        "## Extra — a_in3<=p50 leftover after days",
        "",
        ctx["pin3l"]["prose"],
        "",
        _md_table(ctx["pin3l"]["rows"]),
        "",
        "## Extra — days leftover after issued+size+lag1",
        "",
        ctx["pidsl"]["prose"],
        "",
        _md_table(ctx["pidsl"]["rows"]),
        "",
        "## Extra — pending+CN leftover after days",
        "",
        ctx["ppcnd"]["prose"],
        "",
        _md_table(ctx["ppcnd"]["rows"]),
        "",
        "## Extra — n_tx-defined leftover after days",
        "",
        ctx["pntnn"]["prose"],
        "",
        _md_table(ctx["pntnn"]["rows"]),
        "",
        "## Extra — issued+days defined leftover after days",
        "",
        ctx["pissd"]["prose"],
        "",
        _md_table(ctx["pissd"]["rows"]),
        "",
        "## Extra — DSO-defined leftover after days",
        "",
        ctx["pdson"]["prose"],
        "",
        _md_table(ctx["pdson"]["rows"]),
        "",
        "## Extra — pending-defined days>0 leftover after days",
        "",
        ctx["ppdp"]["prose"],
        "",
        _md_table(ctx["ppdp"]["rows"]),
        "",
        "## Extra — log1p complete-case leftover after days",
        "",
        ctx["plcc"]["prose"],
        "",
        _md_table(ctx["plcc"]["rows"]),
        "",
        "## Extra — DSO-defined issued>0 leftover after days",
        "",
        ctx["pdsip"]["prose"],
        "",
        _md_table(ctx["pdsip"]["rows"]),
        "",
        "## Extra — pending-defined issued>0 leftover after days",
        "",
        ctx["ppip"]["prose"],
        "",
        _md_table(ctx["ppip"]["rows"]),
        "",
        "## Extra — log1p DSO-defined leftover after days",
        "",
        ctx["pldso"]["prose"],
        "",
        _md_table(ctx["pldso"]["rows"]),
        "",
        "## What failed / next (held for wave note)",
        "",
    ]
    for x in ctx["failed"]:
        lines.append(f"- {x}")
    lines.extend(
        [
            "",
            f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts: coverage, twins, singles, leftover-days, "
            f"size/lag1 bars, terciles, Q6, Y7 add-on, CN confirm, fold 4, holdout, same-n, "
            f"log1p, intensity, fat tail, book-only.",
            "",
            "Night quotes unchanged. Do not grow TURNOVER. Do not put issued on the 15-col card.",
        ]
    )
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    ts = _now_iso()
    p1 = ctx["p1"]
    rows = [
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_e_ar_issued",
            "value": ctx["p3"]["y3"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={ctx['p3']['days']:.4f} size={ctx['p3']['size']:.4f}",
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
            "metric": "auroc_issued_resid_days_rank",
            "value": ctx["p4"]["y3_rank"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"ols={ctx['p4']['y3_ols']:.4f} fake={ctx['p4']['y3_fake']}",
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
            "metric": "auroc_issued_resid_lag1_rank",
            "value": ctx["p8"]["rank"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"ols={ctx['p8']['ols']:.4f} addon_close={ctx['p8']['addon_close']}",
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
            "metric": "auroc_e_ar_issued_fold4",
            "value": ctx["p10"]["f4_iss"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"lag1_f4={ctx['p10']['f4_lag']}",
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
            "metric": "rho_issued_vs_days",
            "value": ctx["p2"]["rho_days"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"rho_size={ctx['p2']['rho_size']:.4f} rho_lag1={ctx['p2']['rho_lag1']:.4f}",
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
            "metric": "auroc_log1p_issued_resid_days_rank",
            "value": ctx["pl"]["rank"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"ols={ctx['pl']['ols']:.4f} dies={ctx['pl']['dies']}",
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
            "metric": "auroc_issued_pos_resid_days_rank",
            "value": ctx["ppos"]["rank"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"fake={ctx['ppos']['fake']} raw={ctx['ppos']['raw']}",
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
            "metric": "rho_log1p_issued_vs_lag1",
            "value": ctx["pll"]["rho"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"twin={ctx['pll']['twin']} leftover={ctx['pll']['rank']}",
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
            "metric": "rho_issued_vs_login3_company_median",
            "value": ctx["pmed"]["rho_size"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"SIZE={ctx['pmed']['size_flag']} row_rho={ctx['p2']['rho_size']:.4f}",
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
            "metric": "auroc_issued_resid_days_size_lag1_rank",
            "value": ctx["ptrip"]["rank"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"dies={ctx['ptrip']['dies']}",
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
            "metric": "auroc_q6_resid_days_lag1_issued_lag1_rank",
            "value": ctx["pq6t"]["rank"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"dies={ctx['pq6t']['dies']}",
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
            "metric": "auroc_issued_bothpos_resid_days_rank",
            "value": ctx["pbp"]["days_rank"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"lag_rank={ctx['pbp']['lag_rank']:.4f} rho={ctx['pbp']['rho']:.4f}",
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
    new = []
    for r in rows:
        key = (r["agent"], r["x_families"], r["y"], r["model"], r["split"], r["metric"])
        if key in seen:
            continue
        new.append(r)
        seen.add(key)
    if not new:
        print("registry: no new rows")
        return
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        for r in new:
            writer.writerow({k: r.get(k, "") for k in header})
    print(f"registry appended {len(new)} rows")


def write_wave(ctx: dict) -> None:
    d = ctx["decision"]
    lines = [
        "# Wave 4 — issued leftover after days",
        "",
        f"Agent `{AGENT}`. Train group-fold seed {FOLD_SEED}. Holdout 72 coverage only.",
        "",
        "## Files written",
        "",
        "- `analysis/evaluate/issued_qa.py`",
        "- `analysis/outputs/issued_qa.md`",
        "- `analysis/outputs/issued_leftover.png`",
        "- append-only `analysis/experiments/registry.csv`",
        "- this note",
        "",
        "Did not touch `dso_qa.*`, `n_tx_qa.*`, `credit_note_qa.*`, `invoices.py`, "
        "`product/`, parquet / duckdb, `build_targets`, parent journal, LIVE, canvas, "
        "TURNOVER, or the 15-col card. Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. "
        "Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617.",
        "",
        "## Locked verdict",
        "",
        "| object | decision |",
        "| --- | --- |",
        f"| issued leftover after days (Y3) | **{d['headline_tag']}** |",
        "| 15-col Y3 card | **KEEP off the card** |",
        f"| TURNOVER add-on (now after lag1) | **{d['y7_addon']}** |",
        "| issued_lag1 Q6 / TURNOVER | **KEEP (locked)** |",
        "| y_issued | **PARK** |",
        "",
        d["headline"],
        "",
        "## What failed / next",
        "",
    ]
    for x in ctx["failed"]:
        lines.append(f"- {x}")
    WAVE_NOTE.parent.mkdir(parents=True, exist_ok=True)
    WAVE_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {WAVE_NOTE}")


def run() -> dict:
    t0 = time.time()
    print(f"issued_qa start seed={FOLD_SEED} holdout=72 agent={AGENT} wave={WRITE_WAVE}")
    panel = attach_folds(load_panel())
    tr = panel[panel["split"] == "train"].copy()
    assert_no_holdout(tr["company_id"])
    print(f"train CM={len(tr):,} companies={tr['company_id'].nunique()} holdout CM={(panel['split']=='holdout').sum()}")
    book = book_invoice_ids(connect())
    print("pass 1 coverage")
    p1 = pass1_cov(tr, book)
    print("pass 2 twins")
    p2 = pass2_twins(tr)
    print("pass 3 singles")
    p3 = pass3_singles(tr)
    print("pass 4 leftover days")
    p4 = pass4_leftover(tr)
    print("pass 5 size / lag1 bars")
    p5 = pass5_more_bars(tr)
    print("pass 6 terciles")
    p6 = pass6_terciles(tr)
    print("pass 7 Q6")
    p7 = pass7_q6(tr)
    print("pass 8 Y7 add-on")
    p8 = pass8_y7_addon(tr)
    print("pass 9 CN confirm")
    p9 = pass9_cn(tr)
    print("pass 10 fold 4")
    p10 = pass10_fold4(tr)
    print("extra holdout")
    ph = pass_holdout(panel, book)
    print("extra same-n")
    ps = pass_samen(tr)
    print("extra log1p")
    pl = pass_log1p(tr)
    print("extra intensity")
    pi = pass_intensity(tr)
    print("extra fat")
    pf = pass_fat(tr)
    print("extra book-only")
    pd = pass_dark_erp(tr, book)
    print("extra dark stub")
    pst = pass_dark_stub(tr, book)
    print("extra log1p days+size")
    plb = pass_log1p_both(tr)
    print("extra fold leftover")
    pfl = pass_fold_left(tr)
    print("extra demean leftover")
    pdm = pass_demean(tr)
    print("extra same-n leftover days")
    psd = pass_samen_days(tr)
    print("extra holdout SIZE")
    phs = pass_holdout_size(panel)
    print("extra T2+T3 fake")
    pt23 = pass_t23_fake(tr)
    print("extra issued>0 leftover")
    ppos = pass_pos_only(tr)
    print("extra log1p vs lag1")
    pll = pass_log1p_lag(tr)
    print("extra short leftover fake")
    psf = pass_short_fake(tr)
    print("extra leftover after n_tx")
    pnt = pass_after_ntx(tr)
    print("extra Y7 leftover after days")
    py7d = pass_y7_days_left(tr)
    print("extra n_tx+days leftover fake")
    pnf = pass_ntx_days_fake(tr)
    print("extra early6 coverage")
    pe6 = pass_early6(tr)
    print("extra zero-issued flag")
    pz = pass_zero_flag(tr)
    print("extra log1p issued>0 leftover")
    plp = pass_log1p_pos(tr)
    print("extra leftover days+lag1")
    pdl = pass_days_lag1_both(tr)
    print("extra CN leftover after now")
    pcn = pass_cn_now(tr)
    print("extra holdout early6")
    phe = pass_holdout_early(panel)
    print("extra leftover after pending")
    ppend = pass_after_pending(tr)
    print("extra fold4 leftover days")
    pf4d = pass_fold4_days(tr)
    print("extra lag1 leftover after now")
    pln = pass_lag1_after_now(tr)
    print("extra leftover after CN")
    pacn = pass_after_cn(tr)
    print("extra issued/in3 leftover")
    pratio = pass_issued_over_in3(tr)
    print("extra trail terciles leftover")
    ptrail = pass_trail_terciles(tr)
    print("extra log1p after days+lag1")
    pldl = pass_log1p_days_lag1(tr)
    print("extra train SIZE coverage")
    ptsc = pass_train_size_cov(tr)
    print("extra company-median ρ")
    pmed = pass_company_median_rho(tr)
    print("extra lag3 leftover")
    pl3 = pass_lag3(tr)
    print("extra Y5 coverage")
    py5 = pass_y5_cov(tr)
    print("extra holdout trail")
    pht = pass_holdout_trail(panel)
    print("extra fold leftover after lag1")
    pfl1 = pass_fold_lag1(tr)
    print("extra Y7 leftover days+lag1")
    py7dl = pass_y7_days_lag1(tr)
    print("extra acf1 by SIZE")
    pacf = pass_acf_size(tr)
    print("extra zero share by SIZE")
    pzsh = pass_zero_share(tr)
    print("extra days leftover after log1p")
    pidl = pass_inv_days_log1p(tr)
    print("extra SIZE×trail leftover grid")
    pgrid = pass_size_trail_grid(tr)
    print("extra leftover days+size+lag1")
    ptrip = pass_triple(tr)
    print("extra Q6 leftover days_lag1+issued_lag1")
    pq6t = pass_q6_triple(tr)
    print("extra leftover after DSO")
    pdso = pass_after_dso(tr)
    print("extra holdout zero share")
    phz = pass_holdout_zero(panel)
    print("extra fold 1 isolate")
    pf1 = pass_fold1(tr)
    print("extra resid ICC")
    pricc = pass_resid_icc(tr)
    print("extra calendar leftover")
    pcal = pass_calendar(tr)
    print("extra issued vs months_so_far")
    psof = pass_rho_sofar(tr)
    print("extra ρ by SIZE")
    prsz = pass_rho_by_size(tr)
    print("extra n_tx leftover after issued")
    pintx = pass_inv_ntx(tr)
    print("extra leftover after intensity")
    paim = pass_after_intensity(tr)
    print("extra Y7 leftover after CN")
    py7cn = pass_y7_after_cn(tr)
    print("extra SIZE T2 leftover")
    pt2 = pass_t2_only(tr)
    print("extra calendar-month leftover")
    pmo = pass_month(tr)
    print("extra now vs lag1 ρ by SIZE")
    plrs = pass_lag1_rho_size(tr)
    print("extra days leftover after n_tx")
    pdnt = pass_days_after_ntx(tr)
    print("extra company-mean leftover")
    pcm = pass_company_mean(tr)
    print("extra holdout year coverage")
    phy = pass_holdout_year(panel)
    print("extra leftover days+lag1+CN")
    pdlc = pass_days_lag1_cn(tr)
    print("extra quarter leftover")
    pq = pass_quarter(tr)
    print("extra lag1 leftover after days")
    plad = pass_lag1_after_days(tr)
    print("extra Y7 leftover after size")
    py7s = pass_y7_after_size(tr)
    print("extra issued>0 vs lag1")
    pplr = pass_pos_lag1_rho(tr)
    print("extra train vs holdout gap")
    pgap = pass_split_gap(panel)
    print("extra T1 zero leftover")
    pt1z = pass_t1_zero(tr)
    print("extra both>0 leftover")
    pbp = pass_both_pos(tr)
    print("extra leftover days+n_tx+lag1")
    pdnl = pass_days_ntx_lag1(tr)
    print("extra size leftover after issued")
    psai = pass_size_after_issued(tr)
    print("extra leftover after days_lag1 all")
    pdla = pass_days_lag1_all(tr)
    print("extra leftover after size fold-wise")
    pfsz = pass_fold_after_size(tr)
    print("extra Y7 both>0 leftover")
    py7b = pass_y7_both_pos(tr)
    print("extra leftover after pending+CN")
    ppcn = pass_pending_cn(tr)
    print("extra holdout both>0")
    phb = pass_holdout_both(panel)
    print("extra days leftover both>0")
    pidb = pass_inv_days_both(tr)
    print("extra leftover after n_tx fold-wise")
    pfnt = pass_fold_after_ntx(tr)
    print("extra group leftover")
    pgrp = pass_group_left(tr)
    print("extra first-month cohort leftover")
    pcoh = pass_cohort(tr)
    print("extra Y7 leftover after n_tx")
    py7n = pass_y7_after_ntx(tr)
    print("extra resid acf1")
    prac = pass_resid_acf(tr)
    print("extra holdout SIZE×trail")
    phst = pass_holdout_size_trail(panel)
    print("extra both>0 intensity leftover")
    pib = pass_intensity_both(tr)
    print("extra 2025-cohort leftover")
    p25 = pass_cohort_2025(tr)
    print("extra kitchen-sink leftover")
    pkit = pass_kitchen(tr)
    print("extra intensity leftover after issued")
    pii = pass_inv_intensity(tr)
    print("extra issued vs DSO by SIZE")
    prds = pass_rho_dso_size(tr)
    print("extra 2024-cohort leftover")
    p24 = pass_cohort_2024(tr)
    print("extra Y7 leftover after pending")
    py7p = pass_y7_after_pending(tr)
    print("extra issued vs CN by SIZE")
    prcn = pass_rho_cn_size(tr)
    print("extra leftover after days+DSO")
    pdd = pass_days_dso(tr)
    print("extra log1p leftover days+n_tx+lag1")
    plk = pass_log1p_kitchen(tr)
    print("extra ERP-zero leftover")
    perz = pass_erp_zeros(tr, book)
    print("extra mid-trail leftover")
    pmid = pass_mid_trail(tr)
    print("extra Y7 leftover days+size")
    py7ds = pass_y7_days_size(tr)
    print("extra issued vs pending by SIZE")
    prpd = pass_rho_pending_size(tr)
    print("extra long-trail leftover")
    plng = pass_long_trail(tr)
    print("extra Y7 leftover days+lag1+size")
    py7t = pass_y7_triple(tr)
    print("extra issued vs n_tx by SIZE")
    prnt = pass_rho_ntx_size(tr)
    print("extra leftover by company acf1")
    pacs = pass_acf_split(tr)
    print("extra Y7 leftover pending+CN")
    py7pc = pass_y7_pending_cn(tr)
    print("extra p99 train vs holdout")
    pp99 = pass_p99_gap(panel)
    print("extra n_grid>=18 leftover")
    pngrid = pass_long_book(tr)
    print("extra Y7 leftover after DSO")
    py7dso = pass_y7_after_dso(tr)
    print("extra leftover days+pending+lag1")
    pdpl = pass_days_pending_lag1(tr)
    print("extra n_grid<12 leftover")
    psg = pass_short_grid(tr)
    print("extra leftover days+CN+lag1")
    pdcl = pass_days_cn_lag1(tr)
    print("extra Y7 leftover days+pending")
    py7dp = pass_y7_days_pending(tr)
    print("extra leftover days+n_tx+size")
    pdns = pass_days_ntx_size(tr)
    print("extra days leftover after issued+n_tx")
    pidn = pass_inv_days_both_bars(tr)
    print("extra SIZE T3 leftover")
    pt3 = pass_t3_only(tr)
    print("extra leftover by issued median")
    pmeds = pass_issued_median_split(tr)
    print("extra Y7 leftover days+CN")
    py7dc = pass_y7_days_cn(tr)
    print("extra leftover issued bands")
    pbnd = pass_issued_bands(tr)
    print("extra days leftover after issued+size")
    pids = pass_inv_days_size(tr)
    print("extra leftover issued quartiles")
    pqrt = pass_issued_quartiles(tr)
    print("extra Y7 leftover days+n_tx")
    py7dn = pass_y7_days_ntx(tr)
    print("extra leftover p25-p75")
    piqr = pass_issued_iqr(tr)
    print("extra days leftover after issued+lag1")
    pidl2 = pass_inv_days_lag1(tr)
    print("extra n_tx>0 leftover")
    pnp = pass_ntx_pos(tr)
    print("extra p10-p90 leftover")
    pwins = pass_winsor(tr)
    print("extra days>0 leftover")
    pdpos = pass_days_pos(tr)
    print("extra lag1>0 leftover")
    plagp = pass_lag1_pos(tr)
    print("extra onset leftover")
    ponset = pass_onset(tr)
    print("extra stop leftover")
    pstop = pass_stop(tr)
    print("extra issued>0 days>0 leftover")
    pidp = pass_iss_days_pos(tr)
    print("extra issued>0 SIZE T1 leftover")
    pit1 = pass_iss_pos_t1(tr)
    print("extra issued>0 SIZE T2 leftover")
    pit2 = pass_iss_pos_t2(tr)
    print("extra issued>0 mid leftover")
    pimid = pass_iss_pos_mid(tr)
    print("extra issued>0 T2+T3 leftover")
    pit23 = pass_iss_pos_t23(tr)
    print("extra either-pos leftover")
    peith = pass_either_pos(tr)
    print("extra complete-case leftover")
    pcc = pass_complete(tr)
    print("extra days>p50 leftover")
    pdhi = pass_days_hi(tr)
    print("extra days<=p50 leftover")
    pdlo = pass_days_lo(tr)
    print("extra n_tx>p50 leftover")
    pnth = pass_ntx_hi(tr)
    print("extra n_tx<=p50 leftover")
    pntl = pass_ntx_lo(tr)
    print("extra a_in3>p50 leftover")
    pin3h = pass_in3_hi(tr)
    print("extra CN-defined leftover")
    pcnn = pass_cn_nn(tr)
    print("extra pending-defined leftover")
    ppnn = pass_pending_nn(tr)
    print("extra a_in3<=p50 leftover")
    pin3l = pass_in3_lo(tr)
    print("extra days leftover after issued+size+lag1")
    pidsl = pass_inv_days_size_lag1(tr)
    print("extra pending+CN leftover")
    ppcnd = pass_pending_cn_days(tr)
    print("extra n_tx-defined leftover")
    pntnn = pass_ntx_nn(tr)
    print("extra issued+days defined leftover")
    pissd = pass_iss_days_nn(tr)
    print("extra DSO-defined leftover")
    pdson = pass_dso_nn(tr)
    print("extra pending-defined days>0 leftover")
    ppdp = pass_pend_days_pos(tr)
    print("extra log1p complete-case leftover")
    plcc = pass_log1p_complete(tr)
    print("extra DSO-defined issued>0 leftover")
    pdsip = pass_dso_iss_pos(tr)
    print("extra pending-defined issued>0 leftover")
    ppip = pass_pend_iss_pos(tr)
    print("extra log1p DSO-defined leftover")
    pldso = pass_log1p_dso(tr)
    png = make_png(tr, p4)
    decision = decide(p1, p2, p3, p4, p5, p6, p7, p8, p10)
    failed = [
        f"Y3 leftover after days rank {_f(p4['y3_rank'])} OLS {_f(p4['y3_ols'])} fake={p4['y3_fake']}",
        f"inverse days leftover {_f(p4['inv_rank'])}",
        f"leftover after size {_f(p5['size_rank'])} after lag1 {_f(p5['lag_rank'])}",
        f"terciles T1 {_f(p6['t1'])} T2+T3 {_f(p6['t23'])}",
        f"Q6 short leftover days_lag1 {_f(p7['short_days_l1'])} issued_lag1 {_f(p7['short_iss_l1'])}",
        f"Y7 leftover after lag1 {_f(p8['rank'])} addon={decision['y7_addon']}",
        f"fold 4 issued {_f(p10['f4_iss'])} leftover {_f(p10['f4_left'])}",
        f"log1p leftover {_f(pl['rank'])} intensity {_f(pi['rank'])}",
        f"dark stub n={pst['n']}",
        f"log1p leftover days+size {_f(plb['both'])}",
        f"fold leftover {pfl['folds']}",
        f"demean leftover {_f(pdm['demean'])} mean {_f(pdm['mean'])}",
        f"same-n leftover-days {_f(psd['rank'])} fake={psd['fake']}",
        f"T2+T3 leftover {_f(pt23['rank'])} fake={pt23['fake']}",
        f"issued>0 leftover {_f(ppos['rank'])} fake={ppos['fake']}",
        f"log1p vs lag1 ρ={_f(pll['rho'])} leftover {_f(pll['rank'])}",
        f"short leftover days_lag1 {_f(psf['lag_rank'])} fake={psf['lag_fake']}",
        f"leftover after n_tx {_f(pnt['ntx'])} n_tx+days {_f(pnt['both'])}",
        f"Y7 leftover after days {_f(py7d['rank'])} fake={py7d['fake']}",
        f"n_tx+days leftover {_f(pnf['rank'])} fake={pnf['fake']}",
        f"early6 cov {_pp(pe6['early_cov'])} after7 leftover {_f(pe6['late_rank'])}",
        f"zero-flag leftover {_f(pz['rank'])} fake={pz['fake']}",
        f"log1p>0 leftover {_f(plp['rank'])} fake={plp['fake']}",
        f"leftover days+lag1 {_f(pdl['rank'])} dies={pdl['dies']}",
        f"CN leftover after now {_f(pcn['rank'])}",
        f"leftover after pending {_f(ppend['rank'])} pending+days {_f(ppend['both'])}",
        f"fold4 leftover-days Y3 {_f(pf4d['y3'])} Y7 {_f(pf4d['y7'])}",
        f"lag1 leftover after now Y3 {_f(pln['y3'])} Y7 {_f(pln['y7'])}",
        f"leftover after CN {_f(pacn['rank'])}",
        f"issued/in3 leftover {_f(pratio['rank'])} fake={pratio['fake']}",
        f"trail leftover T1 {_f(ptrail['t1'])} T3 {_f(ptrail['t3'])}",
        f"log1p leftover days+lag1 {_f(pldl['rank'])}",
        f"co-median ρ vs log1p(a_in3) {_f(pmed['rho_size'])} SIZE={pmed['size_flag']}",
        f"leftover after days_lag3 {_f(pl3['days_l3'])} after issued_lag3 {_f(pl3['iss_l3'])}",
        f"leftover after lag1 folds {pfl1['folds']}",
        f"Y7 leftover days+lag1 {_f(py7dl['rank'])}",
        f"days leftover after log1p {_f(pidl['rank'])}",
        f"SIZE×trail leftover T1×T1 {_f(pgrid['t1t1'])} T3×T3 {_f(pgrid['t3t3'])}",
        f"leftover days+size+lag1 {_f(ptrip['rank'])}",
        f"Q6 leftover days_lag1+issued_lag1 {_f(pq6t['rank'])}",
        f"leftover after DSO {_f(pdso['rank'])}",
        f"fold 1 issued {_f(pf1['iss'])} days {_f(pf1['days'])}",
        f"rank-resid ICC {_f(pricc['icc'])}",
        f"issued vs months_so_far ρ={_f(psof['rho'])} leftover {_f(psof['rank'])}",
        f"n_tx leftover after issued {_f(pintx['rank'])}",
        f"leftover after intensity {_f(paim['rank'])} fake={paim['fake']}",
        f"Y7 leftover after CN {_f(py7cn['rank'])}",
        f"SIZE T2 leftover {_f(pt2['rank'])} fake={pt2['fake']}",
        f"month leftover median {_f(pmo['median'])}",
        f"now vs lag1 ρ T1 {_f(plrs['t1'])} T3 {_f(plrs['t3'])}",
        f"days leftover after n_tx {_f(pdnt['rank'])}",
        f"co-mean leftover {_f(pcm['rank'])} fake={pcm['fake']}",
        f"leftover days+lag1+CN {_f(pdlc['rank'])}",
        f"quarter leftover Q1 {_f(pq['q1'])} Q3 {_f(pq['q3'])}",
        f"lag1 leftover after days Y3 {_f(plad['y3'])} fake={plad['y3_fake']}",
        f"Y7 leftover after size {_f(py7s['rank'])}",
        f"issued>0 vs lag1 ρ={_f(pplr['rho'])}",
        f"T1 zero leftover {_f(pt1z['rank'])} fake={pt1z['fake']}",
        f"both>0 leftover-days {_f(pbp['days_rank'])} leftover-lag1 {_f(pbp['lag_rank'])}",
        f"leftover days+n_tx+lag1 {_f(pdnl['rank'])}",
        f"size leftover after issued {_f(psai['rank'])}",
        f"leftover after days_lag1 all {_f(pdla['rank'])} fake={pdla['fake']}",
        f"Y7 both>0 leftover-lag1 {_f(py7b['rank'])}",
        f"leftover after pending+CN {_f(ppcn['rank'])}",
        f"days leftover both>0 {_f(pidb['rank'])}",
        f"group leftover n={pgrp['n']} median {_f(pgrp['median'])}",
        f"Y7 leftover after n_tx {_f(py7n['rank'])} fake={py7n['fake']}",
        f"rank-resid acf1 {_f(prac['acf'])}",
        f"both>0 intensity leftover {_f(pib['rank'])} fake={pib['fake']}",
        f"2025-cohort leftover {_f(p25['rank'])} fake={p25['fake']}",
        f"kitchen-sink leftover {_f(pkit['rank'])}",
        f"2024-cohort leftover {_f(p24['rank'])} fake={p24['fake']}",
        f"Y7 leftover after pending {_f(py7p['rank'])}",
        f"leftover after days+DSO {_f(pdd['rank'])} fake={pdd['fake']}",
        f"log1p leftover days+n_tx+lag1 {_f(plk['rank'])}",
        f"ERP-zero leftover {_f(perz['rank'])} fake={perz['fake']}",
        f"mid-trail leftover {_f(pmid['rank'])} fake={pmid['fake']}",
        f"Y7 leftover days+size {_f(py7ds['rank'])}",
        f"long-trail leftover {_f(plng['rank'])} fake={plng['fake']}",
        f"Y7 leftover days+lag1+size {_f(py7t['rank'])}",
        f"leftover high-acf1 {_f(pacs['high'])} low-acf1 {_f(pacs['low'])}",
        f"Y7 leftover pending+CN {_f(py7pc['rank'])}",
        f"n_grid>=18 leftover {_f(pngrid['rank'])} fake={pngrid['fake']}",
        f"Y7 leftover after DSO {_f(py7dso['rank'])}",
        f"leftover days+pending+lag1 {_f(pdpl['rank'])}",
        f"n_grid<12 leftover {_f(psg['rank'])} fake={psg['fake']}",
        f"leftover days+CN+lag1 {_f(pdcl['rank'])}",
        f"Y7 leftover days+pending {_f(py7dp['rank'])} fake={py7dp['fake']}",
        f"leftover days+n_tx+size {_f(pdns['rank'])}",
        f"days leftover after issued+n_tx {_f(pidn['rank'])}",
        f"SIZE T3 leftover {_f(pt3['rank'])} fake={pt3['fake']}",
        f"leftover issued>p50 {_f(pmeds['hi'])} <=p50 {_f(pmeds['lo'])}",
        f"Y7 leftover days+CN {_f(py7dc['rank'])} fake={py7dc['fake']}",
        f"leftover p50-p90 {_f(pbnd['mid'])} p90+ {_f(pbnd['fat'])}",
        f"days leftover after issued+size {_f(pids['rank'])}",
        f"leftover <=p25 {_f(pqrt['lo'])} >p75 {_f(pqrt['hi'])}",
        f"Y7 leftover days+n_tx {_f(py7dn['rank'])} fake={py7dn['fake']}",
        f"leftover p25-p75 {_f(piqr['rank'])} fake={piqr['fake']}",
        f"days leftover after issued+lag1 {_f(pidl2['rank'])}",
        f"n_tx>0 leftover {_f(pnp['rank'])} fake={pnp['fake']}",
        f"p10-p90 leftover {_f(pwins['rank'])} fake={pwins['fake']}",
        f"days>0 leftover {_f(pdpos['rank'])} fake={pdpos['fake']}",
        f"lag1>0 leftover {_f(plagp['rank'])} fake={plagp['fake']}",
        f"onset leftover {_f(ponset['rank'])} fake={ponset['fake']}",
        f"stop leftover {_f(pstop['rank'])} fake={pstop['fake']}",
        f"issued>0 days>0 leftover {_f(pidp['rank'])} fake={pidp['fake']}",
        f"issued>0 T1 leftover {_f(pit1['rank'])} fake={pit1['fake']}",
        f"issued>0 T2 leftover {_f(pit2['rank'])} fake={pit2['fake']}",
        f"issued>0 mid leftover {_f(pimid['rank'])} fake={pimid['fake']}",
        f"issued>0 T2+T3 leftover {_f(pit23['rank'])} fake={pit23['fake']}",
        f"either-pos leftover {_f(peith['rank'])} fake={peith['fake']}",
        f"complete-case leftover {_f(pcc['rank'])} fake={pcc['fake']}",
        f"days>p50 leftover {_f(pdhi['rank'])} fake={pdhi['fake']}",
        f"days<=p50 leftover {_f(pdlo['rank'])} fake={pdlo['fake']}",
        f"n_tx>p50 leftover {_f(pnth['rank'])} fake={pnth['fake']}",
        f"n_tx<=p50 leftover {_f(pntl['rank'])} fake={pntl['fake']}",
        f"a_in3>p50 leftover {_f(pin3h['rank'])} fake={pin3h['fake']}",
        f"CN-defined leftover {_f(pcnn['rank'])} fake={pcnn['fake']}",
        f"pending-defined leftover {_f(ppnn['rank'])} fake={ppnn['fake']}",
        f"a_in3<=p50 leftover {_f(pin3l['rank'])} fake={pin3l['fake']}",
        f"days leftover after issued+size+lag1 {_f(pidsl['rank'])}",
        f"pending+CN leftover {_f(ppcnd['rank'])} fake={ppcnd['fake']}",
        f"n_tx-defined leftover {_f(pntnn['rank'])} fake={pntnn['fake']}",
        f"issued+days leftover {_f(pissd['rank'])} fake={pissd['fake']}",
        f"DSO-defined leftover {_f(pdson['rank'])} fake={pdson['fake']}",
        f"pending days>0 leftover {_f(ppdp['rank'])} fake={ppdp['fake']}",
        f"log1p complete leftover {_f(plcc['rank'])} fake={plcc['fake']}",
        f"DSO issued>0 leftover {_f(pdsip['rank'])} fake={pdsip['fake']}",
        f"pending issued>0 leftover {_f(ppip['rank'])} fake={ppip['fake']}",
        f"log1p DSO leftover {_f(pldso['rank'])} fake={pldso['fake']}",
        f"card: {decision['card']}",
        "do not grow TURNOVER 0.720; do not put issued on the 15-col card",
    ]
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
        "ph": ph,
        "ps": ps,
        "pl": pl,
        "pi": pi,
        "pf": pf,
        "pd": pd,
        "pst": pst,
        "plb": plb,
        "pfl": pfl,
        "pdm": pdm,
        "psd": psd,
        "phs": phs,
        "pt23": pt23,
        "ppos": ppos,
        "pll": pll,
        "psf": psf,
        "pnt": pnt,
        "py7d": py7d,
        "pnf": pnf,
        "pe6": pe6,
        "pz": pz,
        "plp": plp,
        "pdl": pdl,
        "pcn": pcn,
        "phe": phe,
        "ppend": ppend,
        "pf4d": pf4d,
        "pln": pln,
        "pacn": pacn,
        "pratio": pratio,
        "ptrail": ptrail,
        "pldl": pldl,
        "ptsc": ptsc,
        "pmed": pmed,
        "pl3": pl3,
        "py5": py5,
        "pht": pht,
        "pfl1": pfl1,
        "py7dl": py7dl,
        "pacf": pacf,
        "pzsh": pzsh,
        "pidl": pidl,
        "pgrid": pgrid,
        "ptrip": ptrip,
        "pq6t": pq6t,
        "pdso": pdso,
        "phz": phz,
        "pf1": pf1,
        "pricc": pricc,
        "pcal": pcal,
        "psof": psof,
        "prsz": prsz,
        "pintx": pintx,
        "paim": paim,
        "py7cn": py7cn,
        "pt2": pt2,
        "pmo": pmo,
        "plrs": plrs,
        "pdnt": pdnt,
        "pcm": pcm,
        "phy": phy,
        "pdlc": pdlc,
        "pq": pq,
        "plad": plad,
        "py7s": py7s,
        "pplr": pplr,
        "pgap": pgap,
        "pt1z": pt1z,
        "pbp": pbp,
        "pdnl": pdnl,
        "psai": psai,
        "pdla": pdla,
        "pfsz": pfsz,
        "py7b": py7b,
        "ppcn": ppcn,
        "phb": phb,
        "pidb": pidb,
        "pfnt": pfnt,
        "pgrp": pgrp,
        "pcoh": pcoh,
        "py7n": py7n,
        "prac": prac,
        "phst": phst,
        "pib": pib,
        "p25": p25,
        "pkit": pkit,
        "pii": pii,
        "prds": prds,
        "p24": p24,
        "py7p": py7p,
        "prcn": prcn,
        "pdd": pdd,
        "plk": plk,
        "perz": perz,
        "pmid": pmid,
        "py7ds": py7ds,
        "prpd": prpd,
        "plng": plng,
        "py7t": py7t,
        "prnt": prnt,
        "pacs": pacs,
        "py7pc": py7pc,
        "pp99": pp99,
        "pngrid": pngrid,
        "py7dso": py7dso,
        "pdpl": pdpl,
        "psg": psg,
        "pdcl": pdcl,
        "py7dp": py7dp,
        "pdns": pdns,
        "pidn": pidn,
        "pt3": pt3,
        "pmeds": pmeds,
        "py7dc": py7dc,
        "pbnd": pbnd,
        "pids": pids,
        "pqrt": pqrt,
        "py7dn": py7dn,
        "piqr": piqr,
        "pidl2": pidl2,
        "pnp": pnp,
        "pwins": pwins,
        "pdpos": pdpos,
        "plagp": plagp,
        "ponset": ponset,
        "pstop": pstop,
        "pidp": pidp,
        "pit1": pit1,
        "pit2": pit2,
        "pimid": pimid,
        "pit23": pit23,
        "peith": peith,
        "pcc": pcc,
        "pdhi": pdhi,
        "pdlo": pdlo,
        "pnth": pnth,
        "pntl": pntl,
        "pin3h": pin3h,
        "pcnn": pcnn,
        "ppnn": ppnn,
        "pin3l": pin3l,
        "pidsl": pidsl,
        "ppcnd": ppcnd,
        "pntnn": pntnn,
        "pissd": pissd,
        "pdson": pdson,
        "ppdp": ppdp,
        "plcc": plcc,
        "pdsip": pdsip,
        "ppip": ppip,
        "pldso": pldso,
        "decision": decision,
        "failed": failed,
        "elapsed_s": elapsed,
        "png": png,
    }
    write_md(ctx)
    append_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    print(f"DONE elapsed={elapsed:.0f}s tag={decision['headline_tag']} card={decision['card']}")
    print(decision["headline"])
    return ctx


if __name__ == "__main__":
    run()
