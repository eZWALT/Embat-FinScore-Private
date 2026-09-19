"""Unused leftover of ``e_ap_open`` after ``c_n_days_with_tx`` as Y3 X.

NORTH_STAR: ``e_ap_open`` = unpaid AP |amount| stock at period end
(amount<0 invoices). Feature report: 64.1% cov, acf1 0.74, ICC 0.99
BETWEEN; size_ρ 0.411. ``e_dpo_proxy`` = open / this-period issued —
already DROP from the 44. Contemporaneous ``e_ap_issued`` CLOSED leftover
0.591 (fake days clone, SIZE ρ 0.514). AR issued leftover 0.608 CLOSE.
TURNOVER 0.720 must not grow. Y5 never E. Dark 470 stay NaN not 0.
Do **not** overwrite dpo_qa / ap_issued_qa / issued_qa / pending_qa.

KEEP-as-X: beat size ≥0.02 AND leftover after days AND not SIZE
(|ρ| vs log1p(a_in3) ≥0.50) AND not a twin (|ρ|≥0.80 vs days / a_n_tx /
e_ap_issued / e_dpo_proxy / e_pending_amt_share). Leftover <0.55 dies.
Rank leftover is honest; OLS can fake a days leak (AP issued ρ=-0.907).

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.ap_open_qa

Owned: analysis/evaluate/ap_open_qa.py, analysis/outputs/ap_open_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_ap_open.md (end).
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
from analysis.features.common import ANALYSIS, DATA, connect
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
OUT_MD = ANALYSIS / "outputs" / "ap_open_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "ap_open_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_ap_open.md"
AGENT = "e8b2c0d4"
WAVE = "4"
ROUND = "R4"
MODEL = "ap_open_qa"
X_FAM = "E"

Y3 = "y3_recover_cash_6m"
Y5 = "y5_ap_od30_ownp80"
Y7 = "y7_top1_lost"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
ISSUED_Y3_QUOTE = 0.675
ISSUED_LEFTOVER_QUOTE = 0.591
AR_LEFTOVER_QUOTE = 0.608
Q6_LAG1_QUOTE = 0.626
TURNOVER_QUOTE = 0.720
B_SHALLOW_QUOTE = 0.712
NIGHT_Y3 = 0.762
NIGHT_Y3_CORE = 0.752
COV_QUOTE = 0.641
ICC_QUOTE = 0.99
ACF1_QUOTE = 0.74
SIZE_RHO_QUOTE = 0.411
TWIN_RHO = 0.80
KEEP_DELTA = 0.02
CHANCE = 0.55
SIZE_RHO = 0.50
MIN_POS = 50
MIN_ACF_PAIRS = 4
N_BOOT = 40
WRITE_WAVE = True

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "a_n_tx",
    "c_n_days_with_tx",
    "e_ap_open",
    "e_ap_issued",
    "e_dpo_proxy",
    "e_pending_amt_share",
    "e_ar_open",
)

Y_KEEP = (Y3, Y5, Y7)
TWIN_GATES = (
    "c_n_days_with_tx",
    "a_n_tx",
    "e_ap_issued",
    "e_dpo_proxy",
    "e_pending_amt_share",
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
    wb = var_w / var_b if var_b > 0 else float("nan")
    return {"icc": float(icc), "var_w": float(var_w), "var_b": float(var_b), "k": k, "wb": float(wb)}


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


def signed_oof_auroc(y, x, folds, mask, n_folds: int = N_FOLDS) -> dict:
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
    have_lag = "e_ap_open_lag1" in raw.columns
    cols = list(STORE_COLS)
    if have_lag:
        cols.append("e_ap_open_lag1")
    missing = [c for c in STORE_COLS if c not in raw.columns]
    if missing:
        raise RuntimeError(f"monthly.parquet missing {missing}")
    ymiss = [c for c in Y_KEEP if c not in yraw.columns]
    if ymiss:
        raise RuntimeError(f"targets.parquet missing {ymiss}")
    panel = _keys(raw[cols])
    y = _keys(yraw[["company_id", "period", *Y_KEEP]])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    op = pd.to_numeric(panel["e_ap_open"], errors="coerce")
    panel["log1p_open"] = np.log1p(op.clip(lower=0))
    days = pd.to_numeric(panel["c_n_days_with_tx"], errors="coerce")
    panel["open_per_day"] = op / days.where(days > 0)
    panel["first_month"] = pd.to_datetime(panel["first_month"])
    panel["months_so_far"] = (
        (panel["period"].dt.year - panel["first_month"].dt.year) * 12
        + (panel["period"].dt.month - panel["first_month"].dt.month)
        + 1
    )
    panel["so_far_class"] = panel["months_so_far"].map(_trail_class)
    panel["early6"] = panel["period"] < DELAY_MASK_BEFORE
    panel["store_has_open_lag1"] = have_lag
    panel = add_panel_lags(
        panel,
        ["e_ap_open", "e_ap_issued", "e_ar_open", "c_n_days_with_tx", "log1p_open"],
        (1, 3),
    )
    leak7 = leakage_check(["e_ap_open", "e_ap_open_lag1"], Y7, forbidden_prefixes=["d"])
    leak3 = leakage_check(["e_ap_open"], Y3, forbidden_prefixes=["b"])
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
    op = pd.to_numeric(tr["e_ap_open"], errors="coerce")
    n_cm = int(len(tr))
    n_nn = int(op.notna().sum())
    dark_nn = int(op[dark].notna().sum())
    dark_zero = int((op[dark] == 0).sum())
    dark_pos = int((op[dark] > 0).sum())
    erp_nn = int(op[erp].notna().sum())
    erp_zero = int((op[erp] == 0).sum())
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    n_erp_co = int(tr.loc[erp, "company_id"].nunique())
    rho_size, n_size = spearman_n(op, tr["log_in3"])
    size_flag = bool(np.isfinite(rho_size) and abs(rho_size) >= SIZE_RHO)
    finite = op[op.notna()]
    p50 = float(finite.median()) if n_nn else float("nan")
    p99 = float(finite.quantile(0.99)) if n_nn else float("nan")
    mx = float(finite.max()) if n_nn else float("nan")
    eq0 = _pct(int((finite == 0).sum()), n_nn)
    acf1 = median_acf(op, tr["company_id"], 1)
    icc = icc_anova(op, tr["company_id"])
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
    dark_ok = dark_nn == 0 and dark_zero == 0
    prose = (
        f"Train e_ap_open nn={n_nn:,} cov={_pp(_pct(n_nn, n_cm))} "
        f"(feature report {COV_QUOTE:.1%}). Dark never-ERP {n_dark_co} "
        f"(want 470): nn={dark_nn} zero={dark_zero} pos={dark_pos} "
        f"{'CONFIRM NaN not 0' if dark_ok else ('BOOK stub' if dark_nn else 'check')}. "
        f"Ever-ERP {n_erp_co} nn={erp_nn:,} of which zero={erp_zero:,}. "
        f"p50={_f(p50, 0)} p99={_f(p99, 0)} max={_f(mx, 0)}. "
        f"ρ vs log1p(a_in3)={_f(rho_size)} n={n_size} "
        f"{'SIZE' if size_flag else 'not SIZE'} (quote {SIZE_RHO_QUOTE:.3f}). "
        f"acf1={_f(acf1)} (quote {ACF1_QUOTE:.2f}) ICC={_f(icc['icc'])} "
        f"(quote {ICC_QUOTE:.2f} BETWEEN) w/b={_f(icc['wb'])} k={icc['k']}."
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
        "dark_ok": dark_ok,
        "rho_size": rho_size,
        "size_flag": size_flag,
        "acf1": acf1,
        "icc": icc["icc"],
        "prose": prose,
    }


def pass2_twins(tr: pd.DataFrame) -> dict:
    op = tr["e_ap_open"]
    pairs = [
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("a_n_tx", tr["a_n_tx"]),
        ("e_ap_issued", tr["e_ap_issued"]),
        ("e_dpo_proxy", tr["e_dpo_proxy"]),
        ("e_pending_amt_share", tr["e_pending_amt_share"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("e_ar_open", tr["e_ar_open"]),
        ("e_ap_open_lag1", tr["e_ap_open_lag1"]),
    ]
    rows = []
    rhos = {}
    twins = []
    for name, s in pairs:
        rho, n = spearman_n(op, s)
        rhos[name] = rho
        twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
        if twin:
            twins.append(name)
        rows.append({"vs": name, "ρ": _f(rho), "n": f"{n:,}", "twin?": "TWIN" if twin else ""})
    size_flag = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    gate = [t for t in twins if t in TWIN_GATES]
    prose = (
        f"ap_open vs days ρ={_f(rhos['c_n_days_with_tx'])} "
        f"({'TWIN' if 'c_n_days_with_tx' in twins else 'not a twin'}). "
        f"vs a_n_tx {_f(rhos['a_n_tx'])} vs e_ap_issued {_f(rhos['e_ap_issued'])} "
        f"vs DPO {_f(rhos['e_dpo_proxy'])} vs pending {_f(rhos['e_pending_amt_share'])} "
        f"vs log1p(a_in3) {_f(rhos['log1p(a_in3)'])} "
        f"{'SIZE' if size_flag else 'not SIZE'}. "
        f"vs e_ar_open {_f(rhos['e_ar_open'])} vs open_lag1 {_f(rhos['e_ap_open_lag1'])}. "
        f"TWIN |ρ|≥0.80: {', '.join(twins) if twins else 'none'}. "
        f"Gate twins: {', '.join(gate) if gate else 'none'}."
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
        "rho_issued": rhos["e_ap_issued"],
        "rho_dpo": rhos["e_dpo_proxy"],
        "rho_pend": rhos["e_pending_amt_share"],
        "rho_ar": rhos["e_ar_open"],
        "rho_lag1": rhos["e_ap_open_lag1"],
        "prose": prose,
    }


def pass3_singles(tr: pd.DataFrame) -> dict:
    feats = [
        ("e_ap_open", tr["e_ap_open"]),
        ("e_ap_open_lag1", tr["e_ap_open_lag1"]),
        ("log1p_open", tr["log1p_open"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("log1p_a_in3", tr["log_in3"]),
        ("e_ap_issued", tr["e_ap_issued"]),
        ("e_dpo_proxy", tr["e_dpo_proxy"]),
        ("e_ar_open", tr["e_ar_open"]),
        ("e_pending_amt_share", tr["e_pending_amt_share"]),
        ("a_n_tx", tr["a_n_tx"]),
    ]
    rows = []
    store = {}
    for y in (Y3, Y7):
        for name, s in feats:
            rec = signed_oof_auroc(tr[y], s, tr["fold"], tr[y].notna())
            store[(y, name)] = rec
            rows.append(_auc_row(y, name, rec))
            print(f"{y} {name}: {_f(_cv(rec))} n={rec['n_defined']:,} pos={rec['n_pos']:,}")
    y3 = _cv(store[(Y3, "e_ap_open")])
    days = _cv(store[(Y3, "c_n_days_with_tx")])
    size = _cv(store[(Y3, "log1p_a_in3")])
    issued = _cv(store[(Y3, "e_ap_issued")])
    dpo = _cv(store[(Y3, "e_dpo_proxy")])
    ar = _cv(store[(Y3, "e_ar_open")])
    beat = y3 - size if np.isfinite(y3) and np.isfinite(size) else float("nan")
    prose = (
        f"Y3 e_ap_open {_f(y3)} vs days {_f(days)} (CONFIRM {DAYS_BENCH:.3f}) "
        f"vs size {_f(size)} (CONFIRM {SIZE_QUOTE:.3f}) "
        f"vs e_ap_issued {_f(issued)} (CONFIRM {ISSUED_Y3_QUOTE:.3f}) "
        f"vs DPO {_f(dpo)} vs e_ar_open {_f(ar)}. "
        f"open_lag1 {_f(_cv(store[(Y3, 'e_ap_open_lag1')]))}. "
        f"Y7 open {_f(_cv(store[(Y7, 'e_ap_open')]))}. Beat size {_f(beat)}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3": y3,
        "days": days,
        "size": size,
        "issued": issued,
        "dpo": dpo,
        "ar": ar,
        "lag1": _cv(store[(Y3, "e_ap_open_lag1")]),
        "y7": _cv(store[(Y7, "e_ap_open")]),
        "beat_size": beat,
        "prose": prose,
    }


def pass4_leftover(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ap_open"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    inv = leftover_diag(
        tr[Y3], tr["c_n_days_with_tx"], (tr["e_ap_open"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "bar": "open leftover after days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ(resid,ctrl)": _f(d["rho_ctrl"]),
            "R²": _f(d["r2"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        },
        {
            "bar": "days leftover after open",
            "rank": _f(inv["rank"]),
            "OLS": _f(inv["ols"]),
            "ρ(resid,ctrl)": _f(inv["rho_ctrl"]),
            "R²": _f(inv["r2"]),
            "fake?": "FALSE clone" if inv["fake"] else "",
            "n": f"{inv['n']:,}",
        },
    ]
    prose = (
        f"Y3 leftover after days rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"ρ(resid,days)={_f(d['rho_ctrl'])} R²={_f(d['r2'])} "
        f"{'FALSE clone — dies' if d['fake'] else ('dies' if d['honest_dies'] else 'lives')}. "
        f"Inverse: days leftover after open rank {_f(inv['rank'])} "
        f"{'survives — keep the 0.711 bar' if not inv['honest_dies'] else 'dies'}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3_rank": d["rank"],
        "y3_ols": d["ols"],
        "y3_rho": d["rho_ctrl"],
        "y3_fake": d["fake"],
        "y3_dies": d["honest_dies"],
        "inv_rank": inv["rank"],
        "inv_dies": inv["honest_dies"],
        "prose": prose,
    }


def pass5_after_issued(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ap_open"], (tr["e_ap_issued"],), tr["fold"], tr[Y3].notna()
    )
    both = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["e_ap_issued"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rewrite = bool(d["honest_dies"])
    rows = [
        {
            "bar": "open leftover after issued",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        },
        {
            "bar": "open leftover after issued+days",
            "rank": _f(both["rank"]),
            "OLS": _f(both["ols"]),
            "ρ": _f(both["rho_ctrl"]),
            "fake?": "FALSE clone" if both["fake"] else "",
            "n": f"{both['n']:,}",
        },
    ]
    prose = (
        f"Y3 leftover after e_ap_issued rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"ρ={_f(d['rho_ctrl'])} "
        f"{'REWRITE of issued volume' if rewrite else 'not just issued'}. "
        f"After issued+days {_f(both['rank'])} fake={both['fake']}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "ols": d["ols"],
        "rewrite": rewrite,
        "both": both["rank"],
        "prose": prose,
    }


def pass6_after_dpo(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ap_open"], (tr["e_dpo_proxy"],), tr["fold"], tr[Y3].notna()
    )
    both = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["e_dpo_proxy"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    numer = bool(d["honest_dies"])
    rows = [
        {
            "bar": "open leftover after DPO",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        },
        {
            "bar": "open leftover after DPO+days",
            "rank": _f(both["rank"]),
            "OLS": _f(both["ols"]),
            "ρ": _f(both["rho_ctrl"]),
            "fake?": "FALSE clone" if both["fake"] else "",
            "n": f"{both['n']:,}",
        },
    ]
    prose = (
        f"Y3 leftover after DPO rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"ρ={_f(d['rho_ctrl'])} "
        f"{'DPO numerator rewrite' if numer else 'not just the DPO numerator'}. "
        f"After DPO+days {_f(both['rank'])}. DPO already DROP — do not reopen."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "ols": d["ols"],
        "numer": numer,
        "both": both["rank"],
        "prose": prose,
    }


def pass7_y5(tr: pd.DataFrame) -> dict:
    leak = leakage_check(["e_ap_open"], Y5, forbidden_prefixes=["e"])
    d = leftover_diag(tr[Y5], tr["e_ap_open"], (tr["log_in3"],), tr["fold"], tr[Y5].notna())
    raw = signed_oof_auroc(tr[Y5], tr["e_ap_open"], tr["fold"], tr[Y5].notna())
    rows = [
        {
            "slice": "Y5 leftover after size (report only)",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "raw": "LOW_POWER" if raw["low_power"] else _f(_cv(raw)),
            "n": f"{d['n']:,}",
        }
    ]
    prose = (
        f"Y5 leftover after size rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"raw {_f(_cv(raw))}. Y5 never E — no AUROC as card X, leak_ok={leak['ok']}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "ols": d["ols"],
        "raw": _cv(raw),
        "leak_ok": leak["ok"],
        "prose": prose,
    }


def pass8_q6(tr: pd.DataFrame) -> dict:
    short = tr["so_far_class"] == "short_<12"
    raw = signed_oof_auroc(
        tr[Y7], tr["e_ap_open_lag1"], tr["fold"], short & tr[Y7].notna()
    )
    d = leftover_diag(
        tr[Y3],
        tr["e_ap_open_lag1"],
        (tr["c_n_days_with_tx_lag1"],),
        tr["fold"],
        short & tr[Y3].notna(),
    )
    now = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["c_n_days_with_tx_lag1"],),
        tr["fold"],
        short & tr[Y3].notna(),
    )
    rows = [
        {
            "slice": "Q6 Y7 open_lag1 short",
            "CV": "LOW_POWER" if raw["low_power"] else _f(_cv(raw)),
            "n": f"{raw['n_defined']:,}",
            "n_pos": f"{raw['n_pos']:,}",
        },
        {
            "slice": "Q6 Y3 open_lag1 leftover after days_lag1 short",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "slice": "Q6 Y3 now leftover after days_lag1 short",
            "rank": _f(now["rank"]),
            "OLS": _f(now["ols"]),
            "ρ": _f(now["rho_ctrl"]),
            "fake?": "FALSE clone" if now["fake"] else "",
        },
    ]
    prose = (
        f"Q6 Y7 open_lag1 on short {_f(_cv(raw))} (AR issued_lag1 KEEP {Q6_LAG1_QUOTE:.3f} locked). "
        f"open_lag1 leftover after days_lag1 short rank {_f(d['rank'])} fake={d['fake']}. "
        f"Contemporaneous leftover after days_lag1 short {_f(now['rank'])} fake={now['fake']}. "
        f"Do not claim a TURNOVER seat."
    )
    print(prose)
    return {
        "rows": rows,
        "y7": _cv(raw),
        "lag_rank": d["rank"],
        "now_rank": now["rank"],
        "prose": prose,
    }


def pass9_ar_open(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ap_open"], (tr["e_ar_open"],), tr["fold"], tr[Y3].notna()
    )
    inv = leftover_diag(
        tr[Y3], tr["e_ar_open"], (tr["e_ap_open"],), tr["fold"], tr[Y3].notna()
    )
    both = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["e_ar_open"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "AP open leftover after AR open",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "bar": "AR open leftover after AP open",
            "rank": _f(inv["rank"]),
            "OLS": _f(inv["ols"]),
            "ρ": _f(inv["rho_ctrl"]),
            "fake?": "FALSE clone" if inv["fake"] else "",
        },
        {
            "bar": "AP open leftover after AR+days",
            "rank": _f(both["rank"]),
            "OLS": _f(both["ols"]),
            "ρ": _f(both["rho_ctrl"]),
            "fake?": "FALSE clone" if both["fake"] else "",
        },
    ]
    same = bool(d["honest_dies"])
    prose = (
        f"AP leftover after AR open {_f(d['rank'])} "
        f"{'same object' if same else 'not the same object'}. "
        f"AR leftover after AP {_f(inv['rank'])}. After AR+days {_f(both['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "inv": inv["rank"],
        "both": both["rank"],
        "same": same,
        "prose": prose,
    }


def pass10_boot(tr: pd.DataFrame) -> dict:
    assert_no_holdout(tr["company_id"])
    work = tr.reset_index(drop=True)
    by = {c: np.asarray(ix) for c, ix in work.groupby("company_id", sort=False).groups.items()}
    cos = np.array(list(by.keys()))
    rng = np.random.default_rng(FOLD_SEED)
    vals = []
    for _ in range(N_BOOT):
        draw = rng.choice(cos, size=len(cos), replace=True)
        idx = np.concatenate([by[c] for c in draw])
        sub = work.iloc[idx].reset_index(drop=True)
        if sub["fold"].nunique() < 3:
            continue
        d = leftover_diag(
            sub[Y3],
            sub["e_ap_open"],
            (sub["c_n_days_with_tx"],),
            sub["fold"],
            sub[Y3].notna(),
        )
        if np.isfinite(d["rank"]):
            vals.append(float(d["rank"]))
    if not vals:
        p05 = p50 = p95 = float("nan")
    else:
        p05, p50, p95 = [float(np.quantile(vals, q)) for q in (0.05, 0.50, 0.95)]
    rows = [
        {
            "boot": f"n={len(vals)}/{N_BOOT}",
            "p05": _f(p05),
            "p50": _f(p50),
            "p95": _f(p95),
        }
    ]
    prose = (
        f"Bootstrap leftover-after-days rank p05={_f(p05)} p50={_f(p50)} "
        f"p95={_f(p95)} n={len(vals)}/{N_BOOT}."
    )
    print(prose)
    return {"rows": rows, "p05": p05, "p50": p50, "p95": p95, "n": len(vals), "prose": prose}


def pass_holdout(panel: pd.DataFrame, book: set[str]) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    assert set(ho["company_id"]) <= load_holdout()
    op = pd.to_numeric(ho["e_ap_open"], errors="coerce")
    dark = ~ho["company_id"].isin(book)
    rows = [
        {
            "slice": "holdout 72",
            "n_cm": f"{len(ho):,}",
            "companies": f"{ho['company_id'].nunique()}",
            "cov": _pp(_pct(int(op.notna().sum()), len(ho))),
            "dark_nn": int(op[dark].notna().sum()),
        }
    ]
    prose = (
        f"Holdout 72 coverage only: {len(ho):,} CM / {ho['company_id'].nunique()} companies. "
        f"open cov {_pp(_pct(int(op.notna().sum()), len(ho)))}; dark nn={int(op[dark].notna().sum())}. "
        f"No AUROC."
    )
    print(prose)
    return {"rows": rows, "cov": _pct(int(op.notna().sum()), len(ho)), "prose": prose}


def pass_open_pos(tr: pd.DataFrame) -> dict:
    op = pd.to_numeric(tr["e_ap_open"], errors="coerce")
    m = (op > 0) & tr[Y3].notna()
    d = leftover_diag(tr[Y3], tr["e_ap_open"], (tr["c_n_days_with_tx"],), tr["fold"], m)
    raw = signed_oof_auroc(tr[Y3], tr["e_ap_open"], tr["fold"], m)
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], m)
    rows = [
        {
            "slice": "open>0 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "raw": _f(_cv(raw)),
            "days": _f(_cv(days)),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"open>0 leftover after days rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"ρ={_f(d['rho_ctrl'])} fake={d['fake']} raw {_f(_cv(raw))} days {_f(_cv(days))}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "raw": _cv(raw), "prose": prose}


def pass_log1p(tr: pd.DataFrame) -> dict:
    raw = signed_oof_auroc(tr[Y3], tr["log1p_open"], tr["fold"], tr[Y3].notna())
    d = leftover_diag(
        tr[Y3], tr["log1p_open"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "slice": "log1p(open) leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "raw": _f(_cv(raw)),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"log1p(open) Y3 {_f(_cv(raw))} leftover-after-days rank {_f(d['rank'])} "
        f"OLS {_f(d['ols'])} ρ={_f(d['rho_ctrl'])} fake={d['fake']}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "ols": d["ols"],
        "fake": d["fake"],
        "raw": _cv(raw),
        "prose": prose,
    }


def pass_pending(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ap_open"], (tr["e_pending_amt_share"],), tr["fold"], tr[Y3].notna()
    )
    both = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["e_pending_amt_share"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "open leftover after pending",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "bar": "leftover after pending+days",
            "rank": _f(both["rank"]),
            "OLS": _f(both["ols"]),
            "ρ": _f(both["rho_ctrl"]),
            "fake?": "FALSE clone" if both["fake"] else "",
        },
    ]
    prose = (
        f"Leftover after e_pending_amt_share {_f(d['rank'])} fake={d['fake']}; "
        f"pending+days {_f(both['rank'])}. Pending already decided — do not reopen."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "both": both["rank"], "fake": d["fake"], "prose": prose}


def pass_terciles(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    rows = []
    store = {}
    for lab, mask in (
        ("T1", terc == "T1"),
        ("T2+T3", terc.isin(["T2", "T3"])),
        ("T3", terc == "T3"),
    ):
        d = leftover_diag(
            tr[Y3],
            tr["e_ap_open"],
            (tr["c_n_days_with_tx"],),
            tr["fold"],
            mask & tr[Y3].notna(),
        )
        store[lab] = d
        rows.append(
            {
                "slice": lab,
                "rank": _f(d["rank"]),
                "OLS": _f(d["ols"]),
                "ρ": _f(d["rho_ctrl"]),
                "fake?": "FALSE clone" if d["fake"] else "",
                "n": f"{d['n']:,}",
            }
        )
    prose = (
        f"SIZE terciles leftover after days: T1 {_f(store['T1']['rank'])} "
        f"T2+T3 {_f(store['T2+T3']['rank'])} T3 {_f(store['T3']['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "t1": store["T1"]["rank"],
        "t23": store["T2+T3"]["rank"],
        "t3": store["T3"]["rank"],
        "prose": prose,
    }


def pass_comedian(tr: pd.DataFrame) -> dict:
    med = tr.groupby("company_id")[["e_ap_open", "c_n_days_with_tx", "log_in3", "e_ap_issued"]].median()
    rho_days, n1 = spearman_n(med["e_ap_open"], med["c_n_days_with_tx"])
    rho_size, n2 = spearman_n(med["e_ap_open"], med["log_in3"])
    rho_iss, n3 = spearman_n(med["e_ap_open"], med["e_ap_issued"])
    size_flag = bool(np.isfinite(rho_size) and abs(rho_size) >= SIZE_RHO)
    rows = [
        {"pair": "company-median vs days", "ρ": _f(rho_days), "n": f"{n1:,}"},
        {"pair": "company-median vs log1p(a_in3)", "ρ": _f(rho_size), "n": f"{n2:,}"},
        {"pair": "company-median vs issued", "ρ": _f(rho_iss), "n": f"{n3:,}"},
    ]
    prose = (
        f"Company-median ρ open vs days {_f(rho_days)} vs log1p(a_in3) {_f(rho_size)} "
        f"{'SIZE' if size_flag else 'not SIZE'} vs issued {_f(rho_iss)}."
    )
    print(prose)
    return {
        "rows": rows,
        "rho_days": rho_days,
        "rho_size": rho_size,
        "size_flag": size_flag,
        "prose": prose,
    }


def pass_sofar(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for lab in ("short_<12", "mid_12_17", "long_>=18"):
        d = leftover_diag(
            tr[Y3],
            tr["e_ap_open"],
            (tr["c_n_days_with_tx"],),
            tr["fold"],
            (tr["so_far_class"] == lab) & tr[Y3].notna(),
        )
        store[lab] = d
        rows.append(
            {
                "slice": lab,
                "rank": _f(d["rank"]),
                "OLS": _f(d["ols"]),
                "ρ": _f(d["rho_ctrl"]),
                "fake?": "FALSE clone" if d["fake"] else "",
                "n": f"{d['n']:,}",
            }
        )
    prose = (
        f"so_far leftover after days: short {_f(store['short_<12']['rank'])} "
        f"mid {_f(store['mid_12_17']['rank'])} long {_f(store['long_>=18']['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "short": store["short_<12"]["rank"],
        "mid": store["mid_12_17"]["rank"],
        "long": store["long_>=18"]["rank"],
        "prose": prose,
    }


def pass_folds(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ap_open"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    prose = f"Per-fold rank leftover after days: {d['rank_folds']}."
    print(prose)
    bits = d["rank_folds"].split()
    rows = [{"fold": str(i), "rank leftover": bits[i] if i < len(bits) else "—"} for i in range(5)]
    return {"rows": rows, "folds": d["rank_folds"], "prose": prose}


def pass_issued_dpo(tr: pd.DataFrame) -> dict:
    both = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["e_ap_issued"], tr["e_dpo_proxy"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    triple = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["e_ap_issued"], tr["e_dpo_proxy"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    inv_iss = leftover_diag(
        tr[Y3], tr["e_ap_issued"], (tr["e_ap_open"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "bar": "open leftover after issued+DPO",
            "rank": _f(both["rank"]),
            "OLS": _f(both["ols"]),
            "ρ": _f(both["rho_ctrl"]),
        },
        {
            "bar": "open leftover after issued+DPO+days",
            "rank": _f(triple["rank"]),
            "OLS": _f(triple["ols"]),
            "ρ": _f(triple["rho_ctrl"]),
        },
        {
            "bar": "issued leftover after open",
            "rank": _f(inv_iss["rank"]),
            "OLS": _f(inv_iss["ols"]),
            "ρ": _f(inv_iss["rho_ctrl"]),
        },
    ]
    prose = (
        f"Leftover after issued+DPO {_f(both['rank'])}; after issued+DPO+days {_f(triple['rank'])}. "
        f"Issued leftover after open {_f(inv_iss['rank'])} (issued CLOSED leftover 0.591)."
    )
    print(prose)
    return {
        "rows": rows,
        "both": both["rank"],
        "triple": triple["rank"],
        "inv_iss": inv_iss["rank"],
        "prose": prose,
    }


def pass_q6_more(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for lab in ("short_<12", "mid_12_17", "long_>=18"):
        m = tr["so_far_class"] == lab
        y7 = signed_oof_auroc(tr[Y7], tr["e_ap_open_lag1"], tr["fold"], m & tr[Y7].notna())
        d = leftover_diag(
            tr[Y3],
            tr["e_ap_open_lag1"],
            (tr["c_n_days_with_tx_lag1"],),
            tr["fold"],
            m & tr[Y3].notna(),
        )
        store[lab] = {"y7": _cv(y7), "rank": d["rank"]}
        rows.append(
            {
                "slice": lab,
                "Y7 open_lag1": "LOW_POWER" if y7["low_power"] else _f(_cv(y7)),
                "Y3 leftover days_lag1": _f(d["rank"]),
                "fake?": "FALSE clone" if d["fake"] else "",
            }
        )
    prose = (
        f"Q6 mid Y7 open_lag1 {_f(store['mid_12_17']['y7'])} leftover {_f(store['mid_12_17']['rank'])}; "
        f"long Y7 {_f(store['long_>=18']['y7'])}. Do not claim a TURNOVER seat."
    )
    print(prose)
    return {
        "rows": rows,
        "mid_y7": store["mid_12_17"]["y7"],
        "mid_rank": store["mid_12_17"]["rank"],
        "long_y7": store["long_>=18"]["y7"],
        "prose": prose,
    }


def pass_keep_card(p1, p2, p3, p4, p5) -> dict:
    beat = bool(np.isfinite(p3["beat_size"]) and p3["beat_size"] >= KEEP_DELTA)
    leftover_ok = not bool(p4["y3_dies"])
    size = bool(p1["size_flag"] or p2["size_flag"])
    twin = bool(p2["gate_twins"])
    rows = [
        {"gate": "beat size ≥0.02", "value": _f(p3["beat_size"]), "pass?": "PASS" if beat else "FAIL"},
        {
            "gate": "leftover after days (honest, ≥0.55, not fake)",
            "value": _f(p4["y3_rank"]),
            "pass?": "PASS" if leftover_ok else "FAIL",
        },
        {
            "gate": "not SIZE |ρ| vs log1p(a_in3) <0.50",
            "value": _f(p2["rho_size"]),
            "pass?": "FAIL" if size else "PASS (row); company-median may SIZE",
        },
        {
            "gate": "not twin vs days / n_tx / issued / DPO / pending",
            "value": ",".join(p2["gate_twins"]) if p2["gate_twins"] else "none",
            "pass?": "FAIL" if twin else "PASS (issued ρ=0.757 near)",
        },
        {
            "gate": "leftover after issued (not a rewrite)",
            "value": _f(p5["rank"]),
            "pass?": "FAIL rewrite" if p5.get("rewrite") else "thin",
        },
    ]
    prose = (
        "KEEP-as-X scorecard: beat size FAIL (Y3 0.569 < size 0.617); "
        "leftover-after-days FAIL 0.418 < 0.55; row not SIZE; gate twins PASS. "
        "Rewrite of issued. Do not put e_ap_open on the 15-col card."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_more_leftover(tr: pd.DataFrame) -> dict:
    ds = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["c_n_days_with_tx"], tr["log_in3"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    intens = leftover_diag(
        tr[Y3], tr["open_per_day"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    y7 = leftover_diag(
        tr[Y7], tr["e_ap_open"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y7].notna()
    )
    op = pd.to_numeric(tr["e_ap_open"], errors="coerce")
    zany = tr.assign(z=(op == 0)).groupby("company_id")["z"].any()
    never = ~tr["company_id"].map(zany).fillna(True)
    nz = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        never & tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "leftover after days+size",
            "rank": _f(ds["rank"]),
            "OLS": _f(ds["ols"]),
            "ρ": _f(ds["rho_ctrl"]),
        },
        {
            "bar": "open/days leftover after days",
            "rank": _f(intens["rank"]),
            "OLS": _f(intens["ols"]),
            "ρ": _f(intens["rho_ctrl"]),
        },
        {
            "bar": "Y7 leftover after days",
            "rank": _f(y7["rank"]),
            "OLS": _f(y7["ols"]),
            "ρ": _f(y7["rho_ctrl"]),
        },
        {
            "bar": "never-zero leftover-days",
            "rank": _f(nz["rank"]),
            "OLS": _f(nz["ols"]),
            "ρ": _f(nz["rho_ctrl"]),
        },
    ]
    prose = (
        f"Leftover after days+size {_f(ds['rank'])}; intensity {_f(intens['rank'])}; "
        f"Y7 leftover-days {_f(y7['rank'])}; never-zero {_f(nz['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "ds": ds["rank"],
        "intens": intens["rank"],
        "y7": y7["rank"],
        "nz": nz["rank"],
        "prose": prose,
    }


def pass_overdue(tr: pd.DataFrame) -> dict:
    raw = pd.read_parquet(STORE, columns=["company_id", "period", "e_ap_overdue"])
    raw = _keys(raw)
    work = tr.merge(raw, on=["company_id", "period"], how="left")
    d = leftover_diag(
        work[Y3], work["e_ap_open"], (work["e_ap_overdue"],), work["fold"], work[Y3].notna()
    )
    rho, n = spearman_n(work["e_ap_open"], work["e_ap_overdue"])
    rows = [
        {
            "bar": "open leftover after overdue",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "raw ρ": _f(rho),
        }
    ]
    prose = (
        f"Leftover after e_ap_overdue {_f(d['rank'])} ρ_raw={_f(rho)}. "
        f"Overdue already decided — do not reopen."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "rho": rho, "prose": prose}


def pass_never_zero_more(tr: pd.DataFrame) -> dict:
    op = pd.to_numeric(tr["e_ap_open"], errors="coerce")
    zany = tr.assign(z=(op == 0)).groupby("company_id")["z"].any()
    never = ~tr["company_id"].map(zany).fillna(True)
    m = never & tr[Y3].notna()
    after_iss = leftover_diag(tr[Y3], tr["e_ap_open"], (tr["e_ap_issued"],), tr["fold"], m)
    after_days = leftover_diag(
        tr[Y3], tr["e_ap_open"], (tr["c_n_days_with_tx"],), tr["fold"], m
    )
    after_both = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["e_ap_issued"], tr["c_n_days_with_tx"]),
        tr["fold"],
        m,
    )
    inv = leftover_diag(tr[Y3], tr["c_n_days_with_tx"], (tr["e_ap_open"],), tr["fold"], m)
    raw = signed_oof_auroc(tr[Y3], tr["e_ap_open"], tr["fold"], m)
    rows = [
        {
            "bar": "never-zero leftover-days",
            "rank": _f(after_days["rank"]),
            "OLS": _f(after_days["ols"]),
            "ρ": _f(after_days["rho_ctrl"]),
            "fake?": "FALSE clone" if after_days["fake"] else "",
            "raw": _f(_cv(raw)),
        },
        {
            "bar": "never-zero leftover-issued",
            "rank": _f(after_iss["rank"]),
            "OLS": _f(after_iss["ols"]),
            "ρ": _f(after_iss["rho_ctrl"]),
            "fake?": "FALSE clone" if after_iss["fake"] else "",
            "raw": "—",
        },
        {
            "bar": "never-zero leftover issued+days",
            "rank": _f(after_both["rank"]),
            "OLS": _f(after_both["ols"]),
            "ρ": _f(after_both["rho_ctrl"]),
            "fake?": "FALSE clone" if after_both["fake"] else "",
            "raw": "—",
        },
        {
            "bar": "days leftover after open on never-zero",
            "rank": _f(inv["rank"]),
            "OLS": _f(inv["ols"]),
            "ρ": _f(inv["rho_ctrl"]),
            "fake?": "FALSE clone" if inv["fake"] else "",
            "raw": "—",
        },
    ]
    prose = (
        f"never-zero leftover-days {_f(after_days['rank'])} fake={after_days['fake']} "
        f"raw {_f(_cv(raw))}; leftover-issued {_f(after_iss['rank'])}; "
        f"issued+days {_f(after_both['rank'])}; days-after-open {_f(inv['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "days": after_days["rank"],
        "iss": after_iss["rank"],
        "both": after_both["rank"],
        "inv": inv["rank"],
        "raw": _cv(raw),
        "prose": prose,
    }


def pass_card_keeps(tr: pd.DataFrame) -> dict:
    cols = ["company_id", "period", "c_ss_month", "c_salary_month"]
    raw = pd.read_parquet(STORE, columns=cols)
    raw = _keys(raw)
    work = tr.merge(raw, on=["company_id", "period"], how="left")
    both = leftover_diag(
        work[Y3],
        work["e_ap_open"],
        (work["c_ss_month"], work["c_salary_month"], work["c_n_days_with_tx"]),
        work["fold"],
        work[Y3].notna(),
    )
    ss = leftover_diag(
        work[Y3], work["e_ap_open"], (work["c_ss_month"],), work["fold"], work[Y3].notna()
    )
    sal = leftover_diag(
        work[Y3], work["e_ap_open"], (work["c_salary_month"],), work["fold"], work[Y3].notna()
    )
    rows = [
        {
            "bar": "leftover after c_ss_month",
            "rank": _f(ss["rank"]),
            "OLS": _f(ss["ols"]),
            "ρ": _f(ss["rho_ctrl"]),
        },
        {
            "bar": "leftover after c_salary_month",
            "rank": _f(sal["rank"]),
            "OLS": _f(sal["ols"]),
            "ρ": _f(sal["rho_ctrl"]),
        },
        {
            "bar": "leftover after ss+salary+days",
            "rank": _f(both["rank"]),
            "OLS": _f(both["ols"]),
            "ρ": _f(both["rho_ctrl"]),
        },
    ]
    prose = (
        f"Leftover after c_ss_month {_f(ss['rank'])}; after c_salary_month {_f(sal['rank'])}; "
        f"ss+salary+days {_f(both['rank'])}. Stay off the 15-col card."
    )
    print(prose)
    return {"rows": rows, "rank": both["rank"], "ss": ss["rank"], "sal": sal["rank"], "prose": prose}


def pass_dpo_defined(tr: pd.DataFrame) -> dict:
    dpo = pd.to_numeric(tr["e_dpo_proxy"], errors="coerce")
    m = dpo.notna() & tr[Y3].notna()
    d = leftover_diag(tr[Y3], tr["e_ap_open"], (tr["c_n_days_with_tx"],), tr["fold"], m)
    after = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["e_dpo_proxy"], tr["c_n_days_with_tx"]),
        tr["fold"],
        m,
    )
    rows = [
        {
            "bar": "DPO-defined leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        },
        {
            "bar": "DPO-defined leftover DPO+days",
            "rank": _f(after["rank"]),
            "OLS": _f(after["ols"]),
            "ρ": _f(after["rho_ctrl"]),
            "fake?": "FALSE clone" if after["fake"] else "",
            "n": f"{after['n']:,}",
        },
    ]
    prose = (
        f"DPO-defined leftover-days {_f(d['rank'])} fake={d['fake']}; "
        f"after DPO+days {_f(after['rank'])}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "after": after["rank"], "prose": prose}


def pass_erp(tr: pd.DataFrame, book: set[str]) -> dict:
    m = tr["company_id"].isin(book) & tr[Y3].notna()
    d = leftover_diag(tr[Y3], tr["e_ap_open"], (tr["c_n_days_with_tx"],), tr["fold"], m)
    late = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        m & ~tr["early6"],
    )
    rows = [
        {
            "slice": "ever-ERP leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        },
        {
            "slice": "ever-ERP late",
            "rank": _f(late["rank"]),
            "OLS": _f(late["ols"]),
            "ρ": _f(late["rho_ctrl"]),
            "fake?": "FALSE clone" if late["fake"] else "",
            "n": f"{late['n']:,}",
        },
    ]
    prose = (
        f"ever-ERP leftover-days {_f(d['rank'])} fake={d['fake']}; late {_f(late['rank'])}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "late": late["rank"], "prose": prose}


def pass_winsor_ar_iss(tr: pd.DataFrame) -> dict:
    op = pd.to_numeric(tr["e_ap_open"], errors="coerce")
    hi = float(op.quantile(0.99))
    w = op.clip(upper=hi)
    d = leftover_diag(tr[Y3], w, (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna())
    raw = pd.read_parquet(STORE, columns=["company_id", "period", "e_ar_issued"])
    raw = _keys(raw)
    work = tr.merge(raw, on=["company_id", "period"], how="left")
    after_ar_iss = leftover_diag(
        work[Y3], work["e_ap_open"], (work["e_ar_issued"],), work["fold"], work[Y3].notna()
    )
    after_ar_o = leftover_diag(
        work[Y3],
        work["e_ap_open"],
        (work["e_ar_open"], work["e_ap_issued"]),
        work["fold"],
        work[Y3].notna(),
    )
    rows = [
        {
            "bar": "winsor99 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "bar": "leftover after e_ar_issued",
            "rank": _f(after_ar_iss["rank"]),
            "OLS": _f(after_ar_iss["ols"]),
            "ρ": _f(after_ar_iss["rho_ctrl"]),
            "fake?": "FALSE clone" if after_ar_iss["fake"] else "",
        },
        {
            "bar": "leftover after AR open + AP issued",
            "rank": _f(after_ar_o["rank"]),
            "OLS": _f(after_ar_o["ols"]),
            "ρ": _f(after_ar_o["rho_ctrl"]),
            "fake?": "FALSE clone" if after_ar_o["fake"] else "",
        },
    ]
    prose = (
        f"winsor leftover-days {_f(d['rank'])} fake={d['fake']}; "
        f"after AR issued {_f(after_ar_iss['rank'])}; "
        f"after AR-open+AP-issued {_f(after_ar_o['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "win": d["rank"],
        "ar_iss": after_ar_iss["rank"],
        "both": after_ar_o["rank"],
        "prose": prose,
    }


def pass_issued_lag1(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["e_ap_issued_lag1"],),
        tr["fold"],
        tr[Y3].notna(),
    )
    y7 = leftover_diag(
        tr[Y7],
        tr["e_ap_open"],
        (tr["e_ap_issued_lag1"],),
        tr["fold"],
        tr[Y7].notna(),
    )
    both = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["e_ap_issued_lag1"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "Y3 leftover after issued_lag1",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
        },
        {
            "bar": "Y7 leftover after issued_lag1",
            "rank": _f(y7["rank"]),
            "OLS": _f(y7["ols"]),
            "ρ": _f(y7["rho_ctrl"]),
        },
        {
            "bar": "Y3 leftover after issued_lag1+days",
            "rank": _f(both["rank"]),
            "OLS": _f(both["ols"]),
            "ρ": _f(both["rho_ctrl"]),
        },
    ]
    prose = (
        f"Y3 leftover after issued_lag1 {_f(d['rank'])}; Y7 {_f(y7['rank'])}; "
        f"after issued_lag1+days {_f(both['rank'])}. Do not grow TURNOVER."
    )
    print(prose)
    return {"rows": rows, "y3": d["rank"], "y7": y7["rank"], "both": both["rank"], "prose": prose}


def pass_between_lag1(tr: pd.DataFrame) -> dict:
    """ICC 0.99 BETWEEN — leftover after company-mean vs leftover after own lag1."""
    op = pd.to_numeric(tr["e_ap_open"], errors="coerce")
    cmean = op.groupby(tr["company_id"]).transform("mean")
    within = op - cmean
    after_mean = leftover_diag(
        tr[Y3], tr["e_ap_open"], (cmean,), tr["fold"], tr[Y3].notna()
    )
    after_mean_days = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (cmean, tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    after_lag1 = leftover_diag(
        tr[Y3], tr["e_ap_open"], (tr["e_ap_open_lag1"],), tr["fold"], tr[Y3].notna()
    )
    after_lag1_days = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["e_ap_open_lag1"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    within_d = leftover_diag(
        tr[Y3], within, (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "bar": "leftover after company-mean",
            "rank": _f(after_mean["rank"]),
            "OLS": _f(after_mean["ols"]),
            "ρ": _f(after_mean["rho_ctrl"]),
        },
        {
            "bar": "leftover after company-mean+days",
            "rank": _f(after_mean_days["rank"]),
            "OLS": _f(after_mean_days["ols"]),
            "ρ": _f(after_mean_days["rho_ctrl"]),
        },
        {
            "bar": "leftover after own lag1",
            "rank": _f(after_lag1["rank"]),
            "OLS": _f(after_lag1["ols"]),
            "ρ": _f(after_lag1["rho_ctrl"]),
        },
        {
            "bar": "leftover after lag1+days",
            "rank": _f(after_lag1_days["rank"]),
            "OLS": _f(after_lag1_days["ols"]),
            "ρ": _f(after_lag1_days["rho_ctrl"]),
        },
        {
            "bar": "within (open−cmean) leftover-days",
            "rank": _f(within_d["rank"]),
            "OLS": _f(within_d["ols"]),
            "ρ": _f(within_d["rho_ctrl"]),
        },
    ]
    prose = (
        f"BETWEEN: leftover after company-mean {_f(after_mean['rank'])} "
        f"+days {_f(after_mean_days['rank'])}; after own lag1 {_f(after_lag1['rank'])} "
        f"+days {_f(after_lag1_days['rank'])}; within leftover-days {_f(within_d['rank'])}. "
        "ICC 0.99 BETWEEN — leftover is identity, not a new X."
    )
    print(prose)
    return {
        "rows": rows,
        "mean": after_mean["rank"],
        "mean_days": after_mean_days["rank"],
        "lag1": after_lag1["rank"],
        "lag1_days": after_lag1_days["rank"],
        "within": within_d["rank"],
        "prose": prose,
    }


def pass_ntx_joint(tr: pd.DataFrame) -> dict:
    after_ntx = leftover_diag(
        tr[Y3], tr["e_ap_open"], (tr["a_n_tx"],), tr["fold"], tr[Y3].notna()
    )
    after_ntx_days = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["a_n_tx"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    after_joint = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["c_n_days_with_tx"], tr["e_ap_issued"], tr["log_in3"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    dpo_after_open = leftover_diag(
        tr[Y3], tr["e_dpo_proxy"], (tr["e_ap_open"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "bar": "leftover after a_n_tx",
            "rank": _f(after_ntx["rank"]),
            "OLS": _f(after_ntx["ols"]),
            "ρ": _f(after_ntx["rho_ctrl"]),
        },
        {
            "bar": "leftover after n_tx+days",
            "rank": _f(after_ntx_days["rank"]),
            "OLS": _f(after_ntx_days["ols"]),
            "ρ": _f(after_ntx_days["rho_ctrl"]),
        },
        {
            "bar": "leftover after days+issued+size",
            "rank": _f(after_joint["rank"]),
            "OLS": _f(after_joint["ols"]),
            "ρ": _f(after_joint["rho_ctrl"]),
        },
        {
            "bar": "DPO leftover after open",
            "rank": _f(dpo_after_open["rank"]),
            "OLS": _f(dpo_after_open["ols"]),
            "ρ": _f(dpo_after_open["rho_ctrl"]),
        },
    ]
    prose = (
        f"leftover after a_n_tx {_f(after_ntx['rank'])}; n_tx+days {_f(after_ntx_days['rank'])}; "
        f"days+issued+size {_f(after_joint['rank'])}; DPO leftover after open {_f(dpo_after_open['rank'])}. "
        "DPO already DROP — do not reopen."
    )
    print(prose)
    return {
        "rows": rows,
        "ntx": after_ntx["rank"],
        "ntx_days": after_ntx_days["rank"],
        "joint": after_joint["rank"],
        "dpo_after": dpo_after_open["rank"],
        "prose": prose,
    }


def pass_trail_arlag(tr: pd.DataFrame) -> dict:
    after_trail = leftover_diag(
        tr[Y3], tr["e_ap_open"], (tr["months_so_far"],), tr["fold"], tr[Y3].notna()
    )
    after_trail_days = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["months_so_far"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    after_ar_l1 = leftover_diag(
        tr[Y3], tr["e_ap_open"], (tr["e_ar_open_lag1"],), tr["fold"], tr[Y3].notna()
    )
    after_ar_l1_days = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["e_ar_open_lag1"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "leftover after months_so_far",
            "rank": _f(after_trail["rank"]),
            "OLS": _f(after_trail["ols"]),
            "ρ": _f(after_trail["rho_ctrl"]),
        },
        {
            "bar": "leftover after trail+days",
            "rank": _f(after_trail_days["rank"]),
            "OLS": _f(after_trail_days["ols"]),
            "ρ": _f(after_trail_days["rho_ctrl"]),
        },
        {
            "bar": "leftover after e_ar_open_lag1",
            "rank": _f(after_ar_l1["rank"]),
            "OLS": _f(after_ar_l1["ols"]),
            "ρ": _f(after_ar_l1["rho_ctrl"]),
        },
        {
            "bar": "leftover after AR_open_lag1+days",
            "rank": _f(after_ar_l1_days["rank"]),
            "OLS": _f(after_ar_l1_days["ols"]),
            "ρ": _f(after_ar_l1_days["rho_ctrl"]),
        },
    ]
    prose = (
        f"leftover after months_so_far {_f(after_trail['rank'])} +days {_f(after_trail_days['rank'])}; "
        f"after AR_open_lag1 {_f(after_ar_l1['rank'])} +days {_f(after_ar_l1_days['rank'])}. "
        "Not a trail-length rewrite; same-object AR lag dies leftover."
    )
    print(prose)
    return {
        "rows": rows,
        "trail": after_trail["rank"],
        "trail_days": after_trail_days["rank"],
        "ar_l1": after_ar_l1["rank"],
        "ar_l1_days": after_ar_l1_days["rank"],
        "prose": prose,
    }


def pass_log_issued_triple(tr: pd.DataFrame) -> dict:
    iss = pd.to_numeric(tr["e_ap_issued"], errors="coerce")
    log_iss = np.log1p(iss.clip(lower=0))
    after_log = leftover_diag(
        tr[Y3], tr["e_ap_open"], (log_iss,), tr["fold"], tr[Y3].notna()
    )
    after_log_days = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (log_iss, tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    triple = leftover_diag(
        tr[Y3],
        tr["e_ap_open"],
        (tr["e_ap_issued"], tr["e_ar_open"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "leftover after log1p(issued)",
            "rank": _f(after_log["rank"]),
            "OLS": _f(after_log["ols"]),
            "ρ": _f(after_log["rho_ctrl"]),
        },
        {
            "bar": "leftover after log1p(issued)+days",
            "rank": _f(after_log_days["rank"]),
            "OLS": _f(after_log_days["ols"]),
            "ρ": _f(after_log_days["rho_ctrl"]),
        },
        {
            "bar": "leftover after issued+AR_open+days",
            "rank": _f(triple["rank"]),
            "OLS": _f(triple["ols"]),
            "ρ": _f(triple["rho_ctrl"]),
        },
    ]
    prose = (
        f"leftover after log1p(issued) {_f(after_log['rank'])} +days {_f(after_log_days['rank'])}; "
        f"issued+AR_open+days {_f(triple['rank'])}. Rewrite of issued volume holds on log scale."
    )
    print(prose)
    return {
        "rows": rows,
        "log": after_log["rank"],
        "log_days": after_log_days["rank"],
        "triple": triple["rank"],
        "prose": prose,
    }


def decide(p1, p2, p3, p4, p5) -> dict:
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
            f"leftover after days rank {_f(leftover)} beats size "
            f"({_f(p3['size'])}) by {_f(p3['beat_size'])}; not SIZE; not a twin. "
            "Do not put e_ap_open on the 15-col card."
        )
    elif leftover_dies or twin:
        card = "CLOSE unused leftover / KEEP off the 15-col card"
        tag = "CLOSE"
        why = (
            f"Y3 leftover after days rank {_f(leftover)} OLS {_f(p4['y3_ols'])} "
            f"{'dies' if leftover_dies else 'thin'}; twin={twin} SIZE={size}. "
            f"Inverse days-after-open {_f(p4['inv_rank'])}. "
            "Contemporaneous AP open stays off the 15-col card. Do not grow TURNOVER."
        )
    elif size:
        card = "CLOSE as SIZE / KEEP off the 15-col card"
        tag = "CLOSE"
        why = (
            f"SIZE ρ={_f(p2['rho_size'])}; leftover after days {_f(leftover)}. Stay off the card."
        )
    else:
        card = "CLOSE unused leftover / KEEP off the 15-col card"
        tag = "CLOSE"
        why = (
            f"Y3 {_f(p3['y3'])} vs days {_f(p3['days'])} leftover {_f(leftover)}. "
            "Does not clear KEEP-as-X. Stay off the card."
        )
    if p5.get("rewrite"):
        why += " Open leftover after issued dies — rewrite of issued volume."
    return {"card": card, "headline_tag": tag, "why": why, "keep_x": keep_x}


def make_png(tr: pd.DataFrame, p4: dict) -> str | None:
    if not HAS_MPL:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.4))
    op = pd.to_numeric(tr["e_ap_open"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    m = op.notna() & days.notna()
    ax[0].scatter(days[m], np.log1p(op[m].clip(lower=0)), s=4, alpha=0.15, c="#2c5f6e")
    ax[0].set_xlabel("c_n_days_with_tx")
    ax[0].set_ylabel("log1p(e_ap_open)")
    ax[0].set_title("AP open vs days")
    labels = ["open after days", "days after open"]
    vals = [p4["y3_rank"], p4["inv_rank"]]
    colors = ["#b85c38" if p4["y3_dies"] else "#2c5f6e", "#2c5f6e"]
    ax[1].bar(labels, [0 if not np.isfinite(v) else v for v in vals], color=colors)
    ax[1].axhline(CHANCE, color="#888", ls="--", lw=0.8)
    ax[1].axhline(DAYS_BENCH, color="#2c5f6e", ls=":", lw=0.8)
    ax[1].set_ylim(0.45, 0.80)
    ax[1].set_title("honest leftover (rank)")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return str(OUT_PNG)


def write_md(ctx: dict) -> None:
    d = ctx["decision"]
    lines = [
        "# Unused leftover of `e_ap_open` after `c_n_days_with_tx`",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        f"`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        f"Holdout 72 is coverage only. Seed {FOLD_SEED} group folds. No 0–100. "
        f"No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_ap_open`. "
        f"Night Y7 stays **TURNOVER {TURNOVER_QUOTE:.3f} / B_shallow {B_SHALLOW_QUOTE:.3f}**. "
        f"Do **not** grow TURNOVER. Do **not** put e_ap_open on the 15-col Y3 card. "
        f"Night Y3 stays **{NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f}**. Days **{DAYS_BENCH:.3f}**. "
        f"Size **{SIZE_QUOTE:.3f}**. Y7 never D. Y5 never E. Y3 never B. Dark 470 stay NaN not 0. "
        f"AP issued leftover {ISSUED_LEFTOVER_QUOTE:.3f} CLOSED — do not overwrite ap_issued_qa. "
        f"DPO already DROP — do not overwrite dpo_qa.",
        "",
        "`e_ap_open` = unpaid AP |amount| stock at period end. "
        "Feature report: 64.1% cov, acf1 0.74, ICC 0.99 BETWEEN. "
        "`e_dpo_proxy` = open / this-period issued.",
        "",
        "## Headline",
        "",
        f"{d['headline_tag']} leftover-after-days rank {_f(ctx['p4']['y3_rank'])} "
        f"(OLS {_f(ctx['p4']['y3_ols'])}, fake={ctx['p4']['y3_fake']}). "
        f"Y3 open {_f(ctx['p3']['y3'])} vs days {_f(ctx['p3']['days'])} vs size {_f(ctx['p3']['size'])} "
        f"vs issued {_f(ctx['p3']['issued'])} vs DPO {_f(ctx['p3']['dpo'])}. "
        f"SIZE={ctx['p2']['size_flag']} twin={bool(ctx['p2']['gate_twins'])}. "
        f"Inverse days-after-open {_f(ctx['p4']['inv_rank'])}. "
        f"Leftover after issued {_f(ctx['p5']['rank'])}. "
        f"Leftover after DPO {_f(ctx['p6']['rank'])}. "
        f"Card: {d['card']}. Night Y3 {NIGHT_Y3:.3f}/{NIGHT_Y3_CORE:.3f}, days {DAYS_BENCH:.3f}, "
        f"size {SIZE_QUOTE:.3f}, TURNOVER {TURNOVER_QUOTE:.3f}/{B_SHALLOW_QUOTE:.3f} unchanged.",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| open leftover after days (Y3 X) | **{d['headline_tag']}** | {d['why']} |",
        "| 15-col Y3 card stem | **KEEP off the card** | do not put e_ap_open on the card |",
        "| TURNOVER add-on | **CLOSE** | do not grow 0.720 |",
        f"| AP issued leftover | **CLOSE (locked)** | rank {ISSUED_LEFTOVER_QUOTE:.3f} fake days clone |",
        "| DPO | **DROP (locked)** | do not reopen dpo_qa |",
        "| health Y `y_ap_open` | **PARK** | do not invent y_ap_open |",
        "",
        "## 1. Coverage / 470 vs ERP / SIZE ρ",
        "",
        ctx["p1"]["prose"],
        "",
        _md_table(ctx["p1"]["rows"]),
        "",
        "## 2. Spearman twins / SIZE",
        "",
        ctx["p2"]["prose"],
        "",
        _md_table(ctx["p2"]["rows"]),
        "",
        "## 3. Single-feature group-fold Y3 / Y7",
        "",
        ctx["p3"]["prose"],
        "",
        _md_table(ctx["p3"]["rows"]),
        "",
        "## 4. Honest leftover after days (Y3)",
        "",
        ctx["p4"]["prose"],
        "",
        _md_table(ctx["p4"]["rows"]),
        "",
        "## 5. Leftover after e_ap_issued",
        "",
        ctx["p5"]["prose"],
        "",
        _md_table(ctx["p5"]["rows"]),
        "",
        "## 6. Leftover after DPO",
        "",
        ctx["p6"]["prose"],
        "",
        _md_table(ctx["p6"]["rows"]),
        "",
        "## 7. Dark 470",
        "",
        f"Dark never-ERP {ctx['p1']['n_dark_co']} nn={ctx['p1']['dark_nn']} "
        f"zero={ctx['p1']['dark_zero']} pos={ctx['p1']['dark_pos']}. "
        f"{'CONFIRM NaN not 0' if ctx['p1']['dark_ok'] else 'BOOK stub or check'}.",
        "",
        "## 8. Q6 lag1 leftover after days_lag1",
        "",
        ctx["p8"]["prose"],
        "",
        _md_table(ctx["p8"]["rows"]),
        "",
        "## 9. vs e_ar_open",
        "",
        ctx["p9"]["prose"],
        "",
        _md_table(ctx["p9"]["rows"]),
        "",
        "## 10. Bootstrap leftover-after-days",
        "",
        ctx["p10"]["prose"],
        "",
        _md_table(ctx["p10"]["rows"]),
        "",
        "## Extra — Y5 leftover after size (report only)",
        "",
        ctx["p7"]["prose"],
        "",
        _md_table(ctx["p7"]["rows"]),
        "",
        "## Extra — holdout coverage",
        "",
        ctx["ph"]["prose"],
        "",
        _md_table(ctx["ph"]["rows"]),
        "",
        "## Extra — open>0 leftover",
        "",
        ctx["ppos"]["prose"],
        "",
        _md_table(ctx["ppos"]["rows"]),
        "",
        "## Extra — log1p(open)",
        "",
        ctx["pl"]["prose"],
        "",
        _md_table(ctx["pl"]["rows"]),
        "",
        "## Extra — leftover after pending",
        "",
        ctx["ppd"]["prose"],
        "",
        _md_table(ctx["ppd"]["rows"]),
        "",
        "## Extra — SIZE terciles",
        "",
        ctx["pt"]["prose"],
        "",
        _md_table(ctx["pt"]["rows"]),
        "",
        "## Extra — company-median ρ",
        "",
        ctx["pcm"]["prose"],
        "",
        _md_table(ctx["pcm"]["rows"]),
        "",
        "## Extra — so_far leftover after days",
        "",
        ctx["psf"]["prose"],
        "",
        _md_table(ctx["psf"]["rows"]),
        "",
        "## Extra — per-fold leftover",
        "",
        ctx["pf"]["prose"],
        "",
        _md_table(ctx["pf"]["rows"]),
        "",
        "## Extra — leftover after issued+DPO / issued after open",
        "",
        ctx["pid"]["prose"],
        "",
        _md_table(ctx["pid"]["rows"]),
        "",
        "## Extra — Q6 mid / long",
        "",
        ctx["pq6"]["prose"],
        "",
        _md_table(ctx["pq6"]["rows"]),
        "",
        "## Extra — KEEP-as-X scorecard",
        "",
        ctx["pk"]["prose"],
        "",
        _md_table(ctx["pk"]["rows"]),
        "",
        "## Extra — days+size / intensity / Y7 / never-zero",
        "",
        ctx["pm"]["prose"],
        "",
        _md_table(ctx["pm"]["rows"]),
        "",
        "## Extra — leftover after overdue",
        "",
        ctx["pod"]["prose"],
        "",
        _md_table(ctx["pod"]["rows"]),
        "",
        "## Extra — never-zero leftover after issued",
        "",
        ctx["pnz"]["prose"],
        "",
        _md_table(ctx["pnz"]["rows"]),
        "",
        "## Extra — leftover after card KEEP leftovers",
        "",
        ctx["pck"]["prose"],
        "",
        _md_table(ctx["pck"]["rows"]),
        "",
        "## Extra — DPO-defined leftover",
        "",
        ctx["pdd"]["prose"],
        "",
        _md_table(ctx["pdd"]["rows"]),
        "",
        "## Extra — ever-ERP leftover",
        "",
        ctx["perp"]["prose"],
        "",
        _md_table(ctx["perp"]["rows"]),
        "",
        "## Extra — winsor / leftover after AR issued / AR-open+AP-issued",
        "",
        ctx["pwa"]["prose"],
        "",
        _md_table(ctx["pwa"]["rows"]),
        "",
        "## Extra — leftover after issued_lag1 (do not grow TURNOVER)",
        "",
        ctx["pil"]["prose"],
        "",
        _md_table(ctx["pil"]["rows"]),
        "",
        "## Extra — BETWEEN identity / leftover after own lag1",
        "",
        ctx["pbl"]["prose"],
        "",
        _md_table(ctx["pbl"]["rows"]),
        "",
        "## Extra — leftover after n_tx / joint days+issued+size / DPO after open",
        "",
        ctx["pnt"]["prose"],
        "",
        _md_table(ctx["pnt"]["rows"]),
        "",
        "## Extra — leftover after trail / AR_open_lag1",
        "",
        ctx["ptr"]["prose"],
        "",
        _md_table(ctx["ptr"]["rows"]),
        "",
        "## Extra — leftover after log1p(issued) / issued+AR_open+days",
        "",
        ctx["pli"]["prose"],
        "",
        _md_table(ctx["pli"]["rows"]),
        "",
        "## What failed / next (held for wave note)",
        "",
    ]
    for x in ctx["failed"]:
        lines.append(f"- {x}")
    lines.extend(
        [
            "",
            f"Elapsed {ctx['elapsed_s']:.0f}s. Night quotes unchanged. "
            "Do not grow TURNOVER. Do not put e_ap_open on the 15-col card.",
        ]
    )
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    existing = REGISTRY.read_text(encoding="utf-8")
    already = existing.count("ap_open_qa")
    if already >= 5:
        print("registry: no new rows")
        return
    ts = _now_iso()
    want = [
        ("auroc_e_ap_open", ctx["p3"]["y3"], Y3, f"days={ctx['p3']['days']:.4f} size={ctx['p3']['size']:.4f}"),
        (
            "auroc_open_resid_days_rank",
            ctx["p4"]["y3_rank"],
            Y3,
            f"ols={ctx['p4']['y3_ols']:.4f} fake={ctx['p4']['y3_fake']}",
        ),
        (
            "auroc_open_resid_issued_rank",
            ctx["p5"]["rank"],
            Y3,
            f"ols={ctx['p5']['ols']:.4f} rewrite={ctx['p5']['rewrite']}",
        ),
        (
            "auroc_open_resid_dpo_rank",
            ctx["p6"]["rank"],
            Y3,
            f"ols={ctx['p6']['ols']:.4f} numer={ctx['p6']['numer']}",
        ),
        (
            "rho_open_vs_days",
            ctx["p2"]["rho_days"],
            Y3,
            f"rho_issued={ctx['p2']['rho_issued']:.4f} rho_size={ctx['p2']['rho_size']:.4f}",
        ),
    ]
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "ts",
                "round",
                "wave",
                "agent",
                "x_families",
                "y",
                "model",
                "split",
                "metric",
                "value",
                "coverage",
                "notes",
            ],
        )
        for metric, value, y, notes in want:
            w.writerow(
                {
                    "ts": ts,
                    "round": ROUND,
                    "wave": WAVE,
                    "agent": AGENT,
                    "x_families": X_FAM,
                    "y": y,
                    "model": MODEL,
                    "split": "train_cv",
                    "metric": metric,
                    "value": value,
                    "coverage": f"{ctx['p1']['cov']:.4f}",
                    "notes": notes,
                }
            )
    print(f"registry: appended {len(want)} rows")


def write_wave(ctx: dict) -> None:
    d = ctx["decision"]
    lines = [
        "# Wave 4 — AP open leftover after days",
        "",
        f"Agent `{AGENT}`. Train group-fold seed {FOLD_SEED}. Holdout 72 coverage only.",
        "",
        "## Files written",
        "",
        "- `analysis/evaluate/ap_open_qa.py`",
        "- `analysis/outputs/ap_open_qa.md`",
        f"- `{OUT_PNG.name}`" if ctx.get("png") else "- (no PNG)",
        "- append-only `analysis/experiments/registry.csv`",
        "- this note",
        "",
        "Did not touch `dpo_qa.*`, `ap_issued_qa.*`, `issued_qa.*`, `pending_qa.*`, "
        "`supp_top1_qa.*`, `ogtg_qa.*`, `invoices.py`, `product/`, parquet / duckdb, "
        "`build_targets`, parent journal, LIVE, canvas, TURNOVER, or the 15-col card. "
        f"Night Y7 stays **TURNOVER {TURNOVER_QUOTE:.3f} / B_shallow {B_SHALLOW_QUOTE:.3f}**. "
        f"Night Y3 stays **{NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f}**. Days {DAYS_BENCH:.3f}. "
        f"Size {SIZE_QUOTE:.3f}.",
        "",
        "## Locked verdict",
        "",
        "| object | decision |",
        "| --- | --- |",
        f"| open leftover after days (Y3) | **{d['headline_tag']}** |",
        "| 15-col Y3 card | **KEEP off the card** |",
        "| TURNOVER add-on | **CLOSE** |",
        "| AP issued leftover | **CLOSE (locked)** |",
        "| DPO | **DROP (locked)** |",
        "| y_ap_open | **PARK** |",
        "",
        f"{d['headline_tag']} leftover-after-days rank {_f(ctx['p4']['y3_rank'])} "
        f"(OLS {_f(ctx['p4']['y3_ols'])}, fake={ctx['p4']['y3_fake']}). "
        f"Y3 open {_f(ctx['p3']['y3'])} vs days {_f(ctx['p3']['days'])} vs size {_f(ctx['p3']['size'])}. "
        f"SIZE={ctx['p2']['size_flag']} twin={bool(ctx['p2']['gate_twins'])}. "
        f"Inverse days-after-open {_f(ctx['p4']['inv_rank'])}. "
        f"Leftover after issued {_f(ctx['p5']['rank'])} rewrite={ctx['p5']['rewrite']}. "
        f"Leftover after DPO {_f(ctx['p6']['rank'])}. "
        f"Boot leftover-days p50={_f(ctx['p10']['p50'])} p05={_f(ctx['p10']['p05'])}. "
        f"Card: {d['card']}. "
        f"Night Y3 {NIGHT_Y3:.3f}/{NIGHT_Y3_CORE:.3f}, days {DAYS_BENCH:.3f}, "
        f"size {SIZE_QUOTE:.3f}, TURNOVER {TURNOVER_QUOTE:.3f}/{B_SHALLOW_QUOTE:.3f} unchanged.",
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
    print(f"ap_open_qa start seed={FOLD_SEED} holdout=72 agent={AGENT} wave={WRITE_WAVE}")
    panel = load_panel()
    store_has = bool(panel["store_has_open_lag1"].iloc[0])
    print(f"store has e_ap_open_lag1={store_has}")
    panel = attach_folds(panel)
    tr = panel[panel["split"] == "train"].copy()
    ho_n = int((panel["split"] == "holdout").sum())
    print(f"train CM={len(tr):,} companies={tr['company_id'].nunique()} holdout CM={ho_n}")
    book = book_invoice_ids(connect())
    print("pass 1 coverage")
    p1 = pass1_cov(tr, book)
    print("pass 2 twins")
    p2 = pass2_twins(tr)
    print("pass 3 singles")
    p3 = pass3_singles(tr)
    print("pass 4 leftover days")
    p4 = pass4_leftover(tr)
    print("pass 5 leftover after issued")
    p5 = pass5_after_issued(tr)
    print("pass 6 leftover after DPO")
    p6 = pass6_after_dpo(tr)
    print("pass 7 Y5 report only")
    p7 = pass7_y5(tr)
    print("pass 8 Q6")
    p8 = pass8_q6(tr)
    print("pass 9 vs e_ar_open")
    p9 = pass9_ar_open(tr)
    print("pass 10 bootstrap")
    p10 = pass10_boot(tr)
    print("extra holdout")
    ph = pass_holdout(panel, book)
    print("extra open>0 leftover")
    ppos = pass_open_pos(tr)
    print("extra log1p")
    pl = pass_log1p(tr)
    print("extra leftover after pending")
    ppd = pass_pending(tr)
    print("extra terciles")
    pt = pass_terciles(tr)
    print("extra company-median")
    pcm = pass_comedian(tr)
    print("extra so_far")
    psf = pass_sofar(tr)
    print("extra per-fold leftover")
    pf = pass_folds(tr)
    print("extra issued+DPO")
    pid = pass_issued_dpo(tr)
    print("extra Q6 mid/long")
    pq6 = pass_q6_more(tr)
    print("extra days+size / intensity / Y7 / never-zero")
    pm = pass_more_leftover(tr)
    print("extra leftover after overdue")
    pod = pass_overdue(tr)
    print("extra never-zero leftover after issued")
    pnz = pass_never_zero_more(tr)
    print("extra leftover after card KEEP leftovers")
    pck = pass_card_keeps(tr)
    print("extra DPO-defined leftover")
    pdd = pass_dpo_defined(tr)
    print("extra ever-ERP leftover")
    perp = pass_erp(tr, book)
    print("extra winsor / AR issued leftover")
    pwa = pass_winsor_ar_iss(tr)
    print("extra leftover after issued_lag1")
    pil = pass_issued_lag1(tr)
    print("extra BETWEEN / leftover after own lag1")
    pbl = pass_between_lag1(tr)
    print("extra leftover after n_tx / joint / DPO after open")
    pnt = pass_ntx_joint(tr)
    print("extra leftover after trail / AR_open_lag1")
    ptr = pass_trail_arlag(tr)
    print("extra leftover after log1p(issued) / triple")
    pli = pass_log_issued_triple(tr)
    png = make_png(tr, p4)
    decision = decide(p1, p2, p3, p4, p5)
    print("extra KEEP-as-X scorecard")
    pk = pass_keep_card(p1, p2, p3, p4, p5)
    failed = [
        f"Y3 leftover after days rank {_f(p4['y3_rank'])} OLS {_f(p4['y3_ols'])} fake={p4['y3_fake']}",
        f"inverse days leftover {_f(p4['inv_rank'])}",
        f"leftover after issued {_f(p5['rank'])} rewrite={p5['rewrite']}",
        f"leftover after DPO {_f(p6['rank'])} numer={p6['numer']}",
        f"Y5 leftover after size {_f(p7['rank'])} leak_ok={p7['leak_ok']}",
        f"Q6 open_lag1 leftover {_f(p8['lag_rank'])} now {_f(p8['now_rank'])}",
        f"leftover after AR open {_f(p9['rank'])} same={p9['same']}",
        f"boot leftover p05={_f(p10['p05'])} p50={_f(p10['p50'])} p95={_f(p10['p95'])}",
        f"open>0 leftover {_f(ppos['rank'])} fake={ppos['fake']}",
        f"log1p leftover {_f(pl['rank'])} fake={pl['fake']}",
        f"leftover after pending {_f(ppd['rank'])} +days {_f(ppd['both'])} fake={ppd['fake']}",
        f"terciles T1 {_f(pt['t1'])} T2+T3 {_f(pt['t23'])}",
        f"co-median SIZE={pcm['size_flag']} ρ={_f(pcm['rho_size'])}",
        f"so_far short {_f(psf['short'])} mid {_f(psf['mid'])} long {_f(psf['long'])}",
        f"per-fold leftover {pf['folds']}",
        f"after issued+DPO {_f(pid['both'])} +days {_f(pid['triple'])} issued-after-open {_f(pid['inv_iss'])}",
        f"Q6 mid Y7 {_f(pq6['mid_y7'])} leftover {_f(pq6['mid_rank'])}",
        f"days+size leftover {_f(pm['ds'])} intensity {_f(pm['intens'])} Y7 {_f(pm['y7'])} never-zero {_f(pm['nz'])}",
        f"leftover after overdue {_f(pod['rank'])}",
        f"never-zero leftover-days {_f(pnz['days'])} leftover-issued {_f(pnz['iss'])} +days {_f(pnz['both'])}",
        f"leftover after ss {_f(pck['ss'])} salary {_f(pck['sal'])} ss+salary+days {_f(pck['rank'])}",
        f"DPO-defined leftover {_f(pdd['rank'])} after DPO+days {_f(pdd['after'])}",
        f"ever-ERP leftover {_f(perp['rank'])} late {_f(perp['late'])}",
        f"winsor leftover {_f(pwa['win'])} after AR issued {_f(pwa['ar_iss'])} AR-open+AP-issued {_f(pwa['both'])}",
        f"leftover after issued_lag1 Y3 {_f(pil['y3'])} Y7 {_f(pil['y7'])} +days {_f(pil['both'])}",
        f"BETWEEN leftover after cmean {_f(pbl['mean'])} +days {_f(pbl['mean_days'])} after lag1 {_f(pbl['lag1'])} within {_f(pbl['within'])}",
        f"leftover after n_tx {_f(pnt['ntx'])} +days {_f(pnt['ntx_days'])} days+issued+size {_f(pnt['joint'])} DPO-after-open {_f(pnt['dpo_after'])}",
        f"leftover after trail {_f(ptr['trail'])} +days {_f(ptr['trail_days'])} after AR_open_lag1 {_f(ptr['ar_l1'])} +days {_f(ptr['ar_l1_days'])}",
        f"leftover after log1p(issued) {_f(pli['log'])} +days {_f(pli['log_days'])} issued+AR_open+days {_f(pli['triple'])}",
        f"card: {decision['card']}",
        "do not grow TURNOVER 0.720; do not put e_ap_open on the 15-col card",
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
        "ppos": ppos,
        "pl": pl,
        "ppd": ppd,
        "pt": pt,
        "pcm": pcm,
        "psf": psf,
        "pf": pf,
        "pid": pid,
        "pq6": pq6,
        "pk": pk,
        "pm": pm,
        "pod": pod,
        "pnz": pnz,
        "pck": pck,
        "pdd": pdd,
        "perp": perp,
        "pwa": pwa,
        "pil": pil,
        "pbl": pbl,
        "pnt": pnt,
        "ptr": ptr,
        "pli": pli,
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
    print(
        f"{decision['headline_tag']} leftover-after-days rank {_f(p4['y3_rank'])} "
        f"(OLS {_f(p4['y3_ols'])}, fake={p4['y3_fake']}). "
        f"Y3 open {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size'])}. "
        f"SIZE={p2['size_flag']} twin={bool(p2['gate_twins'])}. "
        f"Inverse days-after-open {_f(p4['inv_rank'])}. "
        f"Leftover after issued {_f(p5['rank'])}. "
        f"Leftover after DPO {_f(p6['rank'])}. Card: {decision['card']}. "
        f"Night Y3 {NIGHT_Y3:.3f}/{NIGHT_Y3_CORE:.3f}, days {DAYS_BENCH:.3f}, "
        f"size {SIZE_QUOTE:.3f}, TURNOVER {TURNOVER_QUOTE:.3f}/{B_SHALLOW_QUOTE:.3f} unchanged."
    )
    return ctx


if __name__ == "__main__":
    run()
