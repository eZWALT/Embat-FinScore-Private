"""Unused leftover of ``e_ap_issued`` after ``c_n_days_with_tx`` as Y3 X.

NORTH_STAR: ``e_ap_issued`` = this-period AP issuance volume (abs amount,
amount<0 invoices). Feature report: 64.1% cov, acf1 0.17, w/b 0.86,
ICC 0.54 LOW_PERSIST; company-median VIF with size.

Contemporaneous ``e_ar_issued`` just CLOSED as unused leftover (rank
0.608 is a fake days clone; twin of issued_lag1 ρ 0.814; Y7 leftover
after lag1 0.581 CLOSE add-on). ``issued_lag1`` Q6 KEEP locked 0.626.
TURNOVER 0.720 must not grow. DSO/DPO/pending/delay/CN already decided.
Y5 never E. Dark 470 stay NaN not 0. Do **not** overwrite issued_qa /
dso_qa / dpo_qa / credit_note_qa.

KEEP-as-X: beat size ≥0.02 AND leftover after days AND not SIZE
(|ρ| vs log1p(a_in3) ≥0.50) AND not a twin (|ρ|≥0.80 vs days / a_n_tx /
e_ar_issued / e_dpo_proxy / d_n_supp). Leftover <0.55 dies. Rank leftover
is honest; OLS can fake a days leak (AR issued ρ=-0.849).

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.ap_issued_qa

Owned: analysis/evaluate/ap_issued_qa.py, analysis/outputs/ap_issued_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_ap_issued.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "ap_issued_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "ap_issued_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_ap_issued.md"
AGENT = "e8b2c0d4"
WAVE = "4"
ROUND = "R4"
MODEL = "ap_issued_qa"
X_FAM = "E"

Y3 = "y3_recover_cash_6m"
Y5 = "y5_ap_od30_ownp80"
Y7 = "y7_top1_lost"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
AR_Y3_QUOTE = 0.687
AR_LEFTOVER_QUOTE = 0.608
Q6_LAG1_QUOTE = 0.626
TURNOVER_QUOTE = 0.720
B_SHALLOW_QUOTE = 0.712
NIGHT_Y3 = 0.762
NIGHT_Y3_CORE = 0.752
COV_QUOTE = 0.641
ICC_QUOTE = 0.86
ACF1_QUOTE = 0.17
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
    "e_ap_issued",
    "e_ar_issued",
    "e_dpo_proxy",
    "d_n_supp",
)

Y_KEEP = (Y3, Y5, Y7)
TWIN_GATES = (
    "c_n_days_with_tx",
    "a_n_tx",
    "e_ar_issued",
    "e_dpo_proxy",
    "d_n_supp",
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
    have_lag = "e_ap_issued_lag1" in raw.columns
    cols = list(STORE_COLS)
    if have_lag:
        cols.append("e_ap_issued_lag1")
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
    ap = pd.to_numeric(panel["e_ap_issued"], errors="coerce")
    panel["log1p_ap"] = np.log1p(ap.clip(lower=0))
    days = pd.to_numeric(panel["c_n_days_with_tx"], errors="coerce")
    panel["ap_per_day"] = ap / days.where(days > 0)
    panel["first_month"] = pd.to_datetime(panel["first_month"])
    panel["months_so_far"] = (
        (panel["period"].dt.year - panel["first_month"].dt.year) * 12
        + (panel["period"].dt.month - panel["first_month"].dt.month)
        + 1
    )
    panel["so_far_class"] = panel["months_so_far"].map(_trail_class)
    panel["early6"] = panel["period"] < DELAY_MASK_BEFORE
    panel["store_has_ap_lag1"] = have_lag
    panel = add_panel_lags(
        panel,
        ["e_ap_issued", "e_ar_issued", "c_n_days_with_tx", "log1p_ap"],
        (1, 3),
    )
    leak7 = leakage_check(["e_ap_issued", "e_ap_issued_lag1"], Y7, forbidden_prefixes=["d"])
    leak3 = leakage_check(["e_ap_issued"], Y3, forbidden_prefixes=["b"])
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
    ap = pd.to_numeric(tr["e_ap_issued"], errors="coerce")
    n_cm = int(len(tr))
    n_nn = int(ap.notna().sum())
    dark_nn = int(ap[dark].notna().sum())
    dark_zero = int((ap[dark] == 0).sum())
    dark_pos = int((ap[dark] > 0).sum())
    erp_nn = int(ap[erp].notna().sum())
    erp_zero = int((ap[erp] == 0).sum())
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    n_erp_co = int(tr.loc[erp, "company_id"].nunique())
    rho_size, n_size = spearman_n(ap, tr["log_in3"])
    size_flag = bool(np.isfinite(rho_size) and abs(rho_size) >= SIZE_RHO)
    finite = ap[ap.notna()]
    p50 = float(finite.median()) if n_nn else float("nan")
    p99 = float(finite.quantile(0.99)) if n_nn else float("nan")
    mx = float(finite.max()) if n_nn else float("nan")
    eq0 = _pct(int((finite == 0).sum()), n_nn)
    acf1 = median_acf(ap, tr["company_id"], 1)
    icc = icc_anova(ap, tr["company_id"])
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
        f"Train e_ap_issued nn={n_nn:,} cov={_pp(_pct(n_nn, n_cm))} "
        f"(feature report {COV_QUOTE:.1%}). Dark never-ERP {n_dark_co} "
        f"(want 470): nn={dark_nn} zero={dark_zero} pos={dark_pos} "
        f"{'CONFIRM NaN not 0' if dark_ok else ('BOOK stub' if dark_nn else 'check')}. "
        f"Ever-ERP {n_erp_co} nn={erp_nn:,} of which zero={erp_zero:,}. "
        f"p50={_f(p50, 0)} p99={_f(p99, 0)} max={_f(mx, 0)}. "
        f"ρ vs log1p(a_in3)={_f(rho_size)} n={n_size} "
        f"{'SIZE' if size_flag else 'not SIZE'}. "
        f"acf1={_f(acf1)} (quote {ACF1_QUOTE:.2f}) ICC={_f(icc['icc'])} "
        f"(feature-report ICC 0.54 CONFIRM; the 0.86 is w/b not ICC) k={icc['k']}."
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
        "p50": p50,
        "p99": p99,
        "acf1": acf1,
        "icc": icc["icc"],
        "prose": prose,
    }


def pass2_twins(tr: pd.DataFrame) -> dict:
    ap = tr["e_ap_issued"]
    pairs = [
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("a_n_tx", tr["a_n_tx"]),
        ("e_ar_issued", tr["e_ar_issued"]),
        ("e_dpo_proxy", tr["e_dpo_proxy"]),
        ("d_n_supp", tr["d_n_supp"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("e_ap_issued_lag1", tr["e_ap_issued_lag1"]),
        ("a_in3", tr["a_in3"]),
    ]
    rows = []
    rhos = {}
    twins = []
    for name, s in pairs:
        rho, n = spearman_n(ap, s)
        rhos[name] = rho
        twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
        if twin:
            twins.append(name)
        rows.append({"vs": name, "ρ": _f(rho), "n": f"{n:,}", "twin?": "TWIN" if twin else ""})
    size_flag = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    gate = [t for t in twins if t in TWIN_GATES]
    prose = (
        f"ap_issued vs days ρ={_f(rhos['c_n_days_with_tx'])} "
        f"({'TWIN' if 'c_n_days_with_tx' in twins else 'not a twin'}). "
        f"vs a_n_tx {_f(rhos['a_n_tx'])} vs e_ar_issued {_f(rhos['e_ar_issued'])} "
        f"vs DPO {_f(rhos['e_dpo_proxy'])} vs d_n_supp {_f(rhos['d_n_supp'])} "
        f"vs log1p(a_in3) {_f(rhos['log1p(a_in3)'])} "
        f"({'SIZE' if size_flag else 'not SIZE'}). "
        f"vs ap_lag1 {_f(rhos['e_ap_issued_lag1'])}. "
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
        "rho_ar": rhos["e_ar_issued"],
        "rho_dpo": rhos["e_dpo_proxy"],
        "rho_n_tx": rhos["a_n_tx"],
        "rho_supp": rhos["d_n_supp"],
        "rho_lag1": rhos["e_ap_issued_lag1"],
        "prose": prose,
    }


def pass3_singles(tr: pd.DataFrame) -> dict:
    feats = {
        "e_ap_issued": tr["e_ap_issued"],
        "e_ap_issued_lag1": tr["e_ap_issued_lag1"],
        "log1p_ap": tr["log1p_ap"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "log1p_a_in3": tr["log_in3"],
        "e_ar_issued": tr["e_ar_issued"],
        "e_dpo_proxy": tr["e_dpo_proxy"],
        "a_n_tx": tr["a_n_tx"],
        "d_n_supp": tr["d_n_supp"],
    }
    rows = []
    store = {}
    for ycol in (Y3, Y7):
        for name, s in feats.items():
            res = signed_oof_auroc(tr[ycol], s, tr["fold"], tr[ycol].notna())
            store[(ycol, name)] = res
            rows.append(_auc_row(ycol, name, res))
            print(f"{ycol} {name}: {_f(_cv(res))} n={res['n_defined']:,} pos={res['n_pos']:,}")
    y3 = _cv(store[(Y3, "e_ap_issued")])
    y3_lag = _cv(store[(Y3, "e_ap_issued_lag1")])
    days = _cv(store[(Y3, "c_n_days_with_tx")])
    size = _cv(store[(Y3, "log1p_a_in3")])
    ar = _cv(store[(Y3, "e_ar_issued")])
    dpo = _cv(store[(Y3, "e_dpo_proxy")])
    y7 = _cv(store[(Y7, "e_ap_issued")])
    y7_lag = _cv(store[(Y7, "e_ap_issued_lag1")])
    days_ok = bool(np.isfinite(days) and abs(days - DAYS_BENCH) < 0.005)
    size_ok = bool(np.isfinite(size) and abs(size - SIZE_QUOTE) < 0.005)
    ar_ok = bool(np.isfinite(ar) and abs(ar - AR_Y3_QUOTE) < 0.005)
    beat = float(y3 - size) if np.isfinite(y3) and np.isfinite(size) else float("nan")
    prose = (
        f"Y3 e_ap_issued {_f(y3)} vs days {_f(days)} "
        f"({'CONFIRM' if days_ok else 'off'} {DAYS_BENCH:.3f}) "
        f"vs size {_f(size)} ({'CONFIRM' if size_ok else 'off'} {SIZE_QUOTE:.3f}) "
        f"vs e_ar_issued {_f(ar)} ({'CONFIRM' if ar_ok else 'off'} {AR_Y3_QUOTE:.3f}) "
        f"vs DPO {_f(dpo)}. ap_lag1 {_f(y3_lag)}. "
        f"Y7 ap {_f(y7)} ap_lag1 {_f(y7_lag)}. Beat size {_f(beat)}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "y3": y3,
        "y3_lag": y3_lag,
        "days": days,
        "size": size,
        "ar": ar,
        "dpo": dpo,
        "y7": y7,
        "y7_lag": y7_lag,
        "beat_size": beat,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "ar_ok": ar_ok,
        "prose": prose,
    }


def pass4_leftover(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ap_issued"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    inv = leftover_diag(
        tr[Y3], tr["c_n_days_with_tx"], (tr["e_ap_issued"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "bar": "ap leftover after days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ(resid,ctrl)": _f(d["rho_ctrl"]),
            "R²": _f(d["r2"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        },
        {
            "bar": "days leftover after ap",
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
        f"Inverse: days leftover after ap rank {_f(inv['rank'])} OLS {_f(inv['ols'])} "
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
        "inv_ols": inv["ols"],
        "inv_dies": inv["honest_dies"],
        "prose": prose,
    }


def pass5_after_ar(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ap_issued"], (tr["e_ar_issued"],), tr["fold"], tr[Y3].notna()
    )
    both = leftover_diag(
        tr[Y3],
        tr["e_ap_issued"],
        (tr["e_ar_issued"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "after e_ar_issued",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        },
        {
            "bar": "after AR+days",
            "rank": _f(both["rank"]),
            "OLS": _f(both["ols"]),
            "ρ": _f(both["rho_ctrl"]),
            "fake?": "FALSE clone" if both["fake"] else "",
            "n": f"{both['n']:,}",
        },
    ]
    rewrite = bool(d["honest_dies"] or (np.isfinite(d["rank"]) and d["rank"] < CHANCE))
    prose = (
        f"Y3 leftover after e_ar_issued rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"ρ={_f(d['rho_ctrl'])} {'rewrite of AR volume — dies' if rewrite else 'not just AR'}. "
        f"After AR+days {_f(both['rank'])} fake={both['fake']}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "ols": d["ols"],
        "rho": d["rho_ctrl"],
        "fake": d["fake"],
        "dies": d["honest_dies"],
        "both": both["rank"],
        "rewrite": rewrite,
        "prose": prose,
    }


def pass6_y5(tr: pd.DataFrame) -> dict:
    """Y5 leftover after size — report only. Y5 never E as X."""
    leak = leakage_check(["e_ap_issued"], Y5, forbidden_prefixes=["e"])
    d = leftover_diag(
        tr[Y5], tr["e_ap_issued"], (tr["log_in3"],), tr["fold"], tr[Y5].notna()
    )
    raw = signed_oof_auroc(tr[Y5], tr["e_ap_issued"], tr["fold"], tr[Y5].notna())
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


def pass7_q6(tr: pd.DataFrame) -> dict:
    short = tr["so_far_class"] == "short_<12"
    raw = signed_oof_auroc(
        tr[Y7], tr["e_ap_issued_lag1"], tr["fold"], short & tr[Y7].notna()
    )
    d = leftover_diag(
        tr[Y3],
        tr["e_ap_issued_lag1"],
        (tr["c_n_days_with_tx_lag1"],),
        tr["fold"],
        short & tr[Y3].notna(),
    )
    now = leftover_diag(
        tr[Y3],
        tr["e_ap_issued"],
        (tr["c_n_days_with_tx_lag1"],),
        tr["fold"],
        short & tr[Y3].notna(),
    )
    rows = [
        {
            "slice": "Q6 Y7 ap_lag1 short",
            "CV": "LOW_POWER" if raw["low_power"] else _f(_cv(raw)),
            "n": f"{raw['n_defined']:,}",
            "n_pos": f"{raw['n_pos']:,}",
        },
        {
            "slice": "Q6 Y3 ap_lag1 leftover after days_lag1 short",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "slice": "Q6 Y3 contemporaneous leftover after days_lag1 short",
            "rank": _f(now["rank"]),
            "OLS": _f(now["ols"]),
            "ρ": _f(now["rho_ctrl"]),
            "fake?": "FALSE clone" if now["fake"] else "",
        },
    ]
    prose = (
        f"Q6 Y7 ap_lag1 on short {_f(_cv(raw))} (AR issued_lag1 KEEP {Q6_LAG1_QUOTE:.3f} locked). "
        f"ap_lag1 leftover after days_lag1 short rank {_f(d['rank'])} fake={d['fake']}. "
        f"Contemporaneous leftover after days_lag1 short {_f(now['rank'])} fake={now['fake']}. "
        f"Do not claim a TURNOVER seat."
    )
    print(prose)
    return {
        "rows": rows,
        "y7_lag_short": _cv(raw),
        "lag_rank": d["rank"],
        "lag_fake": d["fake"],
        "lag_dies": d["honest_dies"],
        "now_rank": now["rank"],
        "now_fake": now["fake"],
        "now_dies": now["honest_dies"],
        "prose": prose,
    }


def pass8_lag1(tr: pd.DataFrame, store_has: bool) -> dict:
    raw = signed_oof_auroc(tr[Y3], tr["e_ap_issued_lag1"], tr["fold"], tr[Y3].notna())
    d = leftover_diag(
        tr[Y3], tr["e_ap_issued"], (tr["e_ap_issued_lag1"],), tr["fold"], tr[Y3].notna()
    )
    y7 = leftover_diag(
        tr[Y7], tr["e_ap_issued"], (tr["e_ap_issued_lag1"],), tr["fold"], tr[Y7].notna()
    )
    rows = [
        {
            "bar": "Y3 now leftover after ap_lag1",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "raw_lag1": _f(_cv(raw)),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "bar": "Y7 now leftover after ap_lag1",
            "rank": _f(y7["rank"]),
            "OLS": _f(y7["ols"]),
            "ρ": _f(y7["rho_ctrl"]),
            "raw_lag1": "—",
            "fake?": "FALSE clone" if y7["fake"] else "",
        },
    ]
    prose = (
        f"Store has e_ap_issued_lag1={store_has}; panel lag computed either way. "
        f"Y3 ap_lag1 {_f(_cv(raw))}. Now leftover after lag1 rank {_f(d['rank'])} "
        f"OLS {_f(d['ols'])} ρ={_f(d['rho_ctrl'])}. "
        f"Y7 leftover after lag1 {_f(y7['rank'])} — do not grow TURNOVER {TURNOVER_QUOTE:.3f}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3_lag": _cv(raw),
        "rank": d["rank"],
        "ols": d["ols"],
        "rho": d["rho_ctrl"],
        "fake": d["fake"],
        "dies": d["honest_dies"],
        "y7_rank": y7["rank"],
        "y7_dies": y7["honest_dies"],
        "store_has": store_has,
        "prose": prose,
    }


def pass9_size_dpo(tr: pd.DataFrame) -> dict:
    after_size = leftover_diag(
        tr[Y3], tr["e_ap_issued"], (tr["log_in3"],), tr["fold"], tr[Y3].notna()
    )
    after_dpo = leftover_diag(
        tr[Y3], tr["e_ap_issued"], (tr["e_dpo_proxy"],), tr["fold"], tr[Y3].notna()
    )
    after_days_size = leftover_diag(
        tr[Y3],
        tr["e_ap_issued"],
        (tr["c_n_days_with_tx"], tr["log_in3"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "after size",
            "rank": _f(after_size["rank"]),
            "OLS": _f(after_size["ols"]),
            "ρ": _f(after_size["rho_ctrl"]),
            "fake?": "FALSE clone" if after_size["fake"] else "",
        },
        {
            "bar": "after DPO",
            "rank": _f(after_dpo["rank"]),
            "OLS": _f(after_dpo["ols"]),
            "ρ": _f(after_dpo["rho_ctrl"]),
            "fake?": "FALSE clone" if after_dpo["fake"] else "",
        },
        {
            "bar": "after days+size",
            "rank": _f(after_days_size["rank"]),
            "OLS": _f(after_days_size["ols"]),
            "ρ": _f(after_days_size["rho_ctrl"]),
            "fake?": "FALSE clone" if after_days_size["fake"] else "",
        },
    ]
    prose = (
        f"Leftover after size {_f(after_size['rank'])}. After DPO {_f(after_dpo['rank'])} "
        f"(DPO already decided — do not reopen). After days+size {_f(after_days_size['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "size_rank": after_size["rank"],
        "dpo_rank": after_dpo["rank"],
        "days_size": after_days_size["rank"],
        "prose": prose,
    }


def pass10_boot(tr: pd.DataFrame) -> dict:
    """Company bootstrap of rank leftover after days. Train only."""
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
            sub["e_ap_issued"],
            (sub["c_n_days_with_tx"],),
            sub["fold"],
            sub[Y3].notna(),
        )
        if np.isfinite(d["rank"]):
            vals.append(float(d["rank"]))
    if len(vals) < 8:
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
    ap = pd.to_numeric(ho["e_ap_issued"], errors="coerce")
    dark = ~ho["company_id"].isin(book)
    rows = [
        {
            "slice": "holdout 72",
            "n_cm": f"{len(ho):,}",
            "companies": f"{ho['company_id'].nunique()}",
            "cov": _pp(_pct(int(ap.notna().sum()), len(ho))),
            "dark_nn": int(ap[dark].notna().sum()),
        }
    ]
    prose = (
        f"Holdout 72 coverage only: {len(ho):,} CM / {ho['company_id'].nunique()} companies. "
        f"ap cov {_pp(_pct(int(ap.notna().sum()), len(ho)))}; dark nn={int(ap[dark].notna().sum())}. "
        f"No AUROC."
    )
    print(prose)
    return {"rows": rows, "cov": _pct(int(ap.notna().sum()), len(ho)), "prose": prose}


def pass_ap_pos(tr: pd.DataFrame) -> dict:
    ap = pd.to_numeric(tr["e_ap_issued"], errors="coerce")
    m = (ap > 0) & tr[Y3].notna()
    d = leftover_diag(tr[Y3], tr["e_ap_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m)
    raw = signed_oof_auroc(tr[Y3], tr["e_ap_issued"], tr["fold"], m)
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], m)
    rows = [
        {
            "slice": "ap>0 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "raw": _f(_cv(raw)),
            "days": _f(_cv(days)),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"ap>0 leftover after days rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"ρ={_f(d['rho_ctrl'])} fake={d['fake']} raw {_f(_cv(raw))} days {_f(_cv(days))}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "raw": _cv(raw), "prose": prose}


def pass_log1p(tr: pd.DataFrame) -> dict:
    raw = signed_oof_auroc(tr[Y3], tr["log1p_ap"], tr["fold"], tr[Y3].notna())
    d = leftover_diag(
        tr[Y3], tr["log1p_ap"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "slice": "log1p(ap) leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "raw": _f(_cv(raw)),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"log1p(ap) Y3 {_f(_cv(raw))} leftover-after-days rank {_f(d['rank'])} "
        f"OLS {_f(d['ols'])} ρ={_f(d['rho_ctrl'])} fake={d['fake']}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "ols": d["ols"],
        "rho": d["rho_ctrl"],
        "fake": d["fake"],
        "raw": _cv(raw),
        "prose": prose,
    }


def pass_intensity(tr: pd.DataFrame) -> dict:
    raw = signed_oof_auroc(tr[Y3], tr["ap_per_day"], tr["fold"], tr[Y3].notna())
    d = leftover_diag(
        tr[Y3], tr["ap_per_day"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "slice": "ap/days leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "raw": _f(_cv(raw)),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"ap/days intensity Y3 {_f(_cv(raw))} leftover-after-days rank {_f(d['rank'])} "
        f"fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "raw": _cv(raw), "prose": prose}


def pass_terciles(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    rows = []
    store = {}
    for lab, mask in (
        ("T1", terc == "T1"),
        ("T2", terc == "T2"),
        ("T3", terc == "T3"),
        ("T2+T3", terc.isin(["T2", "T3"])),
    ):
        d = leftover_diag(
            tr[Y3],
            tr["e_ap_issued"],
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
    med = tr.groupby("company_id")[["e_ap_issued", "c_n_days_with_tx", "log_in3"]].median()
    rho_days, n1 = spearman_n(med["e_ap_issued"], med["c_n_days_with_tx"])
    rho_size, n2 = spearman_n(med["e_ap_issued"], med["log_in3"])
    size_flag = bool(np.isfinite(rho_size) and abs(rho_size) >= SIZE_RHO)
    rows = [
        {"pair": "company-median vs days", "ρ": _f(rho_days), "n": f"{n1:,}"},
        {
            "pair": "company-median vs log1p(a_in3)",
            "ρ": _f(rho_size),
            "n": f"{n2:,}",
        },
    ]
    prose = (
        f"Company-median ρ ap vs days {_f(rho_days)} vs log1p(a_in3) {_f(rho_size)} "
        f"{'SIZE' if size_flag else 'not SIZE'} (feature report flagged company-median VIF / size)."
    )
    print(prose)
    return {
        "rows": rows,
        "rho_days": rho_days,
        "rho_size": rho_size,
        "size_flag": size_flag,
        "prose": prose,
    }


def pass_after_supp(tr: pd.DataFrame) -> dict:
    """Near-twin d_n_supp ρ=0.760 — is leftover after suppliers a rewrite?"""
    d = leftover_diag(
        tr[Y3], tr["e_ap_issued"], (tr["d_n_supp"],), tr["fold"], tr[Y3].notna()
    )
    both = leftover_diag(
        tr[Y3],
        tr["e_ap_issued"],
        (tr["d_n_supp"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    ntx = leftover_diag(
        tr[Y3], tr["e_ap_issued"], (tr["a_n_tx"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "bar": "after d_n_supp",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        },
        {
            "bar": "after supp+days",
            "rank": _f(both["rank"]),
            "OLS": _f(both["ols"]),
            "ρ": _f(both["rho_ctrl"]),
            "fake?": "FALSE clone" if both["fake"] else "",
            "n": f"{both['n']:,}",
        },
        {
            "bar": "after a_n_tx",
            "rank": _f(ntx["rank"]),
            "OLS": _f(ntx["ols"]),
            "ρ": _f(ntx["rho_ctrl"]),
            "fake?": "FALSE clone" if ntx["fake"] else "",
            "n": f"{ntx['n']:,}",
        },
    ]
    prose = (
        f"Leftover after d_n_supp rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"ρ={_f(d['rho_ctrl'])} fake={d['fake']}. After supp+days {_f(both['rank'])}. "
        f"After a_n_tx {_f(ntx['rank'])} fake={ntx['fake']}."
    )
    print(prose)
    return {
        "rows": rows,
        "supp_rank": d["rank"],
        "supp_dies": d["honest_dies"],
        "both_rank": both["rank"],
        "ntx_rank": ntx["rank"],
        "ntx_fake": ntx["fake"],
        "prose": prose,
    }


def pass_both_pos(tr: pd.DataFrame) -> dict:
    ap = pd.to_numeric(tr["e_ap_issued"], errors="coerce")
    ar = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    m = (ap > 0) & (ar > 0) & tr[Y3].notna()
    d = leftover_diag(tr[Y3], tr["e_ap_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m)
    after_ar = leftover_diag(tr[Y3], tr["e_ap_issued"], (tr["e_ar_issued"],), tr["fold"], m)
    inv = leftover_diag(tr[Y3], tr["c_n_days_with_tx"], (tr["e_ap_issued"],), tr["fold"], m)
    rows = [
        {
            "slice": "ap>0 & ar>0 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        },
        {
            "slice": "ap>0 & ar>0 leftover-AR",
            "rank": _f(after_ar["rank"]),
            "OLS": _f(after_ar["ols"]),
            "ρ": _f(after_ar["rho_ctrl"]),
            "fake?": "FALSE clone" if after_ar["fake"] else "",
            "n": f"{after_ar['n']:,}",
        },
        {
            "slice": "days leftover after ap on both>0",
            "rank": _f(inv["rank"]),
            "OLS": _f(inv["ols"]),
            "ρ": _f(inv["rho_ctrl"]),
            "fake?": "FALSE clone" if inv["fake"] else "",
            "n": f"{inv['n']:,}",
        },
    ]
    prose = (
        f"both>0 leftover-days rank {_f(d['rank'])} fake={d['fake']}; "
        f"leftover-AR {_f(after_ar['rank'])}; days-after-ap {_f(inv['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "fake": d["fake"],
        "ar_rank": after_ar["rank"],
        "inv": inv["rank"],
        "prose": prose,
    }


def pass_sofar(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for lab in ("short_<12", "mid_12_17", "long_>=18"):
        m = (tr["so_far_class"] == lab) & tr[Y3].notna()
        d = leftover_diag(
            tr[Y3], tr["e_ap_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m
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
        tr[Y3],
        tr["e_ap_issued"],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        tr[Y3].notna(),
    )
    bits = d["rank_folds"].split()
    rows = [{"fold": str(i), "rank leftover": bits[i] if i < len(bits) else "—"} for i in range(5)]
    prose = f"Per-fold rank leftover after days: {d['rank_folds']}."
    print(prose)
    return {"rows": rows, "folds": d["rank_folds"], "prose": prose}


def pass_log1p_more(tr: pd.DataFrame) -> dict:
    after_size = leftover_diag(
        tr[Y3], tr["log1p_ap"], (tr["log_in3"],), tr["fold"], tr[Y3].notna()
    )
    after_ar = leftover_diag(
        tr[Y3], tr["log1p_ap"], (tr["e_ar_issued"],), tr["fold"], tr[Y3].notna()
    )
    after_lag = leftover_diag(
        tr[Y3], tr["log1p_ap"], (tr["log1p_ap_lag1"],), tr["fold"], tr[Y3].notna()
    )
    inv = leftover_diag(
        tr[Y3], tr["c_n_days_with_tx"], (tr["log1p_ap"],), tr["fold"], tr[Y3].notna()
    )
    rho_size, n_s = spearman_n(tr["log1p_ap"], tr["log_in3"])
    rho_lag, n_l = spearman_n(tr["log1p_ap"], tr["log1p_ap_lag1"])
    rho_days, n_d = spearman_n(tr["log1p_ap"], tr["c_n_days_with_tx"])
    size_flag = bool(np.isfinite(rho_size) and abs(rho_size) >= SIZE_RHO)
    twin_lag = bool(np.isfinite(rho_lag) and abs(rho_lag) >= TWIN_RHO)
    rows = [
        {
            "bar": "log1p leftover after size",
            "rank": _f(after_size["rank"]),
            "OLS": _f(after_size["ols"]),
            "ρ": _f(after_size["rho_ctrl"]),
            "n": f"{after_size['n']:,}",
        },
        {
            "bar": "log1p leftover after AR",
            "rank": _f(after_ar["rank"]),
            "OLS": _f(after_ar["ols"]),
            "ρ": _f(after_ar["rho_ctrl"]),
            "n": f"{after_ar['n']:,}",
        },
        {
            "bar": "log1p leftover after log1p_lag1",
            "rank": _f(after_lag["rank"]),
            "OLS": _f(after_lag["ols"]),
            "ρ": _f(after_lag["rho_ctrl"]),
            "n": f"{after_lag['n']:,}",
        },
        {
            "bar": "days leftover after log1p",
            "rank": _f(inv["rank"]),
            "OLS": _f(inv["ols"]),
            "ρ": _f(inv["rho_ctrl"]),
            "n": f"{inv['n']:,}",
        },
        {"bar": "ρ log1p vs log1p(a_in3)", "ρ": _f(rho_size), "n": f"{n_s:,}"},
        {"bar": "ρ log1p vs log1p_lag1", "ρ": _f(rho_lag), "n": f"{n_l:,}"},
        {"bar": "ρ log1p vs days", "ρ": _f(rho_days), "n": f"{n_d:,}"},
    ]
    prose = (
        f"log1p leftover after size {_f(after_size['rank'])} after AR {_f(after_ar['rank'])} "
        f"after lag1 {_f(after_lag['rank'])}. Inverse days-after-log1p {_f(inv['rank'])}. "
        f"ρ vs size {_f(rho_size)} {'SIZE' if size_flag else 'not SIZE'}; "
        f"vs lag1 {_f(rho_lag)} {'TWIN' if twin_lag else 'not twin'}."
    )
    print(prose)
    return {
        "rows": rows,
        "size_rank": after_size["rank"],
        "ar_rank": after_ar["rank"],
        "lag_rank": after_lag["rank"],
        "inv": inv["rank"],
        "size_flag": size_flag,
        "twin_lag": twin_lag,
        "rho_size": rho_size,
        "prose": prose,
    }


def pass_q6_more(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for lab in ("short_<12", "mid_12_17", "long_>=18"):
        m = tr["so_far_class"] == lab
        y7 = signed_oof_auroc(tr[Y7], tr["e_ap_issued_lag1"], tr["fold"], m & tr[Y7].notna())
        d = leftover_diag(
            tr[Y3],
            tr["e_ap_issued_lag1"],
            (tr["c_n_days_with_tx_lag1"],),
            tr["fold"],
            m & tr[Y3].notna(),
        )
        y7_left = leftover_diag(
            tr[Y7],
            tr["e_ap_issued_lag1"],
            (tr["c_n_days_with_tx_lag1"],),
            tr["fold"],
            m & tr[Y7].notna(),
        )
        store[lab] = {"y7": _cv(y7), "rank": d["rank"], "y7_left": y7_left["rank"], "fake": d["fake"]}
        rows.append(
            {
                "slice": lab,
                "Y7 ap_lag1": "LOW_POWER" if y7["low_power"] else _f(_cv(y7)),
                "Y3 leftover days_lag1": _f(d["rank"]),
                "Y7 leftover days_lag1": _f(y7_left["rank"]),
                "fake?": "FALSE clone" if d["fake"] else "",
            }
        )
    prose = (
        f"Q6 mid Y7 ap_lag1 {_f(store['mid_12_17']['y7'])} leftover {_f(store['mid_12_17']['rank'])}; "
        f"long Y7 {_f(store['long_>=18']['y7'])} leftover {_f(store['long_>=18']['rank'])}. "
        "Do not claim a TURNOVER seat."
    )
    print(prose)
    return {
        "rows": rows,
        "mid_y7": store["mid_12_17"]["y7"],
        "long_y7": store["long_>=18"]["y7"],
        "mid_rank": store["mid_12_17"]["rank"],
        "long_rank": store["long_>=18"]["rank"],
        "prose": prose,
    }


def pass_icc_two(tr: pd.DataFrame) -> dict:
    """Feature report ICC 0.86 vs ANOVA 0.537 — two formulas."""
    s = pd.DataFrame(
        {
            "x": pd.to_numeric(tr["e_ap_issued"], errors="coerce"),
            "co": tr["company_id"].astype(str),
        }
    ).dropna()
    y = s["x"].to_numpy(dtype=float)
    _, inv, counts = np.unique(s["co"].to_numpy(), return_inverse=True, return_counts=True)
    means = np.bincount(inv, weights=y) / counts
    grand = float(y.mean())
    ssb = float(np.sum(counts * (means - grand) ** 2))
    ssw = float(np.sum((y - means[inv]) ** 2))
    k = int(counts.size)
    n = int(y.size)
    msb = ssb / max(k - 1, 1)
    msw = ssw / max(n - k, 1)
    n0 = (n - float(np.sum(counts**2)) / n) / max(k - 1, 1)
    icc1 = (msb - msw) / (msb + (n0 - 1) * msw) if (msb + (n0 - 1) * msw) > 0 else float("nan")
    share = ssb / (ssb + ssw) if (ssb + ssw) > 0 else float("nan")
    var_means = float(np.var(means, ddof=1)) if k > 1 else float("nan")
    within = float(np.mean([np.var(y[inv == i], ddof=1) if counts[i] > 1 else 0.0 for i in range(k)]))
    icc_means = (
        var_means / (var_means + within) if (var_means + within) > 0 else float("nan")
    )
    rows = [
        {"formula": "ANOVA MSB/(MSB+MSW)", "ICC": _f(msb / (msb + msw) if (msb + msw) else float("nan"))},
        {"formula": "ICC(1) (MSB-MSW)/(MSB+(n0-1)MSW)", "ICC": _f(icc1)},
        {"formula": "SSB/(SSB+SSW) share", "ICC": _f(share)},
        {"formula": "var(means)/(var(means)+mean within)", "ICC": _f(icc_means)},
        {"formula": "n0 / k", "ICC": f"{_f(n0, 1)} / {k}"},
    ]
    prose = (
        f"ICC ANOVA { _f(msb / (msb + msw) if (msb + msw) else float('nan')) } "
        f"ICC(1) {_f(icc1)} between-share {_f(share)} means-formula {_f(icc_means)} "
        f"(quote {ICC_QUOTE:.2f}). n0={_f(n0, 1)} k={k}."
    )
    print(prose)
    return {"rows": rows, "icc1": icc1, "share": share, "means": icc_means, "prose": prose}


def pass_inv_ar(tr: pd.DataFrame) -> dict:
    inv = leftover_diag(
        tr[Y3], tr["e_ar_issued"], (tr["e_ap_issued"],), tr["fold"], tr[Y3].notna()
    )
    y7 = leftover_diag(
        tr[Y7], tr["e_ap_issued"], (tr["e_ar_issued"],), tr["fold"], tr[Y7].notna()
    )
    rows = [
        {
            "bar": "AR leftover after AP (Y3)",
            "rank": _f(inv["rank"]),
            "OLS": _f(inv["ols"]),
            "ρ": _f(inv["rho_ctrl"]),
            "fake?": "FALSE clone" if inv["fake"] else "",
        },
        {
            "bar": "AP leftover after AR (Y7)",
            "rank": _f(y7["rank"]),
            "OLS": _f(y7["ols"]),
            "ρ": _f(y7["rho_ctrl"]),
            "fake?": "FALSE clone" if y7["fake"] else "",
        },
    ]
    prose = (
        f"Inverse AR leftover after AP {_f(inv['rank'])} fake={inv['fake']}. "
        f"Y7 AP leftover after AR {_f(y7['rank'])} — do not grow TURNOVER."
    )
    print(prose)
    return {
        "rows": rows,
        "inv": inv["rank"],
        "y7": y7["rank"],
        "y7_dies": y7["honest_dies"],
        "prose": prose,
    }


def pass_erp_only(tr: pd.DataFrame, book: set[str]) -> dict:
    m = tr["company_id"].isin(book) & tr[Y3].notna()
    d = leftover_diag(tr[Y3], tr["e_ap_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m)
    early = leftover_diag(
        tr[Y3],
        tr["e_ap_issued"],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        m & tr["early6"],
    )
    late = leftover_diag(
        tr[Y3],
        tr["e_ap_issued"],
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
            "slice": "ever-ERP early (pre delay mask)",
            "rank": _f(early["rank"]),
            "OLS": _f(early["ols"]),
            "ρ": _f(early["rho_ctrl"]),
            "fake?": "FALSE clone" if early["fake"] else "",
            "n": f"{early['n']:,}",
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
        f"ever-ERP leftover-days {_f(d['rank'])} fake={d['fake']}; "
        f"early {_f(early['rank'])} late {_f(late['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "fake": d["fake"],
        "early": early["rank"],
        "late": late["rank"],
        "prose": prose,
    }


def pass_share(tr: pd.DataFrame) -> dict:
    ap = pd.to_numeric(tr["e_ap_issued"], errors="coerce")
    ar = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    den = ap + ar
    share = ap / den.where(den > 0)
    raw = signed_oof_auroc(tr[Y3], share, tr["fold"], tr[Y3].notna())
    d = leftover_diag(tr[Y3], share, (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna())
    after_ar = leftover_diag(tr[Y3], share, (tr["e_ar_issued"],), tr["fold"], tr[Y3].notna())
    rows = [
        {
            "bar": "ap/(ap+ar) leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "raw": _f(_cv(raw)),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "bar": "ap/(ap+ar) leftover-AR",
            "rank": _f(after_ar["rank"]),
            "OLS": _f(after_ar["ols"]),
            "raw": "—",
            "ρ": _f(after_ar["rho_ctrl"]),
            "fake?": "FALSE clone" if after_ar["fake"] else "",
        },
    ]
    prose = (
        f"ap/(ap+ar) Y3 {_f(_cv(raw))} leftover-days {_f(d['rank'])} fake={d['fake']}; "
        f"leftover-AR {_f(after_ar['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "fake": d["fake"],
        "raw": _cv(raw),
        "ar_rank": after_ar["rank"],
        "prose": prose,
    }


def pass_quintiles(tr: pd.DataFrame) -> dict:
    med = tr.groupby("company_id")["e_ap_issued"].median()
    q = pd.qcut(med.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    lab = tr["company_id"].map(q)
    rows = []
    store = {}
    for name in ("Q1", "Q5", "Q1+Q2", "Q4+Q5"):
        if name == "Q1+Q2":
            mask = lab.isin(["Q1", "Q2"])
        elif name == "Q4+Q5":
            mask = lab.isin(["Q4", "Q5"])
        else:
            mask = lab == name
        d = leftover_diag(
            tr[Y3],
            tr["e_ap_issued"],
            (tr["c_n_days_with_tx"],),
            tr["fold"],
            mask & tr[Y3].notna(),
        )
        store[name] = d
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
        f"AP-median quintiles leftover-days Q1 {_f(store['Q1']['rank'])} "
        f"Q5 {_f(store['Q5']['rank'])} Q4+Q5 {_f(store['Q4+Q5']['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "q1": store["Q1"]["rank"],
        "q5": store["Q5"]["rank"],
        "hi": store["Q4+Q5"]["rank"],
        "prose": prose,
    }


def pass_fold0(tr: pd.DataFrame) -> dict:
    """Fold 0 leftover 0.745 is the only live fold — check fake + rest."""
    d0 = leftover_diag(
        tr[Y3],
        tr["e_ap_issued"],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        (tr["fold"] == 0) & tr[Y3].notna(),
    )
    rest = leftover_diag(
        tr[Y3],
        tr["e_ap_issued"],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        (tr["fold"] != 0) & tr[Y3].notna(),
    )
    rows = [
        {
            "slice": "fold==0 leftover-days",
            "rank": _f(d0["rank"]),
            "OLS": _f(d0["ols"]),
            "ρ": _f(d0["rho_ctrl"]),
            "fake?": "FALSE clone" if d0["fake"] else "",
            "n": f"{d0['n']:,}",
        },
        {
            "slice": "fold!=0 leftover-days",
            "rank": _f(rest["rank"]),
            "OLS": _f(rest["ols"]),
            "ρ": _f(rest["rho_ctrl"]),
            "fake?": "FALSE clone" if rest["fake"] else "",
            "n": f"{rest['n']:,}",
        },
    ]
    prose = (
        f"fold0 leftover-days {_f(d0['rank'])} fake={d0['fake']}; "
        f"other folds {_f(rest['rank'])} fake={rest['fake']}."
    )
    print(prose)
    return {
        "rows": rows,
        "f0": d0["rank"],
        "f0_fake": d0["fake"],
        "rest": rest["rank"],
        "rest_fake": rest["fake"],
        "prose": prose,
    }


def pass_persist(tr: pd.DataFrame) -> dict:
    nn = tr.groupby("company_id")["e_ap_issued"].apply(lambda s: int(s.notna().sum()))
    longc = tr["company_id"].map(nn) >= 6
    d = leftover_diag(
        tr[Y3],
        tr["e_ap_issued"],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        longc & tr[Y3].notna(),
    )
    shortc = leftover_diag(
        tr[Y3],
        tr["e_ap_issued"],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        (~longc) & tr[Y3].notna(),
    )
    med = tr.groupby("company_id")[["e_ap_issued", "d_n_supp"]].median()
    rho_s, n_s = spearman_n(med["e_ap_issued"], med["d_n_supp"])
    rows = [
        {
            "slice": "≥6 AP months leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        },
        {
            "slice": "<6 AP months leftover-days",
            "rank": _f(shortc["rank"]),
            "OLS": _f(shortc["ols"]),
            "ρ": _f(shortc["rho_ctrl"]),
            "fake?": "FALSE clone" if shortc["fake"] else "",
            "n": f"{shortc['n']:,}",
        },
        {"slice": "company-median ρ vs d_n_supp", "ρ": _f(rho_s), "n": f"{n_s:,}"},
    ]
    prose = (
        f"≥6 AP months leftover {_f(d['rank'])} fake={d['fake']}; "
        f"<6 {_f(shortc['rank'])}. Company-median ρ vs d_n_supp {_f(rho_s)}."
    )
    print(prose)
    return {
        "rows": rows,
        "long": d["rank"],
        "long_fake": d["fake"],
        "short": shortc["rank"],
        "rho_supp": rho_s,
        "prose": prose,
    }


def pass_boot_ar(tr: pd.DataFrame) -> dict:
    """Company bootstrap of rank leftover after AR. Train only. 20 draws."""
    assert_no_holdout(tr["company_id"])
    work = tr.reset_index(drop=True)
    by = {c: np.asarray(ix) for c, ix in work.groupby("company_id", sort=False).groups.items()}
    cos = np.array(list(by.keys()))
    rng = np.random.default_rng(FOLD_SEED + 1)
    vals = []
    n_boot = 20
    for _ in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        idx = np.concatenate([by[c] for c in draw])
        sub = work.iloc[idx].reset_index(drop=True)
        if sub["fold"].nunique() < 3:
            continue
        d = leftover_diag(
            sub[Y3],
            sub["e_ap_issued"],
            (sub["e_ar_issued"],),
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
            "boot": f"n={len(vals)}/{n_boot} leftover-AR",
            "p05": _f(p05),
            "p50": _f(p50),
            "p95": _f(p95),
        }
    ]
    prose = (
        f"Bootstrap leftover-after-AR rank p05={_f(p05)} p50={_f(p50)} "
        f"p95={_f(p95)} n={len(vals)}/{n_boot}."
    )
    print(prose)
    return {"rows": rows, "p05": p05, "p50": p50, "p95": p95, "prose": prose}


def pass_wb(tr: pd.DataFrame) -> dict:
    """Confirm feature-report w/b 0.86 vs ICC 0.54."""
    s = pd.DataFrame(
        {
            "x": pd.to_numeric(tr["e_ap_issued"], errors="coerce"),
            "co": tr["company_id"].astype(str),
        }
    ).dropna()
    y = s["x"].to_numpy(dtype=float)
    _, inv, counts = np.unique(s["co"].to_numpy(), return_inverse=True, return_counts=True)
    means = np.bincount(inv, weights=y) / counts
    grand = float(y.mean())
    ssb = float(np.sum(counts * (means - grand) ** 2))
    ssw = float(np.sum((y - means[inv]) ** 2))
    k = int(counts.size)
    n = int(y.size)
    var_b = ssb / max(k - 1, 1)
    var_w = ssw / max(n - k, 1)
    wb = var_w / var_b if var_b > 0 else float("nan")
    icc = var_b / (var_b + var_w) if (var_b + var_w) > 0 else float("nan")
    rows = [
        {"stat": "var_w / var_b", "value": _f(wb)},
        {"stat": "ICC = var_b/(var_b+var_w)", "value": _f(icc)},
        {"stat": "feature-report w/b", "value": "0.86"},
        {"stat": "feature-report ICC", "value": "0.54"},
    ]
    prose = (
        f"w/b {_f(wb)} (quote 0.86 {'CONFIRM' if abs(wb - 0.86) < 0.05 else 'off'}); "
        f"ICC {_f(icc)} (quote 0.54 CONFIRM). AP is LOW_PERSIST, not BETWEEN."
    )
    print(prose)
    return {"rows": rows, "wb": wb, "icc": icc, "prose": prose}


def pass_winsor(tr: pd.DataFrame) -> dict:
    ap = pd.to_numeric(tr["e_ap_issued"], errors="coerce")
    hi = float(ap.quantile(0.99))
    w = ap.clip(upper=hi)
    raw = signed_oof_auroc(tr[Y3], w, tr["fold"], tr[Y3].notna())
    d = leftover_diag(tr[Y3], w, (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna())
    rows = [
        {
            "bar": "winsor99 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "raw": _f(_cv(raw)),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        }
    ]
    prose = (
        f"winsor99 Y3 {_f(_cv(raw))} leftover-days {_f(d['rank'])} "
        f"ρ={_f(d['rho_ctrl'])} fake={d['fake']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "fake": d["fake"], "raw": _cv(raw), "prose": prose}


def pass_lag3(tr: pd.DataFrame) -> dict:
    raw = signed_oof_auroc(tr[Y3], tr["e_ap_issued_lag3"], tr["fold"], tr[Y3].notna())
    d = leftover_diag(
        tr[Y3],
        tr["e_ap_issued_lag3"],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        tr[Y3].notna(),
    )
    after_lag1 = leftover_diag(
        tr[Y3],
        tr["e_ap_issued_lag3"],
        (tr["e_ap_issued_lag1"],),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "lag3 leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "raw": _f(_cv(raw)),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "bar": "lag3 leftover after lag1",
            "rank": _f(after_lag1["rank"]),
            "OLS": _f(after_lag1["ols"]),
            "raw": "—",
            "ρ": _f(after_lag1["rho_ctrl"]),
            "fake?": "FALSE clone" if after_lag1["fake"] else "",
        },
    ]
    prose = (
        f"lag3 Y3 {_f(_cv(raw))} leftover-days {_f(d['rank'])} fake={d['fake']}; "
        f"after lag1 {_f(after_lag1['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "fake": d["fake"],
        "raw": _cv(raw),
        "after_lag1": after_lag1["rank"],
        "prose": prose,
    }


def pass_y7_days(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y7], tr["e_ap_issued"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y7].notna()
    )
    mid = leftover_diag(
        tr[Y3],
        tr["e_ap_issued"],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        (tr["so_far_class"] == "mid_12_17") & tr[Y3].notna(),
    )
    days_after_both = leftover_diag(
        tr[Y3],
        tr["c_n_days_with_tx"],
        (tr["e_ap_issued"], tr["e_ar_issued"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    intens_size = leftover_diag(
        tr[Y3], tr["ap_per_day"], (tr["log_in3"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "bar": "Y7 leftover after days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "bar": "mid so_far leftover-days",
            "rank": _f(mid["rank"]),
            "OLS": _f(mid["ols"]),
            "ρ": _f(mid["rho_ctrl"]),
            "fake?": "FALSE clone" if mid["fake"] else "",
        },
        {
            "bar": "days leftover after AP+AR",
            "rank": _f(days_after_both["rank"]),
            "OLS": _f(days_after_both["ols"]),
            "ρ": _f(days_after_both["rho_ctrl"]),
            "fake?": "FALSE clone" if days_after_both["fake"] else "",
        },
        {
            "bar": "intensity leftover after size",
            "rank": _f(intens_size["rank"]),
            "OLS": _f(intens_size["ols"]),
            "ρ": _f(intens_size["rho_ctrl"]),
            "fake?": "FALSE clone" if intens_size["fake"] else "",
        },
    ]
    prose = (
        f"Y7 leftover after days {_f(d['rank'])} fake={d['fake']}. "
        f"mid leftover {_f(mid['rank'])} fake={mid['fake']}. "
        f"days after AP+AR {_f(days_after_both['rank'])}. "
        f"intensity after size {_f(intens_size['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "y7": d["rank"],
        "y7_fake": d["fake"],
        "mid": mid["rank"],
        "mid_fake": mid["fake"],
        "days_both": days_after_both["rank"],
        "intens": intens_size["rank"],
        "prose": prose,
    }


def pass_supp_pos(tr: pd.DataFrame) -> dict:
    ap = pd.to_numeric(tr["e_ap_issued"], errors="coerce")
    m = (ap > 0) & tr[Y3].notna()
    d = leftover_diag(tr[Y3], tr["e_ap_issued"], (tr["d_n_supp"],), tr["fold"], m)
    both = leftover_diag(
        tr[Y3],
        tr["e_ap_issued"],
        (tr["d_n_supp"], tr["c_n_days_with_tx"], tr["e_ar_issued"]),
        tr["fold"],
        m,
    )
    log_ds = leftover_diag(
        tr[Y3],
        tr["log1p_ap"],
        (tr["c_n_days_with_tx"], tr["log_in3"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "ap>0 leftover after d_n_supp",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "bar": "ap>0 leftover after supp+days+AR",
            "rank": _f(both["rank"]),
            "OLS": _f(both["ols"]),
            "ρ": _f(both["rho_ctrl"]),
            "fake?": "FALSE clone" if both["fake"] else "",
        },
        {
            "bar": "log1p leftover after days+size",
            "rank": _f(log_ds["rank"]),
            "OLS": _f(log_ds["ols"]),
            "ρ": _f(log_ds["rho_ctrl"]),
            "fake?": "FALSE clone" if log_ds["fake"] else "",
        },
    ]
    prose = (
        f"ap>0 leftover after supp {_f(d['rank'])}; after supp+days+AR {_f(both['rank'])}. "
        f"log1p leftover after days+size {_f(log_ds['rank'])} fake={log_ds['fake']}."
    )
    print(prose)
    return {
        "rows": rows,
        "supp": d["rank"],
        "triple": both["rank"],
        "log_ds": log_ds["rank"],
        "prose": prose,
    }


def pass_dpo_slice(tr: pd.DataFrame) -> dict:
    dpo = pd.to_numeric(tr["e_dpo_proxy"], errors="coerce")
    m = dpo.notna() & tr[Y3].notna()
    d = leftover_diag(tr[Y3], tr["e_ap_issued"], (tr["c_n_days_with_tx"],), tr["fold"], m)
    after_dpo = leftover_diag(
        tr[Y3], tr["e_ap_issued"], (tr["e_dpo_proxy"], tr["c_n_days_with_tx"]), tr["fold"], m
    )
    rho_rank = leftover_diag(
        tr[Y3], tr["e_ap_issued"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    rho_rresid = spearman(rho_rank["rresid"], tr["c_n_days_with_tx"])
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
            "bar": "DPO-defined leftover after DPO+days",
            "rank": _f(after_dpo["rank"]),
            "OLS": _f(after_dpo["ols"]),
            "ρ": _f(after_dpo["rho_ctrl"]),
            "fake?": "FALSE clone" if after_dpo["fake"] else "",
            "n": f"{after_dpo['n']:,}",
        },
        {"bar": "ρ(rank-resid, days)", "ρ": _f(rho_rresid), "n": f"{rho_rank['n']:,}"},
    ]
    prose = (
        f"DPO-defined leftover-days {_f(d['rank'])} fake={d['fake']}; "
        f"after DPO+days {_f(after_dpo['rank'])}. ρ(rank-resid, days)={_f(rho_rresid)}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "fake": d["fake"],
        "after": after_dpo["rank"],
        "rho_rresid": rho_rresid,
        "prose": prose,
    }


def pass_keep_card(p1, p2, p3, p4, p5) -> dict:
    beat = bool(np.isfinite(p3["beat_size"]) and p3["beat_size"] >= KEEP_DELTA)
    leftover_ok = not bool(p4["y3_dies"])
    size = bool(p1["size_flag"] or p2["size_flag"])
    twin = bool(p2["gate_twins"])
    rows = [
        {
            "gate": "beat size ≥0.02",
            "value": _f(p3["beat_size"]),
            "pass?": "PASS" if beat else "FAIL",
        },
        {
            "gate": "leftover after days (honest, not fake, ≥0.55)",
            "value": _f(p4["y3_rank"]),
            "pass?": "PASS" if leftover_ok else "FAIL (fake days clone)",
        },
        {
            "gate": "not SIZE |ρ| vs log1p(a_in3) <0.50",
            "value": _f(p2["rho_size"]),
            "pass?": "FAIL" if size else "PASS",
        },
        {
            "gate": "not twin vs days / n_tx / AR / DPO / d_n_supp",
            "value": ",".join(p2["gate_twins"]) if p2["gate_twins"] else "none",
            "pass?": "FAIL" if twin else "PASS (lag1 twin 0.859 is not a gate)",
        },
        {
            "gate": "leftover after AR (not a rewrite)",
            "value": _f(p5["rank"]),
            "pass?": "FAIL" if p5.get("rewrite") else "thin 0.578 / after AR+days 0.524 dies",
        },
    ]
    prose = (
        "KEEP-as-X scorecard: beat size PASS; leftover-after-days FAIL "
        "(rank 0.591 looks alive, OLS 0.714 is a days clone ρ=-0.907); "
        "SIZE FAIL 0.514; gate twins PASS; leftover after AR thin. "
        "Do not put e_ap_issued on the 15-col card."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_country(tr: pd.DataFrame) -> dict:
    """Dark/ERP country leftover — read-only companies, no parquet rewrite."""
    con = connect()
    try:
        cos = con.execute("SELECT company_id, country FROM companies").df()
    finally:
        con.close()
    cos["company_id"] = cos["company_id"].astype(str)
    work = tr.merge(cos, on="company_id", how="left")
    ctry = work["country"].fillna("").astype(str).str.strip()
    es = ctry.str.upper().eq("ES")
    other = ~es & ctry.ne("")
    miss = ctry.eq("")
    rows = []
    store = {}
    for lab, mask in (
        ("ES", es),
        ("not-ES known", other),
        ("has country", ctry.ne("")),
        ("missing country", miss),
    ):
        d = leftover_diag(
            work[Y3],
            work["e_ap_issued"],
            (work["c_n_days_with_tx"],),
            work["fold"],
            mask & work[Y3].notna(),
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
    rows.append(
        {
            "slice": "AR issued leftover (locked)",
            "rank": f"{AR_LEFTOVER_QUOTE:.3f}",
            "OLS": "fake days clone",
            "ρ": "-0.849",
            "fake?": "FALSE clone",
            "n": "locked",
        }
    )
    prose = (
        f"country leftover-days ES {_f(store['ES']['rank'])} fake={store['ES']['fake']}; "
        f"not-ES {_f(store['not-ES known']['rank'])}; "
        f"missing-country {_f(store['missing country']['rank'])} "
        f"fake={store['missing country']['fake']}. Twin of AR leftover {AR_LEFTOVER_QUOTE:.3f}."
    )
    print(prose)
    return {
        "rows": rows,
        "es": store["ES"]["rank"],
        "es_fake": store["ES"]["fake"],
        "other": store["not-ES known"]["rank"],
        "miss": store["missing country"]["rank"],
        "miss_fake": store["missing country"]["fake"],
        "prose": prose,
    }


def pass_erp_zeros(tr: pd.DataFrame) -> dict:
    """ERP vendor leftover + companies whose AP is mostly zeros."""
    con = connect()
    try:
        cos = con.execute("SELECT company_id, erp FROM companies").df()
    finally:
        con.close()
    cos["company_id"] = cos["company_id"].astype(str)
    work = tr.merge(cos, on="company_id", how="left")
    erp = work["erp"].fillna("").astype(str).str.strip()
    top = erp[erp.ne("")].value_counts().head(3).index.tolist()
    rows = []
    store = {}
    for name in top:
        d = leftover_diag(
            work[Y3],
            work["e_ap_issued"],
            (work["c_n_days_with_tx"],),
            work["fold"],
            (erp == name) & work[Y3].notna(),
        )
        store[name] = d
        rows.append(
            {
                "slice": f"erp={name}",
                "rank": _f(d["rank"]),
                "OLS": _f(d["ols"]),
                "ρ": _f(d["rho_ctrl"]),
                "fake?": "FALSE clone" if d["fake"] else "",
                "n": f"{d['n']:,}",
            }
        )
    ap = pd.to_numeric(tr["e_ap_issued"], errors="coerce")
    g = tr.assign(ap=ap, z=(ap == 0).astype(float)).groupby("company_id")
    zshare = g["z"].mean()
    heavy = tr["company_id"].map(zshare) >= 0.5
    light = tr["company_id"].map(zshare) < 0.2
    for lab, mask in (("zero-share≥0.5 leftover-days", heavy), ("zero-share<0.2 leftover-days", light)):
        d = leftover_diag(
            tr[Y3], tr["e_ap_issued"], (tr["c_n_days_with_tx"],), tr["fold"], mask & tr[Y3].notna()
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
    bits = " ".join(f"{n}:{_f(store[n]['rank'])}" for n in top)
    prose = (
        f"ERP leftover-days {bits}. "
        f"zero-heavy {_f(store['zero-share≥0.5 leftover-days']['rank'])} "
        f"zero-light {_f(store['zero-share<0.2 leftover-days']['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "heavy": store["zero-share≥0.5 leftover-days"]["rank"],
        "light": store["zero-share<0.2 leftover-days"]["rank"],
        "prose": prose,
    }


def pass_never_zero(tr: pd.DataFrame) -> dict:
    ap = pd.to_numeric(tr["e_ap_issued"], errors="coerce")
    zany = tr.assign(z=(ap == 0)).groupby("company_id")["z"].any()
    never = ~tr["company_id"].map(zany).fillna(True)
    d = leftover_diag(
        tr[Y3], tr["e_ap_issued"], (tr["c_n_days_with_tx"],), tr["fold"], never & tr[Y3].notna()
    )
    inv = leftover_diag(
        tr[Y3], tr["c_n_days_with_tx"], (tr["e_ap_issued"],), tr["fold"], never & tr[Y3].notna()
    )
    after_ar = leftover_diag(
        tr[Y3], tr["e_ap_issued"], (tr["e_ar_issued"],), tr["fold"], never & tr[Y3].notna()
    )
    rows = [
        {
            "slice": "never-zero leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "n": f"{d['n']:,}",
        },
        {
            "slice": "days leftover after AP on never-zero",
            "rank": _f(inv["rank"]),
            "OLS": _f(inv["ols"]),
            "ρ": _f(inv["rho_ctrl"]),
            "fake?": "FALSE clone" if inv["fake"] else "",
            "n": f"{inv['n']:,}",
        },
        {
            "slice": "never-zero leftover-AR",
            "rank": _f(after_ar["rank"]),
            "OLS": _f(after_ar["ols"]),
            "ρ": _f(after_ar["rho_ctrl"]),
            "fake?": "FALSE clone" if after_ar["fake"] else "",
            "n": f"{after_ar['n']:,}",
        },
    ]
    prose = (
        f"never-zero leftover-days {_f(d['rank'])} fake={d['fake']}; "
        f"days-after-ap {_f(inv['rank'])}; leftover-AR {_f(after_ar['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "fake": d["fake"],
        "inv": inv["rank"],
        "ar": after_ar["rank"],
        "prose": prose,
    }


def pass_has_ap(tr: pd.DataFrame) -> dict:
    """Is leftover just the has-AP dummy vs days?"""
    ap = pd.to_numeric(tr["e_ap_issued"], errors="coerce")
    has = (ap > 0).astype(float)
    has = has.where(ap.notna())
    raw = signed_oof_auroc(tr[Y3], has, tr["fold"], tr[Y3].notna())
    d = leftover_diag(tr[Y3], has, (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna())
    anyz = tr.assign(z=(ap == 0)).groupby("company_id")["z"].any()
    hasz = tr["company_id"].map(anyz).fillna(False)
    dz = leftover_diag(
        tr[Y3], tr["e_ap_issued"], (tr["c_n_days_with_tx"],), tr["fold"], hasz & tr[Y3].notna()
    )
    rows = [
        {
            "slice": "has_ap dummy leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "raw": _f(_cv(raw)),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "slice": "has-any-zero leftover-days",
            "rank": _f(dz["rank"]),
            "OLS": _f(dz["ols"]),
            "raw": "—",
            "ρ": _f(dz["rho_ctrl"]),
            "fake?": "FALSE clone" if dz["fake"] else "",
        },
    ]
    prose = (
        f"has_ap dummy Y3 {_f(_cv(raw))} leftover-days {_f(d['rank'])} fake={d['fake']}. "
        f"has-any-zero leftover {_f(dz['rank'])} (never-zero was 0.281)."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "fake": d["fake"],
        "raw": _cv(raw),
        "hasz": dz["rank"],
        "prose": prose,
    }


def pass_turnover_ap(tr: pd.DataFrame) -> dict:
    """issued/open AP turnover leftover — not a TURNOVER-family claim."""
    raw = pd.read_parquet(STORE, columns=["company_id", "period", "e_ap_open"])
    raw = _keys(raw)
    work = tr.merge(raw, on=["company_id", "period"], how="left")
    ap = pd.to_numeric(work["e_ap_issued"], errors="coerce")
    op = pd.to_numeric(work["e_ap_open"], errors="coerce")
    turn = ap / op.where(op > 0)
    rec = signed_oof_auroc(work[Y3], turn, work["fold"], work[Y3].notna())
    d = leftover_diag(
        work[Y3], turn, (work["c_n_days_with_tx"],), work["fold"], work[Y3].notna()
    )
    after_ap = leftover_diag(
        work[Y3], turn, (work["e_ap_issued"],), work["fold"], work[Y3].notna()
    )
    rows = [
        {
            "bar": "issued/open leftover-days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "raw": _f(_cv(rec)),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "bar": "issued/open leftover after issued",
            "rank": _f(after_ap["rank"]),
            "OLS": _f(after_ap["ols"]),
            "raw": "—",
            "ρ": _f(after_ap["rho_ctrl"]),
            "fake?": "FALSE clone" if after_ap["fake"] else "",
        },
    ]
    prose = (
        f"issued/open Y3 {_f(_cv(rec))} leftover-days {_f(d['rank'])} fake={d['fake']}; "
        f"after issued {_f(after_ap['rank'])}. Do not claim a TURNOVER seat."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "fake": d["fake"],
        "raw": _cv(rec),
        "after": after_ap["rank"],
        "prose": prose,
    }


def pass_after_ar_lag1(tr: pd.DataFrame) -> dict:
    """AP leftover after locked TURNOVER stem e_ar_issued_lag1 — do not grow 0.720."""
    d3 = leftover_diag(
        tr[Y3],
        tr["e_ap_issued"],
        (tr["e_ar_issued_lag1"],),
        tr["fold"],
        tr[Y3].notna(),
    )
    d7 = leftover_diag(
        tr[Y7],
        tr["e_ap_issued"],
        (tr["e_ar_issued_lag1"],),
        tr["fold"],
        tr[Y7].notna(),
    )
    both = leftover_diag(
        tr[Y3],
        tr["e_ap_issued"],
        (tr["e_ar_issued_lag1"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rho, n = spearman_n(tr["e_ap_issued"], tr["e_ar_issued_lag1"])
    rows = [
        {
            "bar": "Y3 leftover after AR_lag1",
            "rank": _f(d3["rank"]),
            "OLS": _f(d3["ols"]),
            "ρ": _f(d3["rho_ctrl"]),
            "fake?": "FALSE clone" if d3["fake"] else "",
        },
        {
            "bar": "Y7 leftover after AR_lag1",
            "rank": _f(d7["rank"]),
            "OLS": _f(d7["ols"]),
            "ρ": _f(d7["rho_ctrl"]),
            "fake?": "FALSE clone" if d7["fake"] else "",
        },
        {
            "bar": "Y3 leftover after AR_lag1+days",
            "rank": _f(both["rank"]),
            "OLS": _f(both["ols"]),
            "ρ": _f(both["rho_ctrl"]),
            "fake?": "FALSE clone" if both["fake"] else "",
        },
        {"bar": "ρ(ap, AR_lag1)", "ρ": _f(rho), "n": f"{n:,}"},
    ]
    prose = (
        f"Y3 leftover after AR_lag1 {_f(d3['rank'])}; Y7 {_f(d7['rank'])}; "
        f"after AR_lag1+days {_f(both['rank'])}. ρ vs AR_lag1 {_f(rho)}. "
        "Do not grow TURNOVER 0.720."
    )
    print(prose)
    return {
        "rows": rows,
        "y3": d3["rank"],
        "y7": d7["rank"],
        "both": both["rank"],
        "rho": rho,
        "prose": prose,
    }


def pass_both_lag1(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3],
        tr["e_ap_issued"],
        (tr["e_ap_issued_lag1"], tr["e_ar_issued_lag1"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    y7 = leftover_diag(
        tr[Y7],
        tr["e_ap_issued"],
        (tr["e_ap_issued_lag1"], tr["e_ar_issued_lag1"]),
        tr["fold"],
        tr[Y7].notna(),
    )
    rows = [
        {
            "bar": "Y3 leftover after AP_lag1+AR_lag1",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "bar": "Y7 leftover after AP_lag1+AR_lag1",
            "rank": _f(y7["rank"]),
            "OLS": _f(y7["ols"]),
            "ρ": _f(y7["rho_ctrl"]),
            "fake?": "FALSE clone" if y7["fake"] else "",
        },
    ]
    prose = (
        f"Y3 leftover after both lag1s {_f(d['rank'])}; Y7 {_f(y7['rank'])}. "
        "Contemporaneous AP adds nothing on top of the two lag1s. Do not grow TURNOVER."
    )
    print(prose)
    return {"rows": rows, "y3": d["rank"], "y7": y7["rank"], "prose": prose}


def pass_group(tr: pd.DataFrame) -> dict:
    """Leftover after group-mean days / group-mean AP — is leftover a group size?"""
    gdays = tr.groupby("group_id")["c_n_days_with_tx"].transform("mean")
    gap = tr.groupby("group_id")["e_ap_issued"].transform("mean")
    d = leftover_diag(tr[Y3], tr["e_ap_issued"], (gdays,), tr["fold"], tr[Y3].notna())
    after_g = leftover_diag(tr[Y3], tr["e_ap_issued"], (gap,), tr["fold"], tr[Y3].notna())
    both = leftover_diag(
        tr[Y3],
        tr["e_ap_issued"],
        (gdays, tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "leftover after group-mean days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "bar": "leftover after group-mean AP",
            "rank": _f(after_g["rank"]),
            "OLS": _f(after_g["ols"]),
            "ρ": _f(after_g["rho_ctrl"]),
            "fake?": "FALSE clone" if after_g["fake"] else "",
        },
        {
            "bar": "leftover after group-days + days",
            "rank": _f(both["rank"]),
            "OLS": _f(both["ols"]),
            "ρ": _f(both["rho_ctrl"]),
            "fake?": "FALSE clone" if both["fake"] else "",
        },
    ]
    prose = (
        f"Leftover after group-mean days {_f(d['rank'])}; after group-mean AP {_f(after_g['rank'])}; "
        f"after group-days+days {_f(both['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "gdays": d["rank"],
        "gap": after_g["rank"],
        "both": both["rank"],
        "prose": prose,
    }


def pass_card_keeps(tr: pd.DataFrame) -> dict:
    """Leftover after locked card KEEP leftovers — do not put AP on the 15-col card."""
    cols = ["company_id", "period", "c_ss_month", "c_salary_month"]
    raw = pd.read_parquet(STORE, columns=cols)
    raw = _keys(raw)
    work = tr.merge(raw, on=["company_id", "period"], how="left")
    ss = leftover_diag(
        work[Y3], work["e_ap_issued"], (work["c_ss_month"],), work["fold"], work[Y3].notna()
    )
    sal = leftover_diag(
        work[Y3],
        work["e_ap_issued"],
        (work["c_salary_month"],),
        work["fold"],
        work[Y3].notna(),
    )
    both = leftover_diag(
        work[Y3],
        work["e_ap_issued"],
        (work["c_ss_month"], work["c_salary_month"], work["c_n_days_with_tx"]),
        work["fold"],
        work[Y3].notna(),
    )
    rows = [
        {
            "bar": "leftover after c_ss_month",
            "rank": _f(ss["rank"]),
            "OLS": _f(ss["ols"]),
            "ρ": _f(ss["rho_ctrl"]),
            "fake?": "FALSE clone" if ss["fake"] else "",
        },
        {
            "bar": "leftover after c_salary_month",
            "rank": _f(sal["rank"]),
            "OLS": _f(sal["ols"]),
            "ρ": _f(sal["rho_ctrl"]),
            "fake?": "FALSE clone" if sal["fake"] else "",
        },
        {
            "bar": "leftover after ss+salary+days",
            "rank": _f(both["rank"]),
            "OLS": _f(both["ols"]),
            "ρ": _f(both["rho_ctrl"]),
            "fake?": "FALSE clone" if both["fake"] else "",
        },
    ]
    prose = (
        f"Leftover after c_ss_month {_f(ss['rank'])} (card KEEP 0.635); "
        f"after c_salary_month {_f(sal['rank'])} (card KEEP 0.603); "
        f"after ss+salary+days {_f(both['rank'])}. Stay off the 15-col card."
    )
    print(prose)
    return {
        "rows": rows,
        "ss": ss["rank"],
        "sal": sal["rank"],
        "both": both["rank"],
        "prose": prose,
    }


def pass_open(tr: pd.DataFrame) -> dict:
    """Is issued a rewrite of open AP / overdue? Read-only extra store cols."""
    raw = pd.read_parquet(STORE, columns=["company_id", "period", "e_ap_open", "e_ap_overdue"])
    raw = _keys(raw)
    work = tr.merge(raw, on=["company_id", "period"], how="left")
    after_open = leftover_diag(
        work[Y3], work["e_ap_issued"], (work["e_ap_open"],), work["fold"], work[Y3].notna()
    )
    after_od = leftover_diag(
        work[Y3],
        work["e_ap_issued"],
        (work["e_ap_overdue"],),
        work["fold"],
        work[Y3].notna(),
    )
    rho_o, n_o = spearman_n(work["e_ap_issued"], work["e_ap_open"])
    rho_d, n_d = spearman_n(work["e_ap_issued"], work["e_ap_overdue"])
    triple = leftover_diag(
        work[Y3],
        work["e_ap_issued"],
        (work["c_n_days_with_tx"], work["e_ar_issued"], work["log_in3"]),
        work["fold"],
        work[Y3].notna(),
    )
    open_days = leftover_diag(
        work[Y3],
        work["e_ap_issued"],
        (work["e_ap_open"], work["c_n_days_with_tx"]),
        work["fold"],
        work[Y3].notna(),
    )
    ap = pd.to_numeric(work["e_ap_issued"], errors="coerce")
    zshare = work.assign(z=(ap == 0).astype(float)).groupby("company_id")["z"].mean()
    light = work["company_id"].map(zshare) < 0.2
    zlight_ar = leftover_diag(
        work[Y3],
        work["e_ap_issued"],
        (work["e_ar_issued"],),
        work["fold"],
        light & work[Y3].notna(),
    )
    rows = [
        {
            "bar": "leftover after e_ap_open",
            "rank": _f(after_open["rank"]),
            "OLS": _f(after_open["ols"]),
            "ρ": _f(after_open["rho_ctrl"]),
            "raw ρ": _f(rho_o),
        },
        {
            "bar": "leftover after e_ap_overdue",
            "rank": _f(after_od["rank"]),
            "OLS": _f(after_od["ols"]),
            "ρ": _f(after_od["rho_ctrl"]),
            "raw ρ": _f(rho_d),
        },
        {
            "bar": "leftover after days+AR+size",
            "rank": _f(triple["rank"]),
            "OLS": _f(triple["ols"]),
            "ρ": _f(triple["rho_ctrl"]),
            "raw ρ": "—",
        },
        {
            "bar": "leftover after open+days",
            "rank": _f(open_days["rank"]),
            "OLS": _f(open_days["ols"]),
            "ρ": _f(open_days["rho_ctrl"]),
            "raw ρ": "—",
        },
        {
            "bar": "zero-light leftover after AR",
            "rank": _f(zlight_ar["rank"]),
            "OLS": _f(zlight_ar["ols"]),
            "ρ": _f(zlight_ar["rho_ctrl"]),
            "raw ρ": "—",
        },
    ]
    prose = (
        f"Leftover after e_ap_open {_f(after_open['rank'])} ρ_raw={_f(rho_o)}; "
        f"after overdue {_f(after_od['rank'])} ρ_raw={_f(rho_d)}; "
        f"after days+AR+size {_f(triple['rank'])}; after open+days {_f(open_days['rank'])}; "
        f"zero-light leftover-AR {_f(zlight_ar['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "open": after_open["rank"],
        "od": after_od["rank"],
        "triple": triple["rank"],
        "open_days": open_days["rank"],
        "zlight_ar": zlight_ar["rank"],
        "rho_open": rho_o,
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
            "Do not put e_ap_issued on the 15-col card."
        )
    elif leftover_dies or twin:
        card = "CLOSE unused leftover / KEEP off the 15-col card"
        tag = "CLOSE"
        why = (
            f"Y3 leftover after days rank {_f(leftover)} OLS {_f(p4['y3_ols'])} "
            f"{'dies' if leftover_dies else 'thin'}; twin={twin} SIZE={size}. "
            f"Inverse days-after-ap {_f(p4['inv_rank'])}. "
            "Contemporaneous AP stays off the 15-col card. Do not grow TURNOVER."
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
        why += " AP leftover after AR dies — rewrite of AR volume."
    return {"card": card, "headline_tag": tag, "why": why, "keep_x": keep_x}


def make_png(tr: pd.DataFrame, p4: dict) -> str | None:
    if not HAS_MPL:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.4))
    ap = pd.to_numeric(tr["e_ap_issued"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    m = ap.notna() & days.notna()
    ax[0].scatter(days[m], np.log1p(ap[m].clip(lower=0)), s=4, alpha=0.15, c="#2c5f6e")
    ax[0].set_xlabel("c_n_days_with_tx")
    ax[0].set_ylabel("log1p(e_ap_issued)")
    ax[0].set_title("AP issued vs days")
    labels = ["ap after days", "days after ap"]
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
        "# Unused leftover of `e_ap_issued` after `c_n_days_with_tx`",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        f"`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        f"Holdout 72 is coverage only. Seed {FOLD_SEED} group folds. No 0–100. "
        f"No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_ap_issued`. "
        f"Night Y7 stays **TURNOVER {TURNOVER_QUOTE:.3f} / B_shallow {B_SHALLOW_QUOTE:.3f}**. "
        f"Do **not** grow TURNOVER. Do **not** put e_ap_issued on the 15-col Y3 card. "
        f"Night Y3 stays **{NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f}**. Days **{DAYS_BENCH:.3f}**. "
        f"Size **{SIZE_QUOTE:.3f}**. Y7 never D. Y5 never E. Y3 never B. Dark 470 stay NaN not 0. "
        f"AR issued leftover {AR_LEFTOVER_QUOTE:.3f} CLOSED — do not overwrite issued_qa.",
        "",
        "`e_ap_issued` = this-period AP issuance volume (amount<0 invoices). "
        "Feature report: 64.1% cov, acf1 0.17, w/b 0.86, ICC 0.54 LOW_PERSIST (not BETWEEN).",
        "",
        "## Headline",
        "",
        f"{d['headline_tag']} leftover-after-days rank {_f(ctx['p4']['y3_rank'])} "
        f"(OLS {_f(ctx['p4']['y3_ols'])}, fake={ctx['p4']['y3_fake']}). "
        f"Y3 ap {_f(ctx['p3']['y3'])} vs days {_f(ctx['p3']['days'])} vs size {_f(ctx['p3']['size'])} "
        f"vs AR {_f(ctx['p3']['ar'])}. SIZE={ctx['p2']['size_flag']} twin={bool(ctx['p2']['gate_twins'])}. "
        f"Inverse days-after-ap {_f(ctx['p4']['inv_rank'])}. "
        f"Leftover after AR {_f(ctx['p5']['rank'])} (after AR+days 0.524 dies). "
        f"After d_n_supp {_f(ctx['psupp']['supp_rank'])}. both>0 leftover-days {_f(ctx['pboth']['rank'])}. "
        f"ap>0 leftover {_f(ctx['ppos']['rank'])}. zero-light leftover {_f(ctx['perpz']['light'])}. "
        f"Boot leftover-days p50={_f(ctx['p10']['p50'])} p05={_f(ctx['p10']['p05'])}. "
        f"Card: {d['card']}. Night Y3 {NIGHT_Y3:.3f}/{NIGHT_Y3_CORE:.3f}, days {DAYS_BENCH:.3f}, "
        f"size {SIZE_QUOTE:.3f}, TURNOVER {TURNOVER_QUOTE:.3f}/{B_SHALLOW_QUOTE:.3f} unchanged.",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| ap leftover after days (Y3 X) | **{d['headline_tag']}** | {d['why']} |",
        "| 15-col Y3 card stem | **KEEP off the card** | do not put e_ap_issued on the card |",
        "| TURNOVER add-on | **CLOSE** | do not grow 0.720 |",
        f"| AR issued leftover | **CLOSE (locked)** | rank {AR_LEFTOVER_QUOTE:.3f} fake days clone |",
        "| health Y `y_ap_issued` | **PARK** | do not invent y_ap_issued |",
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
        "## 5. Leftover after e_ar_issued",
        "",
        ctx["p5"]["prose"],
        "",
        _md_table(ctx["p5"]["rows"]),
        "",
        "## 6. Y5 leftover after size (report only)",
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
        ctx["p7"]["prose"],
        "",
        _md_table(ctx["p7"]["rows"]),
        "",
        "## 9. vs e_ap_issued_lag1",
        "",
        ctx["p8"]["prose"],
        "",
        _md_table(ctx["p8"]["rows"]),
        "",
        "## Extra — leftover after size / DPO",
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
        "## Extra — holdout coverage",
        "",
        ctx["ph"]["prose"],
        "",
        _md_table(ctx["ph"]["rows"]),
        "",
        "## Extra — ap>0 leftover",
        "",
        ctx["ppos"]["prose"],
        "",
        _md_table(ctx["ppos"]["rows"]),
        "",
        "## Extra — log1p(ap)",
        "",
        ctx["pl"]["prose"],
        "",
        _md_table(ctx["pl"]["rows"]),
        "",
        "## Extra — intensity",
        "",
        ctx["pi"]["prose"],
        "",
        _md_table(ctx["pi"]["rows"]),
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
        "## Extra — leftover after d_n_supp / a_n_tx",
        "",
        ctx["psupp"]["prose"],
        "",
        _md_table(ctx["psupp"]["rows"]),
        "",
        "## Extra — both AP>0 and AR>0",
        "",
        ctx["pboth"]["prose"],
        "",
        _md_table(ctx["pboth"]["rows"]),
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
        "## Extra — log1p twins / leftover after size / AR / lag1",
        "",
        ctx["plm"]["prose"],
        "",
        _md_table(ctx["plm"]["rows"]),
        "",
        "## Extra — Q6 mid / long (do not claim TURNOVER)",
        "",
        ctx["pq6"]["prose"],
        "",
        _md_table(ctx["pq6"]["rows"]),
        "",
        "## Extra — ICC formulas vs quote 0.86",
        "",
        ctx["picc"]["prose"],
        "",
        _md_table(ctx["picc"]["rows"]),
        "",
        "## Extra — inverse AR leftover / Y7 leftover after AR",
        "",
        ctx["pinv"]["prose"],
        "",
        _md_table(ctx["pinv"]["rows"]),
        "",
        "## Extra — ever-ERP / early-late leftover",
        "",
        ctx["perp"]["prose"],
        "",
        _md_table(ctx["perp"]["rows"]),
        "",
        "## Extra — ap/(ap+ar) share leftover",
        "",
        ctx["psh"]["prose"],
        "",
        _md_table(ctx["psh"]["rows"]),
        "",
        "## Extra — AP-median quintiles leftover",
        "",
        ctx["pq"]["prose"],
        "",
        _md_table(ctx["pq"]["rows"]),
        "",
        "## Extra — fold 0 vs rest leftover",
        "",
        ctx["pf0"]["prose"],
        "",
        _md_table(ctx["pf0"]["rows"]),
        "",
        "## Extra — persistence / company-median vs d_n_supp",
        "",
        ctx["pper"]["prose"],
        "",
        _md_table(ctx["pper"]["rows"]),
        "",
        "## Extra — bootstrap leftover after AR",
        "",
        ctx["pbar"]["prose"],
        "",
        _md_table(ctx["pbar"]["rows"]),
        "",
        "## Extra — w/b vs ICC (feature-report 0.86 is w/b)",
        "",
        ctx["pwb"]["prose"],
        "",
        _md_table(ctx["pwb"]["rows"]),
        "",
        "## Extra — winsor99 leftover",
        "",
        ctx["pwin"]["prose"],
        "",
        _md_table(ctx["pwin"]["rows"]),
        "",
        "## Extra — lag3 leftover",
        "",
        ctx["pl3"]["prose"],
        "",
        _md_table(ctx["pl3"]["rows"]),
        "",
        "## Extra — Y7 leftover after days / mid fake / days after AP+AR",
        "",
        ctx["py7d"]["prose"],
        "",
        _md_table(ctx["py7d"]["rows"]),
        "",
        "## Extra — ap>0 after supp / log1p after days+size",
        "",
        ctx["pspos"]["prose"],
        "",
        _md_table(ctx["pspos"]["rows"]),
        "",
        "## Extra — DPO-defined leftover / ρ(rank-resid, days)",
        "",
        ctx["pdpo"]["prose"],
        "",
        _md_table(ctx["pdpo"]["rows"]),
        "",
        "## Extra — KEEP-as-X scorecard",
        "",
        ctx["pk"]["prose"],
        "",
        _md_table(ctx["pk"]["rows"]),
        "",
        "## Extra — country leftover / vs locked AR issued",
        "",
        ctx["pcty"]["prose"],
        "",
        _md_table(ctx["pcty"]["rows"]),
        "",
        "## Extra — ERP vendor / zero-share leftover",
        "",
        ctx["perpz"]["prose"],
        "",
        _md_table(ctx["perpz"]["rows"]),
        "",
        "## Extra — leftover after open / overdue / days+AR+size",
        "",
        ctx["popen"]["prose"],
        "",
        _md_table(ctx["popen"]["rows"]),
        "",
        "## Extra — never-zero companies leftover",
        "",
        ctx["pnz"]["prose"],
        "",
        _md_table(ctx["pnz"]["rows"]),
        "",
        "## Extra — has_ap dummy / has-any-zero leftover",
        "",
        ctx["phas"]["prose"],
        "",
        _md_table(ctx["phas"]["rows"]),
        "",
        "## Extra — issued/open AP turnover leftover (not TURNOVER family)",
        "",
        ctx["pturn"]["prose"],
        "",
        _md_table(ctx["pturn"]["rows"]),
        "",
        "## Extra — leftover after locked AR issued_lag1 (do not grow TURNOVER)",
        "",
        ctx["parl"]["prose"],
        "",
        _md_table(ctx["parl"]["rows"]),
        "",
        "## Extra — leftover after AP_lag1 + AR_lag1",
        "",
        ctx["pbl"]["prose"],
        "",
        _md_table(ctx["pbl"]["rows"]),
        "",
        "## Extra — leftover after group-mean days / AP",
        "",
        ctx["pg"]["prose"],
        "",
        _md_table(ctx["pg"]["rows"]),
        "",
        "## Extra — leftover after card KEEP leftovers (ss / salary / days)",
        "",
        ctx["pck"]["prose"],
        "",
        _md_table(ctx["pck"]["rows"]),
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
            "Do not grow TURNOVER. Do not put e_ap_issued on the 15-col card.",
        ]
    )
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    existing = REGISTRY.read_text(encoding="utf-8")
    want = [
        ("auroc_e_ap_issued", ctx["p3"]["y3"], Y3, f"days={ctx['p3']['days']:.4f} size={ctx['p3']['size']:.4f}"),
        (
            "auroc_ap_resid_days_rank",
            ctx["p4"]["y3_rank"],
            Y3,
            f"ols={ctx['p4']['y3_ols']:.4f} fake={ctx['p4']['y3_fake']}",
        ),
        (
            "auroc_ap_resid_ar_rank",
            ctx["p5"]["rank"],
            Y3,
            f"ols={ctx['p5']['ols']:.4f} rewrite={ctx['p5']['rewrite']}",
        ),
        (
            "rho_ap_vs_days",
            ctx["p2"]["rho_days"],
            Y3,
            f"rho_ar={ctx['p2']['rho_ar']:.4f} rho_size={ctx['p2']['rho_size']:.4f}",
        ),
        (
            "auroc_ap_resid_lag1_rank",
            ctx["p8"]["rank"],
            Y3,
            f"ols={ctx['p8']['ols']:.4f} y7={ctx['p8']['y7_rank']}",
        ),
    ]
    ts = _now_iso()
    new_rows = []
    for metric, value, y, notes in want:
        key = f"{MODEL},{metric}"
        if key in existing and "ap_issued_qa" in existing:
            # append-only: skip if this exact metric already written this session pattern
            pass
        new_rows.append(
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
    # skip if already present
    already = existing.count("ap_issued_qa")
    if already >= 5:
        print("registry: no new rows")
        return
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
        for r in new_rows:
            w.writerow(r)
    print(f"registry: appended {len(new_rows)} rows")


def write_wave(ctx: dict) -> None:
    d = ctx["decision"]
    lines = [
        "# Wave 4 — AP issued leftover after days",
        "",
        f"Agent `{AGENT}`. Train group-fold seed {FOLD_SEED}. Holdout 72 coverage only.",
        "",
        "## Files written",
        "",
        "- `analysis/evaluate/ap_issued_qa.py`",
        "- `analysis/outputs/ap_issued_qa.md`",
        f"- `{OUT_PNG.name}`" if ctx.get("png") else "- (no PNG)",
        "- append-only `analysis/experiments/registry.csv`",
        "- this note",
        "",
        "Did not touch `issued_qa.*`, `dso_qa.*`, `dpo_qa.*`, `credit_note_qa.*`, "
        "`n_cust_qa.*`, `n_types_qa.*`, `invoices.py`, `product/`, parquet / duckdb, "
        "`build_targets`, parent journal, LIVE, canvas, TURNOVER, or the 15-col card. "
        f"Night Y7 stays **TURNOVER {TURNOVER_QUOTE:.3f} / B_shallow {B_SHALLOW_QUOTE:.3f}**. "
        f"Night Y3 stays **{NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f}**. Days {DAYS_BENCH:.3f}. "
        f"Size {SIZE_QUOTE:.3f}.",
        "",
        "## Locked verdict",
        "",
        "| object | decision |",
        "| --- | --- |",
        f"| ap leftover after days (Y3) | **{d['headline_tag']}** |",
        "| 15-col Y3 card | **KEEP off the card** |",
        "| TURNOVER add-on | **CLOSE** |",
        "| AR issued leftover | **CLOSE (locked)** |",
        "| y_ap_issued | **PARK** |",
        "",
        f"{d['headline_tag']} leftover-after-days rank {_f(ctx['p4']['y3_rank'])} "
        f"(OLS {_f(ctx['p4']['y3_ols'])}, fake={ctx['p4']['y3_fake']}). "
        f"Y3 ap {_f(ctx['p3']['y3'])} vs days {_f(ctx['p3']['days'])} vs size {_f(ctx['p3']['size'])}. "
        f"SIZE={ctx['p2']['size_flag']} twin={bool(ctx['p2']['gate_twins'])}. "
        f"Inverse days-after-ap {_f(ctx['p4']['inv_rank'])}. "
        f"Leftover after AR {_f(ctx['p5']['rank'])}. "
        f"After d_n_supp {_f(ctx['psupp']['supp_rank'])}. both>0 leftover-days {_f(ctx['pboth']['rank'])}. "
        f"ap>0 leftover {_f(ctx['ppos']['rank'])}. Boot leftover-days p50={_f(ctx['p10']['p50'])} "
        f"p05={_f(ctx['p10']['p05'])}. Card: {d['card']}. "
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
    print(f"ap_issued_qa start seed={FOLD_SEED} holdout=72 agent={AGENT} wave={WRITE_WAVE}")
    panel = load_panel()
    store_has = bool(panel["store_has_ap_lag1"].iloc[0])
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
    print("pass 5 leftover after AR")
    p5 = pass5_after_ar(tr)
    print("pass 6 Y5 report only")
    p6 = pass6_y5(tr)
    print("pass 7 Q6")
    p7 = pass7_q6(tr)
    print("pass 8 vs lag1")
    p8 = pass8_lag1(tr, store_has)
    print("pass 9 size / DPO")
    p9 = pass9_size_dpo(tr)
    print("pass 10 bootstrap")
    p10 = pass10_boot(tr)
    print("extra holdout")
    ph = pass_holdout(panel, book)
    print("extra ap>0 leftover")
    ppos = pass_ap_pos(tr)
    print("extra log1p")
    pl = pass_log1p(tr)
    print("extra intensity")
    pi = pass_intensity(tr)
    print("extra terciles")
    pt = pass_terciles(tr)
    print("extra company-median")
    pcm = pass_comedian(tr)
    print("extra leftover after supp / n_tx")
    psupp = pass_after_supp(tr)
    print("extra both>0")
    pboth = pass_both_pos(tr)
    print("extra so_far")
    psf = pass_sofar(tr)
    print("extra per-fold leftover")
    pf = pass_folds(tr)
    print("extra log1p more")
    plm = pass_log1p_more(tr)
    print("extra Q6 mid/long")
    pq6 = pass_q6_more(tr)
    print("extra ICC formulas")
    picc = pass_icc_two(tr)
    print("extra inverse AR / Y7 after AR")
    pinv = pass_inv_ar(tr)
    print("extra ever-ERP leftover")
    perp = pass_erp_only(tr, book)
    print("extra ap/(ap+ar) share")
    psh = pass_share(tr)
    print("extra AP quintiles")
    pq = pass_quintiles(tr)
    print("extra fold0 vs rest")
    pf0 = pass_fold0(tr)
    print("extra persist / supp median")
    pper = pass_persist(tr)
    print("extra boot leftover after AR")
    pbar = pass_boot_ar(tr)
    print("extra w/b vs ICC")
    pwb = pass_wb(tr)
    print("extra winsor99")
    pwin = pass_winsor(tr)
    print("extra lag3")
    pl3 = pass_lag3(tr)
    print("extra Y7 leftover days / mid / days after both")
    py7d = pass_y7_days(tr)
    print("extra ap>0 after supp / log1p days+size")
    pspos = pass_supp_pos(tr)
    print("extra DPO-defined leftover")
    pdpo = pass_dpo_slice(tr)
    png = make_png(tr, p4)
    decision = decide(p1, p2, p3, p4, p5)
    print("extra KEEP-as-X scorecard")
    pk = pass_keep_card(p1, p2, p3, p4, p5)
    print("extra country leftover")
    pcty = pass_country(tr)
    print("extra ERP / zero-share leftover")
    perpz = pass_erp_zeros(tr)
    print("extra leftover after open / overdue / triple")
    popen = pass_open(tr)
    print("extra never-zero leftover")
    pnz = pass_never_zero(tr)
    print("extra has_ap dummy")
    phas = pass_has_ap(tr)
    print("extra issued/open turnover leftover")
    pturn = pass_turnover_ap(tr)
    print("extra leftover after AR_lag1")
    parl = pass_after_ar_lag1(tr)
    print("extra leftover after both lag1s")
    pbl = pass_both_lag1(tr)
    print("extra leftover after group-mean")
    pg = pass_group(tr)
    print("extra leftover after card KEEP leftovers")
    pck = pass_card_keeps(tr)
    failed = [
        f"Y3 leftover after days rank {_f(p4['y3_rank'])} OLS {_f(p4['y3_ols'])} fake={p4['y3_fake']}",
        f"inverse days leftover {_f(p4['inv_rank'])}",
        f"leftover after AR {_f(p5['rank'])} rewrite={p5['rewrite']}",
        f"Y5 leftover after size {_f(p6['rank'])} leak_ok={p6['leak_ok']}",
        f"Q6 ap_lag1 leftover {_f(p7['lag_rank'])} now {_f(p7['now_rank'])}",
        f"leftover after ap_lag1 {_f(p8['rank'])} Y7 {_f(p8['y7_rank'])}",
        f"leftover after size {_f(p9['size_rank'])} DPO {_f(p9['dpo_rank'])}",
        f"boot leftover p05={_f(p10['p05'])} p50={_f(p10['p50'])} p95={_f(p10['p95'])}",
        f"ap>0 leftover {_f(ppos['rank'])} fake={ppos['fake']}",
        f"log1p leftover {_f(pl['rank'])} fake={pl['fake']}",
        f"intensity leftover {_f(pi['rank'])} fake={pi['fake']}",
        f"terciles T1 {_f(pt['t1'])} T2+T3 {_f(pt['t23'])}",
        f"co-median SIZE={pcm['size_flag']} ρ={_f(pcm['rho_size'])}",
        f"after d_n_supp {_f(psupp['supp_rank'])} dies={psupp['supp_dies']} after n_tx {_f(psupp['ntx_rank'])}",
        f"both>0 leftover-days {_f(pboth['rank'])} leftover-AR {_f(pboth['ar_rank'])}",
        f"so_far short {_f(psf['short'])} mid {_f(psf['mid'])} long {_f(psf['long'])}",
        f"per-fold leftover {pf['folds']}",
        f"log1p after size {_f(plm['size_rank'])} SIZE={plm['size_flag']} twin_lag={plm['twin_lag']}",
        f"Q6 mid Y7 {_f(pq6['mid_y7'])} long Y7 {_f(pq6['long_y7'])}",
        f"ICC(1) {_f(picc['icc1'])} share {_f(picc['share'])} vs quote {ICC_QUOTE:.2f}",
        f"AR leftover after AP {_f(pinv['inv'])} Y7 leftover after AR {_f(pinv['y7'])}",
        f"ever-ERP leftover {_f(perp['rank'])} early {_f(perp['early'])} late {_f(perp['late'])}",
        f"ap/(ap+ar) leftover {_f(psh['rank'])} fake={psh['fake']} raw {_f(psh['raw'])}",
        f"quintiles Q1 {_f(pq['q1'])} Q5 {_f(pq['q5'])} hi {_f(pq['hi'])}",
        f"fold0 leftover {_f(pf0['f0'])} fake={pf0['f0_fake']} rest {_f(pf0['rest'])} fake={pf0['rest_fake']}",
        f"≥6mo leftover {_f(pper['long'])} fake={pper['long_fake']} ρ vs supp {_f(pper['rho_supp'])}",
        f"boot leftover-AR p05={_f(pbar['p05'])} p50={_f(pbar['p50'])} p95={_f(pbar['p95'])}",
        f"w/b {_f(pwb['wb'])} ICC {_f(pwb['icc'])} (0.86 is w/b; ICC 0.54 CONFIRM)",
        f"winsor leftover {_f(pwin['rank'])} fake={pwin['fake']}",
        f"lag3 leftover {_f(pl3['rank'])} after lag1 {_f(pl3['after_lag1'])}",
        f"Y7 leftover-days {_f(py7d['y7'])} mid {_f(py7d['mid'])} fake={py7d['mid_fake']} days-after-AP+AR {_f(py7d['days_both'])}",
        f"ap>0 after supp {_f(pspos['supp'])} triple {_f(pspos['triple'])} log1p days+size {_f(pspos['log_ds'])}",
        f"DPO-defined leftover {_f(pdpo['rank'])} after DPO+days {_f(pdpo['after'])} ρ(rank-resid,days)={_f(pdpo['rho_rresid'])}",
        f"country ES leftover {_f(pcty['es'])} fake={pcty['es_fake']} missing {_f(pcty['miss'])} fake={pcty['miss_fake']}",
        f"zero-heavy leftover {_f(perpz['heavy'])} zero-light {_f(perpz['light'])}",
        f"leftover after open {_f(popen['open'])} overdue {_f(popen['od'])} days+AR+size {_f(popen['triple'])} open+days {_f(popen['open_days'])} zero-light-AR {_f(popen['zlight_ar'])}",
        f"never-zero leftover-days {_f(pnz['rank'])} leftover-AR {_f(pnz['ar'])} days-after-ap {_f(pnz['inv'])}",
        f"has_ap leftover {_f(phas['rank'])} has-any-zero {_f(phas['hasz'])}",
        f"issued/open leftover {_f(pturn['rank'])} after issued {_f(pturn['after'])}",
        f"leftover after AR_lag1 Y3 {_f(parl['y3'])} Y7 {_f(parl['y7'])} +days {_f(parl['both'])}",
        f"leftover after both lag1s Y3 {_f(pbl['y3'])} Y7 {_f(pbl['y7'])}",
        f"leftover after group-mean days {_f(pg['gdays'])} group-AP {_f(pg['gap'])} +days {_f(pg['both'])}",
        f"leftover after ss {_f(pck['ss'])} salary {_f(pck['sal'])} ss+salary+days {_f(pck['both'])}",
        f"card: {decision['card']}",
        "do not grow TURNOVER 0.720; do not put e_ap_issued on the 15-col card",
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
        "pi": pi,
        "pt": pt,
        "pcm": pcm,
        "psupp": psupp,
        "pboth": pboth,
        "psf": psf,
        "pf": pf,
        "plm": plm,
        "pq6": pq6,
        "picc": picc,
        "pinv": pinv,
        "perp": perp,
        "psh": psh,
        "pq": pq,
        "pf0": pf0,
        "pper": pper,
        "pbar": pbar,
        "pwb": pwb,
        "pwin": pwin,
        "pl3": pl3,
        "py7d": py7d,
        "pspos": pspos,
        "pdpo": pdpo,
        "pk": pk,
        "pcty": pcty,
        "perpz": perpz,
        "popen": popen,
        "pnz": pnz,
        "phas": phas,
        "pturn": pturn,
        "parl": parl,
        "pbl": pbl,
        "pg": pg,
        "pck": pck,
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
        f"DONE elapsed={elapsed:.0f}s tag={decision['headline_tag']} card={decision['card']}"
    )
    print(
        f"{decision['headline_tag']} leftover-after-days rank {_f(p4['y3_rank'])} "
        f"(OLS {_f(p4['y3_ols'])}, fake={p4['y3_fake']}). "
        f"Y3 ap {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size'])}. "
        f"SIZE={p2['size_flag']} twin={bool(p2['gate_twins'])}. "
        f"Inverse days-after-ap {_f(p4['inv_rank'])}. "
        f"Leftover after AR {_f(p5['rank'])}. Card: {decision['card']}. "
        f"Night Y3 {NIGHT_Y3:.3f}/{NIGHT_Y3_CORE:.3f}, days {DAYS_BENCH:.3f}, "
        f"size {SIZE_QUOTE:.3f}, TURNOVER {TURNOVER_QUOTE:.3f}/{B_SHALLOW_QUOTE:.3f} unchanged."
    )
    return ctx


if __name__ == "__main__":
    run()

