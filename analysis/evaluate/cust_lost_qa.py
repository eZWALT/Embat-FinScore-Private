"""Unused leftover of ``d_cust_lost`` after ``c_n_days_with_tx`` as Y3 X.

NORTH_STAR: ``d_cust_lost`` = |prev \\ cur| of AR counterparties in the
calendar quarter of ``period`` vs the previous quarter (counterparties.py).
Different object from Y7 ``y7_top1_lost`` (invoice-book top1). Feature
report: 53.6% cov, acf1 0.58, ICC 0.98 BETWEEN, size_ρ 0.323; redundant
with ``d_n_cust`` (|ρ| 0.86). ``d_n_cust`` leftover 0.545 DROP.
``d_cust_top1`` leftover 0.525 DROP as Y3 X. Y4 HHI >0.975 footnote KEEP
locked. Y7 never D. TURNOVER 0.720 must not grow. Dark 470 stay NaN not 0.
Do **not** overwrite top1_qa / n_cust_qa / cust_hhi_qa / y4_why / delay_qa.

KEEP-as-X: beat size ≥0.02 AND leftover after days AND not SIZE
(|ρ| vs log1p(a_in3) ≥0.50) AND not a twin (|ρ|≥0.80 vs days / a_n_tx /
d_cust_new / d_n_cust / d_cust_top1 / d_cust_hhi). Leftover <0.55 dies.
Rank leftover is honest; OLS can fake a days leak.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.cust_lost_qa

Owned: analysis/evaluate/cust_lost_qa.py, analysis/outputs/cust_lost_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_cust_lost.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "cust_lost_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "cust_lost_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_cust_lost.md"
AGENT = "e8b2c0d4"
WAVE = "4"
ROUND = "R4"
MODEL = "cust_lost_qa"
X_FAM = "D"

Y3 = "y3_recover_cash_6m"
Y4 = "y4_ds_r_double"
Y7 = "y7_top1_lost"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
N_CUST_Y3_QUOTE = 0.653
N_CUST_LEFTOVER_QUOTE = 0.545
TOP1_Y3_QUOTE = 0.590
TOP1_LEFTOVER_QUOTE = 0.525
DAYS_LAG1_QUOTE = 0.684
TURNOVER_QUOTE = 0.720
B_SHALLOW_QUOTE = 0.712
NIGHT_Y3 = 0.762
NIGHT_Y3_CORE = 0.752
COV_QUOTE = 0.536
ICC_QUOTE = 0.98
ACF1_QUOTE = 0.58
SIZE_RHO_QUOTE = 0.323
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
    "d_cust_lost",
    "d_cust_new",
    "d_n_cust",
    "d_cust_top1",
    "d_cust_hhi",
    "e_ar_issued",
)

Y_KEEP = (Y3, Y4, Y7)
TWIN_GATES = (
    "c_n_days_with_tx",
    "a_n_tx",
    "d_cust_new",
    "d_n_cust",
    "d_cust_top1",
    "d_cust_hhi",
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
    panel["so_far_class"] = panel["months_so_far"].map(_trail_class)
    panel = add_panel_lags(
        panel,
        ["d_cust_lost", "d_cust_new", "d_n_cust", "c_n_days_with_tx", "e_ar_issued", "d_cust_top1"],
        (1, 3),
    )
    leak7 = leakage_check(["e_ar_issued", "e_ar_issued_lag1"], Y7, forbidden_prefixes=["d"])
    leak3 = leakage_check(["d_cust_lost"], Y3, forbidden_prefixes=["b"])
    if not leak7["ok"]:
        raise RuntimeError(f"Y7 never-D leak: {leak7['issues']}")
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 X leak: {leak3['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def pass1_cov(tr: pd.DataFrame, book: set[str]) -> dict:
    dark = ~tr["company_id"].isin(book)
    erp = tr["company_id"].isin(book)
    x = pd.to_numeric(tr["d_cust_lost"], errors="coerce")
    n_cm = int(len(tr))
    n_nn = int(x.notna().sum())
    dark_nn = int(x[dark].notna().sum())
    dark_zero = int((x[dark] == 0).sum())
    dark_pos = int((x[dark] > 0).sum())
    erp_nn = int(x[erp].notna().sum())
    erp_zero = int((x[erp] == 0).sum())
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    n_erp_co = int(tr.loc[erp, "company_id"].nunique())
    rho_size, n_size = spearman_n(x, tr["log_in3"])
    size_flag = bool(np.isfinite(rho_size) and abs(rho_size) >= SIZE_RHO)
    finite = x[x.notna()]
    p50 = float(finite.median()) if n_nn else float("nan")
    p99 = float(finite.quantile(0.99)) if n_nn else float("nan")
    mx = float(finite.max()) if n_nn else float("nan")
    eq0 = _pct(int((finite == 0).sum()), n_nn)
    acf1 = median_acf(x, tr["company_id"], 1)
    icc = icc_anova(x, tr["company_id"])
    rows = [
        {
            "slice": "all train",
            "n_cm": f"{n_cm:,}",
            "nn": f"{n_nn:,}",
            "cov": _pp(_pct(n_nn, n_cm)),
            "eq0": _pp(eq0),
            "p50": _f(p50),
            "p99": _f(p99),
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
        f"Train d_cust_lost nn={n_nn:,} cov={_pp(_pct(n_nn, n_cm))} "
        f"(feature report {COV_QUOTE:.1%}). Dark never-ERP {n_dark_co} "
        f"(want 470): nn={dark_nn} zero={dark_zero} pos={dark_pos} "
        f"{'CONFIRM NaN not 0' if dark_ok else ('BOOK stub' if dark_nn else 'check')}. "
        f"Ever-ERP {n_erp_co} nn={erp_nn:,} of which zero={erp_zero:,}. "
        f"p50={_f(p50)} p99={_f(p99)} max={_f(mx)}. "
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
    x = tr["d_cust_lost"]
    pairs = [
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("a_n_tx", tr["a_n_tx"]),
        ("d_cust_new", tr["d_cust_new"]),
        ("d_n_cust", tr["d_n_cust"]),
        ("d_cust_top1", tr["d_cust_top1"]),
        ("d_cust_hhi", tr["d_cust_hhi"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("e_ar_issued", tr["e_ar_issued"]),
        ("d_cust_lost_lag1", tr["d_cust_lost_lag1"]),
    ]
    rows = []
    rhos = {}
    twins = []
    for name, s in pairs:
        rho, n = spearman_n(x, s)
        rhos[name] = rho
        twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
        if twin:
            twins.append(name)
        rows.append({"vs": name, "ρ": _f(rho), "n": f"{n:,}", "twin?": "TWIN" if twin else ""})
    size_flag = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    gate = [t for t in twins if t in TWIN_GATES]
    prose = (
        f"cust_lost vs days ρ={_f(rhos['c_n_days_with_tx'])} "
        f"({'TWIN' if 'c_n_days_with_tx' in twins else 'not a twin'}). "
        f"vs a_n_tx {_f(rhos['a_n_tx'])} vs d_cust_new {_f(rhos['d_cust_new'])} "
        f"vs d_n_cust {_f(rhos['d_n_cust'])} vs top1 {_f(rhos['d_cust_top1'])} "
        f"vs HHI {_f(rhos['d_cust_hhi'])} vs log1p(a_in3) {_f(rhos['log1p(a_in3)'])} "
        f"{'SIZE' if size_flag else 'not SIZE'}. "
        f"vs issued {_f(rhos['e_ar_issued'])} vs own lag1 {_f(rhos['d_cust_lost_lag1'])}. "
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
        "rho_new": rhos["d_cust_new"],
        "rho_ncust": rhos["d_n_cust"],
        "rho_top1": rhos["d_cust_top1"],
        "rho_hhi": rhos["d_cust_hhi"],
        "rho_lag1": rhos["d_cust_lost_lag1"],
        "prose": prose,
    }


def pass3_singles(tr: pd.DataFrame) -> dict:
    feats = [
        ("d_cust_lost", tr["d_cust_lost"]),
        ("d_cust_lost_lag1", tr["d_cust_lost_lag1"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("log1p_a_in3", tr["log_in3"]),
        ("d_n_cust", tr["d_n_cust"]),
        ("d_cust_top1", tr["d_cust_top1"]),
        ("d_cust_hhi", tr["d_cust_hhi"]),
        ("d_cust_new", tr["d_cust_new"]),
        ("a_n_tx", tr["a_n_tx"]),
        ("e_ar_issued", tr["e_ar_issued"]),
    ]
    rows = []
    store = {}
    for feat, s in feats:
        rec = signed_oof_auroc(tr[Y3], s, tr["fold"], tr[Y3].notna())
        print(f"{Y3} {feat}: {_f(_cv(rec))} n={rec['n_defined']:,} pos={rec['n_pos']:,}")
        rows.append(_auc_row(Y3, feat, rec))
        store[feat] = rec
    y3 = _cv(store["d_cust_lost"])
    days = _cv(store["c_n_days_with_tx"])
    size = _cv(store["log1p_a_in3"])
    nc = _cv(store["d_n_cust"])
    top1 = _cv(store["d_cust_top1"])
    beat_size = y3 - size if np.isfinite(y3) and np.isfinite(size) else float("nan")
    prose = (
        f"Y3 d_cust_lost {_f(y3)} vs days {_f(days)} (CONFIRM {DAYS_BENCH:.3f}) "
        f"vs size {_f(size)} (CONFIRM {SIZE_QUOTE:.3f}) vs d_n_cust {_f(nc)} "
        f"(CONFIRM {N_CUST_Y3_QUOTE:.3f}) vs top1 {_f(top1)} "
        f"(CONFIRM {TOP1_Y3_QUOTE:.3f}) vs new {_f(_cv(store['d_cust_new']))} "
        f"vs HHI {_f(_cv(store['d_cust_hhi']))}. Beat size {_f(beat_size)}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3": y3,
        "days": days,
        "size": size,
        "n_cust": nc,
        "top1": top1,
        "hhi": _cv(store["d_cust_hhi"]),
        "new": _cv(store["d_cust_new"]),
        "lag1": _cv(store["d_cust_lost_lag1"]),
        "issued": _cv(store["e_ar_issued"]),
        "beat_size": beat_size,
        "prose": prose,
    }


def pass4_leftover(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["d_cust_lost"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    inv = leftover_diag(
        tr[Y3], tr["c_n_days_with_tx"], (tr["d_cust_lost"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {
            "bar": "Y3 leftover after days",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "R²": _f(d["r2"]),
            "fake?": "FALSE clone" if d["fake"] else "",
            "dies?": "dies" if d["honest_dies"] else "lives",
        },
        {
            "bar": "inverse: days leftover after lost",
            "rank": _f(inv["rank"]),
            "OLS": _f(inv["ols"]),
            "ρ": _f(inv["rho_ctrl"]),
            "R²": _f(inv["r2"]),
            "fake?": "FALSE clone" if inv["fake"] else "",
            "dies?": "dies" if inv["honest_dies"] else "lives",
        },
    ]
    fake_note = " FALSE clone" if d["fake"] else ""
    prose = (
        f"Y3 leftover after days rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"ρ(resid,days)={_f(d['rho_ctrl'])} R²={_f(d['r2'])} "
        f"{'dies' if d['honest_dies'] else 'lives'}{fake_note}. "
        f"Inverse: days leftover after lost rank {_f(inv['rank'])} "
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


def pass5_after_family(tr: pd.DataFrame) -> dict:
    after_n = leftover_diag(
        tr[Y3], tr["d_cust_lost"], (tr["d_n_cust"],), tr["fold"], tr[Y3].notna()
    )
    after_n_days = leftover_diag(
        tr[Y3],
        tr["d_cust_lost"],
        (tr["d_n_cust"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    after_t = leftover_diag(
        tr[Y3], tr["d_cust_lost"], (tr["d_cust_top1"],), tr["fold"], tr[Y3].notna()
    )
    after_t_days = leftover_diag(
        tr[Y3],
        tr["d_cust_lost"],
        (tr["d_cust_top1"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    after_h = leftover_diag(
        tr[Y3], tr["d_cust_lost"], (tr["d_cust_hhi"],), tr["fold"], tr[Y3].notna()
    )
    after_new = leftover_diag(
        tr[Y3], tr["d_cust_lost"], (tr["d_cust_new"],), tr["fold"], tr[Y3].notna()
    )
    after_new_days = leftover_diag(
        tr[Y3],
        tr["d_cust_lost"],
        (tr["d_cust_new"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rewrite_n = bool(after_n["honest_dies"])
    rows = [
        {"bar": "leftover after d_n_cust", "rank": _f(after_n["rank"]), "OLS": _f(after_n["ols"]), "ρ": _f(after_n["rho_ctrl"])},
        {"bar": "leftover after n_cust+days", "rank": _f(after_n_days["rank"]), "OLS": _f(after_n_days["ols"])},
        {"bar": "leftover after d_cust_top1", "rank": _f(after_t["rank"]), "OLS": _f(after_t["ols"]), "ρ": _f(after_t["rho_ctrl"])},
        {"bar": "leftover after top1+days", "rank": _f(after_t_days["rank"]), "OLS": _f(after_t_days["ols"])},
        {"bar": "leftover after d_cust_hhi", "rank": _f(after_h["rank"]), "OLS": _f(after_h["ols"]), "ρ": _f(after_h["rho_ctrl"])},
        {"bar": "leftover after d_cust_new", "rank": _f(after_new["rank"]), "OLS": _f(after_new["ols"]), "ρ": _f(after_new["rho_ctrl"])},
        {"bar": "leftover after new+days", "rank": _f(after_new_days["rank"]), "OLS": _f(after_new_days["ols"])},
    ]
    prose = (
        f"Y3 leftover after d_n_cust {_f(after_n['rank'])} "
        f"{'REWRITE of n_cust' if rewrite_n else 'not just n_cust'} "
        f"(n_cust leftover {N_CUST_LEFTOVER_QUOTE:.3f} DROP). "
        f"After n_cust+days {_f(after_n_days['rank'])}. "
        f"After top1 {_f(after_t['rank'])} +days {_f(after_t_days['rank'])} "
        f"(top1 leftover {TOP1_LEFTOVER_QUOTE:.3f} DROP). "
        f"After HHI {_f(after_h['rank'])}. After new {_f(after_new['rank'])} "
        f"+days {_f(after_new_days['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "n_cust": after_n["rank"],
        "n_cust_days": after_n_days["rank"],
        "top1": after_t["rank"],
        "top1_days": after_t_days["rank"],
        "hhi": after_h["rank"],
        "new": after_new["rank"],
        "new_days": after_new_days["rank"],
        "rewrite_n": rewrite_n,
        "prose": prose,
    }


def pass6_q6(tr: pd.DataFrame) -> dict:
    short = tr["so_far_class"] == "short_<12"
    raw = signed_oof_auroc(
        tr[Y3], tr["d_cust_lost_lag1"], tr["fold"], short & tr[Y3].notna()
    )
    d = leftover_diag(
        tr[Y3],
        tr["d_cust_lost_lag1"],
        (tr["c_n_days_with_tx_lag1"],),
        tr["fold"],
        short & tr[Y3].notna(),
    )
    now = leftover_diag(
        tr[Y3],
        tr["d_cust_lost"],
        (tr["c_n_days_with_tx_lag1"],),
        tr["fold"],
        short & tr[Y3].notna(),
    )
    days_l1 = signed_oof_auroc(
        tr[Y3], tr["c_n_days_with_tx_lag1"], tr["fold"], short & tr[Y3].notna()
    )
    rows = [
        {"slice": "Q6 Y3 lost_lag1 short", "CV": "LOW_POWER" if raw["low_power"] else _f(_cv(raw)), "n": f"{raw['n_defined']:,}"},
        {"slice": "Q6 leftover after days_lag1 short", "rank": _f(d["rank"]), "OLS": _f(d["ols"]), "fake?": "FALSE clone" if d["fake"] else ""},
        {"slice": "Q6 now leftover after days_lag1 short", "rank": _f(now["rank"]), "OLS": _f(now["ols"])},
        {"slice": "Q6 days_lag1 short", "CV": _f(_cv(days_l1))},
    ]
    prose = (
        f"Q6 Y3 lost_lag1 on short {_f(_cv(raw))}. "
        f"lag1 leftover after days_lag1 short rank {_f(d['rank'])} fake={d['fake']}. "
        f"now leftover after days_lag1 {_f(now['rank'])}. "
        f"days_lag1 short {_f(_cv(days_l1))} (quote {DAYS_LAG1_QUOTE:.3f}). "
        "q6_keep = issued_lag1 / days_lag1 / ss_lag1. Do not claim a TURNOVER seat."
    )
    print(prose)
    return {
        "rows": rows,
        "y3": _cv(raw),
        "lag_rank": d["rank"],
        "now_rank": now["rank"],
        "days_l1": _cv(days_l1),
        "prose": prose,
    }


def pass7_y7_issued(tr: pd.DataFrame) -> dict:
    leak = leakage_check(["d_cust_lost"], Y7, forbidden_prefixes=["d"])
    raw = signed_oof_auroc(tr[Y7], tr["e_ar_issued"], tr["fold"], tr[Y7].notna())
    d = leftover_diag(
        tr[Y7], tr["e_ar_issued"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y7].notna()
    )
    rows = [
        {
            "bar": "Y7 issued raw (report only)",
            "CV": _f(_cv(raw)),
            "leftover-days": _f(d["rank"]),
            "leak_ok": str(leak["ok"]),
        }
    ]
    prose = (
        f"Y7 leftover after issued is the issued leftover (report only). "
        f"issued Y7 {_f(_cv(raw))} leftover-days {_f(d['rank'])}. "
        f"Y7 never D — d_cust_lost leak_ok={leak['ok']}. Do not grow TURNOVER 0.720."
    )
    print(prose)
    return {"rows": rows, "issued": _cv(raw), "leftover": d["rank"], "leak_ok": leak["ok"], "prose": prose}


def pass8_boot(tr: pd.DataFrame) -> dict:
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
            sub["d_cust_lost"],
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
    rows = [{"boot": f"n={len(vals)}/{N_BOOT}", "p05": _f(p05), "p50": _f(p50), "p95": _f(p95)}]
    prose = (
        f"Bootstrap leftover-after-days rank p05={_f(p05)} p50={_f(p50)} "
        f"p95={_f(p95)} n={len(vals)}/{N_BOOT}."
    )
    print(prose)
    return {"rows": rows, "p05": p05, "p50": p50, "p95": p95, "n": len(vals), "prose": prose}


def pass_holdout(panel: pd.DataFrame, book: set[str]) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    assert set(ho["company_id"]) <= load_holdout()
    x = pd.to_numeric(ho["d_cust_lost"], errors="coerce")
    dark = ~ho["company_id"].isin(book)
    rows = [
        {
            "slice": "holdout 72",
            "n_cm": f"{len(ho):,}",
            "companies": f"{ho['company_id'].nunique()}",
            "cov": _pp(_pct(int(x.notna().sum()), len(ho))),
            "dark_nn": int(x[dark].notna().sum()),
        }
    ]
    prose = (
        f"Holdout 72 coverage only: {len(ho):,} CM / {ho['company_id'].nunique()} companies. "
        f"lost cov {_pp(_pct(int(x.notna().sum()), len(ho)))}; dark nn={int(x[dark].notna().sum())}. "
        "No AUROC."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_keep_card(p1, p2, p3, p4) -> dict:
    beat = bool(np.isfinite(p3["beat_size"]) and p3["beat_size"] >= KEEP_DELTA)
    leftover_ok = not bool(p4["y3_dies"])
    size_ok = not bool(p1["size_flag"] or p2["size_flag"])
    twin_ok = not bool(p2["gate_twins"])
    rows = [
        {"gate": "beat size ≥0.02", "ok": "PASS" if beat else "FAIL", "value": _f(p3["beat_size"])},
        {"gate": "leftover after days ≥0.55 and not fake", "ok": "PASS" if leftover_ok else "FAIL", "value": _f(p4["y3_rank"])},
        {"gate": "not SIZE |ρ|≥0.50 vs log1p(a_in3)", "ok": "PASS" if size_ok else "FAIL", "value": _f(p2["rho_size"])},
        {
            "gate": "not twin vs days / n_tx / new / n_cust / top1 / HHI",
            "ok": "PASS" if twin_ok else "FAIL",
            "value": ",".join(p2["gate_twins"]) if p2["gate_twins"] else "none",
        },
    ]
    prose = (
        f"KEEP-as-X scorecard: beat size {'PASS' if beat else 'FAIL'} "
        f"(Y3 {_f(p3['y3'])} vs size {_f(p3['size'])}); "
        f"leftover-after-days {'PASS' if leftover_ok else 'FAIL'} {_f(p4['y3_rank'])}; "
        f"row {'SIZE' if not size_ok else 'not SIZE'}; "
        f"gate twins {'FAIL' if not twin_ok else 'PASS'}. "
        "Do not put d_cust_lost on the 15-col card."
    )
    print(prose)
    return {"rows": rows, "beat": beat, "leftover_ok": leftover_ok, "prose": prose}


def pass_terciles(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    rows = []
    store = {}
    for lab, mask in (("T1", terc == "T1"), ("T2+T3", terc.isin(["T2", "T3"])), ("T3", terc == "T3")):
        d = leftover_diag(
            tr[Y3], tr["d_cust_lost"], (tr["c_n_days_with_tx"],), tr["fold"], mask & tr[Y3].notna()
        )
        store[lab] = d["rank"]
        rows.append({"slice": lab, "rank": _f(d["rank"]), "OLS": _f(d["ols"]), "n": f"{d['n']:,}"})
    prose = (
        f"SIZE terciles leftover after days: T1 {_f(store['T1'])} "
        f"T2+T3 {_f(store['T2+T3'])} T3 {_f(store['T3'])}."
    )
    print(prose)
    return {"rows": rows, "t1": store["T1"], "t23": store["T2+T3"], "prose": prose}


def pass_so_far(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for lab in ("short_<12", "mid_12_17", "long_>=18"):
        m = tr["so_far_class"] == lab
        d = leftover_diag(
            tr[Y3], tr["d_cust_lost"], (tr["c_n_days_with_tx"],), tr["fold"], m & tr[Y3].notna()
        )
        store[lab] = d["rank"]
        rows.append({"slice": lab, "rank": _f(d["rank"]), "OLS": _f(d["ols"]), "n": f"{d['n']:,}"})
    prose = (
        f"so_far leftover after days: short {_f(store['short_<12'])} "
        f"mid {_f(store['mid_12_17'])} long {_f(store['long_>=18'])}."
    )
    print(prose)
    return {"rows": rows, "short": store["short_<12"], "mid": store["mid_12_17"], "prose": prose}


def pass_folds(tr: pd.DataFrame) -> dict:
    bits = []
    rows = []
    for k in range(N_FOLDS):
        m = tr["fold"] == k
        d = leftover_diag(
            tr[Y3], tr["d_cust_lost"], (tr["c_n_days_with_tx"],), tr["fold"], m & tr[Y3].notna()
        )
        bits.append(_f(d["rank"]))
        rows.append({"fold": k, "rank": _f(d["rank"]), "OLS": _f(d["ols"]), "n": f"{d['n']:,}"})
    prose = f"Per-fold rank leftover after days: {' '.join(bits)}."
    print(prose)
    return {"rows": rows, "bits": bits, "prose": prose}


def pass_between(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["d_cust_lost"], errors="coerce")
    cmean = x.groupby(tr["company_id"]).transform("mean")
    within = x - cmean
    after_mean = leftover_diag(tr[Y3], tr["d_cust_lost"], (cmean,), tr["fold"], tr[Y3].notna())
    after_lag1 = leftover_diag(
        tr[Y3], tr["d_cust_lost"], (tr["d_cust_lost_lag1"],), tr["fold"], tr[Y3].notna()
    )
    within_d = leftover_diag(tr[Y3], within, (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna())
    rows = [
        {"bar": "leftover after company-mean", "rank": _f(after_mean["rank"]), "OLS": _f(after_mean["ols"])},
        {"bar": "leftover after own lag1", "rank": _f(after_lag1["rank"]), "OLS": _f(after_lag1["ols"])},
        {"bar": "within leftover-days", "rank": _f(within_d["rank"]), "OLS": _f(within_d["ols"])},
    ]
    prose = (
        f"BETWEEN: leftover after company-mean {_f(after_mean['rank'])}; "
        f"after own lag1 {_f(after_lag1['rank'])}; within leftover-days {_f(within_d['rank'])}. "
        "ICC 0.98 BETWEEN."
    )
    print(prose)
    return {"rows": rows, "mean": after_mean["rank"], "lag1": after_lag1["rank"], "within": within_d["rank"], "prose": prose}


def pass_pos_samen(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["d_cust_lost"], errors="coerce")
    m = x.notna() & tr[Y3].notna()
    days_s = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], m)
    size_s = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], m)
    lost_s = signed_oof_auroc(tr[Y3], tr["d_cust_lost"], tr["fold"], m)
    pos = leftover_diag(
        tr[Y3], tr["d_cust_lost"], (tr["c_n_days_with_tx"],), tr["fold"], (x > 0) & tr[Y3].notna()
    )
    beat = (
        _cv(lost_s) - _cv(size_s)
        if np.isfinite(_cv(lost_s)) and np.isfinite(_cv(size_s))
        else float("nan")
    )
    rows = [
        {"bar": "Y3 lost same-n", "CV": _f(_cv(lost_s)), "n": f"{lost_s['n_defined']:,}"},
        {"bar": "Y3 days same-n", "CV": _f(_cv(days_s))},
        {"bar": "Y3 size same-n", "CV": _f(_cv(size_s))},
        {"bar": "lost>0 leftover-days", "rank": _f(pos["rank"]), "OLS": _f(pos["ols"])},
    ]
    prose = (
        f"same-n Y3 lost {_f(_cv(lost_s))} days {_f(_cv(days_s))} size {_f(_cv(size_s))} "
        f"beat-size {_f(beat)}; lost>0 leftover-days {_f(pos['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "lost": _cv(lost_s),
        "days": _cv(days_s),
        "size": _cv(size_s),
        "beat": beat,
        "pos": pos["rank"],
        "prose": prose,
    }


def pass_card_issued(tr: pd.DataFrame) -> dict:
    ds = leftover_diag(
        tr[Y3],
        tr["d_cust_lost"],
        (tr["c_n_days_with_tx"], tr["log_in3"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    raw = pd.read_parquet(STORE, columns=["company_id", "period", "c_ss_month", "c_salary_month"])
    raw = _keys(raw)
    work = tr.merge(raw, on=["company_id", "period"], how="left")
    card = leftover_diag(
        work[Y3],
        work["d_cust_lost"],
        (work["c_ss_month"], work["c_salary_month"], work["c_n_days_with_tx"]),
        work["fold"],
        work[Y3].notna(),
    )
    after_iss = leftover_diag(
        tr[Y3], tr["d_cust_lost"], (tr["e_ar_issued"],), tr["fold"], tr[Y3].notna()
    )
    after_iss_days = leftover_diag(
        tr[Y3],
        tr["d_cust_lost"],
        (tr["e_ar_issued"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    both = leftover_diag(
        tr[Y3],
        tr["d_cust_lost"],
        (tr["d_n_cust"], tr["d_cust_top1"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {"bar": "leftover after days+size", "rank": _f(ds["rank"]), "OLS": _f(ds["ols"])},
        {"bar": "leftover after ss+salary+days", "rank": _f(card["rank"]), "OLS": _f(card["ols"])},
        {"bar": "leftover after e_ar_issued", "rank": _f(after_iss["rank"]), "OLS": _f(after_iss["ols"])},
        {"bar": "leftover after issued+days", "rank": _f(after_iss_days["rank"]), "OLS": _f(after_iss_days["ols"])},
        {"bar": "leftover after n_cust+top1", "rank": _f(both["rank"]), "OLS": _f(both["ols"])},
    ]
    prose = (
        f"leftover after days+size {_f(ds['rank'])}; ss+salary+days {_f(card['rank'])}; "
        f"after issued {_f(after_iss['rank'])} +days {_f(after_iss_days['rank'])}; "
        f"n_cust+top1 {_f(both['rank'])}. Stay off the 15-col card."
    )
    print(prose)
    return {
        "rows": rows,
        "ds": ds["rank"],
        "card": card["rank"],
        "iss": after_iss["rank"],
        "iss_days": after_iss_days["rank"],
        "both": both["rank"],
        "prose": prose,
    }


def pass_company_med(tr: pd.DataFrame) -> dict:
    med = tr.groupby("company_id").agg(
        lost=("d_cust_lost", "median"),
        days=("c_n_days_with_tx", "median"),
        size=("log_in3", "median"),
        nc=("d_n_cust", "median"),
        new=("d_cust_new", "median"),
    )
    rho_d, n_d = spearman_n(med["lost"], med["days"])
    rho_s, n_s = spearman_n(med["lost"], med["size"])
    rho_n, n_n = spearman_n(med["lost"], med["nc"])
    rho_w, n_w = spearman_n(med["lost"], med["new"])
    size_flag = bool(np.isfinite(rho_s) and abs(rho_s) >= SIZE_RHO)
    twin_n = bool(np.isfinite(rho_n) and abs(rho_n) >= TWIN_RHO)
    rows = [
        {"vs": "days", "ρ": _f(rho_d), "n": f"{n_d:,}"},
        {"vs": "log1p(a_in3)", "ρ": _f(rho_s), "n": f"{n_s:,}", "SIZE?": "SIZE" if size_flag else ""},
        {"vs": "d_n_cust", "ρ": _f(rho_n), "n": f"{n_n:,}", "twin?": "TWIN" if twin_n else ""},
        {"vs": "d_cust_new", "ρ": _f(rho_w), "n": f"{n_w:,}"},
    ]
    prose = (
        f"Company-median ρ lost vs days {_f(rho_d)} vs log1p(a_in3) {_f(rho_s)} "
        f"{'SIZE' if size_flag else 'not SIZE'} vs n_cust {_f(rho_n)} "
        f"{'TWIN' if twin_n else ''} vs new {_f(rho_w)}."
    )
    print(prose)
    return {"rows": rows, "rho_ncust": rho_n, "twin": twin_n, "prose": prose}


def pass_q6_midlong(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for lab in ("mid_12_17", "long_>=18"):
        m = tr["so_far_class"] == lab
        raw = signed_oof_auroc(tr[Y3], tr["d_cust_lost_lag1"], tr["fold"], m & tr[Y3].notna())
        d = leftover_diag(
            tr[Y3],
            tr["d_cust_lost_lag1"],
            (tr["c_n_days_with_tx_lag1"],),
            tr["fold"],
            m & tr[Y3].notna(),
        )
        store[lab] = (_cv(raw), d["rank"])
        rows.append({"slice": f"Q6 Y3 {lab}", "raw": _f(_cv(raw)), "leftover": _f(d["rank"]), "n": f"{raw['n_defined']:,}"})
    prose = (
        f"Q6 mid Y3 lost_lag1 {_f(store['mid_12_17'][0])} leftover {_f(store['mid_12_17'][1])}; "
        f"long Y3 {_f(store['long_>=18'][0])}. Do not claim a TURNOVER seat."
    )
    print(prose)
    return {"rows": rows, "mid": store["mid_12_17"][1], "long": store["long_>=18"][0], "prose": prose}


def pass_rate_ntx(tr: pd.DataFrame) -> dict:
    lost = pd.to_numeric(tr["d_cust_lost"], errors="coerce")
    nc = pd.to_numeric(tr["d_n_cust"], errors="coerce")
    rate = lost / nc.where(nc > 0)
    rate_d = leftover_diag(tr[Y3], rate, (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna())
    rate_raw = signed_oof_auroc(tr[Y3], rate, tr["fold"], tr[Y3].notna())
    after_ntx = leftover_diag(
        tr[Y3], tr["d_cust_lost"], (tr["a_n_tx"],), tr["fold"], tr[Y3].notna()
    )
    after_ntx_days = leftover_diag(
        tr[Y3],
        tr["d_cust_lost"],
        (tr["a_n_tx"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    new_after = leftover_diag(
        tr[Y3], tr["d_cust_new"], (tr["d_cust_lost"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {"bar": "lost/n_cust Y3 raw", "CV": _f(_cv(rate_raw)), "n": f"{rate_raw['n_defined']:,}"},
        {"bar": "lost/n_cust leftover-days", "rank": _f(rate_d["rank"]), "OLS": _f(rate_d["ols"])},
        {"bar": "leftover after a_n_tx", "rank": _f(after_ntx["rank"]), "OLS": _f(after_ntx["ols"])},
        {"bar": "leftover after n_tx+days", "rank": _f(after_ntx_days["rank"]), "OLS": _f(after_ntx_days["ols"])},
        {"bar": "new leftover after lost", "rank": _f(new_after["rank"]), "OLS": _f(new_after["ols"])},
    ]
    prose = (
        f"lost/n_cust raw {_f(_cv(rate_raw))} leftover-days {_f(rate_d['rank'])}; "
        f"leftover after n_tx {_f(after_ntx['rank'])} +days {_f(after_ntx_days['rank'])}; "
        f"new leftover after lost {_f(new_after['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "rate": rate_d["rank"],
        "rate_raw": _cv(rate_raw),
        "ntx": after_ntx["rank"],
        "ntx_days": after_ntx_days["rank"],
        "new_after": new_after["rank"],
        "prose": prose,
    }


def pass_ncust_lag_joint(tr: pd.DataFrame) -> dict:
    after = leftover_diag(
        tr[Y3], tr["d_cust_lost"], (tr["d_n_cust_lag1"],), tr["fold"], tr[Y3].notna()
    )
    after_days = leftover_diag(
        tr[Y3],
        tr["d_cust_lost"],
        (tr["d_n_cust_lag1"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    joint = leftover_diag(
        tr[Y3],
        tr["d_cust_lost"],
        (tr["d_n_cust"], tr["d_cust_new"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {"bar": "leftover after n_cust_lag1", "rank": _f(after["rank"]), "OLS": _f(after["ols"])},
        {"bar": "leftover after n_cust_lag1+days", "rank": _f(after_days["rank"]), "OLS": _f(after_days["ols"])},
        {"bar": "leftover after n_cust+new+days", "rank": _f(joint["rank"]), "OLS": _f(joint["ols"])},
    ]
    prose = (
        f"leftover after n_cust_lag1 {_f(after['rank'])} +days {_f(after_days['rank'])}; "
        f"n_cust+new+days {_f(joint['rank'])}. Twin of the customer book."
    )
    print(prose)
    return {
        "rows": rows,
        "lag": after["rank"],
        "lag_days": after_days["rank"],
        "joint": joint["rank"],
        "prose": prose,
    }


def pass_y4_report(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y4], tr["d_cust_lost"], (tr["d_cust_top1"],), tr["fold"], tr[Y4].notna()
    )
    days = leftover_diag(
        tr[Y4], tr["d_cust_lost"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y4].notna()
    )
    rows = [
        {"bar": "Y4 leftover after top1 (report)", "rank": _f(d["rank"]), "OLS": _f(d["ols"])},
        {"bar": "Y4 leftover after days (report)", "rank": _f(days["rank"]), "OLS": _f(days["ols"])},
    ]
    prose = (
        f"Y4 leftover after top1 {_f(d['rank'])}; after days {_f(days['rank'])}. "
        "Y4 HHI >0.975 footnote KEEP locked — do not overwrite y4_why."
    )
    print(prose)
    return {"rows": rows, "top1": d["rank"], "days": days["rank"], "prose": prose}


def decide(p1, p2, p3, p4, p5) -> dict:
    twin = bool(p2["gate_twins"])
    size = bool(p1["size_flag"] or p2["size_flag"])
    leftover = p4["y3_rank"]
    leftover_dies = bool(p4["y3_dies"])
    beat = bool(np.isfinite(p3["beat_size"]) and p3["beat_size"] >= KEEP_DELTA)
    keep_x = bool(beat and (not leftover_dies) and (not size) and (not twin))
    y3_loses = bool(np.isfinite(p3["y3"]) and np.isfinite(p3["days"]) and p3["y3"] + 0.02 < p3["days"])
    if keep_x:
        card = "KEEP as unused leftover — still off the 15-col card"
        tag = "KEEP"
        why = (
            f"leftover after days rank {_f(leftover)} beats size "
            f"({_f(p3['size'])}) by {_f(p3['beat_size'])}; not SIZE; not a twin. "
            "Do not put d_cust_lost on the 15-col card. Do not grow TURNOVER."
        )
        park = False
    elif leftover_dies and (twin or y3_loses):
        card = "DROP from the 44 as Y3 X / CLOSE unused leftover"
        tag = "DROP"
        why = (
            f"Y3 leftover after days rank {_f(leftover)} OLS {_f(p4['y3_ols'])} "
            f"{'FALSE clone' if p4['y3_fake'] else 'dies'}; twin={twin} SIZE={size}. "
            f"Y3 {_f(p3['y3'])} vs days {_f(p3['days'])}. "
            f"n_cust leftover {N_CUST_LEFTOVER_QUOTE:.3f} DROP; top1 leftover {TOP1_LEFTOVER_QUOTE:.3f} DROP. "
            "Stay off the 15-col card. Do not invent y_cust_lost."
        )
        park = True
    elif leftover_dies or twin:
        card = "CLOSE unused leftover / KEEP off the 15-col card"
        tag = "CLOSE"
        why = (
            f"Y3 leftover after days rank {_f(leftover)} OLS {_f(p4['y3_ols'])} "
            f"{'dies' if leftover_dies else 'thin'}; twin={twin} SIZE={size}. "
            "Stay off the 15-col card. Do not invent y_cust_lost."
        )
        park = True
    else:
        card = "CLOSE unused leftover / KEEP off the 15-col card"
        tag = "CLOSE"
        why = (
            f"Y3 {_f(p3['y3'])} vs days {_f(p3['days'])} leftover {_f(leftover)}. "
            "Does not clear KEEP-as-X. Stay off the card. Do not invent y_cust_lost."
        )
        park = True
    if p5.get("rewrite_n"):
        why += " Lost leftover after n_cust dies — rewrite of customer count."
    return {"card": card, "headline_tag": tag, "why": why, "keep_x": keep_x, "park": park}


def make_png(tr: pd.DataFrame, p4: dict) -> str | None:
    if not HAS_MPL:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.4))
    x = pd.to_numeric(tr["d_cust_lost"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    m = x.notna() & days.notna()
    ax[0].scatter(days[m], x[m], s=4, alpha=0.15, c="#2c5f6e")
    ax[0].set_xlabel("c_n_days_with_tx")
    ax[0].set_ylabel("d_cust_lost")
    ax[0].set_title("lost customers vs days")
    labels = ["lost after days", "days after lost"]
    vals = [p4["y3_rank"], p4["inv_rank"]]
    colors = ["#b85c38" if p4["y3_dies"] else "#2c5f6e", "#2c5f6e"]
    ax[1].bar(labels, [0 if not np.isfinite(v) else v for v in vals], color=colors)
    ax[1].axhline(CHANCE, color="#888", ls="--", lw=0.8)
    ax[1].axhline(DAYS_BENCH, color="#2c5f6e", ls=":", lw=0.8)
    ax[1].set_ylim(0.40, 0.80)
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
        "# Unused leftover of `d_cust_lost` after `c_n_days_with_tx`",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        f"`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        f"Holdout 72 is coverage only. Seed {FOLD_SEED} group folds. No 0–100. "
        f"No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_cust_lost`. "
        f"Night Y7 stays **TURNOVER {TURNOVER_QUOTE:.3f} / B_shallow {B_SHALLOW_QUOTE:.3f}**. "
        f"Do **not** grow TURNOVER. Do **not** put d_cust_lost on the 15-col Y3 card. "
        f"Night Y3 stays **{NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f}**. Days **{DAYS_BENCH:.3f}**. "
        f"Size **{SIZE_QUOTE:.3f}**. Y7 never D. Y3 never B. Dark 470 stay NaN not 0. "
        f"`d_n_cust` leftover {N_CUST_LEFTOVER_QUOTE:.3f} DROP — do not overwrite n_cust_qa. "
        f"`d_cust_top1` leftover {TOP1_LEFTOVER_QUOTE:.3f} DROP — do not overwrite top1_qa. "
        "Y4 HHI >0.975 footnote KEEP locked — do not overwrite y4_why / cust_hhi_qa.",
        "",
        "`d_cust_lost` = |prev \\ cur| of AR counterparties in the calendar quarter "
        "vs the previous quarter. Different object from Y7 `y7_top1_lost`. "
        "Feature report: 53.6% cov, acf1 0.58, ICC 0.98 BETWEEN.",
        "",
        "## Headline",
        "",
        f"{d['headline_tag']} leftover-after-days rank {_f(ctx['p4']['y3_rank'])} "
        f"(OLS {_f(ctx['p4']['y3_ols'])}, fake={ctx['p4']['y3_fake']}). "
        f"Y3 lost {_f(ctx['p3']['y3'])} vs days {_f(ctx['p3']['days'])} vs size {_f(ctx['p3']['size'])} "
        f"vs n_cust {_f(ctx['p3']['n_cust'])} vs top1 {_f(ctx['p3']['top1'])}. "
        f"SIZE={ctx['p2']['size_flag']} twin={bool(ctx['p2']['gate_twins'])}. "
        f"Inverse days-after-lost {_f(ctx['p4']['inv_rank'])}. "
        f"Leftover after n_cust {_f(ctx['p5']['n_cust'])} / top1 {_f(ctx['p5']['top1'])} / new {_f(ctx['p5']['new'])}. "
        f"Card: {d['card']}. "
        f"{'PARK as Y — do not invent y_cust_lost.' if d['park'] else 'KEEP leftover only — still no y_cust_lost invent unless later locked.'} "
        f"Night Y3 {NIGHT_Y3:.3f}/{NIGHT_Y3_CORE:.3f}, days {DAYS_BENCH:.3f}, "
        f"size {SIZE_QUOTE:.3f}, TURNOVER {TURNOVER_QUOTE:.3f}/{B_SHALLOW_QUOTE:.3f} unchanged.",
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        f"| 1 | Who is healthy? | {'PARK as Y — do not invent y_cust_lost' if d['park'] else 'leftover only'}. Dark 470 = NaN, not 0. |",
        f"| 2 | Who is improving? | Q6 lag1 leftover after days_lag1 {_f(ctx['p6']['lag_rank'])}. |",
        f"| 3 | Who is turning? | **{d['headline_tag']}** leftover after days {_f(ctx['p4']['y3_rank'])} vs days {_f(ctx['p3']['days'])}. |",
        f"| 4 | Dip vs fall? | Y4 leftover after top1 {_f(ctx['py4']['top1'])} — footnote KEEP locked, do not overwrite y4_why. |",
        f"| 5 | Why did it change? | Twin screen: {', '.join(ctx['p2']['twins']) if ctx['p2']['twins'] else 'none'}. ρ vs n_cust {_f(ctx['p2']['rho_ncust'])}. |",
        f"| 6 | Months earlier? | lag1 leftover {_f(ctx['p6']['lag_rank'])}; days_lag1 {_f(ctx['p6']['days_l1'])} (quote {DAYS_LAG1_QUOTE:.3f}). |",
        "",
        "## PARK / CLOSE / KEEP / DROP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| lost leftover after days (Y3 X) | **{d['headline_tag']}** | {d['why']} |",
        "| 15-col Y3 card stem | **KEEP off the card** | do not put d_cust_lost on the card |",
        "| TURNOVER add-on | **CLOSE** | do not grow 0.720; Y7 never D |",
        f"| d_n_cust leftover | **DROP (locked)** | rank {N_CUST_LEFTOVER_QUOTE:.3f} |",
        f"| d_cust_top1 leftover | **DROP (locked)** | rank {TOP1_LEFTOVER_QUOTE:.3f} |",
        "| Y4 HHI >0.975 footnote | **KEEP (locked)** | do not overwrite y4_why |",
        "| health Y `y_cust_lost` | **PARK** | do not invent y_cust_lost |",
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
        "## 3. Single-feature group-fold Y3",
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
        "## 5. Leftover after n_cust / top1 / HHI / new",
        "",
        ctx["p5"]["prose"],
        "",
        _md_table(ctx["p5"]["rows"]),
        "",
        "## 6. Q6 lag1 leftover after days_lag1",
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
        "## 8. Y7 leftover after issued (report only; never D)",
        "",
        ctx["p7"]["prose"],
        "",
        _md_table(ctx["p7"]["rows"]),
        "",
        "## 9. Bootstrap leftover-after-days",
        "",
        ctx["p8"]["prose"],
        "",
        _md_table(ctx["p8"]["rows"]),
        "",
        "## Extra — holdout coverage",
        "",
        ctx["ph"]["prose"],
        "",
        _md_table(ctx["ph"]["rows"]),
        "",
        "## Extra — KEEP-as-X scorecard",
        "",
        ctx["pk"]["prose"],
        "",
        _md_table(ctx["pk"]["rows"]),
        "",
        "## Extra — SIZE terciles",
        "",
        ctx["pt"]["prose"],
        "",
        _md_table(ctx["pt"]["rows"]),
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
        "## Extra — BETWEEN identity",
        "",
        ctx["pbl"]["prose"],
        "",
        _md_table(ctx["pbl"]["rows"]),
        "",
        "## Extra — same-n / lost>0",
        "",
        ctx["psn"]["prose"],
        "",
        _md_table(ctx["psn"]["rows"]),
        "",
        "## Extra — Y4 leftover (report; footnote KEEP locked)",
        "",
        ctx["py4"]["prose"],
        "",
        _md_table(ctx["py4"]["rows"]),
        "",
        "## Extra — leftover after days+size / card KEEP / issued / n_cust+top1",
        "",
        ctx["pci"]["prose"],
        "",
        _md_table(ctx["pci"]["rows"]),
        "",
        "## Extra — company-median ρ",
        "",
        ctx["pcm"]["prose"],
        "",
        _md_table(ctx["pcm"]["rows"]),
        "",
        "## Extra — Q6 mid / long",
        "",
        ctx["pq6"]["prose"],
        "",
        _md_table(ctx["pq6"]["rows"]),
        "",
        "## Extra — lost/n_cust rate / leftover after n_tx / new after lost",
        "",
        ctx["prt"]["prose"],
        "",
        _md_table(ctx["prt"]["rows"]),
        "",
        "## Extra — leftover after n_cust_lag1 / n_cust+new+days",
        "",
        ctx["pnl"]["prose"],
        "",
        _md_table(ctx["pnl"]["rows"]),
        "",
        "## What failed / next (held for wave note)",
        "",
    ]
    for x in ctx["failed"]:
        lines.append(f"- {x}")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    existing = REGISTRY.read_text(encoding="utf-8")
    already = existing.count("cust_lost_qa")
    if already >= 5:
        print("registry: no new rows")
        return
    ts = _now_iso()
    want = [
        ("auroc_d_cust_lost", ctx["p3"]["y3"], Y3, f"days={ctx['p3']['days']:.4f} size={ctx['p3']['size']:.4f}"),
        (
            "auroc_lost_resid_days_rank",
            ctx["p4"]["y3_rank"],
            Y3,
            f"ols={ctx['p4']['y3_ols']:.4f} fake={ctx['p4']['y3_fake']}",
        ),
        (
            "auroc_lost_resid_ncust_rank",
            ctx["p5"]["n_cust"],
            Y3,
            f"rewrite={ctx['p5']['rewrite_n']}",
        ),
        (
            "auroc_lost_resid_top1_rank",
            ctx["p5"]["top1"],
            Y3,
            f"new={ctx['p5']['new']:.4f}",
        ),
        (
            "rho_lost_vs_ncust",
            ctx["p2"]["rho_ncust"],
            Y3,
            f"rho_days={ctx['p2']['rho_days']:.4f} rho_size={ctx['p2']['rho_size']:.4f}",
        ),
    ]
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "ts", "round", "wave", "agent", "x_families", "y", "model",
                "split", "metric", "value", "coverage", "notes",
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
        "# Wave 4 — customer-lost leftover after days",
        "",
        f"Agent `{AGENT}`. Train group-fold seed {FOLD_SEED}. Holdout 72 coverage only.",
        "",
        "## Files written",
        "",
        "- `analysis/evaluate/cust_lost_qa.py`",
        "- `analysis/outputs/cust_lost_qa.md`",
        f"- `{OUT_PNG.name}`" if ctx.get("png") else "- (no PNG)",
        "- append-only `analysis/experiments/registry.csv`",
        "- this note",
        "",
        "Did not touch `top1_qa.*`, `cust_hhi_qa.*`, `n_cust_qa.*`, `y4_why.*`, "
        "`delay_qa.*`, `ap_overdue_qa.*`, `ap_open_qa.*`, `dso_qa.*`, `gbm_core.py`, "
        "`counterparties.py`, `gbm_y7_core.py`, `lit_*.md`, `product/`, parquet / duckdb, "
        "`build_targets`, parent journal, LIVE, canvas, TURNOVER, or the 15-col card. "
        f"Night Y7 stays **TURNOVER {TURNOVER_QUOTE:.3f} / B_shallow {B_SHALLOW_QUOTE:.3f}**. "
        f"Night Y3 stays **{NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f}**. Days {DAYS_BENCH:.3f}. "
        f"Size {SIZE_QUOTE:.3f}.",
        "",
        "## Locked verdict",
        "",
        "| object | decision |",
        "| --- | --- |",
        f"| lost leftover after days (Y3) | **{d['headline_tag']}** |",
        "| 15-col Y3 card | **KEEP off the card** |",
        "| TURNOVER add-on | **CLOSE** |",
        "| d_n_cust leftover | **DROP (locked)** |",
        "| d_cust_top1 leftover | **DROP (locked)** |",
        "| y_cust_lost | **PARK** |",
        "",
        f"{d['headline_tag']} leftover-after-days rank {_f(ctx['p4']['y3_rank'])} "
        f"(OLS {_f(ctx['p4']['y3_ols'])}, fake={ctx['p4']['y3_fake']}). "
        f"Y3 lost {_f(ctx['p3']['y3'])} vs days {_f(ctx['p3']['days'])} vs size {_f(ctx['p3']['size'])}. "
        f"SIZE={ctx['p2']['size_flag']} twin={bool(ctx['p2']['gate_twins'])}. "
        f"Inverse days-after-lost {_f(ctx['p4']['inv_rank'])}. "
        f"Leftover after n_cust {_f(ctx['p5']['n_cust'])} rewrite={ctx['p5']['rewrite_n']}. "
        f"Leftover after top1 {_f(ctx['p5']['top1'])}. "
        f"Boot leftover-days p50={_f(ctx['p8']['p50'])} p05={_f(ctx['p8']['p05'])}. "
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
    print(f"cust_lost_qa start seed={FOLD_SEED} holdout=72 agent={AGENT} wave={WRITE_WAVE}")
    panel = load_panel()
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
    print("pass 5 leftover after n_cust / top1 / new")
    p5 = pass5_after_family(tr)
    print("pass 6 Q6")
    p6 = pass6_q6(tr)
    print("pass 7 Y7 issued report only")
    p7 = pass7_y7_issued(tr)
    print("pass 8 bootstrap")
    p8 = pass8_boot(tr)
    print("extra holdout")
    ph = pass_holdout(panel, book)
    print("extra KEEP-as-X scorecard")
    pk = pass_keep_card(p1, p2, p3, p4)
    print("extra terciles")
    pt = pass_terciles(tr)
    print("extra so_far")
    psf = pass_so_far(tr)
    print("extra per-fold leftover")
    pf = pass_folds(tr)
    print("extra BETWEEN")
    pbl = pass_between(tr)
    print("extra same-n / lost>0")
    psn = pass_pos_samen(tr)
    print("extra Y4 report")
    py4 = pass_y4_report(tr)
    print("extra leftover after days+size / card / issued")
    pci = pass_card_issued(tr)
    print("extra company-median")
    pcm = pass_company_med(tr)
    print("extra Q6 mid/long")
    pq6 = pass_q6_midlong(tr)
    print("extra lost/n_cust rate / n_tx")
    prt = pass_rate_ntx(tr)
    print("extra leftover after n_cust_lag1 / joint")
    pnl = pass_ncust_lag_joint(tr)
    png = make_png(tr, p4)
    decision = decide(p1, p2, p3, p4, p5)
    failed = [
        f"Y3 lost {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size'])} beat={_f(p3['beat_size'])}",
        f"leftover-after-days rank {_f(p4['y3_rank'])} OLS {_f(p4['y3_ols'])} fake={p4['y3_fake']} dies={p4['y3_dies']}",
        f"inverse days-after-lost {_f(p4['inv_rank'])}",
        f"leftover after n_cust {_f(p5['n_cust'])} rewrite={p5['rewrite_n']} +days {_f(p5['n_cust_days'])}",
        f"leftover after top1 {_f(p5['top1'])} +days {_f(p5['top1_days'])} HHI {_f(p5['hhi'])} new {_f(p5['new'])}",
        f"Q6 lag1 leftover {_f(p6['lag_rank'])} days_lag1 {_f(p6['days_l1'])}",
        f"Y7 issued leftover-days {_f(p7['leftover'])} leak_ok={p7['leak_ok']}",
        f"boot leftover-days p05={_f(p8['p05'])} p50={_f(p8['p50'])} p95={_f(p8['p95'])}",
        f"terciles T1 {_f(pt['t1'])} T2+T3 {_f(pt['t23'])}",
        f"so_far short {_f(psf['short'])} mid {_f(psf['mid'])}",
        f"per-fold {' '.join(pf['bits'])}",
        f"BETWEEN cmean {_f(pbl['mean'])} lag1 {_f(pbl['lag1'])} within {_f(pbl['within'])}",
        f"same-n lost {_f(psn['lost'])} days {_f(psn['days'])} beat={_f(psn['beat'])} lost>0 {_f(psn['pos'])}",
        f"Y4 leftover-top1 {_f(py4['top1'])} leftover-days {_f(py4['days'])}",
        f"leftover after days+size {_f(pci['ds'])} card {_f(pci['card'])} issued {_f(pci['iss'])} n_cust+top1 {_f(pci['both'])}",
        f"company-median ρ vs n_cust {_f(pcm['rho_ncust'])} twin={pcm['twin']}",
        f"Q6 mid leftover {_f(pq6['mid'])} long raw {_f(pq6['long'])}",
        f"lost/n_cust leftover-days {_f(prt['rate'])} raw {_f(prt['rate_raw'])} leftover-ntx {_f(prt['ntx'])} new-after-lost {_f(prt['new_after'])}",
        f"leftover after n_cust_lag1 {_f(pnl['lag'])} +days {_f(pnl['lag_days'])} n_cust+new+days {_f(pnl['joint'])}",
        f"card: {decision['card']}",
        "do not grow TURNOVER 0.720; do not put d_cust_lost on the 15-col card; PARK y_cust_lost",
    ]
    ctx = {
        "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6, "p7": p7, "p8": p8,
        "ph": ph, "pk": pk, "pt": pt, "psf": psf, "pf": pf, "pbl": pbl, "psn": psn, "py4": py4,
        "pci": pci, "pcm": pcm, "pq6": pq6, "prt": prt, "pnl": pnl,
        "png": png, "decision": decision, "failed": failed,
    }
    write_md(ctx)
    append_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    elapsed = time.time() - t0
    print(f"DONE elapsed={elapsed:.0f}s tag={decision['headline_tag']} card={decision['card']}")
    print(
        f"{decision['headline_tag']} leftover-after-days rank {_f(p4['y3_rank'])} "
        f"(OLS {_f(p4['y3_ols'])}, fake={p4['y3_fake']}). "
        f"Y3 lost {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size'])}. "
        f"SIZE={p2['size_flag']} twin={bool(p2['gate_twins'])}. "
        f"Inverse days-after-lost {_f(p4['inv_rank'])}. "
        f"Leftover after n_cust {_f(p5['n_cust'])}. Leftover after top1 {_f(p5['top1'])}. "
        f"Card: {decision['card']}. "
        f"Night Y3 {NIGHT_Y3:.3f}/{NIGHT_Y3_CORE:.3f}, days {DAYS_BENCH:.3f}, "
        f"size {SIZE_QUOTE:.3f}, TURNOVER {TURNOVER_QUOTE:.3f}/{B_SHALLOW_QUOTE:.3f} unchanged."
    )
    return ctx


if __name__ == "__main__":
    run()


