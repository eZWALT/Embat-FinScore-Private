"""Unused leftover of `d_supp_hhi` on the 44 — monopoly tail, or twin?

NORTH_STAR: Family D `d_supp_hhi` = sum_i (amt_i / tot)^2 over AP
invoice counterparties in the trailing 6-month window (same window as
Javier concentration). Needs identified supplier CPs. Dark 470 stay
**NaN not 0**. Javier 14: concentration is **top1**, not HHI.

Y5 already PARK `d_supp_hhi` as Y5 X (size AUROC 0.663). Unused leftover:
(1) same monopoly tail as Y4 customer HHI, or a different object?
(2) leftover after `d_cust_hhi` / after `d_supp_top1` / after days as Y3 X?
(3) should the 44 lose `d_supp_hhi`?

KEEP-as-X: beat size ≥0.02 AND leftover after the honest bar AND not SIZE
(|ρ|≥0.50) AND not a twin (|ρ|≥0.80 vs `d_supp_top1` / `d_cust_hhi` /
`d_tx_cp_share`). Leftover <0.55 dies.

Honest bars: Y3 after days; Y4 after `d_cust_hhi_lag3` / body-only
(drop >0.975); Y5 AP after the tail and after size — diagnostic only,
do not revive trees.

Night quotes unchanged: Y3 0.762 / 0.752; days 0.711; size 0.617;
Y7 TURNOVER 0.720 / 0.712. Y7 never D. Y5 never E. Y3 never B.
Off the 15-col card. Do not grow TURNOVER. Do not invent `y_supp_hhi`.
Do not merge with Y4. Do not reopen Y4/Y5 trees.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.supp_hhi_qa

Owned: analysis/evaluate/supp_hhi_qa.py, analysis/outputs/supp_hhi_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_supp_hhi.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "supp_hhi_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "supp_hhi_quintiles.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "c9e14b20"
WAVE = "4"
ROUND = "R4"
MODEL = "supp_hhi_qa"
X_FAM = "D"
WRITE_WAVE = True
WAVE_PATH = ROOT / "overnight" / "waves" / "wave4_supp_hhi.md"

Y3 = "y3_recover_cash_6m"
Y4 = "y4_ds_r_double"
Y5 = "y5_ap_od30_ownp80"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
NIGHT_Y3 = 0.762
NIGHT_Y3_CORE = 0.752
TURNOVER_QUOTE = 0.720
B_SHALLOW_QUOTE = 0.712
Y4_HHI_LAG3 = 0.605
Y4_BODY = 0.445
Y4_TOP1_RHO = 0.991
Y5_TAIL_HI = 0.027
Y5_TAIL_REST = 0.086
Y5_LEFTOVER = 0.651
Y5_SIZE_AUROC = 0.663
KEEP_DELTA = 0.02
CHANCE = 0.55
SIZE_RHO = 0.50
SIZE_PARK = 0.60
TWIN_RHO = 0.80
ICC_TRAIT = 0.85
ICC_SHOCK = 0.50
MIN_POS = 50
MIN_ACF_PAIRS = 4
N_DARK_WANT = 470
TAIL_CUT = 0.975
OWN_P_HI = 0.80
OWN_P_LO = 0.20
MIN_OWN_HIST = 6
Y5_LEFTOVER_QUOTE = 0.651

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "a_io_ratio",
    "c_n_days_with_tx",
    "d_supp_hhi",
    "d_supp_top1",
    "d_n_supp",
    "d_cust_hhi",
    "d_cust_top1",
    "d_n_cust",
    "d_tx_cp_share",
)

Y_KEEP = (Y3, Y4, Y5)


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


def pearson(a, b) -> float:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    ).dropna()
    if len(d) < 8 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="pearson"))


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
        return {"icc": float("nan"), "eta2": float("nan"), "var_w": float("nan"), "var_b": float("nan"), "k": 0, "n": 0}
    y = s["x"].to_numpy(dtype=float)
    _, inv, counts = np.unique(s["co"].to_numpy(), return_inverse=True, return_counts=True)
    if counts.size < 2:
        return {"icc": float("nan"), "eta2": float("nan"), "var_w": float("nan"), "var_b": float("nan"), "k": 0, "n": 0}
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
    return {"icc": float(icc), "eta2": float(eta2), "var_w": float(var_w), "var_b": float(var_b), "k": k, "n": n}


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


def rank_resid(y: pd.Series, x: pd.Series) -> pd.Series:
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


def company_mean(series: pd.Series, company: pd.Series) -> pd.Series:
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    return d.groupby("co")["x"].transform("mean")


def size_rank_auroc(df: pd.DataFrame, col: str, mask: pd.Series) -> dict:
    """Does this column rank large vs small firms? Train-median a_in3 cut (y5_why)."""
    x = pd.to_numeric(df[col], errors="coerce")
    size = pd.to_numeric(df["a_in3"], errors="coerce")
    m = mask & x.notna() & size.notna()
    if int(m.sum()) < 50:
        return {"auroc": float("nan"), "rho": float("nan"), "n": int(m.sum())}
    med = float(size[m].median())
    large = (size[m] > med).astype(float)
    auc = auroc(large, x[m])
    auc2 = max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan")
    return {
        "auroc": float(auc2) if np.isfinite(auc2) else float("nan"),
        "rho": spearman(x[m], np.log1p(size[m].abs())),
        "n": int(m.sum()),
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
        "train": _f(res["train_auc"]) if not res["low_power"] else "—",
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
        s = pd.to_numeric(out[c], errors="coerce")
        g = s.groupby(cid, sort=False)
        for k in lags:
            extra[f"{c}_lag{k}"] = g.shift(k)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    return out


def _expanding_quantile_skipna(s: pd.Series, q: float, min_periods: int) -> pd.Series:
    out = np.full(len(s), np.nan, dtype=float)
    vals: list[float] = []
    for i, v in enumerate(s.to_numpy(dtype=float)):
        if np.isfinite(v):
            vals.append(float(v))
        if len(vals) >= min_periods:
            out[i] = float(np.quantile(vals, q))
    return pd.Series(out, index=s.index)


def add_own_cuts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-company expanding p80 / p20 (months ≤ t, min 6 finite). Never pooled."""
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    extra = {}
    x = pd.to_numeric(out["d_supp_hhi"], errors="coerce")
    p80 = x.groupby(out["company_id"], sort=False).transform(
        lambda s: _expanding_quantile_skipna(s, OWN_P_HI, MIN_OWN_HIST)
    )
    extra["d_supp_hhi_hi"] = pd.Series(
        np.where(x.notna() & p80.notna(), (x > p80).astype(float), np.nan),
        index=out.index,
    )
    io = pd.to_numeric(out["a_io_ratio"], errors="coerce")
    p20 = io.groupby(out["company_id"], sort=False).transform(
        lambda s: _expanding_quantile_skipna(s, OWN_P_LO, MIN_OWN_HIST)
    )
    extra["a_io_ratio_lo"] = pd.Series(
        np.where(io.notna() & p20.notna(), (io < p20).astype(float), np.nan),
        index=out.index,
    )
    return pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)


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
    leak3 = leakage_check(["d_supp_hhi", "c_n_days_with_tx"], Y3, forbidden_prefixes=["b"])
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 X leak: {leak3['issues']}")
    leak4 = leakage_check(["d_supp_hhi", "d_cust_hhi"], Y4, forbidden_prefixes=["f"])
    if not leak4["ok"]:
        raise RuntimeError(f"Y4 X leak: {leak4['issues']}")
    leak5 = leakage_check(["d_supp_hhi", "d_supp_top1"], Y5, forbidden_prefixes=["e"])
    if not leak5["ok"]:
        raise RuntimeError(f"Y5 X leak: {leak5['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def _two_by_two(lab: pd.DataFrame, pos: pd.Series) -> dict:
    both = pos & lab["a_io_ratio_lo"].notna() & lab["d_supp_hhi_hi"].notna()
    lo = lab.loc[both, "a_io_ratio_lo"] == 1
    hh = lab.loc[both, "d_supp_hhi_hi"] == 1
    n = int(both.sum())
    neither = int((~lo & ~hh).sum())
    return {
        "n": n,
        "cash_only": int((lo & ~hh).sum()),
        "hhi_only": int((~lo & hh).sum()),
        "both": int((lo & hh).sum()),
        "neither": neither,
        "neither_share": (neither / n) if n else float("nan"),
    }


# ---------------------------------------------------------------------------
# 1. Coverage; 470 dark NaN vs invoice-book; ever-n
# ---------------------------------------------------------------------------
def cut1_coverage(panel: pd.DataFrame, book: set[str], dark: dict) -> dict:
    rows = []
    for split, sl in (
        ("train", panel["split"] == "train"),
        ("holdout", panel["split"] == "holdout"),
    ):
        d = panel.loc[sl]
        p = pd.to_numeric(d["d_supp_hhi"], errors="coerce")
        n = len(d)
        nn = int(p.notna().sum())
        na = int(p.isna().sum())
        n_co = int(d["company_id"].nunique())
        ever = int(d.loc[p.notna(), "company_id"].nunique())
        book_m = d["company_id"].isin(book)
        dark_m = ~book_m
        dark_nn = int(p[dark_m].notna().sum())
        dark_zero = int((p[dark_m] == 0).sum())
        book_nn = int(p[book_m].notna().sum())
        book_na = int(p[book_m].isna().sum())
        n_dark_co = int(d.loc[dark_m, "company_id"].nunique())
        n_book_co = int(d.loc[book_m, "company_id"].nunique())
        n_eq0 = int((p == 0).sum())
        if split == "train":
            assert_no_holdout(d["company_id"])
        rows.append(
            {
                "split": split,
                "cm": f"{n:,}",
                "companies": f"{n_co:,}",
                "HHI nn": f"{nn:,}",
                "cov": _pp(nn / n if n else float("nan")),
                "NaN": f"{na:,}",
                "ever-n": f"{ever:,}",
                "ever-ERP / never": f"{n_book_co} / {n_dark_co}",
                "dark nn / zero": f"{dark_nn} / {dark_zero}",
                "ERP nn / NaN": f"{book_nn:,} / {book_na:,}",
                "HHI==0": f"{n_eq0:,}",
            }
        )
        print(
            f"1 {split}: cov={nn/n if n else float('nan'):.4f} nn={nn} na={na} "
            f"ever={ever} dark_co={n_dark_co} dark_nn={dark_nn} dark_zero={dark_zero}"
        )

    tr = panel[panel["split"] == "train"]
    p = pd.to_numeric(tr["d_supp_hhi"], errors="coerce")
    top1 = pd.to_numeric(tr["d_supp_top1"], errors="coerce")
    n_supp = pd.to_numeric(tr["d_n_supp"], errors="coerce")
    train_dark_ids = set(dark.get("train_dark_ids") or set())
    if not train_dark_ids:
        train_dark_ids = set(tr.loc[~tr["company_id"].isin(book), "company_id"].astype(str))
    n_dark_co = len(train_dark_ids)
    dark_cm = tr["company_id"].isin(train_dark_ids)
    dark_nn = int(p[dark_cm].notna().sum())
    dark_zero = int((p[dark_cm] == 0).sum())
    dark_top1_nn = int(top1[dark_cm].notna().sum())
    dark_ok = dark_nn == 0 and dark_zero == 0 and n_dark_co == N_DARK_WANT
    n0_erp = int((tr.loc[~dark_cm, "d_n_supp"] == 0).sum())
    hhi_when_n0 = int(p[(~dark_cm) & (n_supp == 0)].notna().sum())
    acf1 = median_acf(p, tr["company_id"], 1)
    acf3 = median_acf(p, tr["company_id"], 3)
    size_rho = spearman(p, tr["log_in3"])
    prose = (
        f"Train `d_supp_hhi` coverage {_pp(_pct(int(p.notna().sum()), len(tr)))} "
        f"({int(p.notna().sum()):,}/{len(tr):,}); ever-n "
        f"{tr.loc[p.notna(), 'company_id'].nunique()} companies. "
        f"Dark {n_dark_co} (want {N_DARK_WANT}): HHI non-null {dark_nn} "
        f"zero-filled {dark_zero} top1 nn {dark_top1_nn}. "
        f"470 stay NaN not 0: {'CONFIRM' if dark_ok else 'FAIL'}. "
        f"ERP n_supp==0 months {n0_erp:,}; HHI defined on those {hhi_when_n0} "
        f"(want 0 — n=0 keeps HHI NaN). acf1={_f(acf1)} size ρ={_f(size_rho)}."
    )
    print(prose)
    return {
        "rows": rows,
        "prose": prose,
        "train_cov": _pct(int(p.notna().sum()), len(tr)),
        "train_nn": int(p.notna().sum()),
        "train_na": int(p.isna().sum()),
        "ever_n": int(tr.loc[p.notna(), "company_id"].nunique()),
        "n_dark_co": n_dark_co,
        "dark_nn": dark_nn,
        "dark_zero": dark_zero,
        "dark_top1_nn": dark_top1_nn,
        "dark_ok": dark_ok,
        "n0_erp": n0_erp,
        "hhi_when_n0": hhi_when_n0,
        "acf1": acf1,
        "acf3": acf3,
        "size_rho": size_rho,
        "hold_cov": _pct(
            int(pd.to_numeric(panel.loc[panel["split"] == "holdout", "d_supp_hhi"], errors="coerce").notna().sum()),
            int((panel["split"] == "holdout").sum()),
        ),
        "n_train": len(tr),
        "n_train_co": int(tr["company_id"].nunique()),
        "confirm_470": bool(dark.get("confirm_470")),
    }


# ---------------------------------------------------------------------------
# 2. Spearman twins
# ---------------------------------------------------------------------------
def cut2_twins(tr: pd.DataFrame) -> dict:
    p = tr["d_supp_hhi"]
    pairs = [
        ("d_cust_hhi", tr["d_cust_hhi"]),
        ("d_cust_top1", tr["d_cust_top1"]),
        ("d_supp_top1", tr["d_supp_top1"]),
        ("d_tx_cp_share", tr["d_tx_cp_share"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("d_n_supp", tr["d_n_supp"]),
        ("d_cust_hhi_lag3", tr["d_cust_hhi_lag3"]),
        ("d_supp_top1_lag3", tr["d_supp_top1_lag3"]),
        ("a_io_ratio", tr["a_io_ratio"]),
    ]
    official = {"d_cust_hhi", "d_supp_top1", "d_tx_cp_share"}
    rhos = {}
    pears = {}
    ns = {}
    twins = []
    rows = []
    for name, s in pairs:
        rho = spearman(p, s)
        pr = pearson(p, s)
        d = pd.DataFrame({"a": pd.to_numeric(p, errors="coerce"), "b": pd.to_numeric(s, errors="coerce")}).dropna()
        rhos[name] = rho
        pears[name] = pr
        ns[name] = int(len(d))
        twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO and name in official)
        if twin:
            twins.append(name)
        size_flag = name == "log1p(a_in3)" and bool(np.isfinite(rho) and abs(rho) >= SIZE_RHO)
        rows.append(
            {
                "vs": name,
                "n": f"{len(d):,}",
                "Spearman": _f(rho),
                "Pearson": _f(pr),
                "twin ≥0.80": "YES" if twin else ("SIZE" if size_flag else "no"),
            }
        )
        print(f"2 ρ vs {name}: {_f(rho)} n={len(d)} twin={twin}")
    size_flag = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    top1_twin = bool(np.isfinite(rhos["d_supp_top1"]) and abs(rhos["d_supp_top1"]) >= TWIN_RHO)
    prose = (
        f"ρ vs `d_supp_top1` {_f(rhos['d_supp_top1'])} n={ns['d_supp_top1']:,} "
        f"({'TWIN — HHI is the weaker rewrite' if top1_twin else 'not a twin'}). "
        f"vs `d_cust_hhi` {_f(rhos['d_cust_hhi'])} vs `d_tx_cp_share` {_f(rhos['d_tx_cp_share'])} "
        f"vs size {_f(rhos['log1p(a_in3)'])} ({'SIZE' if size_flag else 'not SIZE'}). "
        f"Javier: concentration is top1. Y4 customer HHI↔top1 ρ {Y4_TOP1_RHO}."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "pears": pears,
        "ns": ns,
        "twins": twins,
        "any_twin": bool(twins),
        "top1_twin": top1_twin,
        "size_flag": size_flag,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 3. Quintiles + >0.975 tail on Y3 / Y4 / Y5 AP
# ---------------------------------------------------------------------------
def _quintile_block(df: pd.DataFrame, mask: pd.Series, x_col: str, y_col: str) -> dict:
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    m = mask & x.notna() & y.notna()
    d = df.loc[m, [x_col]].copy()
    d[y_col] = y[m].to_numpy()
    if int(m.sum()) < 40 or d[x_col].nunique() < 5:
        return {"rows": [], "up": False, "down": False, "n": int(m.sum()), "bins": 0}
    cats, bins = pd.qcut(d[x_col], 5, retbins=True, duplicates="drop")
    d = d.copy()
    d["q"] = cats
    rows = []
    for i, (q, g) in enumerate(d.groupby("q", observed=True), start=1):
        rows.append(
            {
                "y": y_col,
                "Q": i,
                "interval": str(q),
                "n": int(len(g)),
                "n_pos": int((g[y_col] == 1).sum()),
                "P(Y=1)": _f(float(g[y_col].mean()), 3),
                "median HHI": _f(float(g[x_col].median()), 4),
            }
        )
    rates = [float(r["P(Y=1)"]) if r["P(Y=1)"] != "—" else float("nan") for r in rows]
    up = bool(len(rates) >= 3 and all(a <= b + 1e-9 for a, b in zip(rates, rates[1:])))
    down = bool(len(rates) >= 3 and all(a >= b - 1e-9 for a, b in zip(rates, rates[1:])))
    return {"rows": rows, "up": up, "down": down, "n": int(m.sum()), "bins": len(rows), "raw_rates": rates}


def cut3_tail(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["d_supp_hhi"], errors="coerce")
    x3 = pd.to_numeric(tr["d_supp_hhi_lag3"], errors="coerce")
    q_rows = []
    q_store = {}
    for y in (Y3, Y4, Y5):
        q = _quintile_block(tr, tr[y].notna(), "d_supp_hhi", y)
        q_store[y] = q
        q_rows.extend(q["rows"])
        print(f"3 quintiles {y}: n={q['n']} up={q['up']} down={q['down']}")

    tail_rows = []
    body_store = {}
    for y, xcol, xx in (
        (Y3, "d_supp_hhi", x),
        (Y4, "d_supp_hhi", x),
        (Y4, "d_supp_hhi_lag3", x3),
        (Y5, "d_supp_hhi", x),
    ):
        yv = pd.to_numeric(tr[y], errors="coerce")
        m = yv.notna() & xx.notna()
        hi = m & (xx > TAIL_CUT)
        lo = m & (xx <= TAIL_CUT)
        rate_hi = float(yv[hi].mean()) if int(hi.sum()) else float("nan")
        rate_lo = float(yv[lo].mean()) if int(lo.sum()) else float("nan")
        n_hi = int(hi.sum())
        n_pos_hi = int((hi & (yv == 1)).sum())
        n_lo = int(lo.sum())
        n_pos_lo = int((lo & (yv == 1)).sum())
        tail_auc = auroc(yv[m], (xx[m] > TAIL_CUT).astype(float))
        body = signed_oof_auroc(tr[y], xx, tr["fold"], lo)
        body_store[(y, xcol)] = body
        protective = bool(np.isfinite(rate_hi) and np.isfinite(rate_lo) and rate_hi < rate_lo)
        tail_rows.append(
            {
                "y": y,
                "x": xcol,
                "n tail / pos": f"{n_hi} / {n_pos_hi}",
                "P(Y=1) tail": _f(rate_hi),
                "P(Y=1) rest": _f(rate_lo),
                "tail AUROC": _f(tail_auc),
                "body CV": "LOW_POWER" if body["low_power"] else _f(body["cv"]),
                "body n / pos": f"{body['n_defined']} / {body['n_pos']}",
                "shape": "protective" if protective else ("crash" if (np.isfinite(rate_hi) and rate_hi > rate_lo) else "?"),
            }
        )
        print(
            f"3 tail {y} {xcol}>0.975 n={n_hi} pos={n_pos_hi} "
            f"rate={rate_hi:.4f} rest={rate_lo:.4f} body={_cv(body):.4f}"
        )

    y5_hi = next(r for r in tail_rows if r["y"] == Y5 and r["x"] == "d_supp_hhi")
    # parse rates from the store rather than formatted strings
    y5_m = tr[Y5].notna() & x.notna()
    y5_rate_hi = float(tr.loc[y5_m & (x > TAIL_CUT), Y5].mean()) if int((y5_m & (x > TAIL_CUT)).sum()) else float("nan")
    y5_rate_lo = float(tr.loc[y5_m & (x <= TAIL_CUT), Y5].mean()) if int((y5_m & (x <= TAIL_CUT)).sum()) else float("nan")
    y5_confirm = bool(
        np.isfinite(y5_rate_hi)
        and np.isfinite(y5_rate_lo)
        and abs(y5_rate_hi - Y5_TAIL_HI) < 0.015
        and abs(y5_rate_lo - Y5_TAIL_REST) < 0.015
        and y5_rate_hi < y5_rate_lo
    )
    y4_body = _cv(body_store[(Y4, "d_supp_hhi")])
    y4_body_lag3 = _cv(body_store[(Y4, "d_supp_hhi_lag3")])
    y3_body = _cv(body_store[(Y3, "d_supp_hhi")])
    y5_body = _cv(body_store[(Y5, "d_supp_hhi")])
    y4_dead = bool(np.isfinite(y4_body) and y4_body < CHANCE)
    prose = (
        f"Y5 AP tail >0.975 P(Y=1)={_f(y5_rate_hi)} vs rest {_f(y5_rate_lo)} "
        f"(quote {Y5_TAIL_HI:.1%} vs {Y5_TAIL_REST:.1%}) — "
        f"{'CONFIRM protective' if y5_confirm else 'CHECK vs quote'}. "
        f"Body HHI≤0.975 CV Y3 {_f(y3_body)} Y4 {_f(y4_body)} "
        f"(lag3 {_f(y4_body_lag3)}; Y4 customer body {Y4_BODY}) Y5 {_f(y5_body)}. "
        f"{'Y4 body dead like customer 0.445' if y4_dead else 'Y4 body still ranks'}."
    )
    print(prose)
    return {
        "q_rows": q_rows,
        "q_store": q_store,
        "tail_rows": tail_rows,
        "body_store": body_store,
        "y5_rate_hi": y5_rate_hi,
        "y5_rate_lo": y5_rate_lo,
        "y5_confirm": y5_confirm,
        "y3_body": y3_body,
        "y4_body": y4_body,
        "y4_body_lag3": y4_body_lag3,
        "y5_body": y5_body,
        "y4_dead": y4_dead,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 4. Single-feature group-fold AUROC
# ---------------------------------------------------------------------------
def cut4_singles(tr: pd.DataFrame) -> dict:
    feats = [
        ("d_supp_hhi", tr["d_supp_hhi"]),
        ("d_supp_top1", tr["d_supp_top1"]),
        ("d_cust_hhi", tr["d_cust_hhi"]),
        ("d_cust_hhi_lag3", tr["d_cust_hhi_lag3"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("d_n_supp", tr["d_n_supp"]),
        ("d_tx_cp_share", tr["d_tx_cp_share"]),
        ("d_supp_hhi_lag1", tr["d_supp_hhi_lag1"]),
        ("d_supp_hhi_lag3", tr["d_supp_hhi_lag3"]),
    ]
    rows = []
    store = {}
    size_on = {}
    for y in (Y3, Y4, Y5):
        lab = tr[y].notna()
        for name, s in feats:
            res = signed_oof_auroc(tr[y], s, tr["fold"], lab)
            store[(y, name)] = res
            rows.append(_auc_row(y, name, res))
            print(f"4 {y} {name}: {_f(_cv(res))} n={res['n_defined']} pos={res['n_pos']}")
        # size AUROC on rows where supp HHI is defined (Y5 PARK gate)
        m = lab & tr["d_supp_hhi"].notna() & tr["log_in3"].notna()
        size_on[y] = auroc(tr.loc[m, y], tr.loc[m, "log_in3"])
    # y5_why SIZE_PARK: does the column rank large vs small? (not size-vs-Y)
    size_rank = {}
    for y in (Y3, Y4, Y5):
        size_rank[y] = size_rank_auroc(tr, "d_supp_hhi", tr[y].notna())
    size_y3 = _cv(store[(Y3, "log1p(a_in3)")])
    days_y3 = _cv(store[(Y3, "c_n_days_with_tx")])
    supp_y3 = _cv(store[(Y3, "d_supp_hhi")])
    supp_y4 = _cv(store[(Y4, "d_supp_hhi")])
    supp_y5 = _cv(store[(Y5, "d_supp_hhi")])
    top1_y3 = _cv(store[(Y3, "d_supp_top1")])
    top1_y4 = _cv(store[(Y4, "d_supp_top1")])
    top1_y5 = _cv(store[(Y5, "d_supp_top1")])
    cust3_y4 = _cv(store[(Y4, "d_cust_hhi_lag3")])
    size_y4 = _cv(store[(Y4, "log1p(a_in3)")])
    size_y5 = _cv(store[(Y5, "log1p(a_in3)")])
    beat_y3 = bool(np.isfinite(supp_y3) and np.isfinite(size_y3) and supp_y3 >= size_y3 + KEEP_DELTA)
    beat_y4 = bool(np.isfinite(supp_y4) and np.isfinite(size_y4) and supp_y4 >= size_y4 + KEEP_DELTA)
    beat_y5 = bool(np.isfinite(supp_y5) and np.isfinite(size_y5) and supp_y5 >= size_y5 + KEEP_DELTA)
    y5_size_park = bool(np.isfinite(size_rank[Y5]["auroc"]) and size_rank[Y5]["auroc"] >= SIZE_PARK)
    prose = (
        f"Y3 supp HHI {_f(supp_y3)} vs size {_f(size_y3)} (night {SIZE_QUOTE}) "
        f"days {_f(days_y3)} (night {DAYS_BENCH}) top1 {_f(top1_y3)}. "
        f"Y4 supp HHI {_f(supp_y4)} vs size {_f(size_y4)} cust_hhi_lag3 {_f(cust3_y4)} "
        f"(night {Y4_HHI_LAG3}) top1 {_f(top1_y4)}. "
        f"Y5 AP supp HHI {_f(supp_y5)} vs size {_f(size_y5)} top1 {_f(top1_y5)}; "
        f"size-vs-Y on defined {_f(size_on[Y5])}; "
        f"size-rank AUROC {_f(size_rank[Y5]['auroc'])} (y5_why quote {Y5_SIZE_AUROC}; "
        f"{'SIZE_PARK' if y5_size_park else 'not size-park'}). "
        f"Beat-size Y3={beat_y3} Y4={beat_y4} Y5={beat_y5}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "size_on": size_on,
        "size_rank": size_rank,
        "supp_y3": supp_y3,
        "supp_y4": supp_y4,
        "supp_y5": supp_y5,
        "top1_y3": top1_y3,
        "top1_y4": top1_y4,
        "top1_y5": top1_y5,
        "size_y3": size_y3,
        "size_y4": size_y4,
        "size_y5": size_y5,
        "days_y3": days_y3,
        "cust3_y4": cust3_y4,
        "beat_y3": beat_y3,
        "beat_y4": beat_y4,
        "beat_y5": beat_y5,
        "y5_size_park": y5_size_park,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 5. Honest leftover
# ---------------------------------------------------------------------------
def _leftover_specs(tr: pd.DataFrame, y: str, feat: pd.Series, specs: list[tuple]) -> dict:
    lab = tr[y].notna()
    rows = []
    store = {}
    infos = {}
    for name, xs in specs:
        resid, info = ols_resid(feat, *xs)
        res = signed_oof_auroc(tr[y], resid, tr["fold"], lab)
        store[name] = res
        infos[name] = info
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
        print(f"5 leftover {y} {name}: {_f(_cv(res))} R2={_f(info['r2'])}")
    return {"rows": rows, "store": store, "infos": infos}


def cut5_leftover(tr: pd.DataFrame) -> dict:
    p = tr["d_supp_hhi"]
    tail_flag = pd.Series(np.where(p.notna(), (p > TAIL_CUT).astype(float), np.nan), index=tr.index)
    y3 = _leftover_specs(
        tr,
        Y3,
        p,
        [
            ("after days", (tr["c_n_days_with_tx"],)),
            ("after size", (tr["log_in3"],)),
            ("after days+size", (tr["c_n_days_with_tx"], tr["log_in3"])),
            ("after d_supp_top1", (tr["d_supp_top1"],)),
            ("after d_cust_hhi", (tr["d_cust_hhi"],)),
        ],
    )
    y4 = _leftover_specs(
        tr,
        Y4,
        p,
        [
            ("after d_cust_hhi_lag3", (tr["d_cust_hhi_lag3"],)),
            ("after d_cust_hhi", (tr["d_cust_hhi"],)),
            ("after d_supp_top1", (tr["d_supp_top1"],)),
            ("after size", (tr["log_in3"],)),
            ("after tail-flag", (tail_flag,)),
        ],
    )
    y5 = _leftover_specs(
        tr,
        Y5,
        p,
        [
            ("after size", (tr["log_in3"],)),
            ("after tail-flag", (tail_flag,)),
            ("after size+tail", (tr["log_in3"], tail_flag)),
            ("after d_supp_top1", (tr["d_supp_top1"],)),
            ("after days", (tr["c_n_days_with_tx"],)),
        ],
    )
    # rank leftover + body-only already in cut3
    lab3 = tr[Y3].notna()
    lab4 = tr[Y4].notna()
    lab5 = tr[Y5].notna()
    r_days = signed_oof_auroc(tr[Y3], rank_resid(p, tr["c_n_days_with_tx"]), tr["fold"], lab3)
    r_cust = signed_oof_auroc(tr[Y4], rank_resid(p, tr["d_cust_hhi_lag3"]), tr["fold"], lab4)
    r_size = signed_oof_auroc(tr[Y5], rank_resid(p, tr["log_in3"]), tr["fold"], lab5)
    r_top1 = signed_oof_auroc(tr[Y3], rank_resid(p, tr["d_supp_top1"]), tr["fold"], lab3)
    y3["rows"].append(
        {
            "y": Y3,
            "residual": "rank-resid after days",
            "n": f"{r_days['n_defined']:,}",
            "n_pos": f"{r_days['n_pos']:,}",
            "CV": "LOW_POWER" if r_days["low_power"] else _f(r_days["cv"]),
            "R²": "—",
            "folds": fold_bits(r_days) if not r_days["low_power"] else "—",
        }
    )
    y4["rows"].append(
        {
            "y": Y4,
            "residual": "rank-resid after cust_hhi_lag3",
            "n": f"{r_cust['n_defined']:,}",
            "n_pos": f"{r_cust['n_pos']:,}",
            "CV": "LOW_POWER" if r_cust["low_power"] else _f(r_cust["cv"]),
            "R²": "—",
            "folds": fold_bits(r_cust) if not r_cust["low_power"] else "—",
        }
    )
    y5["rows"].append(
        {
            "y": Y5,
            "residual": "rank-resid after size",
            "n": f"{r_size['n_defined']:,}",
            "n_pos": f"{r_size['n_pos']:,}",
            "CV": "LOW_POWER" if r_size["low_power"] else _f(r_size["cv"]),
            "R²": "—",
            "folds": fold_bits(r_size) if not r_size["low_power"] else "—",
        }
    )
    after_days = _cv(y3["store"]["after days"])
    after_cust = _cv(y4["store"]["after d_cust_hhi_lag3"])
    after_size = _cv(y5["store"]["after size"])
    after_tail_y5 = _cv(y5["store"]["after tail-flag"])
    after_top1_y3 = _cv(y3["store"]["after d_supp_top1"])
    after_top1_y4 = _cv(y4["store"]["after d_supp_top1"])
    after_top1_y5 = _cv(y5["store"]["after d_supp_top1"])
    lives_y3 = bool(np.isfinite(after_days) and after_days >= CHANCE)
    lives_y4 = bool(np.isfinite(after_cust) and after_cust >= CHANCE)
    lives_y5 = bool(np.isfinite(after_size) and after_size >= CHANCE)
    died_y3 = bool(np.isfinite(after_days) and after_days < CHANCE)
    died_y4 = bool(np.isfinite(after_cust) and after_cust < CHANCE)
    died_y5 = bool(np.isfinite(after_size) and after_size < CHANCE)
    prose = (
        f"Y3 leftover after days {_f(after_days)} "
        f"({'lives' if lives_y3 else 'dies <0.55'}); "
        f"after top1 {_f(after_top1_y3)}. "
        f"Y4 leftover after cust_hhi_lag3 {_f(after_cust)} "
        f"({'lives' if lives_y4 else 'dies <0.55'}); after top1 {_f(after_top1_y4)}. "
        f"Y5 leftover after size {_f(after_size)} "
        f"({'lives' if lives_y5 else 'dies <0.55'}); after tail-flag {_f(after_tail_y5)} "
        f"after top1 {_f(after_top1_y5)}. Rank-ortho days {_f(_cv(r_days))} "
        f"cust {_f(_cv(r_cust))} size {_f(_cv(r_size))} top1 {_f(_cv(r_top1))}."
    )
    print(prose)
    return {
        "rows": y3["rows"] + y4["rows"] + y5["rows"],
        "y3": y3,
        "y4": y4,
        "y5": y5,
        "after_days": after_days,
        "after_cust": after_cust,
        "after_size": after_size,
        "after_tail_y5": after_tail_y5,
        "after_top1_y3": after_top1_y3,
        "after_top1_y4": after_top1_y4,
        "after_top1_y5": after_top1_y5,
        "rank_days": _cv(r_days),
        "rank_cust": _cv(r_cust),
        "rank_size": _cv(r_size),
        "rank_top1": _cv(r_top1),
        "lives_y3": lives_y3,
        "lives_y4": lives_y4,
        "lives_y5": lives_y5,
        "died_y3": died_y3,
        "died_y4": died_y4,
        "died_y5": died_y5,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 6. Twin vs top1 — drop weaker
# ---------------------------------------------------------------------------
def cut6_twin(tr: pd.DataFrame, twins: dict, singles: dict) -> dict:
    rho = twins["rhos"]["d_supp_top1"]
    # Javier 14 locked concentration as top1. When |ρ|≥0.80, HHI is the rewrite
    # even if mean CV is 0.006 higher (0.572 vs 0.566 this run).
    weaker = "d_supp_hhi" if twins["top1_twin"] else None
    mh = float(np.nanmean([singles["supp_y3"], singles["supp_y4"], singles["supp_y5"]]))
    mt = float(np.nanmean([singles["top1_y3"], singles["top1_y4"], singles["top1_y5"]]))
    prose = (
        f"ρ(HHI, top1)={_f(rho)}. "
        f"{'TWIN ≥0.80 — DROP weaker `d_supp_hhi` (Javier concentration is top1).' if weaker else 'Not a twin; keep both in the screen.'} "
        f"Mean CV HHI {_f(mh)} top1 {_f(mt)} (HHI +0.006 is not a KEEP). "
        f"Y4 customer HHI↔top1 ρ {Y4_TOP1_RHO}."
    )
    print(prose)
    return {"rho": rho, "weaker": weaker, "drop_hhi": weaker == "d_supp_hhi", "mean_hhi": mh, "mean_top1": mt, "prose": prose}


# ---------------------------------------------------------------------------
# 7. SIZE terciles — does the tail survive inside T1?
# ---------------------------------------------------------------------------
def cut7_terciles(tr: pd.DataFrame) -> dict:
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    x = pd.to_numeric(tr["d_supp_hhi"], errors="coerce")
    defined = size.notna()
    try:
        tercile = pd.qcut(size[defined], 3, labels=["T1", "T2", "T3"], duplicates="drop")
        terc = pd.Series(index=tr.index, dtype=object)
        terc.loc[defined] = tercile.astype(str)
        cats = ["T1", "T2", "T3"]
    except ValueError:
        terc = None
        cats = []
    rows = []
    t1_survive = {}
    for y in (Y3, Y4, Y5):
        yv = pd.to_numeric(tr[y], errors="coerce")
        if terc is None:
            continue
        for i, cat in enumerate(cats, start=1):
            sl = yv.notna() & x.notna() & (terc == cat)
            hi = sl & (x > TAIL_CUT)
            lo = sl & (x <= TAIL_CUT)
            rate_hi = float(yv[hi].mean()) if int(hi.sum()) else float("nan")
            rate_lo = float(yv[lo].mean()) if int(lo.sum()) else float("nan")
            body = signed_oof_auroc(tr[y], x, tr["fold"], lo)
            raw = signed_oof_auroc(tr[y], x, tr["fold"], sl)
            rec = {
                "y": y,
                "tercile": f"T{i}",
                "n": int(sl.sum()),
                "n_pos": int((sl & (yv == 1)).sum()),
                "n tail / pos": f"{int(hi.sum())} / {int((hi & (yv == 1)).sum())}",
                "P(Y=1) tail": _f(rate_hi),
                "P(Y=1) rest": _f(rate_lo),
                "raw CV": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "body CV": "LOW_POWER" if body["low_power"] else _f(body["cv"]),
            }
            rows.append(rec)
            if i == 1:
                t1_survive[y] = {
                    "rate_hi": rate_hi,
                    "rate_lo": rate_lo,
                    "n_hi": int(hi.sum()),
                    "n_pos_hi": int((hi & (yv == 1)).sum()),
                    "raw": _cv(raw),
                    "body": _cv(body),
                    "protective": bool(np.isfinite(rate_hi) and np.isfinite(rate_lo) and rate_hi < rate_lo),
                }
            print(
                f"7 {y} T{i}: n={rec['n']} tail={int(hi.sum())} "
                f"rate_hi={rate_hi if np.isfinite(rate_hi) else float('nan'):.4f} "
                f"raw={_cv(raw)}"
            )
    t1 = t1_survive.get(Y5, {})
    prose = (
        f"Y5 T1 tail P(Y=1)={_f(t1.get('rate_hi'))} vs rest {_f(t1.get('rate_lo'))} "
        f"(n_tail={t1.get('n_hi', 0)} pos={t1.get('n_pos_hi', 0)}). "
        f"{'Protective tail survives inside T1' if t1.get('protective') else 'T1 tail is not the protective story / LOW_POWER'}. "
        f"Y3 T1 raw {_f((t1_survive.get(Y3) or {}).get('raw'))} Y4 T1 raw {_f((t1_survive.get(Y4) or {}).get('raw'))}."
    )
    print(prose)
    return {"rows": rows, "t1": t1_survive, "prose": prose}


# ---------------------------------------------------------------------------
# 8. Q6 lag1/lag3 on short books
# ---------------------------------------------------------------------------
def cut8_q6(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for y in (Y3, Y4, Y5):
        for sl_name, sl in (
            ("short_<12", tr["so_far_class"] == "short_<12"),
            ("long_>=18", tr["so_far_class"] == "long_>=18"),
            ("all", pd.Series(True, index=tr.index)),
        ):
            lab = sl & tr[y].notna()
            for feat in ("d_supp_hhi", "d_supp_hhi_lag1", "d_supp_hhi_lag3"):
                res = signed_oof_auroc(tr[y], tr[feat], tr["fold"], lab)
                store[(y, sl_name, feat)] = res
                rows.append(
                    {
                        "y": y,
                        "slice": sl_name,
                        "feature": feat,
                        "n": f"{res['n_defined']:,}",
                        "n_pos": f"{res['n_pos']:,}",
                        "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                        "sd": _f(res["sd"]) if not res["low_power"] else "—",
                        "sign": res["train_sign"] if not res["low_power"] else "—",
                    }
                )
                print(f"8 {y} {sl_name} {feat}: {_f(_cv(res))} pos={res['n_pos']}")
    short_y4_lag3 = _cv(store[(Y4, "short_<12", "d_supp_hhi_lag3")])
    short_y4_lag1 = _cv(store[(Y4, "short_<12", "d_supp_hhi_lag1")])
    short_y5_lag1 = _cv(store[(Y5, "short_<12", "d_supp_hhi_lag1")])
    all_y4_lag3 = _cv(store[(Y4, "all", "d_supp_hhi_lag3")])
    q6_close = True  # Y4 Q6 HHI was CLOSE / LOW_POWER 21.7%
    prose = (
        f"Y4 short lag3 {_f(short_y4_lag3)} lag1 {_f(short_y4_lag1)} "
        f"(Y4 Q6 customer HHI was CLOSE / LOW_POWER 21.7%). "
        f"Y4 all lag3 {_f(all_y4_lag3)}. Y5 short lag1 {_f(short_y5_lag1)}. "
        f"Q6 stays CLOSE unless a short-book lag clears size+0.02 — it does not."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "short_y4_lag3": short_y4_lag3,
        "short_y4_lag1": short_y4_lag1,
        "short_y5_lag1": short_y5_lag1,
        "all_y4_lag3": all_y4_lag3,
        "q6_close": q6_close,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 9. ICC / company-demean
# ---------------------------------------------------------------------------
def cut9_icc(tr: pd.DataFrame) -> dict:
    x = tr["d_supp_hhi"]
    icc = icc_anova(x, tr["company_id"])
    demean = company_demean(x, tr["company_id"])
    cmean = company_mean(x, tr["company_id"])
    rows = []
    store = {}
    for y in (Y3, Y4, Y5):
        lab = tr[y].notna()
        d = signed_oof_auroc(tr[y], demean, tr["fold"], lab)
        m = signed_oof_auroc(tr[y], cmean, tr["fold"], lab)
        store[(y, "demean")] = d
        store[(y, "mean")] = m
        rows.append(_auc_row(y, "company-demean", d))
        rows.append(_auc_row(y, "company-mean", m))
        print(f"9 {y} demean={_f(_cv(d))} mean={_f(_cv(m))}")
    trait = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_TRAIT)
    shock = bool(np.isfinite(icc["icc"]) and icc["icc"] < ICC_SHOCK)
    prose = (
        f"ICC {_f(icc['icc'])} η² {_f(icc['eta2'])} k={icc['k']} "
        f"({'TRAIT ≥0.85' if trait else ('shock <0.50' if shock else 'mixed')}). "
        f"Y3 demean {_f(_cv(store[(Y3, 'demean')]))} mean {_f(_cv(store[(Y3, 'mean')]))}. "
        f"Y5 demean {_f(_cv(store[(Y5, 'demean')]))} mean {_f(_cv(store[(Y5, 'mean')]))}."
    )
    print(prose)
    return {
        "icc": icc,
        "rows": rows,
        "store": store,
        "trait": trait,
        "shock": shock,
        "d3": _cv(store[(Y3, "demean")]),
        "m3": _cv(store[(Y3, "mean")]),
        "d4": _cv(store[(Y4, "demean")]),
        "m4": _cv(store[(Y4, "mean")]),
        "d5": _cv(store[(Y5, "demean")]),
        "m5": _cv(store[(Y5, "mean")]),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 10. Dark 470: HHI defined? Confirm NaN not 0
# ---------------------------------------------------------------------------
def cut10_dark(tr: pd.DataFrame, book: set[str], dark: dict) -> dict:
    train_dark = set(dark.get("train_dark_ids") or set())
    x = pd.to_numeric(tr["d_supp_hhi"], errors="coerce")
    top1 = pd.to_numeric(tr["d_supp_top1"], errors="coerce")
    n_cust = pd.to_numeric(tr["d_cust_hhi"], errors="coerce")
    dark_m = tr["company_id"].isin(train_dark)
    book_m = tr["company_id"].isin(book)
    ghost = tr["company_id"] == "COMP_0962"
    rows = [
        {
            "slice": "dark 470 CM",
            "cm": f"{int(dark_m.sum()):,}",
            "HHI nn": int(x[dark_m].notna().sum()),
            "HHI==0": int((x[dark_m] == 0).sum()),
            "top1 nn": int(top1[dark_m].notna().sum()),
            "cust HHI nn": int(n_cust[dark_m].notna().sum()),
        },
        {
            "slice": "invoice-book CM",
            "cm": f"{int(book_m.sum()):,}",
            "HHI nn": int(x[book_m].notna().sum()),
            "HHI==0": int((x[book_m] == 0).sum()),
            "top1 nn": int(top1[book_m].notna().sum()),
            "cust HHI nn": int(n_cust[book_m].notna().sum()),
        },
        {
            "slice": "COMP_0962 (refund ghost)",
            "cm": f"{int(ghost.sum()):,}",
            "HHI nn": int(x[ghost].notna().sum()),
            "HHI==0": int((x[ghost] == 0).sum()),
            "top1 nn": int(top1[ghost].notna().sum()),
            "cust HHI nn": int(n_cust[ghost].notna().sum()),
        },
    ]
    ok = int(x[dark_m].notna().sum()) == 0 and int((x[dark_m] == 0).sum()) == 0
    prose = (
        f"Dark 470 HHI defined {int(x[dark_m].notna().sum())} zero {int((x[dark_m] == 0).sum())}. "
        f"{'CONFIRM NaN not 0 — HHI needs invoice CPs.' if ok else 'FAIL — dark HHI leaked a number'}. "
        f"COMP_0962 HHI nn={int(x[ghost].notna().sum())} (refund-only ghost; live-dark)."
    )
    print(prose)
    return {"rows": rows, "ok": ok, "prose": prose, "dark_nn": int(x[dark_m].notna().sum())}


# ---------------------------------------------------------------------------
# 11. Y5 leftover 65% — drop protective tail
# ---------------------------------------------------------------------------
def cut11_y5_leftover(tr: pd.DataFrame) -> dict:
    pos = tr[Y5] == 1
    full = _two_by_two(tr, pos)
    x = pd.to_numeric(tr["d_supp_hhi"], errors="coerce")
    drop_tail = pos & ~(x > TAIL_CUT)
    dropped = _two_by_two(tr, drop_tail)
    # also: 2x2 among labeled AP excluding tail months (pos and neg)
    labeled = tr[Y5].notna() & ~(x > TAIL_CUT)
    # leftover share among remaining positives
    delta = (
        dropped["neither_share"] - full["neither_share"]
        if np.isfinite(dropped["neither_share"]) and np.isfinite(full["neither_share"])
        else float("nan")
    )
    increased = bool(np.isfinite(delta) and delta > 0.005)
    rows = [
        {
            "slice": "all AP pos (quote 65.1%)",
            "n 2×2": full["n"],
            "cash_only": full["cash_only"],
            "hhi_only": full["hhi_only"],
            "both": full["both"],
            "neither": full["neither"],
            "leftover": _pp(full["neither_share"]),
        },
        {
            "slice": "drop HHI>0.975 pos",
            "n 2×2": dropped["n"],
            "cash_only": dropped["cash_only"],
            "hhi_only": dropped["hhi_only"],
            "both": dropped["both"],
            "neither": dropped["neither"],
            "leftover": _pp(dropped["neither_share"]),
        },
    ]
    n_tail_pos = int((pos & (x > TAIL_CUT)).sum())
    n_lab_tail = int((tr[Y5].notna() & (x > TAIL_CUT)).sum())
    prose = (
        f"AP leftover 2×2 neither {full['neither']}/{full['n']} = {_pp(full['neither_share'])} "
        f"(quote {_pp(Y5_LEFTOVER_QUOTE)}). After dropping the protective tail "
        f"({n_tail_pos} pos / {n_lab_tail} labeled months) leftover "
        f"{dropped['neither']}/{dropped['n']} = {_pp(dropped['neither_share'])} "
        f"(Δ {_f(delta, 3)}). "
        f"{'Dropping the tail increases unexplained share.' if increased else 'Dropping the tail does not raise leftover (tail held almost no positives).'} "
        f"Do not invent a new Y."
    )
    print(prose)
    return {
        "rows": rows,
        "full": full,
        "dropped": dropped,
        "delta": delta,
        "increased": increased,
        "n_tail_pos": n_tail_pos,
        "n_lab_tail": n_lab_tail,
        "labeled_no_tail": int(labeled.sum()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra A — holdout coverage / mix (LOW_POWER)
# ---------------------------------------------------------------------------
def extra_holdout(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    x = pd.to_numeric(ho["d_supp_hhi"], errors="coerce")
    rows = []
    for y in (Y3, Y4, Y5):
        yv = pd.to_numeric(ho[y], errors="coerce")
        m = yv.notna() & x.notna()
        hi = m & (x > TAIL_CUT)
        lo = m & (x <= TAIL_CUT)
        rows.append(
            {
                "y": y,
                "n labeled": int(yv.notna().sum()),
                "n_pos": int((yv == 1).sum()),
                "HHI nn": int(m.sum()),
                "n tail / pos": f"{int(hi.sum())} / {int((hi & (yv == 1)).sum())}",
                "P(Y=1) tail": _f(float(yv[hi].mean()) if int(hi.sum()) else float("nan")),
                "P(Y=1) rest": _f(float(yv[lo].mean()) if int(lo.sum()) else float("nan")),
            }
        )
    # Y4 mix: we do not rebuild crash/spike; coverage only
    prose = (
        f"Holdout 72 is coverage / mix only (Y4 train crash 80% / holdout flipped "
        f"38% crash / 88% spike — LOW_POWER). Do not quote holdout AUROC. "
        + " ".join(
            f"{r['y']} labeled {r['n labeled']} pos {r['n_pos']} HHI-nn {r['HHI nn']}."
            for r in rows
        )
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra B — 2-col z-avg of top1 + HHI (CLOSE expected)
# ---------------------------------------------------------------------------
def extra_zavg(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for y in (Y3, Y4, Y5):
        lab = tr[y].notna()
        yv = pd.to_numeric(tr[y], errors="coerce")
        xs = {
            "d_supp_hhi": pd.to_numeric(tr["d_supp_hhi"], errors="coerce"),
            "d_supp_top1": pd.to_numeric(tr["d_supp_top1"], errors="coerce"),
        }
        both = lab
        for s in xs.values():
            both = both & s.notna()
        n_pos = int((both & (yv == 1)).sum())
        if n_pos < MIN_POS:
            res = {
                "cv": float("nan"),
                "sd": float("nan"),
                "low_power": True,
                "n_defined": int(both.sum()),
                "n_pos": n_pos,
                "n_neg": int((both & (yv == 0)).sum()),
                "folds": [],
                "train_sign": 0,
                "train_auc": float("nan"),
            }
        else:
            aucs = []
            fold_rows = []
            for k in range(N_FOLDS):
                trm = both & (tr["fold"] != k)
                va = both & (tr["fold"] == k)
                parts = []
                for _, s in xs.items():
                    sign = choose_sign(yv[trm], s[trm])
                    mu = float(s[trm].mean())
                    sd = float(s[trm].std(ddof=0))
                    z = (s - mu) / sd if sd and np.isfinite(sd) and sd > 0 else s * 0.0
                    parts.append(sign * z)
                score = sum(parts)
                auc = auroc(yv[va], score[va])
                aucs.append(auc)
                fold_rows.append(
                    {
                        "fold": k,
                        "auroc": float(auc) if np.isfinite(auc) else float("nan"),
                        "n_va": int(va.sum()),
                        "n_pos": int((va & (yv == 1)).sum()),
                    }
                )
            finite = [a for a in aucs if np.isfinite(a)]
            res = {
                "cv": float(np.mean(finite)) if finite else float("nan"),
                "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
                "low_power": False,
                "n_defined": int(both.sum()),
                "n_pos": n_pos,
                "n_neg": int((both & (yv == 0)).sum()),
                "folds": fold_rows,
                "train_sign": 1,
                "train_auc": float("nan"),
            }
        store[y] = res
        hhi_cc = signed_oof_auroc(tr[y], tr["d_supp_hhi"], tr["fold"], both)
        top_cc = signed_oof_auroc(tr[y], tr["d_supp_top1"], tr["fold"], both)
        gap_hhi = (
            _cv(res) - _cv(hhi_cc)
            if np.isfinite(_cv(res)) and np.isfinite(_cv(hhi_cc))
            else float("nan")
        )
        just_hhi = bool(np.isfinite(gap_hhi) and gap_hhi < KEEP_DELTA)
        rows.append(
            {
                "y": y,
                "zavg CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "HHI-CC": "LOW_POWER" if hhi_cc["low_power"] else _f(hhi_cc["cv"]),
                "top1-CC": "LOW_POWER" if top_cc["low_power"] else _f(top_cc["cv"]),
                "gap vs HHI": _f(gap_hhi),
                "just HHI": str(just_hhi),
                "n / pos": f"{res['n_defined']} / {res['n_pos']}",
            }
        )
        print(f"B zavg {y}: {_f(_cv(res))} HHI-CC {_f(_cv(hhi_cc))} top1-CC {_f(_cv(top_cc))}")
    prose = (
        f"2-col z-avg of `d_supp_hhi` + `d_supp_top1` is a twin stack. "
        f"Y4 customer z-avg was CLOSE. Do not put this on any card. "
        f"Y3 {_f(_cv(store[Y3]))} Y4 {_f(_cv(store[Y4]))} Y5 {_f(_cv(store[Y5]))}."
    )
    print(prose)
    return {"rows": rows, "store": store, "prose": prose}


# ---------------------------------------------------------------------------
# Extra C — same-n leftover vs top1 (is leftover just the twin?)
# ---------------------------------------------------------------------------
def extra_samen(tr: pd.DataFrame) -> dict:
    p = tr["d_supp_hhi"]
    t1 = tr["d_supp_top1"]
    rows = []
    store = {}
    for y in (Y3, Y4, Y5):
        both = tr[y].notna() & p.notna() & t1.notna()
        raw = signed_oof_auroc(tr[y], p, tr["fold"], both)
        top = signed_oof_auroc(tr[y], t1, tr["fold"], both)
        resid, info = ols_resid(p, t1)
        lef = signed_oof_auroc(tr[y], resid, tr["fold"], both)
        store[y] = {"raw": raw, "top": top, "lef": lef, "r2": info["r2"]}
        rows.append(_auc_row(y, "HHI same-n", raw))
        rows.append(_auc_row(y, "top1 same-n", top))
        rows.append(_auc_row(y, "HHI resid after top1", lef))
        print(f"C same-n {y}: HHI {_f(_cv(raw))} top1 {_f(_cv(top))} leftover {_f(_cv(lef))} R2={_f(info['r2'])}")
    artifact = bool(
        np.isfinite(store[Y3]["r2"]) and store[Y3]["r2"] >= 0.90
    )
    prose = (
        f"Same-n HHI vs top1: Y3 leftover after top1 {_f(_cv(store[Y3]['lef']))} "
        f"R²={_f(store[Y3]['r2'])}. "
        f"{'Near-identity — leftover after top1 is not a new object.' if artifact else 'R² < 0.90 — not a copy, still check leftover CV.'}"
    )
    print(prose)
    return {"rows": rows, "store": store, "artifact": artifact, "prose": prose}


# ---------------------------------------------------------------------------
# Extra D — tail companies on non-tail months (Y4 shape)
# ---------------------------------------------------------------------------
def extra_tail_cos(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["d_supp_hhi"], errors="coerce")
    tail_cos = set(tr.loc[x > TAIL_CUT, "company_id"].astype(str))
    rows = []
    for y in (Y3, Y4, Y5):
        yv = pd.to_numeric(tr[y], errors="coerce")
        m = yv.notna() & x.notna() & tr["company_id"].isin(tail_cos)
        hi = m & (x > TAIL_CUT)
        lo = m & (x <= TAIL_CUT)
        rows.append(
            {
                "y": y,
                "tail cos": int(tr.loc[m, "company_id"].nunique()),
                "tail months / pos": f"{int(hi.sum())} / {int((hi & (yv == 1)).sum())}",
                "P(Y=1) tail mo": _f(float(yv[hi].mean()) if int(hi.sum()) else float("nan")),
                "non-tail mo / pos": f"{int(lo.sum())} / {int((lo & (yv == 1)).sum())}",
                "P(Y=1) non-tail mo": _f(float(yv[lo].mean()) if int(lo.sum()) else float("nan")),
            }
        )
    prose = (
        f"Companies that ever hit supp HHI>0.975: {len(tail_cos)}. "
        "If those same books are quiet on non-tail months, the object is the bin, not a gradient."
    )
    print(prose)
    return {"rows": rows, "n_cos": len(tail_cos), "prose": prose}


# ---------------------------------------------------------------------------
# Extra E — size-rank AUROC replica (y5_why 0.663)
# ---------------------------------------------------------------------------
def extra_size_rank(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for y, col in (
        (Y5, "d_supp_hhi"),
        (Y5, "d_supp_top1"),
        (Y5, "d_n_supp"),
        (Y3, "d_supp_hhi"),
        (Y4, "d_supp_hhi"),
    ):
        rec = size_rank_auroc(tr, col, tr[y].notna())
        store[(y, col)] = rec
        rows.append(
            {
                "y": y,
                "feature": col,
                "size-rank AUROC": _f(rec["auroc"]),
                "size ρ": _f(rec["rho"]),
                "n": f"{rec['n']:,}",
                "SIZE_PARK": str(bool(np.isfinite(rec["auroc"]) and rec["auroc"] >= SIZE_PARK)),
            }
        )
        print(f"E size-rank {y} {col}: {_f(rec['auroc'])} ρ={_f(rec['rho'])}")
    y5 = store[(Y5, "d_supp_hhi")]["auroc"]
    match = bool(np.isfinite(y5) and abs(y5 - Y5_SIZE_AUROC) < 0.02)
    prose = (
        f"Y5 AP size-rank AUROC of `d_supp_hhi` {_f(y5)} "
        f"(y5_why quote {Y5_SIZE_AUROC}) — {'CONFIRM' if match else 'CHECK'}. "
        f"This is large-vs-small ranking, not size-vs-Y. PARK as Y5 X stands."
    )
    print(prose)
    return {"rows": rows, "store": store, "y5": y5, "match": match, "prose": prose}


# ---------------------------------------------------------------------------
# Extra F — Y3 0.653 pocket: leftover after days inside T2/T3
# ---------------------------------------------------------------------------
def extra_y3_pocket(tr: pd.DataFrame) -> dict:
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    defined = size.notna()
    tercile = pd.qcut(size[defined], 3, labels=["T1", "T2", "T3"], duplicates="drop")
    terc = pd.Series(index=tr.index, dtype=object)
    terc.loc[defined] = tercile.astype(str)
    p = tr["d_supp_hhi"]
    rows = []
    store = {}
    for cat in ("T1", "T2", "T3", "T2+T3"):
        if cat == "T2+T3":
            sl = (terc == "T2") | (terc == "T3")
        else:
            sl = terc == cat
        lab = sl & tr[Y3].notna()
        raw = signed_oof_auroc(tr[Y3], p, tr["fold"], lab)
        days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], lab)
        resid, info = ols_resid(p, tr["c_n_days_with_tx"])
        lef = signed_oof_auroc(tr[Y3], resid, tr["fold"], lab)
        store[cat] = {"raw": raw, "days": days, "lef": lef, "r2": info["r2"]}
        rows.append(
            {
                "slice": cat,
                "n / pos": f"{raw['n_defined']} / {raw['n_pos']}",
                "HHI CV": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "days CV": "LOW_POWER" if days["low_power"] else _f(days["cv"]),
                "leftover days": "LOW_POWER" if lef["low_power"] else _f(lef["cv"]),
            }
        )
        print(f"F Y3 {cat}: HHI {_f(_cv(raw))} days {_f(_cv(days))} leftover {_f(_cv(lef))}")
    t23 = store["T2+T3"]
    dies = bool(np.isfinite(_cv(t23["lef"])) and _cv(t23["lef"]) < CHANCE)
    prose = (
        f"Y3 T2+T3 HHI {_f(_cv(t23['raw']))} leftover-after-days {_f(_cv(t23['lef']))} "
        f"vs days {_f(_cv(t23['days']))}. "
        f"{'Pocket dies after days — the 0.653 is activity, not leftover turning.' if dies else 'Pocket leftover lives — unexpected.'}"
    )
    print(prose)
    return {"rows": rows, "store": store, "dies": dies, "lef_t23": _cv(t23["lef"]), "prose": prose}


# ---------------------------------------------------------------------------
# Extra G — leftover after d_n_supp (ρ −0.726; Y3 n_supp 0.699)
# ---------------------------------------------------------------------------
def extra_n_supp(tr: pd.DataFrame) -> dict:
    p = tr["d_supp_hhi"]
    rows = []
    store = {}
    for y in (Y3, Y4, Y5):
        resid, info = ols_resid(p, tr["d_n_supp"])
        lef = signed_oof_auroc(tr[y], resid, tr["fold"], tr[y].notna())
        ncv = signed_oof_auroc(tr[y], tr["d_n_supp"], tr["fold"], tr[y].notna())
        store[y] = {"lef": lef, "n": ncv, "r2": info["r2"]}
        rows.append(_auc_row(y, "d_n_supp", ncv))
        rows.append(_auc_row(y, "HHI resid after n_supp", lef))
        print(f"G {y}: n_supp {_f(_cv(ncv))} leftover {_f(_cv(lef))} R2={_f(info['r2'])}")
    prose = (
        f"ρ(HHI, n_supp)={_f(spearman(p, tr['d_n_supp']))}. "
        f"Y3 n_supp {_f(_cv(store[Y3]['n']))} leftover after n_supp {_f(_cv(store[Y3]['lef']))} "
        f"R²={_f(store[Y3]['r2'])} — not a linear count rewrite. Extra L: leftover after "
        f"n_supp+days dies (days was leaking through the 0.637)."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "y3_lef": _cv(store[Y3]["lef"]),
        "y3_n": _cv(store[Y3]["n"]),
        "r2": store[Y3]["r2"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra H — Y3 leftover after days of the company-mean (style)
# ---------------------------------------------------------------------------
def extra_style(tr: pd.DataFrame) -> dict:
    p = tr["d_supp_hhi"]
    cmean = company_mean(p, tr["company_id"])
    resid, info = ols_resid(cmean, tr["c_n_days_with_tx"])
    lef = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    raw = signed_oof_auroc(tr[Y3], cmean, tr["fold"], tr[Y3].notna())
    days_after, info2 = ols_resid(tr["c_n_days_with_tx"], p)
    days_inv = signed_oof_auroc(tr[Y3], days_after, tr["fold"], tr[Y3].notna())
    rows = [
        _auc_row(Y3, "company-mean HHI", raw),
        _auc_row(Y3, "mean resid after days", lef),
        _auc_row(Y3, "days resid after HHI", days_inv),
    ]
    prose = (
        f"Y3 company-mean HHI {_f(_cv(raw))} leftover after days {_f(_cv(lef))} "
        f"R²={_f(info['r2'])}. Inverse: days after HHI {_f(_cv(days_inv))} "
        f"R²={_f(info2['r2'])} (days 0.711 should survive). "
        f"{'Style leftover dies — who-has-concentrated-suppliers is days.' if (np.isfinite(_cv(lef)) and _cv(lef) < CHANCE) else 'Style leftover still ranks.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "mean": _cv(raw),
        "lef": _cv(lef),
        "days_inv": _cv(days_inv),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra I — holdout tail mix (coverage only)
# ---------------------------------------------------------------------------
def extra_hold_tail(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    x = pd.to_numeric(ho["d_supp_hhi"], errors="coerce")
    rows = []
    for y in (Y3, Y4, Y5):
        yv = pd.to_numeric(ho[y], errors="coerce")
        m = yv.notna() & x.notna()
        hi = m & (x > TAIL_CUT)
        lo = m & (x <= TAIL_CUT)
        rows.append(
            {
                "y": y,
                "hold n / pos": f"{int(yv.notna().sum())} / {int((yv == 1).sum())}",
                "HHI nn": int(m.sum()),
                "tail n / pos": f"{int(hi.sum())} / {int((hi & (yv == 1)).sum())}",
                "P(Y=1) tail": _f(float(yv[hi].mean()) if int(hi.sum()) else float("nan")),
                "P(Y=1) rest": _f(float(yv[lo].mean()) if int(lo.sum()) else float("nan")),
            }
        )
    y4_pos = int((pd.to_numeric(ho[Y4], errors="coerce") == 1).sum())
    prose = (
        f"Holdout Y4 positives {y4_pos} (train mix flipped 38% crash / 88% spike — LOW_POWER). "
        "Do not quote holdout AUROC. Tail n is a coverage check only."
    )
    print(prose)
    return {"rows": rows, "y4_pos": y4_pos, "prose": prose}


# ---------------------------------------------------------------------------
# Extra J — Y3 body leftover after days (body CV 0.661)
# ---------------------------------------------------------------------------
def extra_y3_body_days(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["d_supp_hhi"], errors="coerce")
    body = tr[Y3].notna() & x.notna() & (x <= TAIL_CUT)
    raw = signed_oof_auroc(tr[Y3], x, tr["fold"], body)
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], body)
    resid, info = ols_resid(x, tr["c_n_days_with_tx"])
    lef = signed_oof_auroc(tr[Y3], resid, tr["fold"], body)
    rows = [
        _auc_row(Y3, "HHI body ≤0.975", raw),
        _auc_row(Y3, "days on body", days),
        _auc_row(Y3, "HHI body resid after days", lef),
    ]
    dies = bool(np.isfinite(_cv(lef)) and _cv(lef) < CHANCE)
    prose = (
        f"Y3 body HHI {_f(_cv(raw))} leftover-after-days {_f(_cv(lef))} "
        f"vs days {_f(_cv(days))} R²={_f(info['r2'])}. "
        f"{'Body gradient dies after days — not a monopoly-tail leftover.' if dies else 'Body leftover lives.'}"
    )
    print(prose)
    return {"rows": rows, "raw": _cv(raw), "lef": _cv(lef), "days": _cv(days), "dies": dies, "prose": prose}


# ---------------------------------------------------------------------------
# Extra K — Y4 short lag3 0.606 leftover after cust HHI (Q6 tease)
# ---------------------------------------------------------------------------
def extra_q6_honest(tr: pd.DataFrame) -> dict:
    short = tr["so_far_class"] == "short_<12"
    rows = []
    store = {}
    for y, bar, barname in (
        (Y4, tr["d_cust_hhi_lag3"], "cust_hhi_lag3"),
        (Y3, tr["c_n_days_with_tx"], "days"),
        (Y5, tr["log_in3"], "size"),
    ):
        lab = short & tr[y].notna()
        raw = signed_oof_auroc(tr[y], tr["d_supp_hhi_lag3"], tr["fold"], lab)
        resid, info = ols_resid(tr["d_supp_hhi_lag3"], bar)
        lef = signed_oof_auroc(tr[y], resid, tr["fold"], lab)
        store[y] = {"raw": raw, "lef": lef, "r2": info["r2"]}
        rows.append(
            {
                "y": y,
                "bar": barname,
                "lag3 CV": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "leftover": "LOW_POWER" if lef["low_power"] else _f(lef["cv"]),
                "n / pos": f"{raw['n_defined']} / {raw['n_pos']}",
                "R²": _f(info["r2"]),
            }
        )
        print(f"K short {y} lag3 {_f(_cv(raw))} leftover-after-{barname} {_f(_cv(lef))}")
    q6_keep = bool(
        np.isfinite(_cv(store[Y4]["lef"]))
        and _cv(store[Y4]["lef"]) >= CHANCE
        and np.isfinite(_cv(store[Y4]["raw"]))
        and _cv(store[Y4]["raw"]) >= SIZE_QUOTE + KEEP_DELTA
    )
    prose = (
        f"Y4 short lag3 {_f(_cv(store[Y4]['raw']))} leftover after cust_hhi_lag3 "
        f"{_f(_cv(store[Y4]['lef']))}. Y3 short lag3 leftover after days "
        f"{_f(_cv(store[Y3]['lef']))}. "
        f"{'Unexpected Q6 KEEP' if q6_keep else 'Q6 CLOSE — short-book lag is not leftover after the honest bar.'}"
    )
    print(prose)
    return {"rows": rows, "store": store, "q6_keep": q6_keep, "prose": prose}


# ---------------------------------------------------------------------------
# Extra L — leftover after n_supp + days (0.637 after n_supp only)
# ---------------------------------------------------------------------------
def extra_n_and_days(tr: pd.DataFrame) -> dict:
    p = tr["d_supp_hhi"]
    resid, info = ols_resid(p, tr["d_n_supp"], tr["c_n_days_with_tx"])
    lef = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    rnk = signed_oof_auroc(
        tr[Y3],
        rank_resid(rank_resid(p, tr["d_n_supp"]), tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    # fold-wise leftover after days (is 0.464 one fold?)
    days_res, _ = ols_resid(p, tr["c_n_days_with_tx"])
    days_lef = signed_oof_auroc(tr[Y3], days_res, tr["fold"], tr[Y3].notna())
    rows = [
        _auc_row(Y3, "HHI resid after n_supp+days", lef),
        _auc_row(Y3, "rank resid n_supp then days", rnk),
        _auc_row(Y3, "OLS resid after days (folds)", days_lef),
    ]
    dies = bool(np.isfinite(_cv(lef)) and _cv(lef) < CHANCE)
    prose = (
        f"Y3 leftover after n_supp+days {_f(_cv(lef))} R²={_f(info['r2'])} "
        f"(after n_supp-only was 0.637). Rank-ortho {_f(_cv(rnk))}. "
        f"Days-leftover folds {fold_bits(days_lef)}. "
        f"{'Dies after n_supp+days — count+activity eat the 0.653.' if dies else 'Still ranks after n_supp+days — but twin of top1, still DROP.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "lef": _cv(lef),
        "rnk": _cv(rnk),
        "r2": info["r2"],
        "dies": dies,
        "folds": fold_bits(days_lef),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 12 + decide
# ---------------------------------------------------------------------------
def decide(ctx: dict) -> dict:
    c2, c3, c4, c5, c6, c8, c9, c11 = (
        ctx["c2"],
        ctx["c3"],
        ctx["c4"],
        ctx["c5"],
        ctx["c6"],
        ctx["c8"],
        ctx["c9"],
        ctx["c11"],
    )
    keep_y3 = bool(
        c4["beat_y3"]
        and c5["lives_y3"]
        and not c2["size_flag"]
        and not c2["any_twin"]
        and not c4["y5_size_park"]
    )
    keep_y4 = bool(
        c4["beat_y4"]
        and c5["lives_y4"]
        and not c2["size_flag"]
        and not c2["any_twin"]
        and np.isfinite(c3["y4_body"])
        and c3["y4_body"] >= CHANCE
    )
    keep_y5 = False  # diagnostic only; trees stay PARK

    if c6["drop_hhi"]:
        y3 = "DROP"
        y3_why = (
            f"twin of `d_supp_top1` ρ={_f(c2['rhos']['d_supp_top1'])} — HHI is the weaker rewrite. "
            f"Y3 {_f(c4['supp_y3'])} leftover-after-days {_f(c5['after_days'])}."
        )
        y4 = "DROP"
        y4_why = (
            f"same twin. Y4 {_f(c4['supp_y4'])} leftover after cust_hhi_lag3 {_f(c5['after_cust'])}; "
            f"body CV {_f(c3['y4_body'])} (customer body {Y4_BODY})."
        )
        y5 = "PARK"
        y5_why = (
            f"already PARK as Y5 X (size-rank AUROC {_f(c4['size_rank'][Y5]['auroc'])}, "
            f"quote {Y5_SIZE_AUROC}). Twin of top1. "
            f"Tail is protective {_f(c3['y5_rate_hi'])} vs {_f(c3['y5_rate_lo'])}."
        )
    elif c5["died_y3"] and not c4["beat_y3"]:
        y3 = "DROP"
        y3_why = (
            f"Y3 {_f(c4['supp_y3'])} vs size {_f(c4['size_y3'])} / days {_f(c4['days_y3'])}; "
            f"leftover after days {_f(c5['after_days'])} dies <0.55."
        )
        y4 = "CLOSE" if not (c5["died_y4"] and c3["y4_dead"]) else "DROP"
        y4_why = (
            f"Y4 {_f(c4['supp_y4'])} leftover after cust_hhi_lag3 {_f(c5['after_cust'])}; "
            f"body {_f(c3['y4_body'])}."
        )
        y5 = "PARK"
        y5_why = (
            f"PARK as Y5 X. Raw {_f(c4['supp_y5'])} leftover-after-size {_f(c5['after_size'])}. "
            f"Do not revive trees."
        )
    else:
        y3 = "KEEP" if keep_y3 else ("CLOSE" if c5["lives_y3"] else "DROP")
        y3_why = (
            f"Y3 {_f(c4['supp_y3'])} leftover-after-days {_f(c5['after_days'])} "
            f"vs size {_f(c4['size_y3'])} days {_f(c4['days_y3'])}."
        )
        y4 = "KEEP" if keep_y4 else ("CLOSE" if c5["lives_y4"] else "DROP")
        y4_why = (
            f"Y4 {_f(c4['supp_y4'])} leftover after cust_hhi_lag3 {_f(c5['after_cust'])}; "
            f"body {_f(c3['y4_body'])}."
        )
        y5 = "PARK"
        y5_why = (
            f"PARK as Y5 X (size-rank AUROC {_f(c4['size_rank'][Y5]['auroc'])}). "
            f"Leftover after size {_f(c5['after_size'])}."
        )

    lose_44 = bool(
        (c6["drop_hhi"] or (c5["died_y3"] and (c5["died_y4"] or c3["y4_dead"])))
        and y3 in {"DROP", "CLOSE"}
        and y4 in {"DROP", "CLOSE"}
        and y5 == "PARK"
    )
    if lose_44:
        overall = "DROP"
        on44 = "DROP from the 44"
        on44_why = (
            f"twin or leftover dies. Y3 leftover {_f(c5['after_days'])}; "
            f"Y4 leftover {_f(c5['after_cust'])} body {_f(c3['y4_body'])}; "
            f"Y5 PARK. Javier concentration is top1."
        )
    elif keep_y3 or keep_y4:
        overall = "KEEP"
        on44 = "KEEP on the 44"
        on44_why = "clears KEEP-as-X on at least one honest bar."
    else:
        overall = "CLOSE"
        on44 = "CLOSE / PARK on the 44"
        on44_why = "fails KEEP-as-X; diagnostic only."

    q3 = y3
    q5 = "PARK" if y5 == "PARK" else y4
    q6 = "CLOSE" if c8["q6_close"] else "KEEP"
    q6_why = (
        f"short lag3 Y4 {_f(c8['short_y4_lag3'])} lag1 {_f(c8['short_y4_lag1'])} "
        f"— Y4 customer HHI Q6 was CLOSE / LOW_POWER 21.7%."
    )
    if c6["drop_hhi"] and c3["y5_confirm"]:
        object_kind = "top1 rewrite; Y5 tail is protective (not a Y4 crash)"
    elif c6["drop_hhi"]:
        object_kind = "twin of top1"
    elif c3["y5_confirm"]:
        object_kind = "different object (protective / leftover)"
    else:
        object_kind = "monopoly tail (Y4-like)"
    table = [
        {
            "object": "d_supp_hhi as Y3 X / the 15-col card",
            "decision": y3,
            "why": y3_why + " Stays off the 15-col card.",
        },
        {
            "object": "d_supp_hhi as Y4 X",
            "decision": y4,
            "why": y4_why + " Do not merge with Y4 / do not reopen trees.",
        },
        {
            "object": "d_supp_hhi as Y5 X",
            "decision": y5,
            "why": y5_why + " Do not revive trees.",
        },
        {
            "object": "d_supp_hhi on the 44-col keep list",
            "decision": on44,
            "why": on44_why,
        },
        {
            "object": "same monopoly tail as Y4 customer HHI?",
            "decision": "NO — different object" if c3["y5_confirm"] else "YES — tail-shaped",
            "why": (
                f"Y5 tail is protective {_f(c3['y5_rate_hi'])} vs {_f(c3['y5_rate_lo'])} "
                f"(CONFIRM {c3['y5_confirm']}). Y4 customer tail was a crash (22.1% vs 11.5%). "
                f"Body Y4 {_f(c3['y4_body'])} vs customer body {Y4_BODY}."
            ),
        },
        {
            "object": "twin of d_supp_top1",
            "decision": "DROP weaker (HHI)" if c6["drop_hhi"] else "not a twin",
            "why": c6["prose"],
        },
        {
            "object": "y_supp_hhi / merge with Y4",
            "decision": "PARK",
            "why": "do not invent y_supp_hhi. Do not merge with Y4.",
        },
        {
            "object": "Q6 lag1/lag3 on short books",
            "decision": q6,
            "why": q6_why,
        },
        {
            "object": "ICC / trait vs month shock",
            "decision": "TRAIT" if c9["trait"] else ("shock" if c9["shock"] else "mixed"),
            "why": c9["prose"],
        },
        {
            "object": "Y5 leftover 65% after dropping tail",
            "decision": "document only",
            "why": c11["prose"],
        },
        {
            "object": "2-col z-avg top1+HHI",
            "decision": "CLOSE",
            "why": "Y4 z-avg was CLOSE. Do not put on the card.",
        },
    ]
    headline = (
        f"**{overall}** as X. Y3 **{y3}** leftover-after-days {_f(c5['after_days'])}. "
        f"Y4 **{y4}** leftover after cust_hhi_lag3 {_f(c5['after_cust'])} "
        f"body {_f(c3['y4_body'])}. Y5 **{y5}** leftover-after-size {_f(c5['after_size'])}. "
        f"{'SIZE' if c2['size_flag'] else 'not SIZE'} (ρ={_f(c2['rhos']['log1p(a_in3)'])}); "
        f"{'twin of d_supp_top1 ρ=' + _f(c2['rhos']['d_supp_top1']) if c2['top1_twin'] else 'not a |ρ|≥0.80 twin'}. "
        f"Tail vs body: Y5 protective {_f(c3['y5_rate_hi'])} vs {_f(c3['y5_rate_lo'])} "
        f"(CONFIRM {c3['y5_confirm']}); Y4 body {_f(c3['y4_body'])} "
        f"{'dead like 0.445' if c3['y4_dead'] else 'still ranks'}. "
        f"Object: {object_kind}. "
        f"44 should {'lose' if lose_44 else 'keep (diagnostic only)'} `d_supp_hhi`. "
        f"Y5 leftover 65% → {_pp(c11['dropped']['neither_share'])} after dropping tail "
        f"(Δ {_f(c11['delta'], 3)}). Q6 {q6}. "
        f"Night quotes unchanged: Y3 {NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f}; days {DAYS_BENCH:.3f}; "
        f"size {SIZE_QUOTE:.3f}; TURNOVER {TURNOVER_QUOTE:.3f} / {B_SHALLOW_QUOTE:.3f}."
    )
    print("DECIDE", headline)
    return {
        "overall": overall,
        "y3": y3,
        "y3_why": y3_why,
        "y4": y4,
        "y4_why": y4_why,
        "y5": y5,
        "y5_why": y5_why,
        "q3": q3,
        "q5": q5,
        "q6": q6,
        "q6_why": q6_why,
        "lose_44": lose_44,
        "on44": on44,
        "on44_why": on44_why,
        "object_kind": object_kind,
        "keep_y3": keep_y3,
        "keep_y4": keep_y4,
        "keep_y5": keep_y5,
        "table": table,
        "headline": headline,
    }


def plot_png(tr: pd.DataFrame, c3: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    x = pd.to_numeric(tr["d_supp_hhi"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))

    # left: Y5 AP quintiles
    y = pd.to_numeric(tr[Y5], errors="coerce")
    m = x.notna() & y.notna()
    d = pd.DataFrame({"x": x[m], "y": y[m]})
    try:
        d["q"] = pd.qcut(d["x"], 5, duplicates="drop")
        rates = [float(g["y"].mean()) for _, g in d.groupby("q", observed=True)]
        ns = [len(g) for _, g in d.groupby("q", observed=True)]
        axes[0].bar(range(1, len(rates) + 1), rates, color="#3d5a80")
        axes[0].axhline(float(d["y"].mean()), color="#ee6c4d", ls="--", lw=1, label="train labeled mean")
        for i, (r, n) in enumerate(zip(rates, ns), start=1):
            axes[0].text(i, r + 0.004, f"{r:.3f}\nn={n}", ha="center", va="bottom", fontsize=8)
        axes[0].set_ylim(0, max(rates + [0.12]) + 0.03)
    except ValueError:
        axes[0].text(0.5, 0.5, "quintile fail", ha="center")
    axes[0].set_title("Y5 AP rate by d_supp_hhi quintile (train)")
    axes[0].set_xlabel("supplier HHI quintile")
    axes[0].set_ylabel("y5_ap_od30_ownp80")
    axes[0].legend(fontsize=8)

    # right: tail vs rest for three Ys
    labels = []
    tail_r = []
    rest_r = []
    for yname, lab in ((Y3, "Y3"), (Y4, "Y4"), (Y5, "Y5 AP")):
        yv = pd.to_numeric(tr[yname], errors="coerce")
        mm = yv.notna() & x.notna()
        hi = mm & (x > TAIL_CUT)
        lo = mm & (x <= TAIL_CUT)
        labels.append(lab)
        tail_r.append(float(yv[hi].mean()) if int(hi.sum()) else 0.0)
        rest_r.append(float(yv[lo].mean()) if int(lo.sum()) else 0.0)
    xpos = np.arange(len(labels))
    axes[1].bar(xpos - 0.18, tail_r, 0.36, label="HHI>0.975", color="#ee6c4d")
    axes[1].bar(xpos + 0.18, rest_r, 0.36, label="body ≤0.975", color="#3d5a80")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Tail vs body P(Y=1) — Y5 tail is protective")
    axes[1].set_ylabel("P(Y=1)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def write_md(ctx: dict) -> None:
    d = ctx["decision"]
    c1, c2, c3, c4, c5 = ctx["c1"], ctx["c2"], ctx["c3"], ctx["c4"], ctx["c5"]
    c6, c7, c8, c9, c10, c11 = ctx["c6"], ctx["c7"], ctx["c8"], ctx["c9"], ctx["c10"], ctx["c11"]
    lines = [
        "# Unused leftover of `d_supp_hhi` on the 44",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage / mix only. Seed 20260918 group folds. No 0–100. "
        "No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_supp_hhi`. "
        "Do not merge with Y4. Do not reopen Y4/Y5 trees. Off the 15-col Y3 card. "
        f"Night Y3 stays **{NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f}**. Days **{DAYS_BENCH:.3f}**. "
        f"Size **{SIZE_QUOTE:.3f}**. Y7 TURNOVER **{TURNOVER_QUOTE:.3f} / {B_SHALLOW_QUOTE:.3f}**. "
        "Y7 never D. Y5 never E. Y3 never B. Do not grow TURNOVER.",
        "",
        "`d_supp_hhi` = Herfindahl of AP invoice counterparties in the trailing 6-month "
        "window (Family D). Needs identified supplier CPs. Dark 470 stay **NaN not 0**. "
        "Javier 14: concentration is **top1**, not HHI. Y5 already PARK this column as Y5 X "
        f"(size AUROC {Y5_SIZE_AUROC}).",
        "",
        "## Headline",
        "",
        d["headline"],
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        "| 1 | Who is healthy? | **PARK** as a health Y. Do not invent `y_supp_hhi`. Dark 470 = NaN, not 0. |",
        "| 2 | Who is improving? | Not a supplier-HHI gradient. |",
        f"| 3 | Who is turning? | **{d['y3']}** as Y3 X — leftover after days {_f(c5['after_days'])} vs days {_f(c4['days_y3'])}. |",
        f"| 4 | Dip vs fall? | **{d['y4']}** as Y4 X — leftover after cust HHI lag3 {_f(c5['after_cust'])}; body {_f(c3['y4_body'])}. Do not merge with Y4. |",
        f"| 5 | Why did it change? | Y5 tail is **protective** {_f(c3['y5_rate_hi'])} vs {_f(c3['y5_rate_lo'])} (CONFIRM {c3['y5_confirm']}). Object: {d['object_kind']}. |",
        f"| 6 | Months earlier? | **{d['q6']}** — {d['q6_why']} |",
        "",
        "## PARK / CLOSE / KEEP / DROP-from-44",
        "",
        _md_table(d["table"], ["object", "decision", "why"]),
        "",
        "## 1. Coverage; 470 dark NaN vs invoice-book; ever-n",
        "",
        c1["prose"],
        "",
        _md_table(c1["rows"]),
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| train CM / companies | {c1['n_train']:,} / {c1['n_train_co']:,} |",
        f"| d_supp_hhi defined | {c1['train_nn']:,} ({_pp(c1['train_cov'])}) |",
        f"| ever-n companies | {c1['ever_n']:,} |",
        f"| never-ERP companies | {c1['n_dark_co']} (want {N_DARK_WANT}; confirm_470={c1['confirm_470']}) |",
        f"| dark HHI non-null / zero-filled | {c1['dark_nn']} / {c1['dark_zero']} |",
        f"| dark 0-fill | {'NO — CONFIRM' if c1['dark_ok'] else 'YES — FAIL'} |",
        f"| ERP n_supp==0 / HHI defined there | {c1['n0_erp']:,} / {c1['hhi_when_n0']} |",
        f"| holdout coverage (check only) | {_pp(c1['hold_cov'])} |",
        f"| acf1 / acf3 | {_f(c1['acf1'])} / {_f(c1['acf3'])} |",
        f"| size ρ vs log1p(a_in3) | {_f(c1['size_rho'])} |",
        "",
        "## 2. Spearman twins (|ρ|≥0.80)",
        "",
        c2["prose"],
        "",
        _md_table(c2["rows"]),
        "",
        "## 3. Quintiles + >0.975 tail on Y3 / Y4 / Y5 AP",
        "",
        c3["prose"],
        "",
        "Train labeled cuts. Not monotone unless noted. Body CV is signed group-fold on HHI≤0.975.",
        "",
        _md_table(c3["q_rows"]),
        "",
        _md_table(c3["tail_rows"]),
        "",
        f"Y5 protective tail CONFIRM vs quote 2.7% / 8.6%: **{c3['y5_confirm']}**. "
        f"Y4 body dead like customer 0.445: **{c3['y4_dead']}**.",
        "",
        "## 4. Single-feature train group-fold AUROC",
        "",
        f"Sign from the train side of each fold. Seed {FOLD_SEED}. "
        f"Night Y3 size **{SIZE_QUOTE}** (replica {_f(c4['size_y3'])}); "
        f"days **{DAYS_BENCH}** (replica {_f(c4['days_y3'])}). "
        f"Y4 customer HHI lag3 night **{Y4_HHI_LAG3}** (replica {_f(c4['cust3_y4'])}). "
        "Do not quote holdout.",
        "",
        c4["prose"],
        "",
        _md_table(c4["rows"]),
        "",
        "## 5. Honest leftover after the bar",
        "",
        "OLS residual of `d_supp_hhi` on the bar (train-defined slope). Leftover <0.55 dies. "
        "Y5 leftover is diagnostic only — do not revive trees.",
        "",
        c5["prose"],
        "",
        _md_table(c5["rows"]),
        "",
        "## 6. Twin vs `d_supp_top1`",
        "",
        c6["prose"],
        "",
        "## 7. SIZE terciles — tail inside T1?",
        "",
        c7["prose"],
        "",
        _md_table(c7["rows"]),
        "",
        "## 8. Q6 lag1 / lag3 on short books",
        "",
        c8["prose"],
        "",
        _md_table(c8["rows"]),
        "",
        "## 9. ICC / company-demean (trait vs month shock)",
        "",
        c9["prose"],
        "",
        _md_table(c9["rows"]),
        "",
        "## 10. Dark 470 — HHI defined?",
        "",
        c10["prose"],
        "",
        _md_table(c10["rows"]),
        "",
        "## 11. Y5 leftover 65% after dropping the protective tail",
        "",
        c11["prose"],
        "",
        _md_table(c11["rows"]),
        "",
        "Do not invent a new Y from the leftover cell.",
        "",
        "## Extra A. Holdout coverage / mix (LOW_POWER)",
        "",
        ctx["hold"]["prose"],
        "",
        _md_table(ctx["hold"]["rows"]),
        "",
        "## Extra B. 2-col z-avg of top1 + HHI",
        "",
        ctx["zavg"]["prose"],
        "",
        _md_table(ctx["zavg"]["rows"]),
        "",
        "CLOSE. Do not put on the card.",
        "",
        "## Extra C. Same-n leftover after top1",
        "",
        ctx["samen"]["prose"],
        "",
        _md_table(ctx["samen"]["rows"]),
        "",
        "## Extra D. Tail companies on non-tail months",
        "",
        ctx["tailcos"]["prose"],
        "",
        _md_table(ctx["tailcos"]["rows"]),
        "",
        "## Extra E. Size-rank AUROC (y5_why 0.663 replica)",
        "",
        ctx["sizerank"]["prose"],
        "",
        _md_table(ctx["sizerank"]["rows"]),
        "",
        "## Extra F. Y3 T2/T3 pocket leftover after days",
        "",
        ctx["pocket"]["prose"],
        "",
        _md_table(ctx["pocket"]["rows"]),
        "",
        "## Extra G. Leftover after `d_n_supp`",
        "",
        ctx["nsupp"]["prose"],
        "",
        _md_table(ctx["nsupp"]["rows"]),
        "",
        "## Extra H. Company-mean leftover after days + inverse days",
        "",
        ctx["style"]["prose"],
        "",
        _md_table(ctx["style"]["rows"]),
        "",
        "## Extra I. Holdout tail mix (LOW_POWER)",
        "",
        ctx["holdtail"]["prose"],
        "",
        _md_table(ctx["holdtail"]["rows"]),
        "",
        "## Extra J. Y3 body leftover after days",
        "",
        ctx["y3body"]["prose"],
        "",
        _md_table(ctx["y3body"]["rows"]),
        "",
        "## Extra K. Short-book lag3 leftover (Q6 honesty)",
        "",
        ctx["q6h"]["prose"],
        "",
        _md_table(ctx["q6h"]["rows"]),
        "",
        "## Extra L. Leftover after n_supp + days",
        "",
        ctx["nandd"]["prose"],
        "",
        _md_table(ctx["nandd"]["rows"]),
        "",
        "## What failed / next (held for wave note)",
        "",
        *([f"- {x}" for x in ctx["failed"]] if ctx["failed"] else ["- (none)"]),
        "",
        f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts 1–12 plus extras "
        "(holdout mix, z-avg, same-n leftover, tail-company months, "
        "size-rank, Y3 T2/T3 pocket, n_supp leftover, style, holdout tail, "
        "Y3 body leftover, short-book Q6 leftover, n_supp+days).",
        "",
        "## What this module did not do",
        "",
        "- Did not change night Y3 0.762 / 0.752, days 0.711, size 0.617, or Y7 TURNOVER 0.720 / 0.712.",
        "- Did not put supp HHI on the 15-col Y3 card. Did not grow TURNOVER.",
        "- Did not reopen Y4/Y5 trees. Did not invent `y_supp_hhi`. Did not merge with Y4.",
        "- Did not use Family E as Y5 X. Did not use Family B as Y3 X. Did not use Family F as Y4 X.",
        "- Did not write 0–100 / pillars. Did not touch `product/`.",
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
    c1, c2, c3, c4, c5, d = ctx["c1"], ctx["c2"], ctx["c3"], ctx["c4"], ctx["c5"], ctx["decision"]
    rows = [
        _reg_row(
            "d_supp_hhi_cov",
            c1["train_cov"],
            f"{c1['train_cov']:.4f}",
            "-",
            f"ever_n={c1['ever_n']} dark_nn={c1['dark_nn']} dark_co={c1['n_dark_co']} ok={c1['dark_ok']}",
            "train",
        ),
        _reg_row(
            "rho_supp_hhi_vs_top1",
            c2["rhos"]["d_supp_top1"],
            "1.0000",
            "-",
            f"cust_hhi={c2['rhos']['d_cust_hhi']:.4f} tx_cp={c2['rhos']['d_tx_cp_share']:.4f} "
            f"size={c2['rhos']['log1p(a_in3)']:.4f} twin={c2['top1_twin']}",
            "train",
        ),
        _reg_row(
            "auroc_d_supp_hhi",
            c4["supp_y3"],
            f"{c1['train_cov']:.4f}",
            Y3,
            f"vs_size={c4['size_y3']:.4f} vs_days={c4['days_y3']:.4f} leftover_days={c5['after_days']:.4f} y3={d['y3']}",
        ),
        _reg_row(
            "auroc_d_supp_hhi",
            c4["supp_y4"],
            f"{c1['train_cov']:.4f}",
            Y4,
            f"vs_size={c4['size_y4']:.4f} vs_cust_lag3={c4['cust3_y4']:.4f} leftover_cust={c5['after_cust']:.4f} "
            f"body={c3['y4_body']:.4f} y4={d['y4']}",
        ),
        _reg_row(
            "auroc_d_supp_hhi",
            c4["supp_y5"],
            f"{c1['train_cov']:.4f}",
            Y5,
            f"vs_size={c4['size_y5']:.4f} size_on_def={c4['size_on'][Y5]:.4f} leftover_size={c5['after_size']:.4f} "
            f"y5={d['y5']}",
        ),
        _reg_row(
            "y5_supp_hhi_tail_rate",
            c3["y5_rate_hi"],
            "1.0000",
            Y5,
            f"rest={c3['y5_rate_lo']:.4f} confirm={c3['y5_confirm']} quote=0.027/0.086",
            "train",
        ),
        _reg_row(
            "y5_leftover_after_drop_tail",
            ctx["c11"]["dropped"]["neither_share"],
            "1.0000",
            Y5,
            f"full={ctx['c11']['full']['neither_share']:.4f} delta={ctx['c11']['delta']} "
            f"increased={ctx['c11']['increased']}",
            "train",
        ),
        _reg_row(
            "supp_hhi_lose_44",
            1.0 if d["lose_44"] else 0.0,
            f"{c1['train_cov']:.4f}",
            "-",
            f"overall={d['overall']} y3={d['y3']} y4={d['y4']} y5={d['y5']} kind={d['object_kind']}",
            "train",
        ),
        _reg_row(
            "icc_d_supp_hhi",
            ctx["c9"]["icc"]["icc"],
            "1.0000",
            "-",
            f"eta2={ctx['c9']['icc']['eta2']:.4f} trait={ctx['c9']['trait']} demean_y3={ctx['c9']['d3']:.4f}",
            "train",
        ),
        _reg_row(
            "size_rank_d_supp_hhi",
            ctx["sizerank"]["y5"],
            "1.0000",
            Y5,
            f"quote=0.663 match={ctx['sizerank']['match']} park={ctx['c4']['y5_size_park']}",
            "train",
        ),
        _reg_row(
            "y3_body_leftover_days",
            ctx["y3body"]["lef"],
            "1.0000",
            Y3,
            f"body_raw={ctx['y3body']['raw']:.4f} days={ctx['y3body']['days']:.4f} dies={ctx['y3body']['dies']}",
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


def write_wave(ctx: dict) -> None:
    d = ctx["decision"]
    c1, c2, c3, c4, c5, c6, c11 = (
        ctx["c1"],
        ctx["c2"],
        ctx["c3"],
        ctx["c4"],
        ctx["c5"],
        ctx["c6"],
        ctx["c11"],
    )
    WAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = (
        f"# Wave 4 — unused leftover of `d_supp_hhi`\n\n"
        f"- **When:** {_now_iso()}\n"
        f"- **Agent:** `{AGENT}`\n"
        f"- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`\n"
        f"- **Re-run:** `python -m analysis.evaluate.supp_hhi_qa`\n"
        f"- **Holdout:** 72 companies, seed 20260918. Rates / AUROC on train. Holdout = coverage.\n"
        f"- **Owned:** `analysis/evaluate/supp_hhi_qa.py`, `analysis/outputs/supp_hhi_qa.md`, "
        f"`analysis/outputs/supp_hhi_quintiles.png`, append-only registry, this note.\n\n"
        f"## Decision\n\n"
        f"{d['headline']}\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| Y3 X | **{d['y3']}** |\n"
        f"| Y4 X | **{d['y4']}** |\n"
        f"| Y5 X | **{d['y5']}** |\n"
        f"| the 44 | **{d['on44']}** |\n"
        f"| twin vs top1 | **{'DROP HHI' if c6['drop_hhi'] else 'not a twin'}** ρ={_f(c2['rhos']['d_supp_top1'])} |\n"
        f"| Y5 tail | **{'CONFIRM protective' if c3['y5_confirm'] else 'check'}** {_f(c3['y5_rate_hi'])} vs {_f(c3['y5_rate_lo'])} |\n"
        f"| Y4 body | **{_f(c3['y4_body'])}** (customer body 0.445) |\n\n"
        f"Coverage train {_pp(c1['train_cov'])} ever-n {c1['ever_n']}. "
        f"Dark 470 NaN not 0: {c1['dark_ok']}. "
        f"Leftover Y3 after days {_f(c5['after_days'])}; Y4 after cust_hhi_lag3 {_f(c5['after_cust'])}; "
        f"Y5 after size {_f(c5['after_size'])}. "
        f"Y5 65% leftover → {_pp(c11['dropped']['neither_share'])} after drop-tail "
        f"(Δ {_f(c11['delta'], 3)}; increased={c11['increased']}).\n\n"
        f"## What failed\n\n"
        + ("\n".join(f"- {x}" for x in ctx["failed"]) if ctx["failed"] else "- (none)")
        + "\n\n"
        f"## Next idea\n\n"
        f"- If the 44 drops HHI, keep `d_supp_top1` only as the Javier concentration object "
        f"(already PARK as Y5 X / SIZE). Do not stack HHI+top1.\n"
        f"- Do not invent `y_supp_hhi`. Do not merge with Y4.\n\n"
        f"Elapsed {ctx['elapsed_s']:.0f}s. Night quotes unchanged.\n"
    )
    WAVE_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {WAVE_PATH}")


def run() -> dict:
    t0 = time.time()
    print(f"supp_hhi_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    panel = attach_folds(panel)
    con = connect()
    try:
        book = book_invoice_ids(con)
        dark = dark_population(con)
    finally:
        con.close()
    panel = add_panel_lags(
        panel,
        ["d_supp_hhi", "d_supp_top1", "d_cust_hhi", "d_cust_top1"],
        (1, 3),
    )
    print("own-cuts expanding p80/p20 (Y5 leftover 2×2)…")
    panel = add_own_cuts(panel)
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
    c2 = cut2_twins(tr)
    c3 = cut3_tail(tr)
    c4 = cut4_singles(tr)
    c5 = cut5_leftover(tr)
    c6 = cut6_twin(tr, c2, c4)
    c7 = cut7_terciles(tr)
    c8 = cut8_q6(tr)
    c9 = cut9_icc(tr)
    c10 = cut10_dark(tr, book, dark)
    c11 = cut11_y5_leftover(tr)
    hold = extra_holdout(panel)
    zavg = extra_zavg(tr)
    samen = extra_samen(tr)
    tailcos = extra_tail_cos(tr)
    sizerank = extra_size_rank(tr)
    pocket = extra_y3_pocket(tr)
    nsupp = extra_n_supp(tr)
    style = extra_style(tr)
    holdtail = extra_hold_tail(panel)
    y3body = extra_y3_body_days(tr)
    q6h = extra_q6_honest(tr)
    nandd = extra_n_and_days(tr)
    png_ok = plot_png(tr, c3)

    if c2["top1_twin"]:
        failed.append(
            f"twin of d_supp_top1 ρ={c2['rhos']['d_supp_top1']:.3f} — DROP weaker HHI"
        )
    if c5["died_y3"]:
        failed.append(f"Y3 leftover after days {c5['after_days']:.3f} dies <0.55")
    if c5["died_y4"] or c3["y4_dead"]:
        failed.append(
            f"Y4 leftover after cust_hhi_lag3 {c5['after_cust']:.3f}; body {c3['y4_body']:.3f}"
        )
    if c4["y5_size_park"]:
        failed.append(
            f"Y5 size-rank AUROC {c4['size_rank'][Y5]['auroc']:.3f} ≥0.60 — PARK as Y5 X"
        )
    if c3["y5_confirm"]:
        failed.append("Y5 tail CONFIRM protective — not a Y4-style crash")
    if not c11["increased"]:
        failed.append("dropping the protective tail does not raise the 65% leftover (almost no tail pos)")
    if pocket["dies"]:
        failed.append(f"Y3 T2+T3 leftover after days {pocket['lef_t23']:.3f} dies — 0.653 is activity")
    if y3body["dies"]:
        failed.append(f"Y3 body leftover after days {y3body['lef']:.3f} dies")
    if sizerank["match"]:
        failed.append(f"Y5 size-rank AUROC {sizerank['y5']:.3f} CONFIRM vs 0.663")
    if not q6h["q6_keep"]:
        failed.append("Q6 short lag3 leftover after the honest bar dies — CLOSE")
    if nandd["dies"]:
        failed.append(f"Y3 leftover after n_supp+days {nandd['lef']:.3f} dies")
    elif np.isfinite(nandd["lef"]) and nandd["lef"] >= CHANCE:
        failed.append(
            f"Y3 leftover after n_supp+days {nandd['lef']:.3f} still ranks — still DROP as top1 twin"
        )

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
        "hold": hold,
        "zavg": zavg,
        "samen": samen,
        "tailcos": tailcos,
        "sizerank": sizerank,
        "pocket": pocket,
        "nsupp": nsupp,
        "style": style,
        "holdtail": holdtail,
        "y3body": y3body,
        "q6h": q6h,
        "nandd": nandd,
        "png_ok": png_ok,
        "failed": failed,
        "elapsed_s": time.time() - t0,
        "book": book,
        "dark": dark,
    }
    ctx["decision"] = decide(ctx)
    write_md(ctx)
    append_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    print(f"supp_hhi_qa done in {ctx['elapsed_s']:.0f}s overall={ctx['decision']['overall']} wave={WRITE_WAVE}")
    return ctx


if __name__ == "__main__":
    run()
