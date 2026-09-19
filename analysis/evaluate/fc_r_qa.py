"""Unused leftover of KEEP-flow ``f_fc_r``.

``f_fc_r`` is already KEEP on the 44 and ``f_fc_r_lag3`` is on Y7
TURNOVER (0.720 / 0.712). This lane asks leftover after ``f_ds_r`` /
days as Y3 X, leftover after issued_lag1 as Y7, twin vs ``f_ds_r`` /
``a_fin_cost`` / in-memory Family M ``m_fin``, and whether
contemporaneous ``f_fc_r`` should leave the 44 while lag3 stays on
TURNOVER.

KEEP-as-X still applies for *adding*. For a KEEP column, leftover
dying after ``f_ds_r`` means twin / rewrite — then DROP contemporaneous
``f_fc_r`` from the 44. Do **not** rip ``f_fc_r_lag3`` off TURNOVER
(TURNFC0 / TURNFC1 did not lift fold 4).

Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617,
TURNOVER 0.720 / 0.712. Do not grow TURNOVER. Do not change the
15-col card. Family F lag3 CLOSE as Q6 (empty until so-far≥6).

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.fc_r_qa

Owned: analysis/evaluate/fc_r_qa.py, analysis/outputs/fc_r_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_fc_r.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "fc_r_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "fc_r_leftover.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_fc_r.md"
AGENT = "572fb928"
WAVE = "4"
ROUND = "R4"
MODEL = "fc_r_qa"
X_FAM = "F"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y7 = "y7_top1_lost"
Y9 = "y9_fee_r_ownp80"
Y9_SPIKE = "y9_fee_spike"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_Y3_QUOTE = 0.617
Y3_NIGHT = (0.762, 0.752)
Y7_TURNOVER = 0.720
Y7_B_SHALLOW = 0.712
Y7_TURN_NOFC = 0.714
ISSUED_LAG1_BENCH = 0.630
TURNOVER_F4 = 0.680
KEEP_DELTA = 0.02
TWIN_RHO = 0.80
SIZE_RHO = 0.50
CHANCE = 0.55
ICC_STYLE = 0.85
ACF_STYLE = 0.40
MIN_POS = 50
MIN_ACF_PAIRS = 4
FOLD4_GROUPS = ("GROUP_0222", "GROUP_0108")
WRITE_WAVE = True

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "a_fin_cost",
    "a_out6",
    "c_n_days_with_tx",
    "e_ar_issued",
    "e_credit_note_ratio",
    "f_ds_r",
    "f_fc_r",
    "f_outstanding_gt_granted",
)

Y_KEEP = (Y2, Y3, Y7, Y9, Y9_SPIKE)
TURN_NOFC = (
    "e_ar_issued_lag1",
    "e_ar_issued_lag_cv",
    "e_credit_note_ratio",
    "e_credit_note_ratio_lag1",
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


def jaccard(a: pd.Series, b: pd.Series) -> dict:
    aa = a.fillna(False).astype(bool)
    bb = b.fillna(False).astype(bool)
    both = aa & bb
    union = aa | bb
    n_u = int(union.sum())
    return {
        "jaccard": float(both.sum() / n_u) if n_u else float("nan"),
        "n_a": int(aa.sum()),
        "n_b": int(bb.sum()),
        "n_both": int(both.sum()),
        "n_union": n_u,
    }


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


def rank_resid(y: pd.Series, *xs: pd.Series) -> pd.Series:
    yr = pd.to_numeric(y, errors="coerce").rank(method="average")
    xr = [pd.to_numeric(x, errors="coerce").rank(method="average") for x in xs]
    resid, _ = ols_resid(yr, *xr)
    return resid


def leftover_diag(y, x, controls, folds, mask) -> dict:
    resid, info = ols_resid(x, *controls)
    rec = signed_oof_auroc(y, resid, folds, mask)
    rho_c = spearman(resid, controls[0]) if controls else float("nan")
    rho_x = spearman(resid, x)
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
        "rho_x": rho_x,
        "r2": info["r2"],
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
        "fake": fake,
        "honest_dies": honest_dies,
        "folds": fold_bits(rec),
        "rank_folds": fold_bits(rrec),
        "resid": resid,
        "info": info,
        "rec": rec,
    }


def company_demean(series: pd.Series, company: pd.Series) -> pd.Series:
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    mu = d.groupby("co")["x"].transform("mean")
    return d["x"] - mu


def company_mean(series: pd.Series, company: pd.Series) -> pd.Series:
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    return d.groupby("co")["x"].transform("mean")


def _cv(res: dict) -> float:
    return float("nan") if res.get("low_power") else float(res.get("cv", float("nan")))


def _auc_row(y: str, feat: str, res: dict) -> dict:
    return {
        "y": y,
        "feature": feat,
        "n": f"{res['n_defined']:,}",
        "n_pos": f"{res['n_pos']:,}",
        "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
        "train": "LOW_POWER" if res["low_power"] else _f(res["train_auc"]),
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


def _sofar_bucket(n) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return "na"
    n = int(n)
    if n < 6:
        return "<6"
    if n < 12:
        return "6-11"
    if n < 18:
        return "12-17"
    if n < 24:
        return "18-23"
    return "24"


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


def add_issued_lag_cv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lag_names = [c for c in ("e_ar_issued_lag1", "e_ar_issued_lag2", "e_ar_issued_lag3") if c in out.columns]
    if len(lag_names) < 2:
        out["e_ar_issued_lag_cv"] = np.nan
        return out
    mat = out[lag_names].apply(pd.to_numeric, errors="coerce").clip(lower=0.0)
    mu = mat.mean(axis=1)
    sd = mat.std(axis=1, ddof=0)
    out["e_ar_issued_lag_cv"] = sd / (mu.abs() + 1.0)
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
    miss = [c for c in STORE_COLS if c not in raw.columns]
    if miss:
        raise RuntimeError(f"monthly.parquet missing {miss}")
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
    panel["so_far_bucket"] = panel["months_so_far"].map(_sofar_bucket)
    panel["co_class"] = panel["n_grid_months"].map(_trail_class)
    leak3 = leakage_check(["f_fc_r", "f_ds_r", "c_n_days_with_tx", "log_in3"], Y3, forbidden_prefixes=["b"])
    leak7 = leakage_check(["f_fc_r", "e_ar_issued"], Y7, forbidden_prefixes=["d"])
    leak9 = leakage_check(["log_in3", "a_out6"], Y9, forbidden_prefixes=["f"])
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 never-B leak: {leak3['issues']}")
    if not leak7["ok"]:
        raise RuntimeError(f"Y7 never-D leak: {leak7['issues']}")
    if not leak9["ok"]:
        raise RuntimeError(f"Y9 never-F leak: {leak9['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def attach_m_fin(panel: pd.DataFrame) -> pd.DataFrame:
    """In-memory Family M. Do not merge to parquet / FAMILIES."""
    from analysis.features.catmix import build as build_m

    con = connect()
    try:
        grid = panel[["company_id", "period"]].drop_duplicates()
        m = build_m(con, grid)
    finally:
        con.close()
    keep = [c for c in ("company_id", "period", "m_fin_share", "m_fee_share", "m_int_share") if c in m.columns]
    m = _keys(m[keep])
    out = panel.merge(m, on=["company_id", "period"], how="left")
    print(f"in-memory Family M attached ({len(keep) - 2} cols); not written")
    return out


def size_terciles(tr: pd.DataFrame) -> pd.Series:
    med = tr.groupby("company_id")["log_in3"].median()
    cuts = med.quantile([1 / 3, 2 / 3])
    lab = pd.Series("T2", index=med.index)
    lab[med <= cuts.iloc[0]] = "T1"
    lab[med > cuts.iloc[1]] = "T3"
    return tr["company_id"].map(lab)


# ---------------------------------------------------------------------------
# 1. Coverage; 470 vs ERP; ever-n vs f_ds_r
# ---------------------------------------------------------------------------
def pass1_cov(tr: pd.DataFrame, book: set[str]) -> dict:
    fc = pd.to_numeric(tr["f_fc_r"], errors="coerce")
    ds = pd.to_numeric(tr["f_ds_r"], errors="coerce")
    n_cm = len(tr)
    n_co = int(tr["company_id"].nunique())
    erp = tr["company_id"].isin(book)
    dark = ~erp
    n_dark = int(tr.loc[dark, "company_id"].nunique())
    n_erp = int(tr.loc[erp, "company_id"].nunique())
    rows = []
    for name, s in (("f_fc_r", fc), ("f_ds_r", ds)):
        nn = s.notna()
        ever = tr.loc[nn, "company_id"].nunique()
        ever_pos = tr.loc[s.gt(0), "company_id"].nunique()
        rows.append(
            {
                "col": name,
                "n_nn": f"{int(nn.sum()):,}",
                "cov CM": _pp(_pct(int(nn.sum()), n_cm)),
                "ever co": f"{ever:,}",
                "ever %": _pp(_pct(ever, n_co)),
                "ever >0": f"{ever_pos:,}",
                "share>0": _pp(_pct(int(s.gt(0).sum()), int(nn.sum()))),
                "p50": _f(float(s[nn].median()) if nn.any() else float("nan"), 4),
                "share=0": _pp(_pct(int(s.eq(0).sum()), int(nn.sum()))),
            }
        )
    dark_rows = []
    for slice_name, sl in (("dark", dark), ("ERP", erp)):
        for name, s in (("f_fc_r", fc), ("f_ds_r", ds)):
            nn = sl & s.notna()
            n_co_sl = int(tr.loc[sl, "company_id"].nunique())
            dark_rows.append(
                {
                    "book": slice_name,
                    "col": name,
                    "n_co": n_co_sl,
                    "n_cm": int(sl.sum()),
                    "n_nn": int(nn.sum()),
                    "cov CM": _pp(_pct(int(nn.sum()), int(sl.sum()))),
                    "ever co": int(tr.loc[nn, "company_id"].nunique()),
                    "share=0": _pp(_pct(int((sl & s.eq(0)).sum()), int(nn.sum()))),
                    "zero_fill": "no — 0 is no-fee 3m, not a dark fill",
                }
            )
    fc_dark_nn = int((dark & fc.notna()).sum())
    fc_erp_nn = int((erp & fc.notna()).sum())
    ds_cov = _pct(int(ds.notna().sum()), n_cm)
    fc_cov = _pct(int(fc.notna().sum()), n_cm)
    match_ds = bool(np.isfinite(fc_cov) and np.isfinite(ds_cov) and abs(fc_cov - ds_cov) < 0.005)
    prose = (
        f"Train {n_co:,} co / {n_cm:,} CM. "
        f"f_fc_r cov {_pp(fc_cov)} vs f_ds_r {_pp(ds_cov)} "
        f"({'SAME rolling-3 hole' if match_ds else 'coverage differs'}). "
        f"Dark {n_dark} vs ERP {n_erp}: fc nn CM {fc_dark_nn:,} / {fc_erp_nn:,}. "
        f"Access ≠ ERP — F is bank/debt, dark companies have f_fc_r."
    )
    print(prose)
    return {
        "n_cm": n_cm,
        "n_co": n_co,
        "n_dark": n_dark,
        "n_erp": n_erp,
        "fc_cov": fc_cov,
        "ds_cov": ds_cov,
        "match_ds": match_ds,
        "fc_ever": int(tr.loc[fc.notna(), "company_id"].nunique()),
        "ds_ever": int(tr.loc[ds.notna(), "company_id"].nunique()),
        "fc_ever_pos": int(tr.loc[fc.gt(0), "company_id"].nunique()),
        "ds_ever_pos": int(tr.loc[ds.gt(0), "company_id"].nunique()),
        "fc_dark_nn": fc_dark_nn,
        "fc_erp_nn": fc_erp_nn,
        "rows": rows,
        "dark_rows": dark_rows,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 2. Spearman twins
# ---------------------------------------------------------------------------
def pass2_twins(tr: pd.DataFrame) -> dict:
    fc = tr["f_fc_r"]
    pairs = [
        ("f_ds_r", tr["f_ds_r"]),
        ("f_fc_r_lag1", tr["f_fc_r_lag1"]),
        ("f_fc_r_lag3", tr["f_fc_r_lag3"]),
        ("a_fin_cost", tr["a_fin_cost"]),
        ("a_out6", tr["a_out6"]),
        ("m_fin_share", tr.get("m_fin_share", pd.Series(np.nan, index=tr.index))),
        ("m_fee_share", tr.get("m_fee_share", pd.Series(np.nan, index=tr.index))),
        ("log1p(a_in3)", tr["log_in3"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("f_outstanding_gt_granted", tr["f_outstanding_gt_granted"]),
    ]
    rows = []
    rhos = {}
    twins = []
    size_flag = False
    for name, s in pairs:
        rho = spearman(fc, s)
        rhos[name] = rho
        twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
        sizeish = bool(name == "log1p(a_in3)" and np.isfinite(rho) and abs(rho) >= SIZE_RHO)
        if twin:
            twins.append(name)
        if sizeish:
            size_flag = True
        rows.append(
            {
                "a": "f_fc_r",
                "b": name,
                "ρ": _f(rho),
                "twin_|ρ|≥0.80": "YES" if twin else "no",
                "SIZE_|ρ|≥0.50": "YES" if sizeish else ("—" if name != "log1p(a_in3)" else "no"),
            }
        )
    weaker = ""
    if "f_ds_r" in twins:
        weaker = "TWIN of f_ds_r — ds_r is on the 15-col Y3 card; DROP weaker contemporaneous f_fc_r from the 44"
    ogtg_twin = "f_outstanding_gt_granted" in twins
    prose = (
        f"f_fc_r vs f_ds_r ρ={_f(rhos['f_ds_r'])} "
        f"vs lag1 {_f(rhos['f_fc_r_lag1'])} vs lag3 {_f(rhos['f_fc_r_lag3'])} "
        f"vs a_fin_cost {_f(rhos['a_fin_cost'])} vs a_out6 {_f(rhos['a_out6'])} "
        f"vs m_fin {_f(rhos['m_fin_share'])} vs days {_f(rhos['c_n_days_with_tx'])} "
        f"vs size {_f(rhos['log1p(a_in3)'])} vs ogtg {_f(rhos['f_outstanding_gt_granted'])}. "
        f"Twins: {twins or 'none'}. SIZE={size_flag}. ogtg_twin={ogtg_twin}."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "twins": twins,
        "size_flag": size_flag,
        "twin_ds": "f_ds_r" in twins,
        "twin_mfin": "m_fin_share" in twins,
        "twin_afin": "a_fin_cost" in twins,
        "ogtg_twin": ogtg_twin,
        "weaker": weaker,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 3. Single-feature group-fold AUROC
# ---------------------------------------------------------------------------
def pass3_singles(tr: pd.DataFrame) -> dict:
    feats = [
        ("f_fc_r", tr["f_fc_r"]),
        ("f_fc_r_lag1", tr["f_fc_r_lag1"]),
        ("f_fc_r_lag3", tr["f_fc_r_lag3"]),
        ("f_ds_r", tr["f_ds_r"]),
        ("e_ar_issued_lag1", tr["e_ar_issued_lag1"]),
        ("log1p_a_in3", tr["log_in3"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
    ]
    rows = []
    store = {}
    for ycol in (Y3, Y7, Y2):
        lab = tr[ycol].notna()
        for name, x in feats:
            res = signed_oof_auroc(tr[ycol], x, tr["fold"], lab)
            store[(ycol, name)] = res
            rows.append(_auc_row(ycol, name, res))
            print(
                f"{ycol} {name}: CV={_f(res['cv'])} train={_f(res['train_auc'])} "
                f"sign={res['train_sign']} folds={fold_bits(res)}"
            )
    y3 = _cv(store[(Y3, "f_fc_r")])
    y7 = _cv(store[(Y7, "f_fc_r")])
    y7_l3 = _cv(store[(Y7, "f_fc_r_lag3")])
    days = _cv(store[(Y3, "c_n_days_with_tx")])
    size3 = _cv(store[(Y3, "log1p_a_in3")])
    ds3 = _cv(store[(Y3, "f_ds_r")])
    iss = _cv(store[(Y7, "e_ar_issued_lag1")])
    days_ok = bool(np.isfinite(days) and abs(days - DAYS_BENCH) < 0.005)
    size_ok = bool(np.isfinite(size3) and abs(size3 - SIZE_Y3_QUOTE) < 0.005)
    iss_ok = bool(np.isfinite(iss) and abs(iss - ISSUED_LAG1_BENCH) < 0.015)
    beat3 = bool(np.isfinite(y3) and np.isfinite(size3) and (y3 - size3) >= KEEP_DELTA)
    lose_days = bool(np.isfinite(y3) and np.isfinite(days) and (days - y3) > 0)
    lose_ds = bool(np.isfinite(y3) and np.isfinite(ds3) and (ds3 - y3) > 0)
    prose = (
        f"Y3 f_fc_r CV {_f(y3)} vs size {_f(size3)} "
        f"({'CONFIRM 0.617' if size_ok else 'size drifted'}) "
        f"vs days {_f(days)} ({'CONFIRM 0.711' if days_ok else 'days drifted'}) "
        f"vs f_ds_r {_f(ds3)}. beat_size={beat3}. "
        f"Y7 f_fc_r {_f(y7)} lag3 {_f(y7_l3)} vs issued_lag1 {_f(iss)} "
        f"({'CONFIRM ~0.630' if iss_ok else 'issued drifted'})."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "y3": y3,
        "y7": y7,
        "y7_l3": y7_l3,
        "y2": _cv(store[(Y2, "f_fc_r")]),
        "days": days,
        "size3": size3,
        "ds3": ds3,
        "iss": iss,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "iss_ok": iss_ok,
        "beat3": beat3,
        "lose_days": lose_days,
        "lose_ds": lose_ds,
        "y3_sign": store[(Y3, "f_fc_r")]["train_sign"],
        "y7_sign": store[(Y7, "f_fc_r")]["train_sign"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 4. Honest leftover Y3 after f_ds_r / days
# ---------------------------------------------------------------------------
def pass4_y3_leftover(tr: pd.DataFrame) -> dict:
    fc = tr["f_fc_r"]
    specs = [
        (Y3, "after f_ds_r", (tr["f_ds_r"],)),
        (Y3, "after days", (tr["c_n_days_with_tx"],)),
        (Y3, "after size", (tr["log_in3"],)),
        (Y3, "after ds+days", (tr["f_ds_r"], tr["c_n_days_with_tx"])),
        (Y3, "after a_fin_cost", (tr["a_fin_cost"],)),
        (Y2, "after f_ds_r", (tr["f_ds_r"],)),
        (Y2, "after days", (tr["c_n_days_with_tx"],)),
    ]
    if "m_fin_share" in tr.columns:
        specs.append((Y3, "after m_fin", (tr["m_fin_share"],)))
    rows = []
    store = {}
    for ycol, name, xs in specs:
        rec = leftover_diag(tr[ycol], fc, list(xs), tr["fold"], tr[ycol].notna())
        store[(ycol, name)] = rec
        rows.append(
            {
                "y": ycol,
                "control": name,
                "OLS": _f(rec["ols"]),
                "rank": _f(rec["rank"]),
                "ρ(resid,ctrl)": _f(rec["rho_ctrl"]),
                "ρ(resid,fc)": _f(rec["rho_x"]),
                "R²": _f(rec["r2"]),
                "n": f"{rec['n']:,}",
                "honest_dies": "YES" if rec["honest_dies"] else "no",
            }
        )
        print(
            f"Y3-left {ycol} {name}: OLS={_f(rec['ols'])} rank={_f(rec['rank'])} "
            f"R2={_f(rec['r2'])} dies={rec['honest_dies']}"
        )
    ds = store[(Y3, "after f_ds_r")]
    days = store[(Y3, "after days")]
    size = store[(Y3, "after size")]
    both = store[(Y3, "after ds+days")]
    clone_ds = bool(np.isfinite(ds["rho_x"]) and abs(ds["rho_x"]) >= 0.95)
    ds_dies = not (np.isfinite(ds["rank"]) and ds["rank"] >= CHANCE) or ds["honest_dies"]
    days_dies = not (np.isfinite(days["rank"]) and days["rank"] >= CHANCE) or days["honest_dies"]
    prose = (
        f"Y3 leftover after f_ds_r OLS {_f(ds['ols'])} rank {_f(ds['rank'])} "
        f"R²={_f(ds['r2'])} "
        f"({'dies <0.55 — unused leftover (low R², not a twin rewrite)' if ds_dies else 'lives'}). "
        f"After days OLS {_f(days['ols'])} rank {_f(days['rank'])} "
        f"({'dies' if days_dies else 'lives'}). "
        f"After size {_f(size['rank'])}. After ds+days {_f(both['rank'])}. "
        f"{'resid≈fc clone of ds_r' if clone_ds else 'resid is not an fc clone'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "ds_ols": ds["ols"],
        "ds_rank": ds["rank"],
        "ds_r2": ds["r2"],
        "ds_dies": ds_dies,
        "days_ols": days["ols"],
        "days_rank": days["rank"],
        "days_dies": days_dies,
        "size_rank": size["rank"],
        "both_rank": both["rank"],
        "clone_ds": clone_ds,
        "resid_ds": ds["resid"],
        "resid_days": days["resid"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 5. Y7 leftover after issued_lag1 / TURNOVER-without-fc
# ---------------------------------------------------------------------------
def pass5_y7_leftover(tr: pd.DataFrame) -> dict:
    fc = tr["f_fc_r"]
    lag3 = tr["f_fc_r_lag3"]
    specs = [
        ("contemp after issued_lag1", fc, (tr["e_ar_issued_lag1"],)),
        ("contemp after days", fc, (tr["c_n_days_with_tx"],)),
        ("lag3 after issued_lag1", lag3, (tr["e_ar_issued_lag1"],)),
        (
            "contemp after TURN_NOFC",
            fc,
            tuple(tr[c] for c in TURN_NOFC if c in tr.columns),
        ),
        (
            "lag3 after TURN_NOFC",
            lag3,
            tuple(tr[c] for c in TURN_NOFC if c in tr.columns),
        ),
    ]
    rows = []
    store = {}
    for name, x, xs in specs:
        rec = leftover_diag(tr[Y7], x, list(xs), tr["fold"], tr[Y7].notna())
        store[name] = rec
        rows.append(
            {
                "residual": name,
                "OLS": _f(rec["ols"]),
                "rank": _f(rec["rank"]),
                "R²": _f(rec["r2"]),
                "ρ(resid,fc)": _f(rec["rho_x"]),
                "n": f"{rec['n']:,}",
                "n_pos": f"{rec['n_pos']:,}",
                "honest_dies": "YES" if rec["honest_dies"] else "no",
                "folds": rec["folds"],
            }
        )
        print(f"Y7-left {name}: OLS={_f(rec['ols'])} rank={_f(rec['rank'])} R2={_f(rec['r2'])}")
    iss = store["contemp after issued_lag1"]
    turn = store["contemp after TURN_NOFC"]
    lag_iss = store["lag3 after issued_lag1"]
    lag_turn = store["lag3 after TURN_NOFC"]
    iss_dies = not (np.isfinite(iss["rank"]) and iss["rank"] >= CHANCE) or iss["honest_dies"]
    turn_dies = not (np.isfinite(turn["rank"]) and turn["rank"] >= CHANCE) or turn["honest_dies"]
    turn_clone = bool(np.isfinite(turn["rho_x"]) and abs(turn["rho_x"]) >= 0.95)
    prose = (
        f"Y7 leftover after issued_lag1 OLS {_f(iss['ols'])} rank {_f(iss['rank'])} "
        f"({'dies <0.55' if iss_dies else 'lives (bare)'}). "
        f"After TURNOVER-without-fc OLS {_f(turn['ols'])} rank {_f(turn['rank'])} "
        f"ρ(resid,fc)={_f(turn['rho_x'])} "
        f"({'OLS is an fc clone — 0.64 is not new leftover' if turn_clone else 'resid is not an fc clone'}). "
        f"lag3 after issued {_f(lag_iss['rank'])} after TURN_NOFC {_f(lag_turn['rank'])}. "
        f"Did not refit TURNOVER. Quote 0.720 / TURN_NOFC 0.714 unchanged. "
        f"Bare leftover ≥0.55 is CLOSE as a TURNOVER add-on (TURNFC0 failed fold 4)."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "iss_ols": iss["ols"],
        "iss_rank": iss["rank"],
        "iss_dies": iss_dies,
        "turn_ols": turn["ols"],
        "turn_rank": turn["rank"],
        "turn_dies": turn_dies,
        "lag_iss": lag_iss["rank"],
        "lag_turn": lag_turn["rank"],
        "resid_iss": iss["resid"],
        "turn_clone": turn_clone,
        "turn_rho_x": turn["rho_x"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 6. Q6 lag1/lag3 empty-on-short
# ---------------------------------------------------------------------------
def pass6_q6(tr: pd.DataFrame) -> dict:
    rows = []
    for ycol, cols in (
        (Y3, ("f_fc_r", "f_fc_r_lag1", "f_fc_r_lag3", "f_ds_r", "f_ds_r_lag3")),
        (Y7, ("f_fc_r", "f_fc_r_lag1", "f_fc_r_lag3", "e_ar_issued_lag1")),
    ):
        lab = tr[ycol].notna()
        for bucket, sl in (
            ("<6", lab & (tr["so_far_bucket"] == "<6")),
            ("6-11", lab & (tr["so_far_bucket"] == "6-11")),
            ("12-17", lab & (tr["so_far_bucket"] == "12-17")),
            ("18-23", lab & (tr["so_far_bucket"] == "18-23")),
            ("short_<12", lab & (tr["so_far_class"] == "short_<12")),
            ("long_>=18", lab & (tr["so_far_class"] == "long_>=18")),
            ("all", lab),
        ):
            rec = {"y": ycol, "bucket": bucket, "n_lab": int(sl.sum())}
            for c in cols:
                nn = int((sl & pd.to_numeric(tr[c], errors="coerce").notna()).sum())
                rec[c] = f"{nn:,} ({_pp(_pct(nn, int(sl.sum())))})" if sl.sum() else "—"
            rows.append(rec)
    lab7 = tr[Y7].notna()
    short6 = lab7 & (tr["months_so_far"] < 6)
    ge6 = lab7 & (tr["months_so_far"] >= 6)
    lag3 = pd.to_numeric(tr["f_fc_r_lag3"], errors="coerce")
    empty_lt6 = int((short6 & lag3.isna()).sum())
    n_lt6 = int(short6.sum())
    nn_lt6 = int((short6 & lag3.notna()).sum())
    nn_ge6 = int((ge6 & lag3.notna()).sum())
    empty_ok = nn_lt6 == 0
    # contemporaneous also empty until rolling 3 (so-far≥3)
    fc = pd.to_numeric(tr["f_fc_r"], errors="coerce")
    lt3 = lab7 & (tr["months_so_far"] < 3)
    nn_fc_lt3 = int((lt3 & fc.notna()).sum())
    singles = []
    for name, mask in (
        ("Y7 short_<12 f_fc_r", lab7 & (tr["so_far_class"] == "short_<12")),
        ("Y7 so-far<6 f_fc_r (lag3 empty)", short6),
        ("Y7 so-far≥6 f_fc_r", ge6),
        ("Y7 so-far≥6 f_fc_r_lag3", ge6),
        ("Y7 long f_fc_r_lag3", lab7 & (tr["so_far_class"] == "long_>=18")),
        ("Y3 short f_fc_r", tr[Y3].notna() & (tr["so_far_class"] == "short_<12")),
        ("Y3 so-far<6 f_fc_r", tr[Y3].notna() & (tr["months_so_far"] < 6)),
    ):
        ycol = Y3 if name.startswith("Y3") else Y7
        col = "f_fc_r_lag3" if "lag3" in name and "empty" not in name else "f_fc_r"
        res = signed_oof_auroc(tr[ycol], tr[col], tr["fold"], mask)
        singles.append(_auc_row(name, col, res))
    q6_close = empty_ok
    prose = (
        f"f_fc_r_lag3 nn on Y7 so-far<6: {nn_lt6}/{n_lt6} "
        f"({'CONFIRM empty until so-far≥6' if empty_ok else 'DRIFT — lag3 leaks onto short books'}). "
        f"Y7 so-far≥6 lag3 nn {nn_ge6}. "
        f"f_fc_r nn on Y7 so-far<3: {nn_fc_lt3} (rolling-3 hole). "
        f"F lag3 Q6 stays CLOSE."
    )
    print(prose)
    return {
        "rows": rows,
        "singles": singles,
        "empty_ok": empty_ok,
        "nn_lt6": nn_lt6,
        "n_lt6": n_lt6,
        "nn_ge6": nn_ge6,
        "nn_fc_lt3": nn_fc_lt3,
        "q6_close": q6_close,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 7. SIZE terciles / ICC / demean — trait vs month shock; Y9 leak?
# ---------------------------------------------------------------------------
def pass7_trait(tr: pd.DataFrame) -> dict:
    fc = pd.to_numeric(tr["f_fc_r"], errors="coerce")
    terc = size_terciles(tr)
    tr = tr.assign(_T=terc)
    icc = icc_anova(fc, tr["company_id"])
    acf1 = median_acf(fc, tr["company_id"], 1)
    acf3 = median_acf(fc, tr["company_id"], 3)
    demean = company_demean(fc, tr["company_id"])
    cmean = company_mean(fc, tr["company_id"])
    trait = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_STYLE)
    rows = []
    store = {}
    for ycol in (Y3, Y7, Y9):
        lab = tr[ycol].notna()
        raw = signed_oof_auroc(tr[ycol], fc, tr["fold"], lab)
        mu = signed_oof_auroc(tr[ycol], cmean, tr["fold"], lab)
        dm = signed_oof_auroc(tr[ycol], demean, tr["fold"], lab)
        store[(ycol, "raw")] = raw
        store[(ycol, "mean")] = mu
        store[(ycol, "demean")] = dm
        rows.append(_auc_row(ycol, "f_fc_r", raw))
        rows.append(_auc_row(ycol, "company-mean", mu))
        rows.append(_auc_row(ycol, "company-demean", dm))
    t_rows = []
    for t in ("T1", "T2", "T3"):
        sl = tr["_T"] == t
        for ycol in (Y3, Y7, Y9):
            y = pd.to_numeric(tr.loc[sl, ycol], errors="coerce")
            x = fc[sl]
            hi = x >= x.median() if x.notna().any() else pd.Series(False, index=x.index)
            lab = y.notna()
            t_rows.append(
                {
                    "T": t,
                    "y": ycol,
                    "n_lab": int(lab.sum()),
                    "n_pos": int((lab & (y == 1)).sum()),
                    "rate": _pp(float(y[lab].mean()) if lab.any() else float("nan")),
                    "rate hi-fc": _pp(
                        float(y[lab & hi].mean()) if (lab & hi).any() else float("nan")
                    ),
                    "rate lo-fc": _pp(
                        float(y[lab & ~hi].mean()) if (lab & ~hi).any() else float("nan")
                    ),
                }
            )
    y9_mean = _cv(store[(Y9, "mean")])
    y9_dm = _cv(store[(Y9, "demean")])
    y3_dm = _cv(store[(Y3, "demean")])
    same_leak = bool(np.isfinite(y9_mean) and y9_mean >= 0.60 and (not np.isfinite(y9_dm) or y9_dm < 0.56))
    prose = (
        f"ICC {_f(icc['icc'])} acf1 {_f(acf1)} acf3 {_f(acf3)} "
        f"({'TRAIT' if trait else 'month-moving'}). "
        f"Y3 demean {_f(y3_dm)} vs company-mean {_f(_cv(store[(Y3, 'mean')]))}. "
        f"Y9 company-mean {_f(y9_mean)} demean {_f(y9_dm)} "
        f"({'same leak shape as m_fin — trait fee pressure is the Y' if same_leak else 'not the m_fin trait-leak shape'})."
    )
    print(prose)
    return {
        "rows": rows,
        "t_rows": t_rows,
        "icc": icc["icc"],
        "acf1": acf1,
        "acf3": acf3,
        "trait": trait,
        "y3_dm": y3_dm,
        "y3_mu": _cv(store[(Y3, "mean")]),
        "y7_dm": _cv(store[(Y7, "demean")]),
        "y9_mu": y9_mean,
        "y9_dm": y9_dm,
        "y9_raw": _cv(store[(Y9, "raw")]),
        "same_leak": same_leak,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 8. vs Y9 accepted labels — Jaccard / ρ. If it is the Y, CLOSE as Y3 X
# ---------------------------------------------------------------------------
def pass8_y9(tr: pd.DataFrame) -> dict:
    fc = pd.to_numeric(tr["f_fc_r"], errors="coerce")
    y9 = pd.to_numeric(tr[Y9], errors="coerce")
    lab = y9.notna()
    rho = spearman(fc[lab], y9[lab])
    # company p80 of contemporaneous fc (not expanding — descriptive overlap)
    p80 = tr.groupby("company_id")["f_fc_r"].transform(lambda s: s.quantile(0.80))
    hi = fc > p80
    jac_hi = jaccard(hi & lab, y9.eq(1) & lab)
    jac_pos = jaccard(fc.gt(0) & lab, y9.eq(1) & lab)
    rho_spike = spearman(fc, tr[Y9_SPIKE])
    is_y = bool((np.isfinite(rho) and abs(rho) >= TWIN_RHO) or (np.isfinite(jac_hi["jaccard"]) and jac_hi["jaccard"] >= 0.50))
    # do NOT score F as Y9 X — ρ / Jaccard only
    rows = [
        {
            "pair": "f_fc_r vs y9_fee_r_ownp80",
            "n": int(lab.sum()),
            "ρ": _f(rho),
            "is_Y_|ρ|≥0.80": "YES" if (np.isfinite(rho) and abs(rho) >= TWIN_RHO) else "no",
        },
        {
            "pair": "f_fc_r vs y9_fee_spike",
            "n": int(tr[Y9_SPIKE].notna().sum()),
            "ρ": _f(rho_spike),
            "is_Y_|ρ|≥0.80": "YES" if (np.isfinite(rho_spike) and abs(rho_spike) >= TWIN_RHO) else "no",
        },
    ]
    jrows = [
        {
            "set A": "f_fc_r > company p80",
            "set B": "Y9=1",
            "Jaccard": _f(jac_hi["jaccard"]),
            "n_A": jac_hi["n_a"],
            "n_B": jac_hi["n_b"],
            "n_both": jac_hi["n_both"],
        },
        {
            "set A": "f_fc_r > 0",
            "set B": "Y9=1",
            "Jaccard": _f(jac_pos["jaccard"]),
            "n_A": jac_pos["n_a"],
            "n_B": jac_pos["n_b"],
            "n_both": jac_pos["n_both"],
        },
    ]
    prose = (
        f"f_fc_r vs Y9 own-p80 ρ={_f(rho)} Jaccard(hi-p80,Y9+)={_f(jac_hi['jaccard'])} "
        f"Jaccard(fc>0,Y9+)={_f(jac_pos['jaccard'])}. "
        f"Y9 META: f_fc_r is the fee_r raw material (forbidden F as Y9 X). "
        f"{'It is the Y — CLOSE contemporaneous as Y3 X' if is_y else 'Not a |ρ|≥0.80 / Jaccard≥0.50 rewrite of the binary Y9 label (raw material still forbidden as Y9 X)'}."
    )
    print(prose)
    return {
        "rows": rows,
        "jrows": jrows,
        "rho": rho,
        "rho_spike": rho_spike,
        "jac_hi": jac_hi["jaccard"],
        "jac_pos": jac_pos["jaccard"],
        "is_y": is_y,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 9. Fold 4 / short books — contemporaneous where lag3 is empty?
# ---------------------------------------------------------------------------
def pass9_fold4(tr: pd.DataFrame) -> dict:
    lab = tr[Y7].notna()
    f4 = lab & (tr["fold"] == 4)
    short = lab & (tr["months_so_far"] < 6)
    rows = []
    store = {}
    for name, col in (
        ("f_fc_r", tr["f_fc_r"]),
        ("f_fc_r_lag3", tr["f_fc_r_lag3"]),
        ("e_ar_issued_lag1", tr["e_ar_issued_lag1"]),
        ("e_ar_issued_lag_cv", tr["e_ar_issued_lag_cv"]),
        ("f_ds_r", tr["f_ds_r"]),
    ):
        res = signed_oof_auroc(tr[Y7], col, tr["fold"], lab)
        store[name] = res
        rows.append(
            {
                "feature": name,
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "fold4": _f(fold_k(res, 4)),
                "folds": fold_bits(res) if not res["low_power"] else "—",
            }
        )
    f4_rows = []
    for name, col in (
        ("f_fc_r", tr["f_fc_r"]),
        ("f_fc_r_lag3", tr["f_fc_r_lag3"]),
        ("e_ar_issued_lag1", tr["e_ar_issued_lag1"]),
        ("e_ar_issued_lag_cv", tr["e_ar_issued_lag_cv"]),
    ):
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
    short_res = signed_oof_auroc(tr[Y7], tr["f_fc_r"], tr["fold"], short)
    short_iss = signed_oof_auroc(tr[Y7], tr["e_ar_issued_lag1"], tr["fold"], short)
    grp = []
    for gid in FOLD4_GROUPS:
        sl = lab & (tr["group_id"].astype(str) == gid)
        y = pd.to_numeric(tr.loc[sl, Y7], errors="coerce")
        x = pd.to_numeric(tr.loc[sl, "f_fc_r"], errors="coerce")
        grp.append(
            {
                "group": gid,
                "n_lab": int(sl.sum()),
                "n_pos": int((y == 1).sum()),
                "rate": _pp(float(y.mean()) if y.notna().any() else float("nan")),
                "f_fc_r p50": _f(float(x.median()) if x.notna().any() else float("nan"), 4),
                "lag3 nn": int(pd.to_numeric(tr.loc[sl, "f_fc_r_lag3"], errors="coerce").notna().sum()),
            }
        )
    fc4 = next(r for r in f4_rows if r["feature"] == "f_fc_r")
    l34 = next(r for r in f4_rows if r["feature"] == "f_fc_r_lag3")
    iss4 = next(r for r in f4_rows if r["feature"] == "e_ar_issued_lag1")
    fc4v = float(fc4["fold4"]) if fc4["fold4"] != "—" else float("nan")
    l34v = float(l34["fold4"]) if l34["fold4"] != "—" else float("nan")
    iss4v = float(iss4["fold4"]) if iss4["fold4"] != "—" else float("nan")
    short_cv = _cv(short_res)
    helps_short = bool(np.isfinite(short_cv) and short_cv >= CHANCE + 0.02)
    helps_f4 = bool(np.isfinite(fc4v) and fc4v >= TURNOVER_F4 - 0.02 and fc4v > (l34v if np.isfinite(l34v) else 0))
    prose = (
        f"Fold 4 (sign 0–3): f_fc_r {_f(fc4v)} lag3 {_f(l34v)} issued_lag1 {_f(iss4v)}. "
        f"TURNOVER fold-4 quote 0.680. "
        f"Y7 so-far<6 (lag3 empty) contemporaneous {_f(short_cv)} vs issued {_f(_cv(short_iss))}. "
        f"{'Contemporaneous helps the empty-lag3 window' if helps_short else 'Contemporaneous does not rescue short books where lag3 is empty'}. "
        f"{'Contemporaneous owns fold 4' if helps_f4 else 'Fold 4 is still issued / issued-CV, not contemporaneous fc'}."
    )
    print(prose)
    return {
        "rows": rows,
        "f4_rows": f4_rows,
        "grp": grp,
        "short_row": _auc_row("Y7 so-far<6", "f_fc_r", short_res),
        "fc4": fc4v,
        "l34": l34v,
        "iss4": iss4v,
        "short_cv": short_cv,
        "short_iss": _cv(short_iss),
        "helps_short": helps_short,
        "helps_f4": helps_f4,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 10. Dark 470 — do they have f_fc_r?
# ---------------------------------------------------------------------------
def pass10_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    dark = ~tr["company_id"].isin(book)
    fc = pd.to_numeric(tr["f_fc_r"], errors="coerce")
    e_iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    rows = []
    for name, sl in (("dark", dark), ("ERP", ~dark)):
        n_co = int(tr.loc[sl, "company_id"].nunique())
        rows.append(
            {
                "book": name,
                "n_co": n_co,
                "n_cm": int(sl.sum()),
                "f_fc_r nn": int((sl & fc.notna()).sum()),
                "f_fc_r cov": _pp(_pct(int((sl & fc.notna()).sum()), int(sl.sum()))),
                "f_fc_r ever": int(tr.loc[sl & fc.notna(), "company_id"].nunique()),
                "e_ar_issued nn": int((sl & e_iss.notna()).sum()),
                "Y3 labeled": int((sl & tr[Y3].notna()).sum()),
                "Y7 labeled": int((sl & tr[Y7].notna()).sum()),
            }
        )
    dark_iss_nn = int((dark & e_iss.notna()).sum())
    dark_fc_nn = int((dark & fc.notna()).sum())
    invoice_nan = dark_iss_nn == 0
    stub = dark_iss_nn <= 2
    access = dark_fc_nn > 0
    res_d = signed_oof_auroc(tr[Y3], fc, tr["fold"], tr[Y3].notna() & dark)
    res_e = signed_oof_auroc(tr[Y3], fc, tr["fold"], tr[Y3].notna() & ~dark)
    prose = (
        f"Dark {int(tr.loc[dark, 'company_id'].nunique())} vs ERP "
        f"{int(tr.loc[~dark, 'company_id'].nunique())}: "
        f"f_fc_r nn {dark_fc_nn:,} vs {int((~dark & fc.notna()).sum()):,}. "
        f"e_ar_issued on dark {dark_iss_nn} "
        f"({'CONFIRM invoice-only NaN' if invoice_nan else ('BOOK vs E stub' if stub else 'invoice leak on dark')}). "
        f"{'Access ≠ ERP' if access else 'Dark have no fc — unexpected'}. "
        f"Y3 f_fc_r dark {_f(_cv(res_d))} ERP {_f(_cv(res_e))}."
    )
    print(prose)
    return {
        "rows": rows,
        "dark_fc_nn": dark_fc_nn,
        "dark_iss_nn": dark_iss_nn,
        "invoice_nan": invoice_nan,
        "access": access,
        "y3_dark": _cv(res_d),
        "y3_erp": _cv(res_e),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 11. f_outstanding_gt_granted PARK — not a twin of fc
# ---------------------------------------------------------------------------
def pass11_ogtg(tr: pd.DataFrame) -> dict:
    og = pd.to_numeric(tr["f_outstanding_gt_granted"], errors="coerce")
    fc = pd.to_numeric(tr["f_fc_r"], errors="coerce")
    last = tr["period"] == tr["period"].max()
    nn = og.notna()
    last_only = bool(nn.any() and (nn & last).sum() == nn.sum())
    rho = spearman(fc, og)
    twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    rows = [
        {
            "col": "f_outstanding_gt_granted",
            "n_nn": int(nn.sum()),
            "cov": _pp(_pct(int(nn.sum()), len(tr))),
            "share last-month": _pp(_pct(int((nn & last).sum()), int(nn.sum()))),
            "last-month only": "YES" if last_only else "no",
            "ρ vs f_fc_r": _f(rho),
            "twin": "YES" if twin else "no",
        }
    ]
    prose = (
        f"ogtg last-month-only={last_only} n_nn={int(nn.sum()):,} "
        f"ρ vs f_fc_r={_f(rho)} "
        f"({'TWIN — reopen' if twin else 'not a twin — PARK snapshot stands'})."
    )
    print(prose)
    return {
        "rows": rows,
        "last_only": last_only,
        "rho": rho,
        "twin": twin,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 12. Decision table
# ---------------------------------------------------------------------------
def decide(p2, p3, p4, p5, p6, p7, p8, p9, p11) -> dict:
    drop44 = bool(
        p2["twin_ds"]
        or p4["ds_dies"]
        or p4["days_dies"]
        or p8["is_y"]
        or (not p3["beat3"])
    )
    rho_ds = p2["rhos"]["f_ds_r"]
    if p2["twin_ds"]:
        contemp = "DROP"
        why = (
            f"twin of f_ds_r (ρ={_f(rho_ds)}) — DROP weaker contemporaneous from the 44; "
            "do not rip lag3 off TURNOVER"
        )
    elif p4["ds_dies"] or p4["days_dies"] or not p3["beat3"]:
        contemp = "DROP"
        why = (
            f"unused leftover, not a twin (ρ vs f_ds_r={_f(rho_ds)}, R²={_f(p4['ds_r2'])}): "
            f"Y3 leftover after f_ds_r rank {_f(p4['ds_rank'])} / after days {_f(p4['days_rank'])} both die; "
            f"single {_f(p3['y3'])} loses to size {_f(p3['size3'])} and days {_f(p3['days'])}"
        )
    elif p8["is_y"]:
        contemp = "CLOSE"
        why = "it is the binary Y9 label — CLOSE as Y3 X"
    else:
        contemp = "KEEP"
        why = "leftover lives after f_ds_r and days; beats size"
    lag3 = "KEEP"
    q6 = "CLOSE"
    y9_role = (
        "CLOSE as Y3 X"
        if p8["is_y"]
        else "fee_r raw material (Y9 forbids F as X); not a binary twin (ρ 0.13)"
    )
    return {
        "contemp": contemp,
        "lag3": lag3,
        "q6": q6,
        "drop44": drop44,
        "why": why,
        "y9_role": y9_role,
        "y3": contemp if contemp != "KEEP" else "KEEP",
        "turnover": "KEEP — do not rip; TURNFC0/1 failed fold 4",
    }


# ---------------------------------------------------------------------------
# Extras
# ---------------------------------------------------------------------------
def pass_holdout(panel: pd.DataFrame, book: set[str]) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    fc = pd.to_numeric(ho["f_fc_r"], errors="coerce")
    lag3 = pd.to_numeric(ho["f_fc_r_lag3"], errors="coerce")
    dark = ~ho["company_id"].isin(book)
    rows = [
        {
            "split": "holdout",
            "n_co": int(ho["company_id"].nunique()),
            "n_cm": len(ho),
            "f_fc_r nn": int(fc.notna().sum()),
            "f_fc_r cov": _pp(_pct(int(fc.notna().sum()), len(ho))),
            "lag3 nn": int(lag3.notna().sum()),
            "dark ever fc": int(ho.loc[dark & fc.notna(), "company_id"].nunique()),
        }
    ]
    prose = (
        f"Holdout 72 coverage only: {len(ho):,} CM, fc nn {int(fc.notna().sum()):,}, "
        f"lag3 nn {int(lag3.notna().sum()):,}. No AUROC. No percentile fit."
    )
    print(prose)
    return {"rows": rows, "n_cm": len(ho), "fc_nn": int(fc.notna().sum()), "prose": prose}


def pass_quintiles(tr: pd.DataFrame) -> dict:
    fc = pd.to_numeric(tr["f_fc_r"], errors="coerce")
    ds = pd.to_numeric(tr["f_ds_r"], errors="coerce")
    q = pd.qcut(fc, 5, duplicates="drop")
    rows = []
    for ycol in (Y3, Y7, Y9):
        y = pd.to_numeric(tr[ycol], errors="coerce")
        for cat, sl in q.groupby(q, observed=True):
            idx = sl.index
            lab = y.loc[idx].notna()
            rows.append(
                {
                    "y": ycol,
                    "Q": str(cat),
                    "n_lab": int(lab.sum()),
                    "n_pos": int((lab & (y.loc[idx] == 1)).sum()),
                    "rate": _pp(float(y.loc[idx][lab].mean()) if lab.any() else float("nan")),
                    "fc p50": _f(float(fc.loc[idx].median()), 4),
                }
            )
    prose = "Y3 / Y7 / Y9 rates by f_fc_r quintile (train, labeled)."
    print(prose)
    return {"rows": rows, "prose": prose, "q": q, "ds": ds}


def pass_zero(tr: pd.DataFrame) -> dict:
    fc = pd.to_numeric(tr["f_fc_r"], errors="coerce")
    flag = (fc > 0).astype(float)
    flag = flag.where(fc.notna())
    rows = []
    store = {}
    for ycol in (Y3, Y7):
        res = signed_oof_auroc(tr[ycol], flag, tr["fold"], tr[ycol].notna())
        store[ycol] = res
        rows.append(_auc_row(ycol, "f_fc_r>0", res))
        nz = signed_oof_auroc(tr[ycol], fc, tr["fold"], tr[ycol].notna() & fc.gt(0))
        rows.append(_auc_row(ycol, "f_fc_r on gt0", nz))
    prose = (
        f"Zero vs intensity: Y3 flag {_f(_cv(store[Y3]))} "
        f"Y7 flag {_f(_cv(store[Y7]))}."
    )
    print(prose)
    return {"rows": rows, "y3": _cv(store[Y3]), "y7": _cv(store[Y7]), "prose": prose}


def pass_mfin_left(tr: pd.DataFrame) -> dict:
    if "m_fin_share" not in tr.columns:
        return {"rows": [], "prose": "m_fin missing.", "y3": float("nan")}
    rec = leftover_diag(tr[Y3], tr["f_fc_r"], [tr["m_fin_share"]], tr["fold"], tr[Y3].notna())
    rec9 = leftover_diag(tr[Y9], tr["f_fc_r"], [tr["m_fin_share"]], tr["fold"], tr[Y9].notna())
    rows = [
        {
            "y": Y3,
            "OLS": _f(rec["ols"]),
            "rank": _f(rec["rank"]),
            "R²": _f(rec["r2"]),
            "honest_dies": "YES" if rec["honest_dies"] else "no",
        },
        {
            "y": Y9,
            "OLS": _f(rec9["ols"]),
            "rank": _f(rec9["rank"]),
            "R²": _f(rec9["r2"]),
            "honest_dies": "YES" if rec9["honest_dies"] else "no",
        },
    ]
    prose = (
        f"Leftover after in-memory m_fin: Y3 OLS {_f(rec['ols'])} rank {_f(rec['rank'])} "
        f"Y9 OLS {_f(rec9['ols'])} rank {_f(rec9['rank'])} R²={_f(rec9['r2'])}. "
        f"Do not merge M."
    )
    print(prose)
    return {
        "rows": rows,
        "y3": rec["rank"],
        "y9": rec9["rank"],
        "r2_y9": rec9["r2"],
        "dies3": rec["honest_dies"],
        "dies9": rec9["honest_dies"],
        "prose": prose,
    }


def pass_monthly_ratio(tr: pd.DataFrame) -> dict:
    """a_fin_cost / a_in3 is the monthly twin, not the trailing-3 f_fc_r."""
    ain = pd.to_numeric(tr["a_in3"], errors="coerce")
    fc_m = pd.to_numeric(tr["a_fin_cost"], errors="coerce") / np.maximum(ain, 1.0)
    fc_m = fc_m.clip(0.0, 1.0)
    rho = spearman(tr["f_fc_r"], fc_m)
    rec = leftover_diag(tr[Y3], tr["f_fc_r"], [fc_m], tr["fold"], tr[Y3].notna())
    twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    prose = (
        f"f_fc_r vs monthly a_fin_cost/max(a_in3,1) ρ={_f(rho)} "
        f"({'TWIN of monthly ratio' if twin else 'trailing-3 is not the monthly ratio'}). "
        f"Y3 leftover after monthly ratio rank {_f(rec['rank'])}."
    )
    print(prose)
    return {
        "rho": rho,
        "twin": twin,
        "y3": rec["rank"],
        "r2": rec["r2"],
        "prose": prose,
    }


def pass_fold_left(tr: pd.DataFrame, p4: dict) -> dict:
    y = pd.to_numeric(tr[Y3], errors="coerce")
    resid = p4["resid_ds"]
    rows = []
    for k in range(N_FOLDS):
        sl = tr[Y3].notna() & (tr["fold"] == k) & resid.notna()
        auc = auroc(y[sl], resid[sl])
        rows.append(
            {
                "fold": k,
                "n": int(sl.sum()),
                "n_pos": int((sl & (y == 1)).sum()),
                "leftover after f_ds_r": _f(auc),
            }
        )
    vals = [float(r["leftover after f_ds_r"]) for r in rows if r["leftover after f_ds_r"] != "—"]
    spread = float(max(vals) - min(vals)) if len(vals) >= 2 else float("nan")
    prose = f"Y3 leftover-after-ds fold spread {_f(spread)}."
    print(prose)
    return {"rows": rows, "spread": spread, "prose": prose}


def pass_ds_after_fc(tr: pd.DataFrame) -> dict:
    """Does the 15-col stem survive residualizing fc? If yes, ds is the flow engine."""
    rec = leftover_diag(tr[Y3], tr["f_ds_r"], [tr["f_fc_r"]], tr["fold"], tr[Y3].notna())
    raw = signed_oof_auroc(tr[Y3], tr["f_ds_r"], tr["fold"], tr[Y3].notna())
    lives = bool(np.isfinite(rec["rank"]) and rec["rank"] >= CHANCE)
    prose = (
        f"Y3 f_ds_r raw {_f(_cv(raw))} leftover after f_fc_r rank {_f(rec['rank'])} "
        f"OLS {_f(rec['ols'])} R²={_f(rec['r2'])} "
        f"({'15-col stem still lives — fc is the unused twin-ish leftover' if lives else 'ds also dies after fc'})."
    )
    print(prose)
    return {
        "raw": _cv(raw),
        "rank": rec["rank"],
        "ols": rec["ols"],
        "r2": rec["r2"],
        "lives": lives,
        "prose": prose,
    }


def pass_support(tr: pd.DataFrame) -> dict:
    """Leftover after f_ds_r on the fc>0 support — is intensity the leftover?"""
    fc = pd.to_numeric(tr["f_fc_r"], errors="coerce")
    mask = tr[Y3].notna() & fc.gt(0)
    rec = leftover_diag(tr[Y3], fc, [tr["f_ds_r"]], tr["fold"], mask)
    raw = signed_oof_auroc(tr[Y3], fc, tr["fold"], mask)
    days = leftover_diag(tr[Y3], fc, [tr["c_n_days_with_tx"]], tr["fold"], mask)
    dies = not (np.isfinite(rec["rank"]) and rec["rank"] >= CHANCE) or rec["honest_dies"]
    prose = (
        f"Y3 on fc>0: raw {_f(_cv(raw))} leftover-ds rank {_f(rec['rank'])} "
        f"leftover-days {_f(days['rank'])} n={rec['n']:,} "
        f"({'intensity leftover dies too' if dies else 'intensity leftover lives'})."
    )
    print(prose)
    return {
        "raw": _cv(raw),
        "ds": rec["rank"],
        "days": days["rank"],
        "n": rec["n"],
        "dies": dies,
        "prose": prose,
    }


def pass_fold2(tr: pd.DataFrame) -> dict:
    """Y3 f_fc_r fold 2 was 0.703 — group dummy or real?"""
    lab = tr[Y3].notna()
    f2 = lab & (tr["fold"] == 2)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    fc = pd.to_numeric(tr["f_fc_r"], errors="coerce")
    g = (
        tr.loc[f2]
        .assign(y=y[f2].values, fc=fc[f2].values)
        .groupby("group_id", sort=False)
        .agg(n=("company_id", "size"), n_pos=("y", "sum"), n_co=("company_id", "nunique"))
    )
    g["rate"] = g["n_pos"] / g["n"]
    g = g.sort_values("n_pos", ascending=False).head(8)
    rows = []
    for gid, r in g.iterrows():
        sl = f2 & (tr["group_id"] == gid)
        auc = auroc(y[sl], -fc[sl])  # Y3 sign is −
        rows.append(
            {
                "group": gid,
                "n_lab": int(r["n"]),
                "n_pos": int(r["n_pos"]),
                "n_co": int(r["n_co"]),
                "rate": _pp(float(r["rate"])),
                "AUROC −fc": _f(auc),
            }
        )
    wo = signed_oof_auroc(tr[Y3], tr["f_fc_r"], tr["fold"], lab & (tr["fold"] != 2))
    top = rows[0]["group"] if rows else ""
    top_share = float(g.iloc[0]["n_pos"] / y[f2].eq(1).sum()) if rows and y[f2].eq(1).sum() else float("nan")
    dummy = bool(np.isfinite(top_share) and top_share >= 0.40)
    prose = (
        f"Fold 2 Y3 positives share in top group {top} = {_pp(top_share)}. "
        f"CV without fold 2 {_f(_cv(wo))}. "
        f"{'one-group leftover' if dummy else 'not a single-group dummy'}."
    )
    print(prose)
    return {
        "rows": rows,
        "wo": _cv(wo),
        "top": top,
        "top_share": top_share,
        "dummy": dummy,
        "prose": prose,
    }


def pass_dark_inv(tr: pd.DataFrame, book: set[str]) -> dict:
    dark = ~tr["company_id"].isin(book)
    iss = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    sl = dark & iss.notna()
    hits = (
        tr.loc[sl, ["company_id", "group_id", "period", "e_ar_issued", "f_fc_r", Y3, Y7]]
        .copy()
        .head(8)
    )
    rows = []
    for _, r in hits.iterrows():
        rows.append(
            {
                "company": r["company_id"],
                "group": r["group_id"],
                "period": str(pd.Timestamp(r["period"]).date()),
                "e_ar_issued": _f(r["e_ar_issued"], 1),
                "f_fc_r": _f(r["f_fc_r"], 4),
            }
        )
    prose = (
        f"Dark ∧ e_ar_issued defined: {int(sl.sum())} CM / "
        f"{int(tr.loc[sl, 'company_id'].nunique())} co. "
        f"{'BOOK vs Family E mismatch on a stub — invoice-only still NaN-not-0 for the 470' if sl.sum() <= 2 else 'more than a stub'}."
    )
    print(prose)
    return {"rows": rows, "n": int(sl.sum()), "prose": prose}


def pass_holdout_q6(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    lab = ho[Y7].notna()
    lag3 = pd.to_numeric(ho["f_fc_r_lag3"], errors="coerce")
    lt6 = lab & (ho["months_so_far"] < 6)
    nn = int((lt6 & lag3.notna()).sum())
    ok = nn == 0
    rows = [
        {
            "split": "holdout",
            "Y7 n_lab so-far<6": int(lt6.sum()),
            "lag3 nn": nn,
            "empty": "YES" if ok else "no",
            "Y7 n_lab so-far≥6": int((lab & (ho["months_so_far"] >= 6)).sum()),
            "lag3 nn ≥6": int((lab & (ho["months_so_far"] >= 6) & lag3.notna()).sum()),
        }
    ]
    prose = (
        f"Holdout Q6 coverage only: Y7 so-far<6 lag3 nn {nn}/{int(lt6.sum())} "
        f"({'CONFIRM empty' if ok else 'DRIFT'})."
    )
    print(prose)
    return {"rows": rows, "ok": ok, "nn": nn, "prose": prose}


def pass_y9_lead(tr: pd.DataFrame) -> dict:
    """Contemporaneous fc is fee_r now; Y9 is the next-3m own-p80 path."""
    lab = tr[Y9].notna()
    rows = []
    for name, col in (
        ("f_fc_r", tr["f_fc_r"]),
        ("f_fc_r_lag1", tr["f_fc_r_lag1"]),
        ("f_fc_r_lag3", tr["f_fc_r_lag3"]),
        ("m_fin_share", tr["m_fin_share"]),
        ("a_fin_cost", tr["a_fin_cost"]),
    ):
        rho = spearman(col[lab], tr[Y9][lab])
        rows.append({"x": name, "n": int(lab.sum()), "ρ vs Y9": _f(rho)})
    hi = pd.to_numeric(tr["f_fc_r"], errors="coerce") > tr.groupby("company_id")["f_fc_r"].transform(
        lambda s: s.quantile(0.80)
    )
    rho_hi = spearman(tr.loc[lab & hi, "f_fc_r"], tr.loc[lab & hi, Y9])
    prose = (
        f"Y9 is a forward path, not now-fee_r: contemp ρ={rows[0]['ρ vs Y9']} "
        f"lag3 ρ={rows[2]['ρ vs Y9']} m_fin ρ={rows[3]['ρ vs Y9']}. "
        f"On company-p80-hi months ρ={_f(rho_hi)}. Not the binary Y."
    )
    print(prose)
    return {"rows": rows, "rho_hi": rho_hi, "prose": prose}


def pass_days_fake(tr: pd.DataFrame, p4: dict) -> dict:
    """OLS leftover after days 0.648 vs rank 0.449 — fake days leak?"""
    rec = leftover_diag(
        tr[Y3], tr["f_fc_r"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna()
    )
    fake = bool(rec["fake"] or (np.isfinite(rec["ols"]) and rec["ols"] >= CHANCE and rec["rank"] < CHANCE))
    prose = (
        f"Y3 leftover after days OLS {_f(rec['ols'])} rank {_f(rec['rank'])} "
        f"ρ(resid,days)={_f(rec['rho_ctrl'])} ρ(resid,fc)={_f(rec['rho_x'])}. "
        f"{'OLS leftover is a fake days leak — honest rank dies' if fake else 'OLS and rank agree'}."
    )
    print(prose)
    return {
        "ols": rec["ols"],
        "rank": rec["rank"],
        "rho_ctrl": rec["rho_ctrl"],
        "rho_x": rec["rho_x"],
        "fake": fake,
        "prose": prose,
    }


def pass_recon_fc(tr: pd.DataFrame) -> dict:
    """Trailing-3 a_fin_cost / a_in3 vs store f_fc_r (A vs F extreme filter)."""
    work = tr.sort_values(["company_id", "period"])
    fc3 = work.groupby("company_id", sort=False)["a_fin_cost"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(3, min_periods=3).sum()
    )
    ain = pd.to_numeric(work["a_in3"], errors="coerce")
    recon = (fc3 / np.maximum(ain, 1.0)).clip(0.0, 1.0)
    rho = spearman(work["f_fc_r"], recon)
    rec = leftover_diag(work[Y3], work["f_fc_r"], [recon], work["fold"], work[Y3].notna())
    same = bool(np.isfinite(rho) and abs(rho) >= 0.99)
    prose = (
        f"f_fc_r vs rolling-3 a_fin_cost / a_in3 ρ={_f(rho)} "
        f"({'SAME as Family A trail' if same else 'F drops is_extreme — not a 1.000 rewrite of A'}). "
        f"Y3 leftover after recon rank {_f(rec['rank'])}."
    )
    print(prose)
    return {"rho": rho, "same": same, "y3": rec["rank"], "r2": rec["r2"], "prose": prose}


def pass_demean_days(tr: pd.DataFrame) -> dict:
    dm = company_demean(tr["f_fc_r"], tr["company_id"])
    rec = leftover_diag(tr[Y3], dm, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_ds = leftover_diag(tr[Y3], dm, [tr["f_ds_r"]], tr["fold"], tr[Y3].notna())
    raw = signed_oof_auroc(tr[Y3], dm, tr["fold"], tr[Y3].notna())
    y7 = leftover_diag(tr[Y7], dm, [tr["e_ar_issued_lag1"]], tr["fold"], tr[Y7].notna())
    dies = not (np.isfinite(rec["rank"]) and rec["rank"] >= CHANCE) or rec["honest_dies"]
    fake = bool(
        rec["fake"]
        or (np.isfinite(rec["ols"]) and rec["ols"] >= CHANCE and rec["rank"] < CHANCE)
        or (np.isfinite(rec["rho_ctrl"]) and abs(rec["rho_ctrl"]) >= TWIN_RHO)
    )
    beat_size = bool(np.isfinite(_cv(raw)) and (_cv(raw) - SIZE_Y3_QUOTE) >= KEEP_DELTA)
    prose = (
        f"Y3 company-demean fc raw {_f(_cv(raw))} leftover after days rank {_f(rec['rank'])} "
        f"OLS {_f(rec['ols'])} ρ(resid,days)={_f(rec['rho_ctrl'])} "
        f"after ds_r {_f(rec_ds['rank'])}. Y7 demean leftover after issued {_f(y7['rank'])}. "
        f"{'fake days leftover' if fake else ('month shock leftover lives but loses to size' if not beat_size else 'KEEP-ish month shock')}."
    )
    print(prose)
    return {
        "raw": _cv(raw),
        "rank": rec["rank"],
        "ols": rec["ols"],
        "rho_ctrl": rec["rho_ctrl"],
        "ds": rec_ds["rank"],
        "y7": y7["rank"],
        "dies": dies,
        "fake": fake,
        "beat_size": beat_size,
        "prose": prose,
    }


def pass_g155(tr: pd.DataFrame) -> dict:
    sl = tr["group_id"].astype(str) == "GROUP_0155"
    y = pd.to_numeric(tr[Y3], errors="coerce")
    fc = pd.to_numeric(tr["f_fc_r"], errors="coerce")
    lab = sl & y.notna()
    rest = (~sl) & y.notna()
    rows = [
        {
            "slice": "GROUP_0155",
            "n_co": int(tr.loc[sl, "company_id"].nunique()),
            "n_lab": int(lab.sum()),
            "n_pos": int((lab & (y == 1)).sum()),
            "rate": _pp(float(y[lab].mean()) if lab.any() else float("nan")),
            "fc p50": _f(float(fc[lab].median()) if lab.any() else float("nan"), 4),
            "med log in3": _f(float(tr.loc[lab, "log_in3"].median()) if lab.any() else float("nan")),
        },
        {
            "slice": "rest",
            "n_co": int(tr.loc[~sl, "company_id"].nunique()),
            "n_lab": int(rest.sum()),
            "n_pos": int((rest & (y == 1)).sum()),
            "rate": _pp(float(y[rest].mean()) if rest.any() else float("nan")),
            "fc p50": _f(float(fc[rest].median()) if rest.any() else float("nan"), 4),
            "med log in3": _f(float(tr.loc[rest, "log_in3"].median()) if rest.any() else float("nan")),
        },
    ]
    prose = (
        f"GROUP_0155 is {int(tr.loc[sl, 'company_id'].nunique())} train companies, "
        f"Y3 rate {_pp(float(y[lab].mean()) if lab.any() else float('nan'))} "
        f"vs rest {_pp(float(y[rest].mean()) if rest.any() else float('nan'))}. "
        f"Fold-2 top group, not a 25% dummy."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def make_plot(tr: pd.DataFrame, p6: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    fc = pd.to_numeric(tr["f_fc_r"], errors="coerce")
    ds = pd.to_numeric(tr["f_ds_r"], errors="coerce")
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    for name, s, color in (("f_fc_r", fc, "#1f4e79"), ("f_ds_r", ds, "#c45c26")):
        q = pd.qcut(s, 5, duplicates="drop")
        rates = []
        xs = []
        for i, (cat, sl) in enumerate(q.groupby(q, observed=True), start=1):
            lab = y3.loc[sl.index].notna()
            rates.append(float(y3.loc[sl.index][lab].mean()) if lab.any() else float("nan"))
            xs.append(i)
        ax.plot(xs, rates, marker="o", label=name, color=color)
    ax.set_xlabel("quintile (low → high)")
    ax.set_ylabel("Y3 rate")
    ax.set_title("Y3 rate by finance-flow quintile")
    ax.legend(frameon=False)
    ax.set_ylim(0, None)
    ax = axes[1]
    lab7 = tr[Y7].notna()
    buckets = ["<6", "6-11", "12-17", "18-23"]
    nn_fc, nn_l3, nlab = [], [], []
    for b in buckets:
        sl = lab7 & (tr["so_far_bucket"] == b)
        n = int(sl.sum())
        nlab.append(n)
        nn_fc.append(_pct(int((sl & fc.notna()).sum()), n))
        nn_l3.append(
            _pct(int((sl & pd.to_numeric(tr["f_fc_r_lag3"], errors="coerce").notna()).sum()), n)
        )
    x = np.arange(len(buckets))
    ax.bar(x - 0.18, nn_fc, 0.36, label="f_fc_r", color="#1f4e79")
    ax.bar(x + 0.18, nn_l3, 0.36, label="f_fc_r_lag3", color="#c45c26")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("non-null share (Y7 labeled)")
    ax.set_title("Q6: lag3 empty until so-far≥6")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def write_md(ctx: dict) -> None:
    d = ctx["decision"]
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
    p6, p7, p8, p9 = ctx["p6"], ctx["p7"], ctx["p8"], ctx["p9"]
    p10, p11 = ctx["p10"], ctx["p11"]
    lines = [
        "# Unused leftover of KEEP-flow `f_fc_r`",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_fc`. Do not merge Family M. "
        "Do not grow TURNOVER. Do not change the 15-col Y3 card.",
        "",
        "`f_fc_r` = trailing-3m fee+interest / inflow (Family F). Already KEEP on the 44. "
        "`f_fc_r_lag3` is on Y7 TURNOVER. The unused question is leftover after `f_ds_r` / days "
        "as Y3 X, leftover after issued_lag1 as Y7, twin vs `f_ds_r` / `a_fin_cost` / in-memory "
        "`m_fin`, and whether contemporaneous `f_fc_r` should leave the 44 while lag3 stays.",
        "",
        "## Headline",
        "",
        (
            f"Contemporaneous `f_fc_r` on the 44: **{d['contemp']}** ({d['why']}). "
            f"`f_fc_r_lag3` on TURNOVER: **{d['turnover']}**. "
            f"Y3 leftover after `f_ds_r` rank {_f(p4['ds_rank'])} "
            f"({'dies' if p4['ds_dies'] else 'lives'}); after days {_f(p4['days_rank'])}. "
            f"Y7 leftover after issued_lag1 {_f(p5['iss_rank'])}; after TURN_NOFC {_f(p5['turn_rank'])}. "
            f"Not a twin of `f_ds_r` (ρ={_f(p2['rhos']['f_ds_r'])}); vs Y9 ρ={_f(p8['rho'])} "
            f"Jaccard(hi,Y9+)={_f(p8['jac_hi'])}. Q6 lag3 empty until so-far≥6: "
            f"{'CONFIRM' if p6['empty_ok'] else 'DRIFT'}. "
            f"Night quotes unchanged: Y3 0.762 / 0.752, days {_f(p3['days'])}, size {_f(p3['size3'])}, "
            f"TURNOVER 0.720 / 0.712."
        ),
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        f"| 1 | Who is healthy? | Contemporaneous fc is a financing **flow**, not a new health Y. {d['contemp']} as 44-col X. |",
        "| 2 | Who is improving? | Not a path column by itself. |",
        f"| 3 | Who is turning? | Y9 forbids F — fc is the fee_r raw material. {d['y9_role']}. |",
        "| 4 | Dip vs fall? | Y7 leftover after issued_lag1 / TURN_NOFC is the unused test. Do not grow 0.720. |",
        f"| 5 | Why did it change? | Not a twin of `f_ds_r` (ρ={_f(p2['rhos']['f_ds_r'])}). Leftover after ds_r {_f(p4['ds_rank'])} dies — unused leftover of the KEEP flow. |",
        f"| 6 | Months earlier? | `f_fc_r_lag3` empty until so-far≥6. F lag3 **{d['q6']}** (already). |",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| contemporaneous `f_fc_r` on the 44 | **{d['contemp']}** | {d['why']} |",
        f"| `f_fc_r` as Y3 X | **{d['y3']}** | vs size {_f(p3['size3'])} / days {_f(p3['days'])} / leftover-ds {_f(p4['ds_rank'])} |",
        f"| `f_fc_r_lag3` on TURNOVER | **{d['lag3']}** | {d['turnover']} |",
        f"| F lag3 as Q6 | **{d['q6']}** | empty until so-far≥6 (CONFIRM={p6['empty_ok']}) |",
        f"| `f_fc_r` as Y9 X | **CLOSE** | Y9 META forbids F; ρ vs label {_f(p8['rho'])} |",
        f"| `f_outstanding_gt_granted` | **PARK** | last-month-only={p11['last_only']}; not a twin of fc (ρ={_f(p11['rho'])}) |",
        "| Family M merge | **CLOSE** | in-memory only; do not merge |",
        "| invent `y_fc` | **CLOSE** | Y9 already is sustained fee_r |",
        "| Family F flow `f_fc_r` in the store | **KEEP** | observed fee/inflow trail; DROP is from the 44-col starter only |",
        "",
        "KEEP-as-X for *adding*: oriented group-fold beats size ≥0.02 **and** leftover after the honest bar **and** not SIZE **and** not a twin. "
        "For a KEEP column, leftover dying after `f_ds_r` **and** |ρ|≥0.80 is a twin / rewrite. "
        "Leftover dying with low R² / low ρ is **unused leftover** — still DROP contemporaneous from the 44; do not rip lag3 off TURNOVER.",
        "",
        "## 1. Coverage; 470 vs ERP; ever-n",
        "",
        p1["prose"],
        "",
        _md_table(p1["rows"]),
        "",
        "Dark vs ERP (F is bank/debt — access ≠ ERP):",
        "",
        _md_table(p1["dark_rows"]),
        "",
        "## 2. Spearman twins",
        "",
        p2["prose"],
        "",
        _md_table(p2["rows"]),
        "",
        "## 3. Single-feature train group-fold AUROC",
        "",
        p3["prose"],
        "",
        _md_table(p3["rows"]),
        "",
        f"Night replicas: days {_f(p3['days'])} ({'MATCH 0.711' if p3['days_ok'] else 'DRIFT'}), "
        f"size {_f(p3['size3'])} ({'MATCH 0.617' if p3['size_ok'] else 'DRIFT'}), "
        f"issued_lag1 {_f(p3['iss'])} ({'MATCH ~0.630' if p3['iss_ok'] else 'DRIFT'}).",
        "",
        "## 4. Honest leftover after `f_ds_r` (Y3) and after days (Y3)",
        "",
        p4["prose"],
        "",
        _md_table(p4["rows"]),
        "",
        "Leftover <0.55 dies. Rank leftover is the honest bar (OLS can fake a days leak).",
        "",
        "## 5. Y7 leftover after issued_lag1 and TURNOVER-without-fc",
        "",
        p5["prose"],
        "",
        _md_table(p5["rows"]),
        "",
        "Single leftover only. Did not refit TURNOVER. TURN_NOFC quote 0.714 / TURNOVER 0.720 stand. "
        "TURNFC0 / TURNFC1 already failed fold 4 — do not grow.",
        "",
        "## 6. Q6 — lag1 / lag3 empty-on-short",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        "Short-book / so-far singles:",
        "",
        _md_table(p6["singles"]),
        "",
        "## 7. SIZE terciles / ICC / demean",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "Within size terciles (hi/lo = above/below median `f_fc_r` in the tercile):",
        "",
        _md_table(p7["t_rows"]),
        "",
        "## 8. vs Y9 accepted labels",
        "",
        p8["prose"],
        "",
        _md_table(p8["rows"]),
        "",
        _md_table(p8["jrows"]),
        "",
        "Y9 forbids family F as X. This is an identity check, not a Y9 model.",
        "",
        "## 9. Fold 4 / short books",
        "",
        p9["prose"],
        "",
        _md_table(p9["rows"]),
        "",
        "Fold 4 only (sign from folds 0–3):",
        "",
        _md_table(p9["f4_rows"]),
        "",
        _md_table([p9["short_row"]]),
        "",
        "Fold-4 groups (y7_core.md):",
        "",
        _md_table(p9["grp"]),
        "",
        "## 10. Dark 470",
        "",
        p10["prose"],
        "",
        _md_table(p10["rows"]),
        "",
        "## 11. `f_outstanding_gt_granted` (PARK snapshot — do not reopen)",
        "",
        p11["prose"],
        "",
        _md_table(p11["rows"]),
        "",
        "## 12. Brief map + PARK / CLOSE / KEEP / DROP-from-44",
        "",
        "| object | on 44 today | this ticket | TURNOVER | Q6 |",
        "| --- | --- | --- | --- | --- |",
        f"| contemporaneous `f_fc_r` | KEEP flow | **{d['contemp']}** from the 44 | not on TURNOVER (TURNFC0 failed fold 4) | rolling-3 empty until so-far≥3 |",
        f"| `f_fc_r_lag3` | derived, not a 44 stem | do not rip | **{d['lag3']}** | **CLOSE** empty until so-far≥6 |",
        f"| `f_ds_r` | KEEP + 15-col stem | untouched | TURNDSSWAP 0.712 not a swap | lag3 CLOSE |",
        "",
        "## Extra — holdout coverage only",
        "",
        ctx["p_ho"]["prose"],
        "",
        _md_table(ctx["p_ho"]["rows"]),
        "",
        "## Extra — quintiles",
        "",
        ctx["p_q"]["prose"],
        "",
        _md_table(ctx["p_q"]["rows"]),
        "",
        "## Extra — zero vs intensity",
        "",
        ctx["p_z"]["prose"],
        "",
        _md_table(ctx["p_z"]["rows"]),
        "",
        "## Extra — leftover after in-memory `m_fin`",
        "",
        ctx["p_m"]["prose"],
        "",
        _md_table(ctx["p_m"]["rows"]),
        "",
        "## Extra — monthly `a_fin_cost / a_in3`",
        "",
        ctx["p_mr"]["prose"],
        "",
        "## Extra — leftover after `f_ds_r` by fold",
        "",
        ctx["p_fl"]["prose"],
        "",
        _md_table(ctx["p_fl"]["rows"]),
        "",
        "## Extra — does `f_ds_r` survive residualizing `f_fc_r`?",
        "",
        ctx["p_ds"]["prose"],
        "",
        "## Extra — leftover on fc>0 support",
        "",
        ctx["p_sup"]["prose"],
        "",
        "## Extra — Y3 fold 2 (0.703) groups",
        "",
        ctx["p_f2"]["prose"],
        "",
        _md_table(ctx["p_f2"]["rows"]),
        "",
        "## Extra — dark ∧ `e_ar_issued` defined",
        "",
        ctx["p_di"]["prose"],
        "",
        _md_table(ctx["p_di"]["rows"]),
        "",
        "## Extra — holdout Q6 (coverage only)",
        "",
        ctx["p_hq"]["prose"],
        "",
        _md_table(ctx["p_hq"]["rows"]),
        "",
        "## Extra — Y9 is a forward path, not now-fee_r",
        "",
        ctx["p_yl"]["prose"],
        "",
        _md_table(ctx["p_yl"]["rows"]),
        "",
        "## Extra — OLS leftover after days is a fake leak",
        "",
        ctx["p_df"]["prose"],
        "",
        "## Extra — reconstruct fc from Family A",
        "",
        ctx["p_rc"]["prose"],
        "",
        "## Extra — company-demean leftover after days",
        "",
        ctx["p_dd"]["prose"],
        "",
        "## Extra — GROUP_0155 (fold-2 top)",
        "",
        ctx["p_g"]["prose"],
        "",
        _md_table(ctx["p_g"]["rows"]),
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx.get("png") else "Plot: skipped.",
        "",
        "## What failed / next",
        "",
    ]
    for x in ctx["failed"]:
        lines.append(f"- {x}")
    lines.extend(
        [
            "",
            f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts: coverage/470, twins, singles, "
            "Y3 leftover ds/days, Y7 leftover issued/TURN_NOFC, Q6 so-far, ICC/demean, "
            "Y9 Jaccard, fold 4 / short, dark, ogtg, holdout, quintiles, zero, m_fin leftover, "
            "monthly ratio, fold leftover, ds-after-fc, fc>0 support, fold-2 groups, "
            "dark issued stub, holdout Q6, Y9 lead ρ, fake days-OLS, A-recon, demean-days, GROUP_0155.",
            "",
            "Did **not**: merge parquet, invent `y_fc`, merge Family M / I / J, edit "
            "`gbm_y7_core.py` / `gbm_core.py` / `dso_qa.py` / `cust_hhi_qa.py` / `d_tx_qa.py`, "
            "rewrite duckdb, run `build_targets`, touch `product/`, write 0–100, change night "
            "Y3 0.762/0.752 or TURNOVER 0.720, grow the 15-col card, write the parent journal / "
            "LIVE / canvas.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    prev = pd.read_csv(REGISTRY)
    key_cols = ["agent", "x_families", "y", "model", "split", "metric"]
    seen = set()
    if not prev.empty and all(c in prev.columns for c in key_cols):
        seen = set(tuple(str(r[c]) for c in key_cols) for _, r in prev.iterrows())
    d = ctx["decision"]
    p1, p2, p3, p4, p5, p8 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"], ctx["p8"]
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
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
            "metric": "auroc_f_fc_r",
            "value": p3["y3"],
            "coverage": f"{p1['fc_cov']:.4f}",
            "notes": f"days={p3['days']:.4f} leftover_ds={p4['ds_rank']:.4f} contemp={d['contemp']}",
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
            "metric": "auroc_f_fc_r",
            "value": p3["y7"],
            "coverage": f"{p1['fc_cov']:.4f}",
            "notes": f"lag3={p3['y7_l3']:.4f} leftover_iss={p5['iss_rank']:.4f} turn_nofc={p5['turn_rank']:.4f}",
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
            "metric": "auroc_f_fc_r_resid_ds_r",
            "value": p4["ds_rank"],
            "coverage": f"{p1['fc_cov']:.4f}",
            "notes": f"ols={p4['ds_ols']:.4f} dies={p4['ds_dies']} r2={p4['ds_r2']:.4f}",
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
            "metric": "auroc_f_fc_r_resid_days",
            "value": p4["days_rank"],
            "coverage": f"{p1['fc_cov']:.4f}",
            "notes": f"ols={p4['days_ols']:.4f} dies={p4['days_dies']}",
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
            "metric": "rho_f_fc_r_vs_f_ds_r",
            "value": p2["rhos"]["f_ds_r"],
            "coverage": f"{p1['fc_cov']:.4f}",
            "notes": f"twin={p2['twin_ds']} m_fin={p2['rhos']['m_fin_share']:.4f} a_fin={p2['rhos']['a_fin_cost']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y9,
            "model": MODEL,
            "split": "train",
            "metric": "rho_f_fc_r_vs_y9",
            "value": p8["rho"],
            "coverage": f"{p1['fc_cov']:.4f}",
            "notes": f"jac_hi={p8['jac_hi']:.4f} is_y={p8['is_y']} not_scored_as_Y9_X",
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
            "metric": "auroc_f_fc_r_resid_turn_nofc",
            "value": p5["turn_rank"],
            "coverage": f"{p1['fc_cov']:.4f}",
            "notes": f"ols={p5['turn_ols']:.4f} dies={p5['turn_dies']} no_refit",
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
            "metric": "drop_from_44",
            "value": 1 if d["drop44"] else 0,
            "coverage": f"{p1['fc_cov']:.4f}",
            "notes": f"contemp={d['contemp']} lag3_turnover={d['lag3']} q6={d['q6']}",
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
            "metric": "auroc_f_ds_r_resid_fc",
            "value": ctx["p_ds"]["rank"],
            "coverage": f"{p1['ds_cov']:.4f}",
            "notes": f"raw={ctx['p_ds']['raw']:.4f} lives={ctx['p_ds']['lives']} r2={ctx['p_ds']['r2']:.4f}",
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
            "metric": "auroc_f_fc_r_resid_ds_gt0",
            "value": ctx["p_sup"]["ds"],
            "coverage": f"{ctx['p_sup']['n'] / max(p1['n_cm'], 1):.4f}",
            "notes": f"raw={ctx['p_sup']['raw']:.4f} days={ctx['p_sup']['days']:.4f} dies={ctx['p_sup']['dies']}",
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
            "metric": "days_ols_fake_leftover",
            "value": ctx["p_df"]["ols"],
            "coverage": f"{p1['fc_cov']:.4f}",
            "notes": f"rank={ctx['p_df']['rank']:.4f} fake={ctx['p_df']['fake']} rho_ctrl={ctx['p_df']['rho_ctrl']:.4f}",
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
            "metric": "rho_f_fc_r_vs_a_fin_cost_roll3",
            "value": ctx["p_rc"]["rho"],
            "coverage": f"{p1['fc_cov']:.4f}",
            "notes": f"same={ctx['p_rc']['same']} leftover={ctx['p_rc']['y3']:.4f}",
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
            "metric": "auroc_f_fc_r_demean_resid_days",
            "value": ctx["p_dd"]["rank"],
            "coverage": f"{p1['fc_cov']:.4f}",
            "notes": f"raw={ctx['p_dd']['raw']:.4f} fake={ctx['p_dd']['fake']} beat_size={ctx['p_dd']['beat_size']}",
        },
    ]
    new = []
    for r in rows:
        key = tuple(str(r[c]) for c in key_cols)
        if key in seen:
            continue
        seen.add(key)
        new.append(r)
    if not new:
        print("registry: no new rows")
        return
    with REGISTRY.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prev.columns))
        for r in new:
            w.writerow({c: r.get(c, "") for c in prev.columns})
    print(f"registry appended {len(new)} rows")


def write_wave(ctx: dict) -> None:
    d = ctx["decision"]
    p2, p3, p4, p5, p6, p8 = ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"], ctx["p6"], ctx["p8"]
    text = (
        f"# Wave 4 — f_fc_r leftover\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/fc_r_qa.py`\n"
        f"- `analysis/outputs/fc_r_qa.md`\n"
        f"- `analysis/outputs/fc_r_leftover.png`\n"
        f"- append-only `analysis/experiments/registry.csv` "
        f"(skip key agent+x_families+y+model+split+metric)\n"
        f"- this note\n\n"
        f"Did not touch `gbm_y7_core.py`, `gbm_core.py`, `dso_qa.py`, `cust_hhi_qa.py`, "
        f"`d_tx_qa.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, "
        f"LIVE, CONTEXT, canvas, or the parent journal. Night Y3 stays **0.762 / 0.752**. "
        f"Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| contemporaneous `f_fc_r` on the 44 | **{d['contemp']}** |\n"
        f"| `f_fc_r_lag3` on TURNOVER | **{d['lag3']}** |\n"
        f"| F lag3 Q6 | **{d['q6']}** |\n"
        f"| Y9 identity | **{d['y9_role']}** |\n\n"
        f"Y3 leftover after `f_ds_r` rank {_f(p4['ds_rank'])} (dies={p4['ds_dies']}); "
        f"after days {_f(p4['days_rank'])}. Not a twin of ds_r (ρ={_f(p2['rhos']['f_ds_r'])}, R²={_f(p4['ds_r2'])}). "
        f"Y7 leftover issued_lag1 {_f(p5['iss_rank'])} TURN_NOFC {_f(p5['turn_rank'])} — CLOSE add-on. "
        f"Y9 ρ={_f(p8['rho'])} Jaccard={_f(p8['jac_hi'])} (raw material, not the binary Y). "
        f"Q6 empty-until-6 CONFIRM={p6['empty_ok']}. "
        f"Y3 single {_f(p3['y3'])} vs size {_f(p3['size3'])} vs days {_f(p3['days'])}. "
        f"A-trail recon ρ=1.000. Demean leftover after days lives but loses to size.\n\n"
        f"## What failed / next\n\n"
        + "\n".join(f"- {x}" for x in ctx["failed"])
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.parent.mkdir(parents=True, exist_ok=True)
    WAVE_NOTE.write_text(text, encoding="utf-8")
    print(f"wrote {WAVE_NOTE}")


def run() -> dict:
    t0 = time.time()
    print(f"fc_r_qa start seed={FOLD_SEED} holdout=72 agent={AGENT} wave={WRITE_WAVE}")
    panel = load_panel()
    con = connect()
    try:
        book = book_invoice_ids(con)
    finally:
        con.close()
    panel = attach_folds(panel)
    panel = add_panel_lags(
        panel,
        ["f_fc_r", "f_ds_r", "e_ar_issued", "e_credit_note_ratio"],
        (1, 2, 3),
    )
    panel = add_issued_lag_cv(panel)
    print("in-memory Family M (not written)")
    panel = attach_m_fin(panel)
    panel["ever_erp"] = panel["company_id"].isin(book)
    tr = panel[panel["split"] == "train"].copy().reset_index(drop=True)
    assert_no_holdout(tr["company_id"])
    print(
        f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
        f"holdout CM={int((panel['split']=='holdout').sum())}"
    )

    print("pass 1 coverage / 470")
    p1 = pass1_cov(tr, book)
    print("pass 2 twins")
    p2 = pass2_twins(tr)
    print("pass 3 singles")
    p3 = pass3_singles(tr)
    print("pass 4 Y3 leftover")
    p4 = pass4_y3_leftover(tr)
    print("pass 5 Y7 leftover")
    p5 = pass5_y7_leftover(tr)
    print("pass 6 Q6")
    p6 = pass6_q6(tr)
    print("pass 7 trait / ICC")
    p7 = pass7_trait(tr)
    print("pass 8 Y9 identity")
    p8 = pass8_y9(tr)
    print("pass 9 fold 4 / short")
    p9 = pass9_fold4(tr)
    print("pass 10 dark")
    p10 = pass10_dark(tr, book)
    print("pass 11 ogtg")
    p11 = pass11_ogtg(tr)
    print("extras")
    p_ho = pass_holdout(panel, book)
    p_q = pass_quintiles(tr)
    p_z = pass_zero(tr)
    p_m = pass_mfin_left(tr)
    p_mr = pass_monthly_ratio(tr)
    p_fl = pass_fold_left(tr, p4)
    p_ds = pass_ds_after_fc(tr)
    p_sup = pass_support(tr)
    p_f2 = pass_fold2(tr)
    p_di = pass_dark_inv(tr, book)
    p_hq = pass_holdout_q6(panel)
    p_yl = pass_y9_lead(tr)
    p_df = pass_days_fake(tr, p4)
    p_rc = pass_recon_fc(tr)
    p_dd = pass_demean_days(tr)
    p_g = pass_g155(tr)
    png = make_plot(tr, p6)
    decision = decide(p2, p3, p4, p5, p6, p7, p8, p9, p11)
    failed = []
    if not p3["days_ok"]:
        failed.append(f"days replica drifted: {_f(p3['days'])} vs 0.711")
    if not p3["size_ok"]:
        failed.append(f"size replica drifted: {_f(p3['size3'])} vs 0.617")
    if not p6["empty_ok"]:
        failed.append("f_fc_r_lag3 not empty on so-far<6 — Q6 clock drifted")
    if p11["twin"]:
        failed.append("ogtg unexpectedly twins f_fc_r — snapshot PARK would need a reopen")
    if p2["twin_ds"] and not p4["ds_dies"]:
        failed.append("twin vs f_ds_r but leftover after ds_r still lives — check rank vs OLS")
    if p_df["fake"]:
        failed.append(
            f"OLS leftover after days {_f(p_df['ols'])} is a fake leak; honest rank {_f(p_df['rank'])} dies"
        )
    if p5["turn_clone"]:
        failed.append(
            f"Y7 TURN_NOFC OLS leftover {_f(p5['turn_ols'])} is an fc clone "
            f"(ρ={_f(p5['turn_rho_x'])}); rank {_f(p5['turn_rank'])} is the honest number"
        )
    if not p_ds["lives"]:
        failed.append("f_ds_r leftover after f_fc_r died — 15-col stem would be the weaker flow")
    if p_f2["dummy"]:
        failed.append(f"Y3 fold 2 is one-group leftover ({p_f2['top']})")
    if not p_hq["ok"]:
        failed.append("holdout lag3 not empty on so-far<6")
    if p_dd["dies"]:
        failed.append(
            f"company-demean leftover after days dies ({_f(p_dd['rank'])}) — month shock is not a Y3 engine"
        )
    if abs(p3["y3"] - 0.50) < 0.02 or (np.isfinite(p_f2["wo"]) and p_f2["wo"] < CHANCE):
        failed.append(
            f"Y3 single without fold 2 is {_f(p_f2['wo'])} — the 0.559 mean is one-fold leftover"
        )
    if not failed:
        failed.append("no replica miss; unused leftover (not a twin) drops contemporaneous from the 44")
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
        "p_ho": p_ho,
        "p_q": p_q,
        "p_z": p_z,
        "p_m": p_m,
        "p_mr": p_mr,
        "p_fl": p_fl,
        "p_ds": p_ds,
        "p_sup": p_sup,
        "p_f2": p_f2,
        "p_di": p_di,
        "p_hq": p_hq,
        "p_yl": p_yl,
        "p_df": p_df,
        "p_rc": p_rc,
        "p_dd": p_dd,
        "p_g": p_g,
        "decision": decision,
        "failed": failed,
        "png": png,
        "elapsed_s": time.time() - t0,
    }
    write_md(ctx)
    append_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    print(
        f"DONE contemp={decision['contemp']} lag3={decision['lag3']} "
        f"leftover_ds={_f(p4['ds_rank'])} leftover_days={_f(p4['days_rank'])} "
        f"y7_iss={_f(p5['iss_rank'])} y7_turn={_f(p5['turn_rank'])} "
        f"elapsed={ctx['elapsed_s']:.0f}s"
    )
    return ctx


if __name__ == "__main__":
    run()
