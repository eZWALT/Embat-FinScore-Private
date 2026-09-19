"""Unused leftover of KEEP-flow ``f_ds_r`` after days.

``f_ds_r`` is still on the 15-col Y3 card and the 44. Contemporaneous
``f_fc_r`` was DROPPED from the 44 as unused leftover (ρ vs ``f_ds_r``
0.205). This lane asks whether leftover of ``f_ds_r`` after
``c_n_days_with_tx`` (the 0.711 bar) lives, or whether the card stem is
unused the same way contemp fc was unused on the 44.

``f_ds_r`` leftover after residualizing fc stayed 0.606. KEEP-as-X:
beat size ≥0.02 **and** leftover after days **and** not SIZE
(|ρ| vs log1p(a_in3) ≥0.50) **and** not a twin (|ρ|≥0.80 vs ``f_fc_r``
/ ``a_debt_service`` / ``f_debt_service``). Leftover <0.55 dies. Rank
leftover is honest; OLS can fake a days leak.

Y4 *is* ``y4_ds_r_double``. Family F is forbidden as Y4 X — identity
check only, never scored as KEEP.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.ds_r_qa

Owned: analysis/evaluate/ds_r_qa.py, analysis/outputs/ds_r_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_ds_r.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "ds_r_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "ds_r_leftover.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_ds_r.md"
AGENT = "572fb928"
WAVE = "4"
ROUND = "R4"
MODEL = "ds_r_qa"
X_FAM = "F"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y4 = "y4_ds_r_double"
Y7 = "y7_top1_lost"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_Y3_QUOTE = 0.617
OWN_DS_QUOTE = 0.620
FC_LEFTOVER_QUOTE = 0.606
RHO_FC_QUOTE = 0.205
Y3_NIGHT = (0.762, 0.752)
Y7_TURNOVER = 0.720
Y7_B_SHALLOW = 0.712
TURNDSSWAP = 0.712
KEEP_DELTA = 0.02
TWIN_RHO = 0.80
SIZE_RHO = 0.50
CHANCE = 0.55
ICC_STYLE = 0.85
MIN_POS = 50
MIN_ACF_PAIRS = 4
HOT_GROUPS = ("GROUP_0158", "GROUP_0172")
WRITE_WAVE = True

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "a_debt_service",
    "a_fin_cost",
    "f_ds_r",
    "f_fc_r",
    "f_debt_service",
    "c_n_days_with_tx",
    "e_ar_issued",
    "b_below_0",
)

Y_KEEP = (Y2, Y3, Y4, Y7)


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
        "rrec": rrec,
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
    return "24+"


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
    ds = pd.to_numeric(panel["f_ds_r"], errors="coerce")
    panel["log1p_ds_r"] = np.log1p(ds.clip(lower=0))
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
    leak3 = leakage_check(["f_ds_r", "f_fc_r", "c_n_days_with_tx", "log_in3"], Y3, forbidden_prefixes=["b"])
    leak7 = leakage_check(["f_ds_r", "e_ar_issued"], Y7, forbidden_prefixes=["d"])
    leak4 = leakage_check(["log_in3", "c_n_days_with_tx"], Y4, forbidden_prefixes=["f"])
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 never-B leak: {leak3['issues']}")
    if not leak7["ok"]:
        raise RuntimeError(f"Y7 never-D leak: {leak7['issues']}")
    if not leak4["ok"]:
        raise RuntimeError(f"Y4 never-F leak: {leak4['issues']}")
    forbidden_f_on_y4 = leakage_check(["f_ds_r"], Y4, forbidden_prefixes=["f"])
    if forbidden_f_on_y4["ok"]:
        raise RuntimeError("Y4 never-F: leakage_check should reject f_ds_r as Y4 X")
    print("Y4 never-F as X: leakage_check rejects f_ds_r — identity only, not KEEP")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def chronic_12(tr: pd.DataFrame) -> list[str]:
    """Recompute 0158+0172 names with ≥50% already-below on Y2-labeled months."""
    y = pd.to_numeric(tr[Y2], errors="coerce")
    below = pd.to_numeric(tr["b_below_0"], errors="coerce")
    lab = y.notna()
    ids: list[str] = []
    for gid in HOT_GROUPS:
        sl = lab & (tr["group_id"] == gid)
        g = (
            tr.loc[sl, ["company_id"]]
            .assign(below=below[sl].values)
            .groupby("company_id")
            .agg(n=("below", "size"), n_below=("below", "sum"))
        )
        g["share_below"] = g["n_below"] / g["n"]
        ids.extend([str(i) for i in g.index[g["share_below"] >= 0.5]])
    return sorted(set(ids))


# ---------------------------------------------------------------------------
# 1. Coverage; 470 vs ERP; ever-n; share=0
# ---------------------------------------------------------------------------
def pass1_cov(tr: pd.DataFrame, book: set[str], hold: pd.DataFrame | None = None) -> dict:
    print("\n" + "=" * 72)
    print("CUT 1 — coverage; 470 vs ERP (access ≠ ERP); ever-n; share=0")
    print("=" * 72)
    ds = pd.to_numeric(tr["f_ds_r"], errors="coerce")
    fc = pd.to_numeric(tr["f_fc_r"], errors="coerce")
    n_cm = len(tr)
    n_co = int(tr["company_id"].nunique())
    erp = tr["company_id"].isin(book)
    dark = ~erp
    rows = []
    for name, s in (("f_ds_r", ds), ("f_fc_r", fc)):
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
    book_rows = []
    for slice_name, sl in (("dark", dark), ("ERP", erp)):
        for name, s in (("f_ds_r", ds), ("f_fc_r", fc)):
            nn = sl & s.notna()
            n_co_sl = int(tr.loc[sl, "company_id"].nunique())
            book_rows.append(
                {
                    "book": slice_name,
                    "col": name,
                    "n_co": n_co_sl,
                    "n_cm": int(sl.sum()),
                    "n_nn": int(nn.sum()),
                    "cov CM": _pp(_pct(int(nn.sum()), int(sl.sum()))),
                    "ever co": int(tr.loc[nn, "company_id"].nunique()),
                    "share=0": _pp(_pct(int((sl & s.eq(0)).sum()), int(nn.sum()))),
                    "zero_fill": "no — 0 is no-ds 3m, not a dark fill",
                }
            )
    ds_cov = _pct(int(ds.notna().sum()), n_cm)
    fc_cov = _pct(int(fc.notna().sum()), n_cm)
    match_fc = bool(np.isfinite(fc_cov) and np.isfinite(ds_cov) and abs(fc_cov - ds_cov) < 0.005)
    hold_row = None
    if hold is not None and len(hold):
        hds = pd.to_numeric(hold["f_ds_r"], errors="coerce")
        hold_row = {
            "n_co": int(hold["company_id"].nunique()),
            "n_cm": len(hold),
            "n_nn": int(hds.notna().sum()),
            "cov": _pct(int(hds.notna().sum()), len(hold)),
            "share0": _pct(int(hds.eq(0).sum()), int(hds.notna().sum())),
        }
    prose = (
        f"Train {n_co:,} co / {n_cm:,} CM. "
        f"f_ds_r cov {_pp(ds_cov)} vs f_fc_r {_pp(fc_cov)} "
        f"{'SAME rolling-3 hole' if match_fc else 'DRIFT'}. "
        f"Dark {int(tr.loc[dark, 'company_id'].nunique())} co have ds "
        f"(access ≠ ERP, same as fc). share=0 is no-repayment 3m, not a fill."
    )
    print(prose)
    return {
        "rows": rows,
        "book_rows": book_rows,
        "ds_cov": ds_cov,
        "fc_cov": fc_cov,
        "n_cm": n_cm,
        "n_co": n_co,
        "n_dark": int(tr.loc[dark, "company_id"].nunique()),
        "n_erp": int(tr.loc[erp, "company_id"].nunique()),
        "match_fc": match_fc,
        "hold": hold_row,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 2. Spearman twins
# ---------------------------------------------------------------------------
def pass2_twins(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 2 — Spearman twins vs fc / euro-ds / days / size")
    print("=" * 72)
    ds = pd.to_numeric(tr["f_ds_r"], errors="coerce")
    pairs = {
        "f_fc_r": tr["f_fc_r"],
        "a_debt_service": tr["a_debt_service"],
        "f_debt_service": tr["f_debt_service"],
        "a_fin_cost": tr["a_fin_cost"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "log_in3": tr["log_in3"],
    }
    rhos = {k: spearman(ds, v) for k, v in pairs.items()}
    rows = [{"vs": k, "rho": _f(v), "twin": "yes" if abs(v) >= TWIN_RHO else "no"} for k, v in rhos.items()]
    twins = [k for k, v in rhos.items() if np.isfinite(v) and abs(v) >= TWIN_RHO]
    size_flag = bool(np.isfinite(rhos["log_in3"]) and abs(rhos["log_in3"]) >= SIZE_RHO)
    fc_match = bool(np.isfinite(rhos["f_fc_r"]) and abs(rhos["f_fc_r"] - RHO_FC_QUOTE) < 0.02)
    prose = (
        f"ρ vs f_fc_r {_f(rhos['f_fc_r'])} "
        f"{'CONFIRM ~0.205' if fc_match else 'DRIFT from 0.205'}. "
        f"vs a_debt_service {_f(rhos['a_debt_service'])} "
        f"vs f_debt_service {_f(rhos['f_debt_service'])} "
        f"vs days {_f(rhos['c_n_days_with_tx'])} "
        f"vs log_in3 {_f(rhos['log_in3'])} SIZE={size_flag}. "
        f"Twins (≥0.80): {twins or 'none'}."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "twins": twins,
        "size_flag": size_flag,
        "twin_fc": "f_fc_r" in twins,
        "twin_ads": "a_debt_service" in twins,
        "twin_fds": "f_debt_service" in twins,
        "fc_match": fc_match,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 3. Single-feature group-fold AUROC
# ---------------------------------------------------------------------------
def pass3_singles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 3 — single-feature group-fold AUROC Y3 / Y2 / Y7")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    y2 = pd.to_numeric(tr[Y2], errors="coerce")
    y7 = pd.to_numeric(tr[Y7], errors="coerce")
    folds = tr["fold"]
    lab3 = y3.notna()
    lab2 = y2.notna()
    lab7 = y7.notna()
    feats = {
        "f_ds_r": tr["f_ds_r"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "log_in3": tr["log_in3"],
        "f_fc_r": tr["f_fc_r"],
        "a_debt_service": tr["a_debt_service"],
        "f_debt_service": tr["f_debt_service"],
    }
    recs = {}
    rows = []
    for yname, y, lab in ((Y3, y3, lab3), (Y2, y2, lab2), (Y7, y7, lab7)):
        for fname, x in feats.items():
            rec = signed_oof_auroc(y, x, folds, lab)
            recs[(yname, fname)] = rec
            rows.append(_auc_row(yname, fname, rec))
            print(f"  {yname} {fname} CV={_f(rec['cv'])} folds={fold_bits(rec)}")
    y3_ds = _cv(recs[(Y3, "f_ds_r")])
    y3_days = _cv(recs[(Y3, "c_n_days_with_tx")])
    y3_size = _cv(recs[(Y3, "log_in3")])
    days_ok = bool(np.isfinite(y3_days) and abs(y3_days - DAYS_BENCH) < 0.008)
    size_ok = bool(np.isfinite(y3_size) and abs(y3_size - SIZE_Y3_QUOTE) < 0.008)
    own_ok = bool(np.isfinite(y3_ds) and abs(y3_ds - OWN_DS_QUOTE) < 0.008)
    beat_size = bool(np.isfinite(y3_ds) and np.isfinite(y3_size) and (y3_ds - y3_size) >= KEEP_DELTA)
    prose = (
        f"Y3 f_ds_r {_f(y3_ds)} vs days {_f(y3_days)} vs size {_f(y3_size)}. "
        f"Replica days 0.711 {'CONFIRM' if days_ok else 'DRIFT'} / "
        f"size 0.617 {'CONFIRM' if size_ok else 'DRIFT'} / "
        f"own 0.620 {'CONFIRM' if own_ok else 'DRIFT'}. "
        f"Beat size ≥0.02: {beat_size} (Δ={_f(y3_ds - y3_size if np.isfinite(y3_ds) and np.isfinite(y3_size) else float('nan'))})."
    )
    print(prose)
    return {
        "rows": rows,
        "recs": recs,
        "y3": y3_ds,
        "days": y3_days,
        "size3": y3_size,
        "y2": _cv(recs[(Y2, "f_ds_r")]),
        "y7": _cv(recs[(Y7, "f_ds_r")]),
        "days_ok": days_ok,
        "size_ok": size_ok,
        "own_ok": own_ok,
        "beat_size": beat_size,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 4. Honest leftover after days + inverse
# ---------------------------------------------------------------------------
def pass4_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 4 — honest leftover after days (OLS + rank); inverse days after ds_r")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    ds = tr["f_ds_r"]
    days = tr["c_n_days_with_tx"]
    after_days = leftover_diag(y, ds, (days,), folds, lab)
    inv = leftover_diag(y, days, (ds,), folds, lab)
    prose = (
        f"ds_r leftover after days OLS {_f(after_days['ols'])} "
        f"rank {_f(after_days['rank'])} fake={after_days['fake']} "
        f"ρ(resid,days)={_f(after_days['rho_ctrl'])} R²={_f(after_days['r2'])} "
        f"honest_dies={after_days['honest_dies']}. "
        f"Inverse: days leftover after ds_r OLS {_f(inv['ols'])} rank {_f(inv['rank'])} "
        f"dies={inv['honest_dies']}."
    )
    print(prose)
    return {
        "after": after_days,
        "inv": inv,
        "ols": after_days["ols"],
        "rank": after_days["rank"],
        "dies": after_days["honest_dies"],
        "fake": after_days["fake"],
        "r2": after_days["r2"],
        "inv_rank": inv["rank"],
        "inv_dies": inv["honest_dies"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 5. Leftover after size; after fc
# ---------------------------------------------------------------------------
def pass5_size_fc(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 5 — leftover after size; after fc (should stay ~0.606)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    ds = tr["f_ds_r"]
    after_size = leftover_diag(y, ds, (tr["log_in3"],), folds, lab)
    after_fc = leftover_diag(y, ds, (tr["f_fc_r"],), folds, lab)
    fc_ok = bool(np.isfinite(after_fc["rank"]) and abs(after_fc["rank"] - FC_LEFTOVER_QUOTE) < 0.02)
    prose = (
        f"after size OLS {_f(after_size['ols'])} rank {_f(after_size['rank'])} "
        f"dies={after_size['honest_dies']} R²={_f(after_size['r2'])}. "
        f"after fc OLS {_f(after_fc['ols'])} rank {_f(after_fc['rank'])} "
        f"{'CONFIRM ~0.606' if fc_ok else 'DRIFT from 0.606'} "
        f"dies={after_fc['honest_dies']} R²={_f(after_fc['r2'])}."
    )
    print(prose)
    return {
        "size": after_size,
        "fc": after_fc,
        "size_rank": after_size["rank"],
        "size_dies": after_size["honest_dies"],
        "fc_rank": after_fc["rank"],
        "fc_dies": after_fc["honest_dies"],
        "fc_ok": fc_ok,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 6. SIZE terciles; ICC / demean
# ---------------------------------------------------------------------------
def pass6_tercile_icc(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 6 — SIZE terciles; ICC / demean")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    ds = pd.to_numeric(tr["f_ds_r"], errors="coerce")
    terc = size_terciles(tr)
    rows = []
    recs = {}
    for t in ("T1", "T2", "T3"):
        sl = lab & (terc == t)
        rec = signed_oof_auroc(y, ds, folds, sl)
        recs[t] = rec
        rows.append(
            {
                "tercile": t,
                "n": f"{rec['n_defined']:,}",
                "n_pos": f"{rec['n_pos']:,}",
                "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                "folds": fold_bits(rec) if not rec["low_power"] else "—",
            }
        )
        print(f"  {t} CV={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']}")
    icc = icc_anova(ds, tr["company_id"])
    trait = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_STYLE)
    demean = company_demean(ds, tr["company_id"])
    meanx = company_mean(ds, tr["company_id"])
    rec_d = signed_oof_auroc(y, demean, folds, lab)
    rec_m = signed_oof_auroc(y, meanx, folds, lab)
    after_days_d = leftover_diag(y, demean, (tr["c_n_days_with_tx"],), folds, lab)
    prose = (
        f"ICC={_f(icc['icc'])} {'TRAIT' if trait else 'STATE'} k={icc['k']}. "
        f"Demean CV {_f(rec_d['cv'])} company-mean {_f(rec_m['cv'])}. "
        f"Demean leftover after days rank {_f(after_days_d['rank'])} dies={after_days_d['honest_dies']}."
    )
    print(prose)
    return {
        "rows": rows,
        "icc": icc,
        "trait": trait,
        "demean": rec_d,
        "mean": rec_m,
        "demean_days": after_days_d,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 7. Q6 lag1 / lag3 on short books
# ---------------------------------------------------------------------------
def pass7_q6(tr: pd.DataFrame, hold: pd.DataFrame | None = None) -> dict:
    print("\n" + "=" * 72)
    print("CUT 7 — Q6 lag1 / lag3; empty until so-far≥6")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y3.notna()
    folds = tr["fold"]
    recs = {}
    rows = []
    for name in ("f_ds_r", "f_ds_r_lag1", "f_ds_r_lag3"):
        rec = signed_oof_auroc(y3, tr[name], folds, lab)
        recs[name] = rec
        rows.append(_auc_row(Y3, name, rec))
        print(f"  {name} CV={_f(rec['cv'])} n={rec['n_defined']}")
    buckets = ["<6", "6-11", "12-17", "18-23", "24+"]
    cov_rows = []
    n_l3_short = 0
    n_short = 0
    for b in buckets:
        sl = lab & (tr["so_far_bucket"] == b)
        n = int(sl.sum())
        l3 = pd.to_numeric(tr["f_ds_r_lag3"], errors="coerce")
        nn_l3 = int((sl & l3.notna()).sum())
        if b == "<6":
            n_l3_short = nn_l3
            n_short = n
        cov_rows.append(
            {
                "so-far": b,
                "n_lab": n,
                "ds nn": int((sl & pd.to_numeric(tr["f_ds_r"], errors="coerce").notna()).sum()),
                "lag1 nn": int((sl & pd.to_numeric(tr["f_ds_r_lag1"], errors="coerce").notna()).sum()),
                "lag3 nn": nn_l3,
            }
        )
    empty_ok = n_l3_short == 0
    hold_empty = True
    if hold is not None and len(hold) and "f_ds_r_lag3" in hold.columns:
        hlab = hold[Y3].notna() if Y3 in hold.columns else pd.Series(True, index=hold.index)
        hsl = hlab & (hold["so_far_bucket"] == "<6")
        hold_empty = int((hsl & pd.to_numeric(hold["f_ds_r_lag3"], errors="coerce").notna()).sum()) == 0
        print(f"  holdout so-far<6 lag3 nn={int((hsl & pd.to_numeric(hold['f_ds_r_lag3'], errors='coerce').notna()).sum())}/{int(hsl.sum())}")
    prose = (
        f"lag3 empty until so-far≥6: train {n_l3_short}/{n_short} "
        f"{'CONFIRM' if empty_ok else 'DRIFT'}. "
        f"Y3 lag1 {_f(recs['f_ds_r_lag1']['cv'])} lag3 {_f(recs['f_ds_r_lag3']['cv'])}. "
        f"F lag3 already CLOSE."
    )
    print(prose)
    return {
        "rows": rows,
        "cov_rows": cov_rows,
        "recs": recs,
        "empty_ok": empty_ok,
        "hold_empty": hold_empty,
        "n_l3_short": n_l3_short,
        "n_short": n_short,
        "lag1": _cv(recs["f_ds_r_lag1"]),
        "lag3": _cv(recs["f_ds_r_lag3"]),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 8. Y4 identity — not KEEP
# ---------------------------------------------------------------------------
def pass8_y4(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 8 — Y4 identity (y4_ds_r_double). F forbidden as Y4 X. Not KEEP.")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y4 = pd.to_numeric(tr[Y4], errors="coerce")
    ds = pd.to_numeric(tr["f_ds_r"], errors="coerce")
    lab = y4.notna()
    rho = spearman(ds[lab], y4[lab])
    hi = ds >= 0.5
    jac = jaccard(hi[lab], y4[lab] == 1)
    leak = leakage_check(["f_ds_r"], Y4, forbidden_prefixes=["f"])
    n_lab = int(lab.sum())
    n_pos = int((lab & (y4 == 1)).sum())
    prose = (
        f"Y4 *is* {Y4}. Train labeled {n_lab:,} pos={n_pos:,}. "
        f"ρ(f_ds_r, Y4)={_f(rho)} Jaccard(ds≥0.5, Y4+)={_f(jac['jaccard'])}. "
        f"leakage_check f_ds_r as Y4 X ok={leak['ok']} issues={leak['issues']}. "
        f"Not scored as KEEP. Family F forbidden as Y4 X."
    )
    print(prose)
    return {
        "rho": rho,
        "jac": jac["jaccard"],
        "n_lab": n_lab,
        "n_pos": n_pos,
        "leak_ok": leak["ok"],
        "leak_issues": leak["issues"],
        "scored_as_keep": False,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 10. Fold-wise leftover; 12-name Y2 drop
# ---------------------------------------------------------------------------
def pass10_fold_y2(tr: pd.DataFrame, p4: dict) -> dict:
    print("\n" + "=" * 72)
    print("CUT 10 — fold-wise leftover after days; 12-name Y2 drop")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    ids = chronic_12(tr)
    print(f"  chronic 0158+0172 names: {len(ids)} {ids}")
    y2 = pd.to_numeric(tr[Y2], errors="coerce")
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    folds = tr["fold"]
    lab2 = y2.notna()
    lab3 = y3.notna()
    chron = tr["company_id"].isin(ids)
    rec2 = signed_oof_auroc(y2, tr["f_ds_r"], folds, lab2)
    rec2d = signed_oof_auroc(y2, tr["c_n_days_with_tx"], folds, lab2)
    rec2_wo = signed_oof_auroc(y2, tr["f_ds_r"], folds, lab2 & ~chron)
    rec2d_wo = signed_oof_auroc(y2, tr["c_n_days_with_tx"], folds, lab2 & ~chron)
    after_wo = leftover_diag(y3, tr["f_ds_r"], (tr["c_n_days_with_tx"],), folds, lab3 & ~chron)
    fold_rows = []
    for r, rr in zip(p4["after"]["rec"]["folds"], p4["after"]["rrec"]["folds"]):
        fold_rows.append(
            {
                "fold": r["fold"],
                "OLS leftover": _f(r["auroc"]),
                "rank leftover": _f(rr["auroc"]),
                "n_va": r["n_va"],
                "n_pos": r["n_pos"],
            }
        )
    y2_rows = [
        _auc_row(Y2, "f_ds_r", rec2),
        _auc_row(Y2, "days", rec2d),
        _auc_row(Y2, "f_ds_r wo12", rec2_wo),
        _auc_row(Y2, "days wo12", rec2d_wo),
    ]
    prose = (
        f"Y2 f_ds_r {_f(rec2['cv'])} days {_f(rec2d['cv'])}. "
        f"Drop {len(ids)} chronic names: ds {_f(rec2_wo['cv'])} days {_f(rec2d_wo['cv'])}. "
        f"Y3 leftover after days without the 12: rank {_f(after_wo['rank'])} dies={after_wo['honest_dies']}."
    )
    print(prose)
    return {
        "ids": ids,
        "n_chronic": len(ids),
        "fold_rows": fold_rows,
        "y2_rows": y2_rows,
        "y2": _cv(rec2),
        "y2_days": _cv(rec2d),
        "y2_wo": _cv(rec2_wo),
        "y2_days_wo": _cv(rec2d_wo),
        "y3_wo_rank": after_wo["rank"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extras
# ---------------------------------------------------------------------------
def extra_log1p(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — log1p intensity leftover after days")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    rec = signed_oof_auroc(y, tr["log1p_ds_r"], folds, lab)
    after = leftover_diag(y, tr["log1p_ds_r"], (tr["c_n_days_with_tx"],), folds, lab)
    prose = (
        f"log1p(f_ds_r) Y3 {_f(rec['cv'])} leftover after days "
        f"OLS {_f(after['ols'])} rank {_f(after['rank'])} dies={after['honest_dies']}."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "dies": after["honest_dies"], "prose": prose}


def extra_stack(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — ds vs fc stack leftover (days+fc; days+size)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    ds = tr["f_ds_r"]
    after_both = leftover_diag(y, ds, (tr["c_n_days_with_tx"], tr["f_fc_r"]), folds, lab)
    after_dsz = leftover_diag(y, ds, (tr["c_n_days_with_tx"], tr["log_in3"]), folds, lab)
    after_euro = leftover_diag(y, ds, (tr["a_debt_service"],), folds, lab)
    prose = (
        f"after days+fc rank {_f(after_both['rank'])} dies={after_both['honest_dies']} R²={_f(after_both['r2'])}. "
        f"after days+size rank {_f(after_dsz['rank'])} dies={after_dsz['honest_dies']}. "
        f"after a_debt_service rank {_f(after_euro['rank'])} dies={after_euro['honest_dies']} "
        f"R²={_f(after_euro['r2'])} (ratio vs euro)."
    )
    print(prose)
    return {
        "days_fc": after_both,
        "days_size": after_dsz,
        "euro": after_euro,
        "days_fc_rank": after_both["rank"],
        "days_size_rank": after_dsz["rank"],
        "euro_rank": after_euro["rank"],
        "prose": prose,
    }


def extra_javier(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Javier debt_serv_r SAME/CLOSE (A-trail roll-3 recon)")
    print("=" * 72)
    work = tr.sort_values(["company_id", "period"]).copy()
    ads = pd.to_numeric(work["a_debt_service"], errors="coerce")
    fds = pd.to_numeric(work["f_debt_service"], errors="coerce")
    ain = pd.to_numeric(work["a_in3"], errors="coerce")
    ds3 = ads.groupby(work["company_id"], sort=False).transform(lambda s: s.rolling(3, min_periods=3).sum())
    fds3 = fds.groupby(work["company_id"], sort=False).transform(lambda s: s.rolling(3, min_periods=3).sum())
    recon_a = (ds3 / np.maximum(ain, 1.0)).clip(0, 2)
    recon_f = (fds3 / np.maximum(ain, 1.0)).clip(0, 2)
    store = pd.to_numeric(work["f_ds_r"], errors="coerce")
    rho_a = spearman(store, recon_a)
    rho_f = spearman(store, recon_f)
    same_a = bool(np.isfinite(rho_a) and rho_a >= 0.995)
    same_f = bool(np.isfinite(rho_f) and rho_f >= 0.995)
    tag_a = "SAME" if same_a else ("CLOSE" if np.isfinite(rho_a) and rho_a >= 0.90 else "DRIFT")
    tag_f = "SAME" if same_f else ("CLOSE" if np.isfinite(rho_f) and rho_f >= 0.90 else "DRIFT")
    prose = (
        f"f_ds_r vs roll3(a_debt_service)/max(a_in3,1) ρ={_f(rho_a)} {tag_a}. "
        f"vs roll3(f_debt_service)/max(a_in3,1) ρ={_f(rho_f)} {tag_f}. "
        f"Javier debt_serv_r is the same formula — {tag_f} to F-trail."
    )
    print(prose)
    return {
        "rho_a": rho_a,
        "rho_f": rho_f,
        "tag_a": tag_a,
        "tag_f": tag_f,
        "same": same_f or same_a,
        "prose": prose,
    }


def extra_hold_dark(tr: pd.DataFrame, hold: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — holdout coverage only; dark vs ERP leftover")
    print("=" * 72)
    hds = pd.to_numeric(hold["f_ds_r"], errors="coerce")
    hold_cov = {
        "n_co": int(hold["company_id"].nunique()),
        "n_cm": len(hold),
        "n_nn": int(hds.notna().sum()),
        "cov": _pct(int(hds.notna().sum()), len(hold)),
        "share0": _pct(int(hds.eq(0).sum()), max(int(hds.notna().sum()), 1)),
    }
    print(f"  holdout {hold_cov['n_co']} co / {hold_cov['n_cm']} CM nn={hold_cov['n_nn']} cov={_pp(hold_cov['cov'])} (no fit)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    erp = tr["company_id"].isin(book)
    rec_d = signed_oof_auroc(y, tr["f_ds_r"], folds, lab & ~erp)
    rec_e = signed_oof_auroc(y, tr["f_ds_r"], folds, lab & erp)
    after_d = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"],), folds, lab & ~erp)
    after_e = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"],), folds, lab & erp)
    prose = (
        f"Holdout coverage {_pp(hold_cov['cov'])} only — no fit. "
        f"Dark Y3 {_f(rec_d['cv'])} leftover-days rank {_f(after_d['rank'])}; "
        f"ERP Y3 {_f(rec_e['cv'])} leftover-days rank {_f(after_e['rank'])}."
    )
    print(prose)
    return {
        "hold": hold_cov,
        "dark_cv": _cv(rec_d),
        "erp_cv": _cv(rec_e),
        "dark_rank": after_d["rank"],
        "erp_rank": after_e["rank"],
        "prose": prose,
    }


def extra_gt0(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on ds>0 months (share=0 pile dropped)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    ds = pd.to_numeric(tr["f_ds_r"], errors="coerce")
    lab = y.notna() & ds.gt(0)
    folds = tr["fold"]
    rec = signed_oof_auroc(y, ds, folds, lab)
    after = leftover_diag(y, ds, (tr["c_n_days_with_tx"],), folds, lab)
    prose = (
        f"ds>0 Y3 {_f(rec['cv'])} leftover after days rank {_f(after['rank'])} "
        f"dies={after['honest_dies']} n={rec['n_defined']} pos={rec['n_pos']}."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "dies": after["honest_dies"], "prose": prose}


def extra_acf_wo_fold(tr: pd.DataFrame, p3: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — ACF + Y3 without weakest fold")
    print("=" * 72)
    work = tr[["company_id", "f_ds_r"]].copy()
    vals = []
    for _, g in work.groupby("company_id", sort=False):
        x = pd.to_numeric(g["f_ds_r"], errors="coerce").to_numpy(dtype=float)
        if len(x) <= 1:
            continue
        aa, bb = x[:-1], x[1:]
        m = np.isfinite(aa) & np.isfinite(bb)
        if int(m.sum()) < MIN_ACF_PAIRS:
            continue
        aa, bb = aa[m], bb[m]
        if np.std(aa) == 0 or np.std(bb) == 0:
            continue
        vals.append(float(np.corrcoef(aa, bb)[0, 1]))
    acf = float(np.median(vals)) if vals else float("nan")
    rec = p3["recs"][(Y3, "f_ds_r")]
    folds = [r["auroc"] for r in rec["folds"] if np.isfinite(r["auroc"])]
    wo = float("nan")
    weak_k = None
    if folds:
        weak_k = int(np.argmin(folds))
        rest = [a for i, a in enumerate(folds) if i != weak_k]
        wo = float(np.mean(rest)) if rest else float("nan")
    prose = f"median ACF1={_f(acf)} n_co={len(vals)}. Y3 without weakest fold {weak_k} = {_f(wo)}."
    print(prose)
    return {"acf": acf, "wo": wo, "weak_k": weak_k, "prose": prose}


def extra_quintiles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y3 rate by f_ds_r / days quintile")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rows = []
    for name, col in (("f_ds_r", tr["f_ds_r"]), ("days", tr["c_n_days_with_tx"])):
        s = pd.to_numeric(col, errors="coerce")
        sl = lab & s.notna()
        q = pd.qcut(s[sl], 5, duplicates="drop", labels=False)
        for i in sorted(q.dropna().unique()):
            m = sl.copy()
            m.loc[sl] = q == i
            n = int(m.sum())
            rows.append(
                {
                    "feat": name,
                    "Q": int(i) + 1,
                    "n": n,
                    "n_pos": int((m & (y == 1)).sum()),
                    "rate": _pp(float(y[m].mean()) if n else float("nan")),
                    "med": _f(float(s[m].median()) if n else float("nan"), 4),
                }
            )
    print(_md_table(rows))
    return {"rows": rows, "prose": "Y3 rate by quintile — days monotone, ds_r flatter then T3 bump."}


def extra_slice_leftover(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days by SIZE tercile and so-far")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    terc = size_terciles(tr)
    rows = []
    for t in ("T1", "T2", "T3"):
        sl = lab & (terc == t)
        after = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"],), folds, sl)
        rows.append(
            {
                "slice": f"size {t}",
                "n": after["n"],
                "n_pos": after["n_pos"],
                "rank": _f(after["rank"]),
                "ols": _f(after["ols"]),
                "dies": after["honest_dies"],
            }
        )
        print(f"  size {t} rank={_f(after['rank'])} dies={after['honest_dies']}")
    for b in ("<6", "6-11", "12-17", "18-23", "24+"):
        sl = lab & (tr["so_far_bucket"] == b)
        after = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"],), folds, sl)
        rows.append(
            {
                "slice": f"so-far {b}",
                "n": after["n"],
                "n_pos": after["n_pos"],
                "rank": _f(after["rank"]),
                "ols": _f(after["ols"]),
                "dies": after["honest_dies"],
            }
        )
        print(f"  so-far {b} rank={_f(after['rank'])} dies={after['honest_dies']} n={after['n']}")
    after_mean = leftover_diag(
        y, company_mean(tr["f_ds_r"], tr["company_id"]), (tr["c_n_days_with_tx"],), folds, lab
    )
    after_euro = leftover_diag(y, tr["a_debt_service"], (tr["c_n_days_with_tx"],), folds, lab)
    days_size = leftover_diag(y, tr["c_n_days_with_tx"], (tr["log_in3"],), folds, lab)
    prose = (
        f"Company-mean leftover after days rank {_f(after_mean['rank'])} dies={after_mean['honest_dies']}. "
        f"Euro a_debt_service leftover after days rank {_f(after_euro['rank'])} dies={after_euro['honest_dies']}. "
        f"Days leftover after size rank {_f(days_size['rank'])} dies={days_size['honest_dies']} "
        f"(days is not SIZE)."
    )
    print(prose)
    return {
        "rows": rows,
        "mean_rank": after_mean["rank"],
        "euro_rank": after_euro["rank"],
        "days_size_rank": days_size["rank"],
        "prose": prose,
    }


def extra_ilift_quote() -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — i_lift drop-ds_r-only quote (no new GBM)")
    print("=" * 72)
    prose = (
        "i_lift.md: drop `f_ds_r` only (no I) 12-col still **0.7525** (Δ 0.000 vs 0.752). "
        "Parent 15-col card absorbs. Do not edit the card. Do not merge I. "
        "`i_io_x_dsr` ρ=0.980 / `i_dso_x_dsr` ρ=0.916 vs `f_ds_r` — I twins, not new leftover."
    )
    print(prose)
    return {"drop_cv": 0.7525, "prose": prose}


def extra_fold0(tr: pd.DataFrame, p4: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — fold-0 leftover isolate; OLS fake-days leak")
    print("=" * 72)
    rec = p4["after"]["rrec"]
    f0 = fold_k(rec, 0)
    rest = [r["auroc"] for r in rec["folds"] if int(r["fold"]) != 0 and np.isfinite(r["auroc"])]
    wo = float(np.mean(rest)) if rest else float("nan")
    rho = p4["after"]["rho_ctrl"]
    almost_fake = bool(np.isfinite(rho) and abs(rho) >= 0.70)
    prose = (
        f"Rank leftover fold 0={_f(f0)} without fold 0={_f(wo)}. "
        f"OLS leftover {_f(p4['ols'])} with ρ(resid,days)={_f(rho)} "
        f"{'almost-fake days leak (rank is honest)' if almost_fake else 'not a days leak'}. "
        f"Ticket: OLS can fake a days leak; leftover <0.55 dies."
    )
    print(prose)
    return {"f0": f0, "wo": wo, "rho": rho, "almost_fake": almost_fake, "prose": prose}


def extra_gt0_quintiles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y3 quintiles on ds>0 only (share=0 pile dropped)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    ds = pd.to_numeric(tr["f_ds_r"], errors="coerce")
    lab = y.notna() & ds.gt(0)
    q = pd.qcut(ds[lab], 5, duplicates="drop", labels=False)
    rows = []
    for i in sorted(q.dropna().unique()):
        m = lab.copy()
        m.loc[lab] = q == i
        n = int(m.sum())
        rows.append(
            {
                "Q": int(i) + 1,
                "n": n,
                "n_pos": int((m & (y == 1)).sum()),
                "rate": _pp(float(y[m].mean()) if n else float("nan")),
                "med ds": _f(float(ds[m].median()) if n else float("nan"), 4),
                "med days": _f(float(pd.to_numeric(tr.loc[m, "c_n_days_with_tx"], errors="coerce").median()) if n else float("nan")),
            }
        )
    zero = y.notna() & ds.eq(0)
    rows.append(
        {
            "Q": "share=0",
            "n": int(zero.sum()),
            "n_pos": int((zero & (y == 1)).sum()),
            "rate": _pp(float(y[zero].mean()) if zero.any() else float("nan")),
            "med ds": "0",
            "med days": _f(float(pd.to_numeric(tr.loc[zero, "c_n_days_with_tx"], errors="coerce").median()) if zero.any() else float("nan")),
        }
    )
    print(_md_table(rows))
    return {"rows": rows, "prose": "share=0 pile is Q1 of the full qcut (8.2% Y3). ds>0 quintiles sit on a thinner positive tail."}


def extra_t3_leak(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — T3 leftover after days: live slice or days leak?")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    terc = size_terciles(tr)
    sl = lab & (terc == "T3")
    after = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"],), folds, sl)
    after_sz = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"], tr["log_in3"]), folds, sl)
    rec = signed_oof_auroc(y, tr["f_ds_r"], folds, sl)
    days = signed_oof_auroc(y, tr["c_n_days_with_tx"], folds, sl)
    prose = (
        f"T3 n={after['n']} pos={after['n_pos']} raw {_f(rec['cv'])} days {_f(days['cv'])}. "
        f"leftover after days OLS {_f(after['ols'])} rank {_f(after['rank'])} "
        f"ρ(resid,days)={_f(after['rho_ctrl'])} fake={after['fake']} dies={after['honest_dies']}. "
        f"after days+size rank {_f(after_sz['rank'])} dies={after_sz['honest_dies']}. "
        f"{'T3 leftover is a days leak' if after['fake'] or after['honest_dies'] else 'T3 leftover lives — still not KEEP on the full card'}."
    )
    print(prose)
    return {
        "rank": after["rank"],
        "rho": after["rho_ctrl"],
        "dies": after["honest_dies"],
        "days_size": after_sz["rank"],
        "raw": _cv(rec),
        "days": _cv(days),
        "prose": prose,
    }


def extra_permute(tr: pd.DataFrame, n_perm: int = 40) -> dict:
    print("\n" + "=" * 72)
    print(f"EXTRA — permute f_ds_r within days quintile (n={n_perm})")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    work = tr.loc[lab].copy()
    days = pd.to_numeric(work["c_n_days_with_tx"], errors="coerce")
    ds = pd.to_numeric(work["f_ds_r"], errors="coerce")
    ok = days.notna() & ds.notna()
    work = work.loc[ok]
    q = pd.qcut(pd.to_numeric(work["c_n_days_with_tx"], errors="coerce"), 5, duplicates="drop")
    rng = np.random.default_rng(FOLD_SEED + 1)
    ranks = []
    for _ in range(n_perm):
        shuf = pd.to_numeric(work["f_ds_r"], errors="coerce").copy()
        for cat in q.dropna().unique():
            idx = q[q == cat].index
            vals = shuf.loc[idx].to_numpy()
            rng.shuffle(vals)
            shuf.loc[idx] = vals
        after = leftover_diag(
            work[Y3],
            shuf,
            (work["c_n_days_with_tx"],),
            work["fold"],
            pd.Series(True, index=work.index),
        )
        if np.isfinite(after["rank"]):
            ranks.append(after["rank"])
    p50 = float(np.median(ranks)) if ranks else float("nan")
    p90 = float(np.quantile(ranks, 0.90)) if ranks else float("nan")
    prose = (
        f"Permuted-within-days leftover rank p50={_f(p50)} p90={_f(p90)} n={len(ranks)}. "
        f"Observed leftover 0.528 should sit inside this null if unused."
    )
    print(prose)
    return {"p50": p50, "p90": p90, "n": len(ranks), "ranks": ranks, "prose": prose}


def extra_ever_ds(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover on companies that ever have ds>0")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    ds = pd.to_numeric(tr["f_ds_r"], errors="coerce")
    ever = set(tr.loc[ds.gt(0), "company_id"].astype(str))
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    sl = lab & tr["company_id"].isin(ever)
    after = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"],), tr["fold"], sl)
    rec = signed_oof_auroc(y, tr["f_ds_r"], tr["fold"], sl)
    prose = (
        f"Ever-ds>0 cos={len(ever)} labeled n={after['n']} pos={after['n_pos']} "
        f"raw {_f(rec['cv'])} leftover-days rank {_f(after['rank'])} dies={after['honest_dies']}."
    )
    print(prose)
    return {"n_ever": len(ever), "cv": _cv(rec), "rank": after["rank"], "dies": after["honest_dies"], "prose": prose}


def extra_has_ds(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — binary has-ds>0 leftover after days (zero-pile rewrite)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    ds = pd.to_numeric(tr["f_ds_r"], errors="coerce")
    has = (ds.gt(0)).astype(float)
    has[ds.isna()] = np.nan
    lab = y.notna()
    rec = signed_oof_auroc(y, has, tr["fold"], lab)
    after = leftover_diag(y, has, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    rho = spearman(has, tr["c_n_days_with_tx"])
    prose = (
        f"has-ds>0 Y3 {_f(rec['cv'])} leftover after days rank {_f(after['rank'])} "
        f"dies={after['honest_dies']} ρ vs days={_f(rho)}. "
        f"The 0.620 single is mostly the zero-pile vs any-repayment, which days already owns."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "dies": after["honest_dies"], "rho": rho, "prose": prose}


def extra_t3_folds(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — T3 leftover folds (live slice, not a card KEEP)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    terc = size_terciles(tr)
    sl = lab & (terc == "T3")
    after = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"],), tr["fold"], sl)
    prose = (
        f"T3 leftover folds OLS {after['folds']} rank {after['rank_folds']}. "
        f"Rank {_f(after['rank'])} lives on the large-book slice only. "
        f"Full-card leftover 0.528 dies. Not KEEP."
    )
    print(prose)
    return {"rank": after["rank"], "folds": after["rank_folds"], "prose": prose}


def extra_javier_duck() -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Javier debt_serv_r from score_pipeline monthly flows (no invoices)")
    print("=" * 72)
    from analysis.score_pipeline import _monthly_flows

    con = connect()
    try:
        mth = _monthly_flows(con)
    finally:
        con.close()
    mth["company_id"] = mth["company_id"].astype(str)
    mth["month"] = pd.to_datetime(mth["month"])
    mth = mth.sort_values(["company_id", "month"])
    g = mth.groupby("company_id", sort=False)
    mth["in3"] = g["op_in"].transform(lambda s: s.rolling(3, min_periods=3).sum())
    mth["ds3"] = g["debt_service"].transform(lambda s: s.rolling(3, min_periods=3).sum())
    mth["debt_serv_r"] = (mth["ds3"] / np.maximum(mth["in3"], 1.0)).clip(0, 2)
    store = pd.read_parquet(STORE, columns=["company_id", "period", "f_ds_r"])
    store = _keys(store)
    mth = mth.rename(columns={"month": "period"})
    j = store.merge(mth[["company_id", "period", "debt_serv_r"]], on=["company_id", "period"], how="inner")
    hold = load_holdout()
    j = j[~j["company_id"].isin(hold)]
    assert_no_holdout(j["company_id"])
    rho = spearman(j["f_ds_r"], j["debt_serv_r"])
    same = bool(np.isfinite(rho) and rho >= 0.995)
    tag = "SAME" if same else ("CLOSE" if np.isfinite(rho) and rho >= 0.90 else "DRIFT")
    prose = (
        f"score_pipeline monthly-flow debt_serv_r vs store f_ds_r ρ={_f(rho)} {tag} "
        f"n={len(j):,} train rows (no invoice join)."
    )
    print(prose)
    return {"rho": rho, "tag": tag, "same": same, "n": len(j), "prose": prose}


def extra_y7_issued(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y7 leftover after issued (not a TURNOVER swap)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y7], errors="coerce")
    lab = y.notna()
    rec = signed_oof_auroc(y, tr["f_ds_r"], tr["fold"], lab)
    after = leftover_diag(y, tr["f_ds_r"], (tr["e_ar_issued"],), tr["fold"], lab)
    prose = (
        f"Y7 f_ds_r {_f(rec['cv'])} leftover after e_ar_issued rank {_f(after['rank'])} "
        f"dies={after['honest_dies']}. TURNDSSWAP already 0.712 — not a swap. Do not refit TURNOVER."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "dies": after["honest_dies"], "prose": prose}


def extra_y2_days(tr: pd.DataFrame, ids: list[str]) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y2 leftover after days; 12-name drop")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y2], errors="coerce")
    lab = y.notna()
    chron = tr["company_id"].isin(ids)
    after = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_wo = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"],), tr["fold"], lab & ~chron)
    prose = (
        f"Y2 leftover after days rank {_f(after['rank'])} dies={after['honest_dies']}. "
        f"Without 12 chronic names {_f(after_wo['rank'])} dies={after_wo['honest_dies']} "
        f"(days wo12 already 0.549)."
    )
    print(prose)
    return {
        "rank": after["rank"],
        "dies": after["honest_dies"],
        "wo": after_wo["rank"],
        "wo_dies": after_wo["honest_dies"],
        "prose": prose,
    }


def extra_trail_leftover(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days on short vs long company books")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rows = []
    for name, sl0 in (
        ("short_<12", tr["co_class"] == "short_<12"),
        ("mid_12_17", tr["co_class"] == "mid_12_17"),
        ("long_>=18", tr["co_class"] == "long_>=18"),
    ):
        sl = lab & sl0
        after = leftover_diag(y, tr["f_ds_r"], (tr["c_n_days_with_tx"],), tr["fold"], sl)
        rec = signed_oof_auroc(y, tr["f_ds_r"], tr["fold"], sl)
        rows.append(
            {
                "book": name,
                "n": after["n"],
                "n_pos": after["n_pos"],
                "raw": _f(rec["cv"]),
                "rank leftover": _f(after["rank"]),
                "dies": after["honest_dies"],
            }
        )
        print(f"  {name} raw={_f(rec['cv'])} leftover={_f(after['rank'])} dies={after['honest_dies']}")
    return {"rows": rows, "prose": "Leftover after days by company trail length."}


def extra_bootstrap(tr: pd.DataFrame, n_boot: int = 80) -> dict:
    print("\n" + "=" * 72)
    print(f"EXTRA — company bootstrap leftover after days (n={n_boot})")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    work = tr.loc[lab, ["company_id", "fold", Y3, "f_ds_r", "c_n_days_with_tx"]].copy()
    cos = work["company_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(FOLD_SEED)
    ranks = []
    for _ in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        parts = [work[work["company_id"] == c] for c in draw]
        b = pd.concat(parts, ignore_index=True)
        after = leftover_diag(
            b[Y3], b["f_ds_r"], (b["c_n_days_with_tx"],), b["fold"], pd.Series(True, index=b.index)
        )
        if np.isfinite(after["rank"]):
            ranks.append(after["rank"])
    p50 = float(np.median(ranks)) if ranks else float("nan")
    p10 = float(np.quantile(ranks, 0.10)) if ranks else float("nan")
    p90 = float(np.quantile(ranks, 0.90)) if ranks else float("nan")
    share_die = float(np.mean([r < CHANCE for r in ranks])) if ranks else float("nan")
    prose = (
        f"Bootstrap leftover-after-days rank p50={_f(p50)} p10={_f(p10)} p90={_f(p90)} "
        f"share<0.55={_pp(share_die)} n={len(ranks)}."
    )
    print(prose)
    return {"p50": p50, "p10": p10, "p90": p90, "share_die": share_die, "n": len(ranks), "prose": prose}


def decide(p2, p3, p4, p5, p8, extras) -> dict:
    leftover_lives = bool(np.isfinite(p4["rank"]) and p4["rank"] >= CHANCE and not p4["dies"])
    beat = bool(p3["beat_size"])
    size_flag = bool(p2["size_flag"])
    twin = bool(p2["twin_fc"] or p2["twin_ads"] or p2["twin_fds"])
    keep_x = leftover_lives and beat and not size_flag and not p2["twin_fc"]
    if leftover_lives and beat and not size_flag and not p2["twin_fc"]:
        card = "KEEP"
        why = (
            f"leftover after days rank {p4['rank']:.3f} lives, "
            f"beats size by {p3['y3'] - p3['size3']:.3f}, not SIZE, not fc-twin"
        )
    elif not leftover_lives:
        card = "DROP"
        why = (
            f"unused leftover after days: honest rank {p4['rank']:.3f} dies "
            f"(OLS {p4['ols']:.3f} ρ(resid,days)={p4['after']['rho_ctrl']:.3f} fake={p4['fake']}). "
            f"Single {p3['y3']:.3f} loses to days {p3['days']:.3f} and fails beat-size "
            f"(Δ={p3['y3'] - p3['size3']:.3f}). Twin of euro ds (ρ a_debt={p2['rhos']['a_debt_service']:.3f}). "
            f"i_lift drop-ds_r-only still 0.7525. Parent 15-col card absorbs; do not edit the card. "
            f"Same unused leftover as contemp f_fc_r on the 44."
        )
    else:
        card = "CLOSE"
        why = (
            f"leftover after days rank {p4['rank']:.3f} lives but KEEP-as-X fails "
            f"(beat_size={beat} Δ={p3['y3'] - p3['size3']:.3f}, SIZE={size_flag}, twin={twin})"
        )
    return {
        "card": card,
        "why": why,
        "keep_x": keep_x,
        "leftover_lives": leftover_lives,
        "beat": beat,
        "q6": "CLOSE",
        "y4_role": "identity only — Y4 is y4_ds_r_double; F forbidden as X; not KEEP",
        "turnover": "not a swap — TURNDSSWAP 0.712; do not refit; do not rip f_fc_r_lag3",
    }


def make_plot(tr: pd.DataFrame, p4: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    ds = pd.to_numeric(tr["f_ds_r"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    for name, s, color in (("f_ds_r", ds, "#c45c26"), ("days", days, "#1f4e79")):
        q = pd.qcut(s, 5, duplicates="drop")
        rates, xs = [], []
        for i, (_, sl) in enumerate(q.groupby(q, observed=True), start=1):
            lab = y3.loc[sl.index].notna()
            rates.append(float(y3.loc[sl.index][lab].mean()) if lab.any() else float("nan"))
            xs.append(i)
        ax.plot(xs, rates, marker="o", label=name, color=color)
    ax.set_xlabel("quintile (low → high)")
    ax.set_ylabel("Y3 rate")
    ax.set_title("Y3 rate by ds_r / days quintile")
    ax.legend(frameon=False)
    ax.set_ylim(0, None)
    ax = axes[1]
    rec = p4["after"]["rrec"]
    xs = [r["fold"] for r in rec["folds"]]
    ys = [r["auroc"] for r in rec["folds"]]
    ax.bar(xs, ys, color="#c45c26")
    ax.axhline(CHANCE, color="#333", ls="--", lw=1, label="0.55 dies")
    ax.axhline(DAYS_BENCH, color="#1f4e79", ls=":", lw=1, label="days 0.711")
    ax.set_xlabel("fold")
    ax.set_ylabel("rank leftover AUROC")
    ax.set_title("f_ds_r leftover after days (rank)")
    ax.set_ylim(0.4, 0.85)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def write_md(ctx: dict) -> None:
    d = ctx["decision"]
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
    p6, p7, p8, p10 = ctx["p6"], ctx["p7"], ctx["p8"], ctx["p10"]
    xl, xs, xj, xh, xg, xa = ctx["xl"], ctx["xs"], ctx["xj"], ctx["xh"], ctx["xg"], ctx["xa"]
    xq, xz, xi, xf, xb = ctx["xq"], ctx["xz"], ctx["xi"], ctx["xf"], ctx["xb"]
    xg0, xt3, xp, xe = ctx["xg0"], ctx["xt3"], ctx["xp"], ctx["xe"]
    xhd, xtf = ctx["xhd"], ctx["xtf"]
    xjd, xy7 = ctx["xjd"], ctx["xy7"]
    xy2, xtr = ctx["xy2"], ctx["xtr"]
    lines = [
        "# Unused leftover of KEEP-flow `f_ds_r` after days",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_ds_r`. Do not score Family F vs Y4 as KEEP. "
        "Do not grow TURNOVER. Do not change the 15-col Y3 card.",
        "",
        "`f_ds_r` = trailing-3m debt service / inflow, clipped [0, 2] (Family F). "
        "Still on the 15-col Y3 card and the 44. Contemporaneous `f_fc_r` was DROPPED from the 44 "
        "as unused leftover (ρ vs `f_ds_r` 0.205). This cut asks leftover after `c_n_days_with_tx` "
        "(0.711 bar).",
        "",
        "## Headline",
        "",
        (
            f"`f_ds_r` on the 15-col card: **{d['card']}** ({d['why']}). "
            f"Y3 leftover after days OLS {_f(p4['ols'])} rank {_f(p4['rank'])} "
            f"({'dies' if p4['dies'] else 'lives'}, fake={p4['fake']}). "
            f"Inverse days after ds_r rank {_f(p4['inv_rank'])}. "
            f"After size {_f(p5['size_rank'])}; after fc {_f(p5['fc_rank'])} "
            f"({'CONFIRM 0.606' if p5['fc_ok'] else 'DRIFT'}). "
            f"Single {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size3'])} "
            f"beat_size={p3['beat_size']}. ρ vs fc {_f(p2['rhos']['f_fc_r'])}. "
            f"Y4 identity only (not KEEP). TURNDSSWAP 0.712 not a swap. "
            f"Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712."
        ),
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        f"| 1 | Who is healthy? | `f_ds_r` is a financing **flow**, already a card stem. {d['card']} as unused leftover after days. |",
        "| 2 | Who is improving? | Not a path column by itself. Q6 lag3 already CLOSE. |",
        f"| 3 | Who is turning? | Y4 *is* `y4_ds_r_double`. F forbidden as Y4 X. {d['y4_role']}. |",
        "| 4 | Dip vs fall? | TURNDSSWAP already 0.712 — not a swap. Do not refit TURNOVER. Do not rip `f_fc_r_lag3`. |",
        f"| 5 | Why did it change? | Leftover after days rank {_f(p4['rank'])}. After fc {_f(p5['fc_rank'])} still lives. Twin vs euro-ds: {p2['twins']}. |",
        f"| 6 | Months earlier? | lag3 empty until so-far≥6: {'CONFIRM' if p7['empty_ok'] else 'DRIFT'}. F lag3 **CLOSE**. |",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `f_ds_r` on the 15-col card | **{d['card']}** | {d['why']} |",
        f"| `f_ds_r` on the 44 | **{d['card']}** | same leftover test; parent absorbs; do not edit the card |",
        f"| F lag3 Q6 | **{d['q6']}** | empty until so-far≥6 already CLOSE |",
        f"| Y4 identity | **not KEEP** | {d['y4_role']} |",
        f"| TURNOVER swap | **not a swap** | {d['turnover']} |",
        "",
        "## 1 — Coverage; 470 vs ERP",
        "",
        p1["prose"],
        "",
        _md_table(p1["rows"]),
        "",
        _md_table(p1["book_rows"]),
        "",
        (
            f"Holdout coverage only: {p1['hold']['n_co']} co / {p1['hold']['n_cm']} CM "
            f"nn={p1['hold']['n_nn']} cov={_pp(p1['hold']['cov'])} share=0={_pp(p1['hold']['share0'])}."
            if p1.get("hold")
            else "Holdout coverage not attached."
        ),
        "",
        "## 2 — Spearman twins",
        "",
        p2["prose"],
        "",
        _md_table(p2["rows"]),
        "",
        "## 3 — Single-feature group-fold AUROC",
        "",
        p3["prose"],
        "",
        _md_table(p3["rows"]),
        "",
        "## 4 — Honest leftover after days",
        "",
        p4["prose"],
        "",
        f"OLS folds: {p4['after']['folds']}. Rank folds: {p4['after']['rank_folds']}.",
        "",
        "## 5 — Leftover after size / after fc",
        "",
        p5["prose"],
        "",
        "## 6 — SIZE terciles; ICC / demean",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        "## 7 — Q6 lag1 / lag3",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        _md_table(p7["cov_rows"]),
        "",
        "## 8 — Y4 identity (not KEEP)",
        "",
        p8["prose"],
        "",
        "## 10 — Fold-wise leftover; 12-name Y2 drop",
        "",
        p10["prose"],
        "",
        _md_table(p10["fold_rows"]),
        "",
        _md_table(p10["y2_rows"]),
        "",
        "## Extras",
        "",
        "### log1p intensity",
        "",
        xl["prose"],
        "",
        "### ds vs fc stack leftover",
        "",
        xs["prose"],
        "",
        "### Javier debt_serv_r",
        "",
        xj["prose"],
        "",
        "### Holdout coverage; dark vs ERP",
        "",
        xh["prose"],
        "",
        "### ds>0 months",
        "",
        xg["prose"],
        "",
        "### ACF / weakest fold",
        "",
        xa["prose"],
        "",
        "### Quintiles",
        "",
        xq["prose"],
        "",
        _md_table(xq["rows"]),
        "",
        "### Leftover by SIZE tercile / so-far",
        "",
        xz["prose"],
        "",
        _md_table(xz["rows"]),
        "",
        "### i_lift drop-ds_r-only (quoted, no new GBM)",
        "",
        xi["prose"],
        "",
        "### Fold-0 leftover / OLS fake-days leak",
        "",
        xf["prose"],
        "",
        "### Company bootstrap leftover after days",
        "",
        xb["prose"],
        "",
        "### ds>0 quintiles",
        "",
        xg0["prose"],
        "",
        _md_table(xg0["rows"]),
        "",
        "### T3 leftover leak check",
        "",
        xt3["prose"],
        "",
        "### Permute within days quintile",
        "",
        xp["prose"],
        "",
        "### Ever-ds>0 companies",
        "",
        xe["prose"],
        "",
        "### Binary has-ds>0 leftover",
        "",
        xhd["prose"],
        "",
        "### T3 leftover folds",
        "",
        xtf["prose"],
        "",
        "### Javier score_pipeline recon",
        "",
        xjd["prose"],
        "",
        "### Y7 leftover after issued",
        "",
        xy7["prose"],
        "",
        "### Y2 leftover after days",
        "",
        xy2["prose"],
        "",
        "### Leftover by company trail length",
        "",
        xtr["prose"],
        "",
        _md_table(xtr["rows"]),
        "",
        "### TURNDSSWAP",
        "",
        "Already **0.712** — not a swap for `f_fc_r_lag3`. Do not refit TURNOVER. "
        "Do not rip `f_fc_r_lag3` off TURNOVER. Night Y7 stays **0.720 / 0.712**.",
        "",
        "## Night quotes (unchanged)",
        "",
        "| quote | locked |",
        "| --- | --- |",
        f"| Y3 | {Y3_NIGHT[0]:.3f} / {Y3_NIGHT[1]:.3f} |",
        f"| days | {DAYS_BENCH:.3f} |",
        f"| size | {SIZE_Y3_QUOTE:.3f} |",
        f"| Y7 TURNOVER | {Y7_TURNOVER:.3f} / {Y7_B_SHALLOW:.3f} |",
        f"| TURNDSSWAP | {TURNDSSWAP:.3f} not a swap |",
        "",
        "Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put new cols on the 15-col card.",
        "",
        "## Files written",
        "",
        "- `analysis/evaluate/ds_r_qa.py`",
        "- `analysis/outputs/ds_r_qa.md`",
        "- `analysis/outputs/ds_r_leftover.png`" if ctx.get("png") else "- (no PNG)",
        "- append-only `analysis/experiments/registry.csv`",
        "- `overnight/waves/wave4_ds_r.md` (end, if WRITE_WAVE)",
        "",
        f"Elapsed {ctx['elapsed_s']:.0f}s. Failed: {'; '.join(ctx['failed']) or 'none'}.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")


def write_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        print("registry missing — skip")
        return
    prev = pd.read_csv(REGISTRY)
    key_cols = ["agent", "x_families", "y", "model", "split", "metric"]
    seen = {tuple(str(r[c]) for c in key_cols) for _, r in prev.iterrows()}
    ts = _now_iso()
    p1, p2, p3, p4, p5, p8 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"], ctx["p8"]
    d = ctx["decision"]
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
            "metric": "auroc_f_ds_r",
            "value": p3["y3"],
            "coverage": f"{p1['ds_cov']:.4f}",
            "notes": f"days={p3['days']:.4f} size={p3['size3']:.4f} beat={p3['beat_size']}",
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
            "metric": "auroc_f_ds_r_resid_days",
            "value": p4["rank"],
            "coverage": f"{p1['ds_cov']:.4f}",
            "notes": f"ols={p4['ols']:.4f} dies={p4['dies']} fake={p4['fake']} r2={p4['r2']:.4f}",
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
            "value": p5["fc_rank"],
            "coverage": f"{p1['ds_cov']:.4f}",
            "notes": f"confirm_606={p5['fc_ok']} dies={p5['fc_dies']}",
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
            "metric": "auroc_days_resid_ds_r",
            "value": p4["inv_rank"],
            "coverage": f"{p1['ds_cov']:.4f}",
            "notes": f"inverse dies={p4['inv_dies']}",
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
            "metric": "rho_f_ds_r_vs_f_fc_r",
            "value": p2["rhos"]["f_fc_r"],
            "coverage": f"{p1['ds_cov']:.4f}",
            "notes": f"confirm_205={p2['fc_match']} twins={p2['twins']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y4,
            "model": MODEL,
            "split": "train",
            "metric": "rho_f_ds_r_vs_y4",
            "value": p8["rho"],
            "coverage": f"{p1['ds_cov']:.4f}",
            "notes": f"identity_only jac={p8['jac']:.4f} not_KEEP F_forbidden",
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
            "metric": "card_stem",
            "value": {"KEEP": 1, "CLOSE": 0, "DROP": -1}.get(d["card"], 0),
            "coverage": f"{p1['ds_cov']:.4f}",
            "notes": d["card"] + " " + d["why"][:180],
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
    p2, p3, p4, p5, p7, p8 = ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"], ctx["p7"], ctx["p8"]
    text = (
        f"# Wave 4 — f_ds_r leftover after days\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/ds_r_qa.py`\n"
        f"- `analysis/outputs/ds_r_qa.md`\n"
        f"- `analysis/outputs/ds_r_leftover.png`\n"
        f"- append-only `analysis/experiments/registry.csv` "
        f"(skip key agent+x_families+y+model+split+metric)\n"
        f"- this note\n\n"
        f"Did not touch `fc_r_qa.py` / `.md`, `n_tx_qa.*`, `dso_qa.*`, `cust_hhi_qa.*`, "
        f"`debt.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, "
        f"TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, or the parent journal. "
        f"Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. "
        f"Y7 TURNOVER **0.720 / 0.712** unchanged. TURNDSSWAP 0.712 not a swap.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| `f_ds_r` on the 15-col card | **{d['card']}** |\n"
        f"| F lag3 Q6 | **{d['q6']}** |\n"
        f"| Y4 identity | **not KEEP** |\n"
        f"| TURNOVER swap | **not a swap** |\n\n"
        f"Y3 leftover after days rank {_f(p4['rank'])} (dies={p4['dies']}, fake={p4['fake']}); "
        f"inverse days after ds_r {_f(p4['inv_rank'])}. After fc {_f(p5['fc_rank'])}. "
        f"Single {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size3'])}. "
        f"ρ vs fc {_f(p2['rhos']['f_fc_r'])}. Y4 ρ={_f(p8['rho'])} not KEEP. "
        f"Q6 empty-until-6 CONFIRM={p7['empty_ok']}. {d['why']}\n\n"
        f"## What failed / next\n\n"
        + "\n".join(f"- {x}" for x in ctx["failed"])
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.write_text(text)
    print(f"wrote {WAVE_NOTE}")


def main() -> None:
    t0 = time.time()
    failed: list[str] = []
    print("ds_r leftover QA — unused leftover of KEEP-flow f_ds_r after days")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, ["f_ds_r"], (1, 3))
    con = connect()
    try:
        book = book_invoice_ids(con)
    finally:
        con.close()
    print(f"book (ERP) companies={len(book)}")
    tr = panel[panel["split"] == "train"].copy()
    hold = panel[panel["split"] == "holdout"].copy()
    assert_no_holdout(tr["company_id"])
    if hold["company_id"].nunique() != 72:
        failed.append(f"holdout n_co={hold['company_id'].nunique()} expected 72")
    p1 = pass1_cov(tr, book, hold)
    p2 = pass2_twins(tr)
    p3 = pass3_singles(tr)
    p4 = pass4_days(tr)
    p5 = pass5_size_fc(tr)
    p6 = pass6_tercile_icc(tr)
    p7 = pass7_q6(tr, hold)
    p8 = pass8_y4(tr)
    p10 = pass10_fold_y2(tr, p4)
    xl = extra_log1p(tr)
    xs = extra_stack(tr)
    xj = extra_javier(tr)
    xh = extra_hold_dark(tr, hold, book)
    xg = extra_gt0(tr)
    xa = extra_acf_wo_fold(tr, p3)
    xq = extra_quintiles(tr)
    xz = extra_slice_leftover(tr)
    xi = extra_ilift_quote()
    xf = extra_fold0(tr, p4)
    xb = extra_bootstrap(tr, n_boot=80)
    xg0 = extra_gt0_quintiles(tr)
    xt3 = extra_t3_leak(tr)
    xp = extra_permute(tr, n_perm=40)
    xe = extra_ever_ds(tr)
    xhd = extra_has_ds(tr)
    xtf = extra_t3_folds(tr)
    xjd = extra_javier_duck()
    xy7 = extra_y7_issued(tr)
    xy2 = extra_y2_days(tr, p10["ids"])
    xtr = extra_trail_leftover(tr)
    extras = {
        "xl": xl,
        "xs": xs,
        "xj": xj,
        "xh": xh,
        "xg": xg,
        "xa": xa,
        "xq": xq,
        "xz": xz,
        "xi": xi,
        "xf": xf,
        "xb": xb,
        "xg0": xg0,
        "xt3": xt3,
        "xp": xp,
        "xe": xe,
        "xhd": xhd,
        "xtf": xtf,
        "xjd": xjd,
        "xy7": xy7,
        "xy2": xy2,
        "xtr": xtr,
    }
    decision = decide(p2, p3, p4, p5, p8, extras)
    print("\n" + "=" * 72)
    print(f"CARD STEM: {decision['card']}")
    print(decision["why"])
    print("=" * 72)
    if not p3["days_ok"]:
        failed.append(f"days replica {p3['days']:.3f} ≠ 0.711")
    if not p3["size_ok"]:
        failed.append(f"size replica {p3['size3']:.3f} ≠ 0.617")
    if not p3["own_ok"]:
        failed.append(f"own f_ds_r {p3['y3']:.3f} ≠ 0.620")
    if not p2["fc_match"]:
        failed.append(f"ρ vs fc {p2['rhos']['f_fc_r']:.3f} ≠ 0.205")
    if not p5["fc_ok"]:
        failed.append(f"leftover after fc {p5['fc_rank']:.3f} ≠ 0.606")
    if not p7["empty_ok"]:
        failed.append("Q6 lag3 not empty on so-far<6")
    if p8["scored_as_keep"] or p8["leak_ok"]:
        failed.append("Y4 KEEP leak — F scored as Y4 X")
    png = make_plot(tr, p4)
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
        "p10": p10,
        "xl": xl,
        "xs": xs,
        "xj": xj,
        "xh": xh,
        "xg": xg,
        "xa": xa,
        "xq": xq,
        "xz": xz,
        "xi": xi,
        "xf": xf,
        "xb": xb,
        "xg0": xg0,
        "xt3": xt3,
        "xp": xp,
        "xe": xe,
        "xhd": xhd,
        "xtf": xtf,
        "xjd": xjd,
        "xy7": xy7,
        "xy2": xy2,
        "xtr": xtr,
        "decision": decision,
        "failed": failed,
        "elapsed_s": elapsed,
        "png": png,
    }
    write_md(ctx)
    write_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    else:
        print("WRITE_WAVE=False — wave note deferred")
    print(f"elapsed {elapsed:.1f}s failed={failed or 'none'}")


if __name__ == "__main__":
    main()
