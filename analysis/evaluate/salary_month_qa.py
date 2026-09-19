"""Unused leftover of ``c_salary_month`` after days.

``c_salary_month`` is still on the 15-col Y3 card (quoted 0.671 vs days
0.711). ``f_ds_r`` was DROPPED as unused leftover after days (rank 0.528).
``a_n_tx`` was DROPPED as a days twin. ``c_missed_salary`` is already
CLOSE as Y3 X (0.513). This lane asks leftover of presence salary after
``c_n_days_with_tx``.

KEEP-as-X: beat size ≥0.02 **and** leftover after days **and** not SIZE
(|ρ| vs log1p(a_in3) ≥0.50) **and** not a twin (|ρ|≥0.80 vs days /
``a_n_tx`` / ``c_ss_month`` / ``c_missed_salary`` / ``c_tax_month``).
Leftover <0.55 dies. Rank leftover is honest; OLS can fake a days leak.

Do **not** overwrite ``salary_qa.py`` / ``.md`` (missed-salary). Do not
touch ``ss_qa.*``. Do not edit the 15-col card.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.salary_month_qa

Owned: analysis/evaluate/salary_month_qa.py, analysis/outputs/salary_month_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_salary_month.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "salary_month_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "salary_month_leftover.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_salary_month.md"
AGENT = "572fb928"
WAVE = "4"
ROUND = "R4"
MODEL = "salary_month_qa"
X_FAM = "C"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_Y3_QUOTE = 0.617
OWN_SAL_QUOTE = 0.671
MISSED_QUOTE = 0.513
Y3_NIGHT = (0.762, 0.752)
Y7_TURNOVER = 0.720
Y7_B_SHALLOW = 0.712
KEEP_DELTA = 0.02
TWIN_RHO = 0.80
SIZE_RHO = 0.50
CHANCE = 0.55
ICC_STYLE = 0.85
MIN_POS = 50
MIN_ACF_PAIRS = 4
HOT_GROUPS = ("GROUP_0158", "GROUP_0172")
Q_MONTHS = {1, 4, 7, 10}
WRITE_WAVE = True

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "a_n_tx",
    "c_n_days_with_tx",
    "c_salary_month",
    "c_ss_month",
    "c_missed_salary",
    "c_tax_month",
    "b_below_0",
)

Y_KEEP = (Y2, Y3)


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


def median_acf(series: pd.Series, company: pd.Series, lag: int = 1) -> float:
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
    panel["is_q"] = panel["period"].dt.month.isin(Q_MONTHS).astype(int)
    panel["is_aug"] = (panel["period"].dt.month == 8).astype(int)
    leak3 = leakage_check(
        ["c_salary_month", "c_ss_month", "c_n_days_with_tx", "log_in3", "a_n_tx"],
        Y3,
        forbidden_prefixes=["b"],
    )
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 never-B leak: {leak3['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def chronic_12(tr: pd.DataFrame) -> list[str]:
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
# 1. Coverage; modal; acf1; size ρ
# ---------------------------------------------------------------------------
def pass1_cov(tr: pd.DataFrame, hold: pd.DataFrame | None = None) -> dict:
    print("\n" + "=" * 72)
    print("CUT 1 — coverage; modal; acf1; size ρ")
    print("=" * 72)
    sal = pd.to_numeric(tr["c_salary_month"], errors="coerce")
    n_cm = len(tr)
    n_co = int(tr["company_id"].nunique())
    nn = sal.notna()
    share1 = _pct(int(sal.eq(1).sum()), int(nn.sum()))
    modal0 = _pct(int(sal.eq(0).sum()), int(nn.sum()))
    acf = median_acf(sal, tr["company_id"], 1)
    rho_size = spearman(sal, tr["log_in3"])
    size_flag = bool(np.isfinite(rho_size) and abs(rho_size) >= SIZE_RHO)
    hold_row = None
    if hold is not None and len(hold):
        hs = pd.to_numeric(hold["c_salary_month"], errors="coerce")
        hold_row = {
            "n_co": int(hold["company_id"].nunique()),
            "n_cm": len(hold),
            "n_nn": int(hs.notna().sum()),
            "cov": _pct(int(hs.notna().sum()), len(hold)),
            "share1": _pct(int(hs.eq(1).sum()), int(hs.notna().sum())),
        }
    rows = [
        {
            "col": "c_salary_month",
            "n_nn": f"{int(nn.sum()):,}",
            "cov": _pp(_pct(int(nn.sum()), n_cm)),
            "share=1": _pp(share1),
            "modal=0": _pp(modal0),
            "acf1": _f(acf),
            "ρ size": _f(rho_size),
            "SIZE": size_flag,
        }
    ]
    prose = (
        f"Train {n_co:,} co / {n_cm:,} CM. salary cov {_pp(_pct(int(nn.sum()), n_cm))} "
        f"share=1 {_pp(share1)} modal=0 {_pp(modal0)}. ACF1={_f(acf)}. "
        f"ρ vs log_in3 {_f(rho_size)} SIZE={size_flag}."
    )
    print(prose)
    return {
        "rows": rows,
        "n_cm": n_cm,
        "n_co": n_co,
        "cov": _pct(int(nn.sum()), n_cm),
        "share1": share1,
        "modal0": modal0,
        "acf": acf,
        "rho_size": rho_size,
        "size_flag": size_flag,
        "hold": hold_row,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 2. Spearman twins
# ---------------------------------------------------------------------------
def pass2_twins(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 2 — Spearman twins vs days / a_n_tx / SS / missed / tax")
    print("=" * 72)
    sal = pd.to_numeric(tr["c_salary_month"], errors="coerce")
    pairs = {
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
        "c_ss_month": tr["c_ss_month"],
        "c_missed_salary": tr["c_missed_salary"],
        "c_tax_month": tr["c_tax_month"],
        "log_in3": tr["log_in3"],
    }
    rhos = {k: spearman(sal, v) for k, v in pairs.items()}
    rows = [{"vs": k, "rho": _f(v), "twin": "yes" if abs(v) >= TWIN_RHO else "no"} for k, v in rhos.items()]
    twins = [k for k, v in rhos.items() if np.isfinite(v) and abs(v) >= TWIN_RHO]
    jac_ss = jaccard(sal.eq(1), pd.to_numeric(tr["c_ss_month"], errors="coerce").eq(1))
    prose = (
        f"ρ vs days {_f(rhos['c_n_days_with_tx'])} vs a_n_tx {_f(rhos['a_n_tx'])} "
        f"vs SS {_f(rhos['c_ss_month'])} vs missed {_f(rhos['c_missed_salary'])} "
        f"vs tax {_f(rhos['c_tax_month'])}. Twins (≥0.80): {twins or 'none'}. "
        f"salary∩SS Jaccard {_f(jac_ss['jaccard'])} both={jac_ss['n_both']:,}."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "twins": twins,
        "twin_days": "c_n_days_with_tx" in twins,
        "twin_ntx": "a_n_tx" in twins,
        "twin_ss": "c_ss_month" in twins,
        "twin_miss": "c_missed_salary" in twins,
        "twin_tax": "c_tax_month" in twins,
        "jac_ss": jac_ss,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 3. Single-feature group-fold AUROC
# ---------------------------------------------------------------------------
def pass3_singles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 3 — single-feature group-fold Y3 / Y2")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    y2 = pd.to_numeric(tr[Y2], errors="coerce")
    folds = tr["fold"]
    lab3 = y3.notna()
    lab2 = y2.notna()
    feats = {
        "c_salary_month": tr["c_salary_month"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "log_in3": tr["log_in3"],
        "c_ss_month": tr["c_ss_month"],
        "c_missed_salary": tr["c_missed_salary"],
        "a_n_tx": tr["a_n_tx"],
        "c_tax_month": tr["c_tax_month"],
    }
    recs = {}
    rows = []
    for yname, y, lab in ((Y3, y3, lab3), (Y2, y2, lab2)):
        for fname, x in feats.items():
            rec = signed_oof_auroc(y, x, folds, lab)
            recs[(yname, fname)] = rec
            rows.append(_auc_row(yname, fname, rec))
            print(f"  {yname} {fname} CV={_f(rec['cv'])} folds={fold_bits(rec)}")
    y3_sal = _cv(recs[(Y3, "c_salary_month")])
    y3_days = _cv(recs[(Y3, "c_n_days_with_tx")])
    y3_size = _cv(recs[(Y3, "log_in3")])
    y3_miss = _cv(recs[(Y3, "c_missed_salary")])
    days_ok = bool(np.isfinite(y3_days) and abs(y3_days - DAYS_BENCH) < 0.008)
    size_ok = bool(np.isfinite(y3_size) and abs(y3_size - SIZE_Y3_QUOTE) < 0.008)
    own_ok = bool(np.isfinite(y3_sal) and abs(y3_sal - OWN_SAL_QUOTE) < 0.008)
    miss_ok = bool(np.isfinite(y3_miss) and abs(y3_miss - MISSED_QUOTE) < 0.015)
    beat_size = bool(np.isfinite(y3_sal) and np.isfinite(y3_size) and (y3_sal - y3_size) >= KEEP_DELTA)
    prose = (
        f"Y3 salary {_f(y3_sal)} vs days {_f(y3_days)} vs size {_f(y3_size)}. "
        f"Replica days 0.711 {'CONFIRM' if days_ok else 'DRIFT'} / "
        f"size 0.617 {'CONFIRM' if size_ok else 'DRIFT'} / "
        f"salary 0.671 {'CONFIRM' if own_ok else 'DRIFT'}. "
        f"missed {_f(y3_miss)} {'CONFIRM 0.513' if miss_ok else 'DRIFT'}. "
        f"Beat size ≥0.02: {beat_size} (Δ={_f(y3_sal - y3_size if np.isfinite(y3_sal) and np.isfinite(y3_size) else float('nan'))})."
    )
    print(prose)
    return {
        "rows": rows,
        "recs": recs,
        "y3": y3_sal,
        "days": y3_days,
        "size3": y3_size,
        "y2": _cv(recs[(Y2, "c_salary_month")]),
        "miss": y3_miss,
        "ss": _cv(recs[(Y3, "c_ss_month")]),
        "days_ok": days_ok,
        "size_ok": size_ok,
        "own_ok": own_ok,
        "miss_ok": miss_ok,
        "beat_size": beat_size,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 4. Honest leftover after days + inverse
# ---------------------------------------------------------------------------
def pass4_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 4 — honest leftover after days (OLS + rank); inverse")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    after = leftover_diag(y, tr["c_salary_month"], (tr["c_n_days_with_tx"],), folds, lab)
    inv = leftover_diag(y, tr["c_n_days_with_tx"], (tr["c_salary_month"],), folds, lab)
    prose = (
        f"salary leftover after days OLS {_f(after['ols'])} rank {_f(after['rank'])} "
        f"fake={after['fake']} ρ(resid,days)={_f(after['rho_ctrl'])} R²={_f(after['r2'])} "
        f"honest_dies={after['honest_dies']}. "
        f"Inverse: days leftover after salary OLS {_f(inv['ols'])} rank {_f(inv['rank'])} "
        f"dies={inv['honest_dies']}."
    )
    print(prose)
    return {
        "after": after,
        "inv": inv,
        "ols": after["ols"],
        "rank": after["rank"],
        "dies": after["honest_dies"],
        "fake": after["fake"],
        "r2": after["r2"],
        "inv_rank": inv["rank"],
        "inv_dies": inv["honest_dies"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 5. Leftover after SS; after missed
# ---------------------------------------------------------------------------
def pass5_ss_miss(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 5 — leftover after SS (payroll twin?); after missed_salary")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    after_ss = leftover_diag(y, tr["c_salary_month"], (tr["c_ss_month"],), folds, lab)
    after_miss = leftover_diag(y, tr["c_salary_month"], (tr["c_missed_salary"],), folds, lab)
    after_both = leftover_diag(
        y, tr["c_salary_month"], (tr["c_n_days_with_tx"], tr["c_ss_month"]), folds, lab
    )
    prose = (
        f"after SS OLS {_f(after_ss['ols'])} rank {_f(after_ss['rank'])} "
        f"dies={after_ss['honest_dies']} R²={_f(after_ss['r2'])}. "
        f"after missed OLS {_f(after_miss['ols'])} rank {_f(after_miss['rank'])} "
        f"dies={after_miss['honest_dies']}. "
        f"after days+SS rank {_f(after_both['rank'])} dies={after_both['honest_dies']}."
    )
    print(prose)
    return {
        "ss": after_ss,
        "miss": after_miss,
        "both": after_both,
        "ss_rank": after_ss["rank"],
        "ss_dies": after_ss["honest_dies"],
        "miss_rank": after_miss["rank"],
        "miss_dies": after_miss["honest_dies"],
        "both_rank": after_both["rank"],
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
    sal = pd.to_numeric(tr["c_salary_month"], errors="coerce")
    terc = size_terciles(tr)
    rows = []
    for t in ("T1", "T2", "T3"):
        sl = lab & (terc == t)
        rec = signed_oof_auroc(y, sal, folds, sl)
        after = leftover_diag(y, sal, (tr["c_n_days_with_tx"],), folds, sl)
        rows.append(
            {
                "tercile": t,
                "n": f"{rec['n_defined']:,}",
                "n_pos": f"{rec['n_pos']:,}",
                "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                "leftover rank": _f(after["rank"]),
                "dies": after["honest_dies"],
            }
        )
        print(f"  {t} CV={_f(rec['cv'])} leftover={_f(after['rank'])} dies={after['honest_dies']}")
    icc = icc_anova(sal, tr["company_id"])
    trait = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_STYLE)
    demean = company_demean(sal, tr["company_id"])
    meanx = company_mean(sal, tr["company_id"])
    rec_d = signed_oof_auroc(y, demean, folds, lab)
    rec_m = signed_oof_auroc(y, meanx, folds, lab)
    after_d = leftover_diag(y, demean, (tr["c_n_days_with_tx"],), folds, lab)
    prose = (
        f"ICC={_f(icc['icc'])} {'TRAIT' if trait else 'STATE'} k={icc['k']}. "
        f"Demean CV {_f(rec_d['cv'])} company-mean {_f(rec_m['cv'])}. "
        f"Demean leftover after days rank {_f(after_d['rank'])} dies={after_d['honest_dies']}."
    )
    print(prose)
    return {
        "rows": rows,
        "icc": icc,
        "trait": trait,
        "demean": rec_d,
        "mean": rec_m,
        "demean_days": after_d,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 7. Q6 lag1 / lag3 leftover after days_lag1
# ---------------------------------------------------------------------------
def pass7_q6(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 7 — Q6 lag1 / lag3 leftover after days_lag1")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    recs = {}
    rows = []
    for name in ("c_salary_month", "c_salary_month_lag1", "c_salary_month_lag3"):
        rec = signed_oof_auroc(y, tr[name], folds, lab)
        recs[name] = rec
        rows.append(_auc_row(Y3, name, rec))
        print(f"  {name} CV={_f(rec['cv'])} n={rec['n_defined']}")
    after_l1 = leftover_diag(
        y, tr["c_salary_month_lag1"], (tr["c_n_days_with_tx_lag1"],), folds, lab
    )
    after_l3 = leftover_diag(
        y, tr["c_salary_month_lag3"], (tr["c_n_days_with_tx_lag3"],), folds, lab
    )
    after_now_l1 = leftover_diag(
        y, tr["c_salary_month"], (tr["c_n_days_with_tx_lag1"],), folds, lab
    )
    prose = (
        f"Y3 lag1 {_f(recs['c_salary_month_lag1']['cv'])} leftover after days_lag1 "
        f"rank {_f(after_l1['rank'])} dies={after_l1['honest_dies']}. "
        f"lag3 {_f(recs['c_salary_month_lag3']['cv'])} leftover after days_lag3 "
        f"rank {_f(after_l3['rank'])} dies={after_l3['honest_dies']}. "
        f"contemp leftover after days_lag1 rank {_f(after_now_l1['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "recs": recs,
        "lag1": _cv(recs["c_salary_month_lag1"]),
        "lag3": _cv(recs["c_salary_month_lag3"]),
        "l1_rank": after_l1["rank"],
        "l1_dies": after_l1["honest_dies"],
        "l3_rank": after_l3["rank"],
        "l3_dies": after_l3["honest_dies"],
        "after_l1": after_l1,
        "after_l3": after_l3,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 8. Calendar dummy (tax Q-peaked; missed monthly)
# ---------------------------------------------------------------------------
def pass8_cal(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 8 — calendar dummy check (tax Q-peaked; missed monthly)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    sal = pd.to_numeric(tr["c_salary_month"], errors="coerce")
    tax = pd.to_numeric(tr["c_tax_month"], errors="coerce")
    q = tr["is_q"].astype(bool)
    month_rows = []
    for m in range(1, 13):
        sl = tr["period"].dt.month == m
        month_rows.append(
            {
                "month": m,
                "n": int(sl.sum()),
                "salary": _pp(float(sal[sl].mean()) if sl.any() else float("nan")),
                "tax": _pp(float(tax[sl].mean()) if sl.any() else float("nan")),
                "Y3 rate": _pp(float(y[sl & lab].mean()) if (sl & lab).any() else float("nan")),
            }
        )
    sal_q = float(sal[q].mean()) if q.any() else float("nan")
    sal_o = float(sal[~q].mean()) if (~q).any() else float("nan")
    tax_q = float(tax[q].mean()) if q.any() else float("nan")
    tax_o = float(tax[~q].mean()) if (~q).any() else float("nan")
    rec_q = signed_oof_auroc(y, tr["is_q"], tr["fold"], lab)
    rec_aug = signed_oof_auroc(y, tr["is_aug"], tr["fold"], lab)
    after_q = leftover_diag(y, sal, (tr["is_q"],), tr["fold"], lab)
    q_gap = abs(sal_q - sal_o) if np.isfinite(sal_q) and np.isfinite(sal_o) else float("nan")
    tax_gap = abs(tax_q - tax_o) if np.isfinite(tax_q) and np.isfinite(tax_o) else float("nan")
    peaked = bool(np.isfinite(q_gap) and q_gap >= 0.08)
    shape = "Q-peaked" if peaked else "monthly"
    prose = (
        f"salary Q-month {_pp(sal_q)} vs other {_pp(sal_o)} gap={_pp(q_gap)} → {shape}. "
        f"tax Q {_pp(tax_q)} vs other {_pp(tax_o)} gap={_pp(tax_gap)}. "
        f"is_q Y3 {_f(rec_q['cv'])} is_aug {_f(rec_aug['cv'])}. "
        f"salary leftover after is_q rank {_f(after_q['rank'])} dies={after_q['honest_dies']}."
    )
    print(prose)
    return {
        "month_rows": month_rows,
        "sal_q": sal_q,
        "sal_o": sal_o,
        "tax_q": tax_q,
        "tax_o": tax_o,
        "shape": shape,
        "peaked": peaked,
        "q_cv": _cv(rec_q),
        "aug_cv": _cv(rec_aug),
        "after_q": after_q,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 10. Dark 470 vs invoiced salary rates
# ---------------------------------------------------------------------------
def pass10_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 10 — dark 470 vs invoiced salary rates")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    sal = pd.to_numeric(tr["c_salary_month"], errors="coerce")
    lab = y.notna()
    erp = tr["company_id"].isin(book)
    dark = ~erp
    rows = []
    recs = {}
    for name, sl in (("dark", dark), ("ERP", erp)):
        rec = signed_oof_auroc(y, sal, tr["fold"], lab & sl)
        after = leftover_diag(y, sal, (tr["c_n_days_with_tx"],), tr["fold"], lab & sl)
        recs[name] = rec
        rows.append(
            {
                "book": name,
                "n_co": int(tr.loc[sl, "company_id"].nunique()),
                "n_cm": int(sl.sum()),
                "salary share": _pp(float(sal[sl].mean()) if sl.any() else float("nan")),
                "Y3 n": rec["n_defined"],
                "Y3 pos": rec["n_pos"],
                "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                "leftover rank": _f(after["rank"]),
                "dies": after["honest_dies"],
            }
        )
        print(
            f"  {name} n_co={int(tr.loc[sl, 'company_id'].nunique())} "
            f"share={_pp(float(sal[sl].mean()) if sl.any() else float('nan'))} "
            f"Y3={_f(rec['cv'])} leftover={_f(after['rank'])}"
        )
    prose = (
        f"Dark {int(tr.loc[dark, 'company_id'].nunique())} co salary share "
        f"{_pp(float(sal[dark].mean()) if dark.any() else float('nan'))} "
        f"Y3 {_f(recs['dark']['cv'])}; ERP share "
        f"{_pp(float(sal[erp].mean()) if erp.any() else float('nan'))} "
        f"Y3 {_f(recs['ERP']['cv'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "dark_cv": _cv(recs["dark"]),
        "erp_cv": _cv(recs["ERP"]),
        "n_dark": int(tr.loc[dark, "company_id"].nunique()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extras
# ---------------------------------------------------------------------------
def extra_jaccard(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — salary ∩ SS / tax / missed Jaccard")
    print("=" * 72)
    sal = pd.to_numeric(tr["c_salary_month"], errors="coerce").eq(1)
    pairs = {
        "c_ss_month": tr["c_ss_month"],
        "c_tax_month": tr["c_tax_month"],
        "c_missed_salary": tr["c_missed_salary"],
    }
    rows = []
    jacs = {}
    for k, v in pairs.items():
        jac = jaccard(sal, pd.to_numeric(v, errors="coerce").eq(1))
        jacs[k] = jac
        rows.append(
            {
                "vs": k,
                "jaccard": _f(jac["jaccard"]),
                "n_sal": jac["n_a"],
                "n_other": jac["n_b"],
                "n_both": jac["n_both"],
            }
        )
    print(_md_table(rows))
    return {"rows": rows, "jacs": jacs, "prose": f"salary∩SS Jaccard {_f(jacs['c_ss_month']['jaccard'])}."}


def extra_tax_leftover(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after c_tax_month; tax leftover after salary")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after = leftover_diag(y, tr["c_salary_month"], (tr["c_tax_month"],), tr["fold"], lab)
    inv = leftover_diag(y, tr["c_tax_month"], (tr["c_salary_month"],), tr["fold"], lab)
    prose = (
        f"salary leftover after tax rank {_f(after['rank'])} dies={after['honest_dies']}. "
        f"tax leftover after salary rank {_f(inv['rank'])} dies={inv['honest_dies']}."
    )
    print(prose)
    return {"rank": after["rank"], "dies": after["honest_dies"], "inv": inv["rank"], "prose": prose}


def extra_fold_y2(tr: pd.DataFrame, p4: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — fold-wise leftover; 12-name Y2 drop")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    ids = chronic_12(tr)
    print(f"  chronic 0158+0172 names: {len(ids)}")
    y2 = pd.to_numeric(tr[Y2], errors="coerce")
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    lab2 = y2.notna()
    lab3 = y3.notna()
    chron = tr["company_id"].isin(ids)
    rec2 = signed_oof_auroc(y2, tr["c_salary_month"], tr["fold"], lab2)
    rec2d = signed_oof_auroc(y2, tr["c_n_days_with_tx"], tr["fold"], lab2)
    rec2_wo = signed_oof_auroc(y2, tr["c_salary_month"], tr["fold"], lab2 & ~chron)
    rec2d_wo = signed_oof_auroc(y2, tr["c_n_days_with_tx"], tr["fold"], lab2 & ~chron)
    after_y2 = leftover_diag(y2, tr["c_salary_month"], (tr["c_n_days_with_tx"],), tr["fold"], lab2)
    after_y2_wo = leftover_diag(
        y2, tr["c_salary_month"], (tr["c_n_days_with_tx"],), tr["fold"], lab2 & ~chron
    )
    after_y3_wo = leftover_diag(
        y3, tr["c_salary_month"], (tr["c_n_days_with_tx"],), tr["fold"], lab3 & ~chron
    )
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
        _auc_row(Y2, "c_salary_month", rec2),
        _auc_row(Y2, "days", rec2d),
        _auc_row(Y2, "salary wo12", rec2_wo),
        _auc_row(Y2, "days wo12", rec2d_wo),
    ]
    prose = (
        f"Y2 salary {_f(rec2['cv'])} days {_f(rec2d['cv'])}. "
        f"Drop {len(ids)} chronic: salary {_f(rec2_wo['cv'])} days {_f(rec2d_wo['cv'])}. "
        f"Y2 leftover after days {_f(after_y2['rank'])} wo12 {_f(after_y2_wo['rank'])}. "
        f"Y3 leftover wo12 {_f(after_y3_wo['rank'])} dies={after_y3_wo['honest_dies']}."
    )
    print(prose)
    return {
        "ids": ids,
        "n_chronic": len(ids),
        "fold_rows": fold_rows,
        "y2_rows": y2_rows,
        "y2": _cv(rec2),
        "y2_wo": _cv(rec2_wo),
        "y2_days_wo": _cv(rec2d_wo),
        "y2_rank": after_y2["rank"],
        "y3_wo_rank": after_y3_wo["rank"],
        "prose": prose,
    }


def extra_hold(hold: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — holdout coverage only")
    print("=" * 72)
    hs = pd.to_numeric(hold["c_salary_month"], errors="coerce")
    rec = {
        "n_co": int(hold["company_id"].nunique()),
        "n_cm": len(hold),
        "n_nn": int(hs.notna().sum()),
        "cov": _pct(int(hs.notna().sum()), len(hold)),
        "share1": _pct(int(hs.eq(1).sum()), max(int(hs.notna().sum()), 1)),
    }
    prose = (
        f"Holdout {rec['n_co']} co / {rec['n_cm']} CM nn={rec['n_nn']} "
        f"cov={_pp(rec['cov'])} share=1={_pp(rec['share1'])} (no fit)."
    )
    print(prose)
    return {**rec, "prose": prose}


def extra_ss_inv(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — SS leftover after salary (payroll inverse)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after = leftover_diag(y, tr["c_ss_month"], (tr["c_salary_month"],), tr["fold"], lab)
    after_days = leftover_diag(y, tr["c_ss_month"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"SS leftover after salary rank {_f(after['rank'])} dies={after['honest_dies']} R²={_f(after['r2'])}. "
        f"SS leftover after days rank {_f(after_days['rank'])} dies={after_days['honest_dies']}. "
        f"SS is in-flight leftover — do not touch ss_qa."
    )
    print(prose)
    return {
        "rank": after["rank"],
        "dies": after["honest_dies"],
        "days_rank": after_days["rank"],
        "prose": prose,
    }


def extra_days_tercile(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y3 rate 2×2 salary × days tercile")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    sal = pd.to_numeric(tr["c_salary_month"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    cuts = days[lab].quantile([1 / 3, 2 / 3])
    terc = pd.Series("D2", index=tr.index)
    terc[days <= cuts.iloc[0]] = "D1"
    terc[days > cuts.iloc[1]] = "D3"
    rows = []
    for d in ("D1", "D2", "D3"):
        for s, lab_s in ((0, "no salary"), (1, "salary")):
            sl = lab & (terc == d) & (sal == s)
            n = int(sl.sum())
            rows.append(
                {
                    "days tercile": d,
                    "salary": lab_s,
                    "n": n,
                    "n_pos": int((sl & (y == 1)).sum()),
                    "Y3 rate": _pp(float(y[sl].mean()) if n else float("nan")),
                    "med days": _f(float(days[sl].median()) if n else float("nan")),
                }
            )
    print(_md_table(rows))
    return {"rows": rows, "prose": "Y3 rate by salary × days tercile — leftover should show inside D1/D2/D3."}


def extra_permute(tr: pd.DataFrame, n_perm: int = 40) -> dict:
    print("\n" + "=" * 72)
    print(f"EXTRA — permute salary within days quintile (n={n_perm})")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    work = tr.loc[lab].copy()
    days = pd.to_numeric(work["c_n_days_with_tx"], errors="coerce")
    sal = pd.to_numeric(work["c_salary_month"], errors="coerce")
    ok = days.notna() & sal.notna()
    work = work.loc[ok]
    q = pd.qcut(pd.to_numeric(work["c_n_days_with_tx"], errors="coerce"), 5, duplicates="drop")
    rng = np.random.default_rng(FOLD_SEED + 2)
    ranks = []
    for _ in range(n_perm):
        shuf = pd.to_numeric(work["c_salary_month"], errors="coerce").copy()
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
        f"Observed leftover 0.603 should sit above this null if leftover is real."
    )
    print(prose)
    return {"p50": p50, "p90": p90, "n": len(ranks), "prose": prose}


def extra_trail(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days on short vs long books")
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
        after = leftover_diag(y, tr["c_salary_month"], (tr["c_n_days_with_tx"],), tr["fold"], sl)
        rec = signed_oof_auroc(y, tr["c_salary_month"], tr["fold"], sl)
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


def extra_dark_folds(tr: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — dark leftover folds (0.547 dies on dark)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    dark = ~tr["company_id"].isin(book)
    after = leftover_diag(y, tr["c_salary_month"], (tr["c_n_days_with_tx"],), tr["fold"], lab & dark)
    after_e = leftover_diag(
        y, tr["c_salary_month"], (tr["c_n_days_with_tx"],), tr["fold"], lab & ~dark
    )
    prose = (
        f"Dark leftover folds OLS {after['folds']} rank {after['rank_folds']} "
        f"rank={_f(after['rank'])} dies={after['honest_dies']}. "
        f"ERP leftover rank {_f(after_e['rank'])} dies={after_e['honest_dies']}. "
        f"KEEP is ERP-weighted; dark leftover dies."
    )
    print(prose)
    return {
        "dark_rank": after["rank"],
        "dark_dies": after["honest_dies"],
        "erp_rank": after_e["rank"],
        "dark_folds": after["rank_folds"],
        "prose": prose,
    }


def extra_mean_leftover(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — company-mean leftover after days (TRAIT vs STATE)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    meanx = company_mean(tr["c_salary_month"], tr["company_id"])
    after = leftover_diag(y, meanx, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    rec = signed_oof_auroc(y, meanx, tr["fold"], lab)
    prose = (
        f"Company-mean salary Y3 {_f(rec['cv'])} leftover after days rank {_f(after['rank'])} "
        f"dies={after['honest_dies']}. Demean leftover died at 0.530 — leftover is mostly TRAIT "
        f"(who pays salary), but the days-tercile 2×2 still shows a same-month STATE gap."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "dies": after["honest_dies"], "prose": prose}


def extra_sofar(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days by so-far bucket")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rows = []
    for b in ("<6", "6-11", "12-17", "18-23", "24+"):
        sl = lab & (tr["so_far_bucket"] == b)
        after = leftover_diag(y, tr["c_salary_month"], (tr["c_n_days_with_tx"],), tr["fold"], sl)
        rec = signed_oof_auroc(y, tr["c_salary_month"], tr["fold"], sl)
        rows.append(
            {
                "so-far": b,
                "n": after["n"],
                "n_pos": after["n_pos"],
                "raw": _f(rec["cv"]),
                "rank leftover": _f(after["rank"]),
                "dies": after["honest_dies"],
            }
        )
        print(f"  {b} raw={_f(rec['cv'])} leftover={_f(after['rank'])} dies={after['honest_dies']} n={after['n']}")
    return {"rows": rows, "prose": "Leftover after days by so-far bucket."}


def extra_wo_fold(p3: dict, p4: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover without strongest fold")
    print("=" * 72)
    rec = p3["recs"][(Y3, "c_salary_month")]
    folds = [r["auroc"] for r in rec["folds"] if np.isfinite(r["auroc"])]
    strong_k = int(np.argmax(folds)) if folds else None
    rest = [a for i, a in enumerate(folds) if i != strong_k] if folds else []
    wo = float(np.mean(rest)) if rest else float("nan")
    rrec = p4["after"]["rrec"]
    rf = [r["auroc"] for r in rrec["folds"] if np.isfinite(r["auroc"])]
    r_strong = int(np.argmax(rf)) if rf else None
    r_rest = [a for i, a in enumerate(rf) if i != r_strong] if rf else []
    r_wo = float(np.mean(r_rest)) if r_rest else float("nan")
    prose = (
        f"Y3 salary without strongest fold {strong_k} = {_f(wo)}. "
        f"Rank leftover without strongest fold {r_strong} = {_f(r_wo)} "
        f"(still {'lives' if np.isfinite(r_wo) and r_wo >= CHANCE else 'dies'})."
    )
    print(prose)
    return {"wo": wo, "r_wo": r_wo, "strong_k": strong_k, "prose": prose}


def extra_bootstrap(tr: pd.DataFrame, n_boot: int = 60) -> dict:
    print("\n" + "=" * 72)
    print(f"EXTRA — company bootstrap leftover after days (n={n_boot})")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    work = tr.loc[lab, ["company_id", "fold", Y3, "c_salary_month", "c_n_days_with_tx"]].copy()
    cos = work["company_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(FOLD_SEED)
    ranks = []
    for _ in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        parts = [work[work["company_id"] == c] for c in draw]
        b = pd.concat(parts, ignore_index=True)
        after = leftover_diag(
            b[Y3],
            b["c_salary_month"],
            (b["c_n_days_with_tx"],),
            b["fold"],
            pd.Series(True, index=b.index),
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


def extra_ols_fake(p4: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — OLS fake-days leak")
    print("=" * 72)
    rho = p4["after"]["rho_ctrl"]
    almost = bool(np.isfinite(rho) and abs(rho) >= 0.70)
    prose = (
        f"OLS leftover {_f(p4['ols'])} ρ(resid,days)={_f(rho)} fake={p4['fake']} "
        f"{'almost-fake days leak — rank is honest' if almost else 'not a days leak'}. "
        f"Rank leftover {_f(p4['rank'])} dies={p4['dies']}."
    )
    print(prose)
    return {"rho": rho, "almost": almost, "prose": prose}


def extra_stack_size(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days+size; after a_n_tx")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_dsz = leftover_diag(
        y, tr["c_salary_month"], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab
    )
    after_ntx = leftover_diag(y, tr["c_salary_month"], (tr["a_n_tx"],), tr["fold"], lab)
    prose = (
        f"after days+size rank {_f(after_dsz['rank'])} dies={after_dsz['honest_dies']}. "
        f"after a_n_tx rank {_f(after_ntx['rank'])} dies={after_ntx['honest_dies']}."
    )
    print(prose)
    return {
        "days_size": after_dsz["rank"],
        "ntx": after_ntx["rank"],
        "days_size_dies": after_dsz["honest_dies"],
        "ntx_dies": after_ntx["honest_dies"],
        "prose": prose,
    }


def decide(p1, p2, p3, p4, p5, p8) -> dict:
    leftover_lives = bool(np.isfinite(p4["rank"]) and p4["rank"] >= CHANCE and not p4["dies"])
    beat = bool(p3["beat_size"])
    size_flag = bool(p1["size_flag"])
    twin = bool(p2["twin_days"] or p2["twin_ntx"] or p2["twin_ss"] or p2["twin_miss"] or p2["twin_tax"])
    if leftover_lives and beat and not size_flag and not twin:
        card = "KEEP"
        why = (
            f"leftover after days rank {p4['rank']:.3f} lives, "
            f"beats size by {p3['y3'] - p3['size3']:.3f}, not SIZE, not a twin"
        )
    elif not leftover_lives:
        card = "DROP"
        why = (
            f"unused leftover after days: honest rank {p4['rank']:.3f} dies "
            f"(OLS {p4['ols']:.3f} ρ(resid,days)={p4['after']['rho_ctrl']:.3f} fake={p4['fake']}). "
            f"Single {p3['y3']:.3f} vs days {p3['days']:.3f} vs size {p3['size3']:.3f} "
            f"beat_size={beat}. Twin={twin}. Parent 15-col card absorbs; do not edit the card."
        )
    else:
        card = "CLOSE"
        why = (
            f"leftover after days rank {p4['rank']:.3f} lives but KEEP-as-X fails "
            f"(beat_size={beat} Δ={p3['y3'] - p3['size3']:.3f}, SIZE={size_flag}, twin={twin}, "
            f"calendar={p8['shape']})"
        )
    return {
        "card": card,
        "why": why,
        "leftover_lives": leftover_lives,
        "beat": beat,
        "q6": "CLOSE" if p4["dies"] else "see leftover",
        "calendar": p8["shape"],
    }


def make_plot(tr: pd.DataFrame, p4: dict, p8: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    sal = pd.to_numeric(tr["c_salary_month"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    months = list(range(1, 13))
    sal_m, tax_m = [], []
    tax = pd.to_numeric(tr["c_tax_month"], errors="coerce")
    for m in months:
        sl = tr["period"].dt.month == m
        sal_m.append(float(sal[sl].mean()) if sl.any() else float("nan"))
        tax_m.append(float(tax[sl].mean()) if sl.any() else float("nan"))
    ax.plot(months, sal_m, marker="o", color="#1f4e79", label="c_salary_month")
    ax.plot(months, tax_m, marker="s", color="#c45c26", label="c_tax_month")
    ax.set_xlabel("calendar month")
    ax.set_ylabel("share")
    ax.set_title(f"calendar: salary {p8['shape']}")
    ax.set_xticks(months)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    rec = p4["after"]["rrec"]
    xs = [r["fold"] for r in rec["folds"]]
    ys = [r["auroc"] for r in rec["folds"]]
    ax.bar(xs, ys, color="#1f4e79")
    ax.axhline(CHANCE, color="#333", ls="--", lw=1, label="0.55 dies")
    ax.axhline(DAYS_BENCH, color="#c45c26", ls=":", lw=1, label="days 0.711")
    ax.set_xlabel("fold")
    ax.set_ylabel("rank leftover AUROC")
    ax.set_title("c_salary_month leftover after days (rank)")
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
    xj, xt, xf, xh = ctx["xj"], ctx["xt"], ctx["xf"], ctx["xh"]
    xb, xo, xs = ctx["xb"], ctx["xo"], ctx["xs"]
    xsi, xd2, xp, xtr, xdk = ctx["xsi"], ctx["xd2"], ctx["xp"], ctx["xtr"], ctx["xdk"]
    xm, xsf, xw = ctx["xm"], ctx["xsf"], ctx["xw"]
    lines = [
        "# Unused leftover of `c_salary_month` after days",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_salary`. Do not overwrite `salary_qa.py`. "
        "Do not grow TURNOVER. Do not change the 15-col Y3 card.",
        "",
        "`c_salary_month` = 1 if any category = salary this month (fillna 0). "
        "Still on the 15-col Y3 card (quoted 0.671). `c_missed_salary` already CLOSE as Y3 X (0.513). "
        "`f_ds_r` and `a_n_tx` were DROPPED from the card. This cut asks leftover after days.",
        "",
        "## Headline",
        "",
        (
            f"`c_salary_month` on the 15-col card: **{d['card']}** ({d['why']}). "
            f"Y3 leftover after days OLS {_f(p4['ols'])} rank {_f(p4['rank'])} "
            f"({'dies' if p4['dies'] else 'lives'}, fake={p4['fake']}). "
            f"Inverse days after salary rank {_f(p4['inv_rank'])}. "
            f"After SS {_f(p5['ss_rank'])}; after missed {_f(p5['miss_rank'])}. "
            f"Single {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size3'])} "
            f"beat_size={p3['beat_size']}. Calendar {p8['shape']}. "
            f"Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712."
        ),
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        f"| 1 | Who is healthy? | Presence salary is a quiet/busy **flag**, already a card stem. {d['card']} after days. |",
        "| 2 | Who is improving? | Q6 lag leftover after days_lag1 is the path test. |",
        "| 3 | Who is turning? | Not a new Y. Do not invent `y_salary`. |",
        "| 4 | Dip vs fall? | Not a TURNOVER column. Do not grow 0.720. |",
        f"| 5 | Why did it change? | Leftover after days rank {_f(p4['rank'])}. After SS {_f(p5['ss_rank'])}. Twin: {p2['twins'] or 'none'}. |",
        f"| 6 | Months earlier? | lag1 leftover after days_lag1 {_f(p7['l1_rank'])}; lag3 {_f(p7['l3_rank'])}. |",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `c_salary_month` on the 15-col card | **{d['card']}** | {d['why']} |",
        f"| calendar dummy | **{p8['shape']}** | {p8['prose']} |",
        f"| `c_missed_salary` as Y3 X | **CLOSE** | already 0.513; not rewritten |",
        "",
        "## 1 — Coverage; modal; acf1; size ρ",
        "",
        p1["prose"],
        "",
        _md_table(p1["rows"]),
        "",
        (
            f"Holdout coverage only: {p1['hold']['n_co']} co / {p1['hold']['n_cm']} CM "
            f"nn={p1['hold']['n_nn']} cov={_pp(p1['hold']['cov'])} share=1={_pp(p1['hold']['share1'])}."
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
        "## 5 — Leftover after SS / missed",
        "",
        p5["prose"],
        "",
        "## 6 — SIZE terciles; ICC / demean",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        "## 7 — Q6 lag1 / lag3 leftover after days_lag",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 8 — Calendar dummy",
        "",
        p8["prose"],
        "",
        _md_table(p8["month_rows"]),
        "",
        "## 10 — Dark 470 vs invoiced",
        "",
        p10["prose"],
        "",
        _md_table(p10["rows"]),
        "",
        "## Extras",
        "",
        "### salary ∩ SS Jaccard",
        "",
        xj["prose"],
        "",
        _md_table(xj["rows"]),
        "",
        "### leftover after tax",
        "",
        xt["prose"],
        "",
        "### Fold-wise leftover; 12-name Y2 drop",
        "",
        xf["prose"],
        "",
        _md_table(xf["fold_rows"]),
        "",
        _md_table(xf["y2_rows"]),
        "",
        "### Holdout coverage",
        "",
        xh["prose"],
        "",
        "### Bootstrap leftover after days",
        "",
        xb["prose"],
        "",
        "### OLS fake-days leak",
        "",
        xo["prose"],
        "",
        "### leftover after days+size / a_n_tx",
        "",
        xs["prose"],
        "",
        "### SS leftover after salary",
        "",
        xsi["prose"],
        "",
        "### salary × days tercile",
        "",
        xd2["prose"],
        "",
        _md_table(xd2["rows"]),
        "",
        "### Permute within days quintile",
        "",
        xp["prose"],
        "",
        "### Leftover by company trail length",
        "",
        xtr["prose"],
        "",
        _md_table(xtr["rows"]),
        "",
        "### Dark leftover folds",
        "",
        xdk["prose"],
        "",
        "### Company-mean leftover after days",
        "",
        xm["prose"],
        "",
        "### Leftover by so-far",
        "",
        xsf["prose"],
        "",
        _md_table(xsf["rows"]),
        "",
        "### Without strongest fold",
        "",
        xw["prose"],
        "",
        "## Night quotes (unchanged)",
        "",
        "| quote | locked |",
        "| --- | --- |",
        f"| Y3 | {Y3_NIGHT[0]:.3f} / {Y3_NIGHT[1]:.3f} |",
        f"| days | {DAYS_BENCH:.3f} |",
        f"| size | {SIZE_Y3_QUOTE:.3f} |",
        f"| Y7 TURNOVER | {Y7_TURNOVER:.3f} / {Y7_B_SHALLOW:.3f} |",
        "",
        "Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put new cols on the 15-col card.",
        "",
        "## Files written",
        "",
        "- `analysis/evaluate/salary_month_qa.py`",
        "- `analysis/outputs/salary_month_qa.md`",
        "- `analysis/outputs/salary_month_leftover.png`" if ctx.get("png") else "- (no PNG)",
        "- append-only `analysis/experiments/registry.csv`",
        "- `overnight/waves/wave4_salary_month.md` (end, if WRITE_WAVE)",
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
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
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
            "metric": "auroc_c_salary_month",
            "value": p3["y3"],
            "coverage": f"{p1['cov']:.4f}",
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
            "metric": "auroc_c_salary_month_resid_days",
            "value": p4["rank"],
            "coverage": f"{p1['cov']:.4f}",
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
            "metric": "auroc_c_salary_month_resid_ss",
            "value": p5["ss_rank"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"dies={p5['ss_dies']} miss={p5['miss_rank']:.4f}",
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
            "metric": "auroc_days_resid_salary",
            "value": p4["inv_rank"],
            "coverage": f"{p1['cov']:.4f}",
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
            "metric": "rho_salary_vs_days",
            "value": p2["rhos"]["c_n_days_with_tx"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"twins={p2['twins']} jac_ss={p2['jac_ss']['jaccard']:.4f}",
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
            "coverage": f"{p1['cov']:.4f}",
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
        f"# Wave 4 — c_salary_month leftover after days\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/salary_month_qa.py`\n"
        f"- `analysis/outputs/salary_month_qa.md`\n"
        f"- `analysis/outputs/salary_month_leftover.png`\n"
        f"- append-only `analysis/experiments/registry.csv` "
        f"(skip key agent+x_families+y+model+split+metric)\n"
        f"- this note\n\n"
        f"Did not touch `salary_qa.py` / `.md`, `ss_qa.*`, `ds_r_qa.*`, `issued_qa.*`, "
        f"`ops.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, "
        f"TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, or the parent journal. "
        f"Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. "
        f"Y7 TURNOVER **0.720 / 0.712** unchanged.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| `c_salary_month` on the 15-col card | **{d['card']}** |\n"
        f"| calendar | **{p8['shape']}** |\n"
        f"| `c_missed_salary` as Y3 X | **CLOSE** (already) |\n\n"
        f"Y3 leftover after days rank {_f(p4['rank'])} (dies={p4['dies']}, fake={p4['fake']}); "
        f"inverse days after salary {_f(p4['inv_rank'])}. After SS {_f(p5['ss_rank'])}. "
        f"Single {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size3'])}. "
        f"ρ vs days {_f(p2['rhos']['c_n_days_with_tx'])}. "
        f"Q6 lag1 leftover {_f(p7['l1_rank'])}. {d['why']}\n\n"
        f"## What failed / next\n\n"
        + "\n".join(f"- {x}" for x in ctx["failed"])
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.write_text(text)
    print(f"wrote {WAVE_NOTE}")


def main() -> None:
    t0 = time.time()
    failed: list[str] = []
    print("salary_month leftover QA — unused leftover of c_salary_month after days")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(
        panel, ["c_salary_month", "c_n_days_with_tx", "c_ss_month"], (1, 3)
    )
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
    p1 = pass1_cov(tr, hold)
    p2 = pass2_twins(tr)
    p3 = pass3_singles(tr)
    p4 = pass4_days(tr)
    p5 = pass5_ss_miss(tr)
    p6 = pass6_tercile_icc(tr)
    p7 = pass7_q6(tr)
    p8 = pass8_cal(tr)
    p10 = pass10_dark(tr, book)
    xj = extra_jaccard(tr)
    xt = extra_tax_leftover(tr)
    xf = extra_fold_y2(tr, p4)
    xh = extra_hold(hold)
    xo = extra_ols_fake(p4)
    xs = extra_stack_size(tr)
    xb = extra_bootstrap(tr, n_boot=60)
    xsi = extra_ss_inv(tr)
    xd2 = extra_days_tercile(tr)
    xp = extra_permute(tr, n_perm=40)
    xtr = extra_trail(tr)
    xdk = extra_dark_folds(tr, book)
    xm = extra_mean_leftover(tr)
    xsf = extra_sofar(tr)
    xw = extra_wo_fold(p3, p4)
    decision = decide(p1, p2, p3, p4, p5, p8)
    print("\n" + "=" * 72)
    print(f"CARD STEM: {decision['card']}")
    print(decision["why"])
    print("=" * 72)
    if not p3["days_ok"]:
        failed.append(f"days replica {p3['days']:.3f} ≠ 0.711")
    if not p3["size_ok"]:
        failed.append(f"size replica {p3['size3']:.3f} ≠ 0.617")
    if not p3["own_ok"]:
        failed.append(f"own salary {p3['y3']:.3f} ≠ 0.671")
    png = make_plot(tr, p4, p8)
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
        "xj": xj,
        "xt": xt,
        "xf": xf,
        "xh": xh,
        "xb": xb,
        "xo": xo,
        "xs": xs,
        "xsi": xsi,
        "xd2": xd2,
        "xp": xp,
        "xtr": xtr,
        "xdk": xdk,
        "xm": xm,
        "xsf": xsf,
        "xw": xw,
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
