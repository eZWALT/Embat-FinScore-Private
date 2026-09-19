"""Unused leftover of ``a_in3`` after days as the 44-col size control.

``log1p(a_in3)`` is the night size bar (quoted 0.617). ``c_salary_month``
was KEPT on the 15-col card (leftover after days 0.603). This lane asks
leftover of size after ``c_n_days_with_tx``: KEEP as size control only,
CLOSE as unused leftover / DROP from the 44 as engine X, or leftover
that lives as more than a control.

It *is* the size stem (|ρ| vs log1p(a_in3) = 1). KEEP-as-X (engine):
beat size ≥0.02 **and** leftover after days **and** not SIZE **and**
not a twin (|ρ|≥0.80 vs days / ``a_op_in`` / ``a_in6`` / ``a_in12``).
Leftover <0.55 dies. Rank leftover is honest.

Do **not** put ``a_in3`` on the 15-col card. Do not rewrite
``gbm_core.py``. Do not overwrite ``salary_month_qa.*``.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.in3_qa

Owned: analysis/evaluate/in3_qa.py, analysis/outputs/in3_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_in3.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "in3_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "in3_leftover.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_in3.md"
AGENT = "572fb928"
WAVE = "4"
ROUND = "R4"
MODEL = "in3_qa"
X_FAM = "A"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_Y3_QUOTE = 0.617
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
WRITE_WAVE = True

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "a_op_in",
    "a_in6",
    "a_in12",
    "c_n_days_with_tx",
    "c_salary_month",
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
    ain = pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0)
    aop = pd.to_numeric(panel["a_op_in"], errors="coerce").clip(lower=0)
    panel["log_in3"] = np.log1p(ain)
    panel["log_opin"] = np.log1p(aop)
    panel["log_in6"] = np.log1p(pd.to_numeric(panel["a_in6"], errors="coerce").clip(lower=0))
    panel["log_in12"] = np.log1p(pd.to_numeric(panel["a_in12"], errors="coerce").clip(lower=0))
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
    leak3 = leakage_check(
        ["log_in3", "a_in3", "c_n_days_with_tx", "a_op_in"],
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
# 1. Coverage; acf1; SIZE ρ
# ---------------------------------------------------------------------------
def pass1_cov(tr: pd.DataFrame, hold: pd.DataFrame | None = None) -> dict:
    print("\n" + "=" * 72)
    print("CUT 1 — coverage; acf1 a_in3 vs a_op_in; SIZE ρ")
    print("=" * 72)
    ain = pd.to_numeric(tr["a_in3"], errors="coerce")
    aop = pd.to_numeric(tr["a_op_in"], errors="coerce")
    log = tr["log_in3"]
    n_cm = len(tr)
    n_co = int(tr["company_id"].nunique())
    acf_in3 = median_acf(ain, tr["company_id"], 1)
    acf_op = median_acf(aop, tr["company_id"], 1)
    acf_log = median_acf(log, tr["company_id"], 1)
    pairs = {
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_op_in": aop,
        "log_opin": tr["log_opin"],
        "a_in6": tr["a_in6"],
        "a_in12": tr["a_in12"],
        "log_in6": tr["log_in6"],
        "log_in12": tr["log_in12"],
        "log_in3": log,
    }
    rhos = {k: spearman(log, v) for k, v in pairs.items()}
    twins = [k for k, v in rhos.items() if k != "log_in3" and np.isfinite(v) and abs(v) >= TWIN_RHO]
    rows = [
        {
            "col": "a_in3",
            "n_nn": f"{int(ain.notna().sum()):,}",
            "cov": _pp(_pct(int(ain.notna().sum()), n_cm)),
            "acf1": _f(acf_in3),
        },
        {
            "col": "a_op_in",
            "n_nn": f"{int(aop.notna().sum()):,}",
            "cov": _pp(_pct(int(aop.notna().sum()), n_cm)),
            "acf1": _f(acf_op),
        },
        {
            "col": "log1p(a_in3)",
            "n_nn": f"{int(log.notna().sum()):,}",
            "cov": _pp(_pct(int(log.notna().sum()), n_cm)),
            "acf1": _f(acf_log),
        },
    ]
    rho_rows = [{"vs": k, "rho": _f(v), "twin": "yes" if abs(v) >= TWIN_RHO and k != "log_in3" else "self" if k == "log_in3" else "no"} for k, v in rhos.items()]
    hold_row = None
    if hold is not None and len(hold):
        h = pd.to_numeric(hold["a_in3"], errors="coerce")
        hold_row = {
            "n_co": int(hold["company_id"].nunique()),
            "n_cm": len(hold),
            "n_nn": int(h.notna().sum()),
            "cov": _pct(int(h.notna().sum()), len(hold)),
        }
    persist = bool(np.isfinite(acf_in3) and acf_in3 >= 0.50)
    persist_vs_op = bool(
        persist and (not np.isfinite(acf_op) or acf_in3 > acf_op + 0.3)
    )
    persist_tag = "CONFIRM persist > a_op_in" if persist_vs_op else "check persist"
    prose = (
        f"Train {n_co:,} co / {n_cm:,} CM. a_in3 cov {_pp(_pct(int(ain.notna().sum()), n_cm))} "
        f"acf1={_f(acf_in3)} vs a_op_in acf1={_f(acf_op)} ({persist_tag}). "
        f"ρ log_in3 vs days {_f(rhos['c_n_days_with_tx'])} vs a_op_in {_f(rhos['a_op_in'])} "
        f"vs a_in6 {_f(rhos['a_in6'])} vs a_in12 {_f(rhos['a_in12'])}. Twins: {twins or 'none'}."
    )
    print(prose)
    return {
        "rows": rows,
        "rho_rows": rho_rows,
        "rhos": rhos,
        "twins": twins,
        "n_cm": n_cm,
        "n_co": n_co,
        "cov": _pct(int(ain.notna().sum()), n_cm),
        "acf_in3": acf_in3,
        "acf_op": acf_op,
        "acf_log": acf_log,
        "twin_days": abs(rhos["c_n_days_with_tx"]) >= TWIN_RHO if np.isfinite(rhos["c_n_days_with_tx"]) else False,
        "twin_opin": abs(rhos["a_op_in"]) >= TWIN_RHO if np.isfinite(rhos["a_op_in"]) else False,
        "twin_in6": abs(rhos["a_in6"]) >= TWIN_RHO if np.isfinite(rhos["a_in6"]) else False,
        "twin_in12": abs(rhos["a_in12"]) >= TWIN_RHO if np.isfinite(rhos["a_in12"]) else False,
        "hold": hold_row,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 2. Single-feature group-fold Y3
# ---------------------------------------------------------------------------
def pass2_singles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 2 — single-feature group-fold Y3")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    feats = {
        "log1p(a_in3)": tr["log_in3"],
        "a_in3": tr["a_in3"],
        "a_op_in": tr["a_op_in"],
        "log1p(a_op_in)": tr["log_opin"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "log1p(a_in6)": tr["log_in6"],
        "log1p(a_in12)": tr["log_in12"],
        "c_salary_month": tr["c_salary_month"],
    }
    recs = {}
    rows = []
    for fname, x in feats.items():
        rec = signed_oof_auroc(y, x, folds, lab)
        recs[fname] = rec
        rows.append(_auc_row(Y3, fname, rec))
        print(f"  {fname} CV={_f(rec['cv'])} folds={fold_bits(rec)}")
    y3_size = _cv(recs["log1p(a_in3)"])
    y3_days = _cv(recs["c_n_days_with_tx"])
    y3_raw = _cv(recs["a_in3"])
    y3_op = _cv(recs["a_op_in"])
    days_ok = bool(np.isfinite(y3_days) and abs(y3_days - DAYS_BENCH) < 0.008)
    size_ok = bool(np.isfinite(y3_size) and abs(y3_size - SIZE_Y3_QUOTE) < 0.008)
    beat_size = bool(np.isfinite(y3_size) and np.isfinite(y3_size) and (y3_size - y3_size) >= KEEP_DELTA)
    prose = (
        f"Y3 log1p(a_in3) {_f(y3_size)} vs days {_f(y3_days)} vs raw a_in3 {_f(y3_raw)} "
        f"vs a_op_in {_f(y3_op)}. Replica days 0.711 {'CONFIRM' if days_ok else 'DRIFT'} / "
        f"size 0.617 {'CONFIRM' if size_ok else 'DRIFT'}. "
        f"Engine beat-size vs itself: False (it IS the size bar)."
    )
    print(prose)
    return {
        "rows": rows,
        "recs": recs,
        "size3": y3_size,
        "days": y3_days,
        "raw": y3_raw,
        "opin": y3_op,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "beat_size": False,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 3. Honest leftover after days + inverse
# ---------------------------------------------------------------------------
def pass3_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 3 — honest leftover of log1p(a_in3) after days; inverse")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    after = leftover_diag(y, tr["log_in3"], (tr["c_n_days_with_tx"],), folds, lab)
    inv = leftover_diag(y, tr["c_n_days_with_tx"], (tr["log_in3"],), folds, lab)
    prose = (
        f"log1p(a_in3) leftover after days OLS {_f(after['ols'])} rank {_f(after['rank'])} "
        f"fake={after['fake']} ρ(resid,days)={_f(after['rho_ctrl'])} R²={_f(after['r2'])} "
        f"honest_dies={after['honest_dies']}. "
        f"Inverse: days leftover after size OLS {_f(inv['ols'])} rank {_f(inv['rank'])} "
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
# 4. Same-n leftover of a_in3 vs a_op_in vs log1p
# ---------------------------------------------------------------------------
def pass4_samen(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 4 — same-n leftover of a_in3 vs a_op_in vs log1p(a_in3)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    ok = (
        y.notna()
        & pd.to_numeric(tr["a_in3"], errors="coerce").notna()
        & pd.to_numeric(tr["a_op_in"], errors="coerce").notna()
        & tr["log_in3"].notna()
        & pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce").notna()
    )
    folds = tr["fold"]
    rows = []
    recs = {}
    leftover = {}
    for name, x in (
        ("log1p(a_in3)", tr["log_in3"]),
        ("a_in3", tr["a_in3"]),
        ("a_op_in", tr["a_op_in"]),
        ("log1p(a_op_in)", tr["log_opin"]),
    ):
        rec = signed_oof_auroc(y, x, folds, ok)
        after = leftover_diag(y, x, (tr["c_n_days_with_tx"],), folds, ok)
        recs[name] = rec
        leftover[name] = after
        rows.append(
            {
                "feature": name,
                "n": rec["n_defined"],
                "n_pos": rec["n_pos"],
                "CV": _f(rec["cv"]),
                "leftover rank": _f(after["rank"]),
                "leftover OLS": _f(after["ols"]),
                "dies": after["honest_dies"],
            }
        )
        print(f"  {name} CV={_f(rec['cv'])} leftover rank={_f(after['rank'])} dies={after['honest_dies']}")
    after_raw_log = leftover_diag(y, tr["a_in3"], (tr["log_in3"],), folds, ok)
    after_op_log = leftover_diag(y, tr["a_op_in"], (tr["log_in3"],), folds, ok)
    prose = (
        f"same-n n={int(ok.sum())}. log leftover {_f(leftover['log1p(a_in3)']['rank'])}; "
        f"raw a_in3 leftover {_f(leftover['a_in3']['rank'])}; "
        f"a_op_in leftover {_f(leftover['a_op_in']['rank'])}. "
        f"raw leftover after log1p {_f(after_raw_log['rank'])} dies={after_raw_log['honest_dies']}; "
        f"a_op_in leftover after log1p {_f(after_op_log['rank'])} dies={after_op_log['honest_dies']}."
    )
    print(prose)
    return {
        "rows": rows,
        "leftover": leftover,
        "log_rank": leftover["log1p(a_in3)"]["rank"],
        "raw_rank": leftover["a_in3"]["rank"],
        "op_rank": leftover["a_op_in"]["rank"],
        "raw_after_log": after_raw_log["rank"],
        "op_after_log": after_op_log["rank"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 5. SIZE terciles — leftover inside T1
# ---------------------------------------------------------------------------
def pass5_tercile(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 5 — SIZE terciles; leftover inside T1 (inverse-size recover)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    terc = size_terciles(tr)
    rows = []
    t1 = None
    for t in ("T1", "T2", "T3"):
        sl = lab & (terc == t)
        rec = signed_oof_auroc(y, tr["log_in3"], folds, sl)
        after = leftover_diag(y, tr["log_in3"], (tr["c_n_days_with_tx"],), folds, sl)
        days = signed_oof_auroc(y, tr["c_n_days_with_tx"], folds, sl)
        row = {
            "tercile": t,
            "n": rec["n_defined"],
            "n_pos": rec["n_pos"],
            "size CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
            "days CV": "LOW_POWER" if days["low_power"] else _f(days["cv"]),
            "leftover rank": _f(after["rank"]),
            "dies": after["honest_dies"],
            "Y3 rate": _pp(float(y[sl].mean()) if sl.any() else float("nan")),
        }
        rows.append(row)
        if t == "T1":
            t1 = after
        print(f"  {t} size={_f(rec['cv'])} days={_f(days['cv'])} leftover={_f(after['rank'])} dies={after['honest_dies']}")
    prose = (
        f"T1 leftover after days rank {_f(t1['rank']) if t1 else '—'} "
        f"dies={t1['honest_dies'] if t1 else '—'}. Inverse-size recover lives inside T1 only if leftover lives there."
    )
    print(prose)
    return {"rows": rows, "t1": t1, "t1_rank": t1["rank"] if t1 else float("nan"), "t1_dies": t1["honest_dies"] if t1 else True, "prose": prose}


# ---------------------------------------------------------------------------
# 6. ICC / demean
# ---------------------------------------------------------------------------
def pass6_icc(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 6 — ICC / demean (is size a company trait?)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    log = tr["log_in3"]
    icc = icc_anova(log, tr["company_id"])
    trait = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_STYLE)
    demean = company_demean(log, tr["company_id"])
    meanx = company_mean(log, tr["company_id"])
    rec_d = signed_oof_auroc(y, demean, folds, lab)
    rec_m = signed_oof_auroc(y, meanx, folds, lab)
    after_d = leftover_diag(y, demean, (tr["c_n_days_with_tx"],), folds, lab)
    after_m = leftover_diag(y, meanx, (tr["c_n_days_with_tx"],), folds, lab)
    prose = (
        f"ICC={_f(icc['icc'])} {'TRAIT' if trait else 'STATE'} k={icc['k']}. "
        f"Demean CV {_f(rec_d['cv'])} leftover-days {_f(after_d['rank'])} dies={after_d['honest_dies']}. "
        f"Company-mean CV {_f(rec_m['cv'])} leftover-days {_f(after_m['rank'])} dies={after_m['honest_dies']}."
    )
    print(prose)
    return {
        "icc": icc,
        "trait": trait,
        "demean": rec_d,
        "mean": rec_m,
        "demean_rank": after_d["rank"],
        "mean_rank": after_m["rank"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 7. Q6 lag1 leftover after days_lag1
# ---------------------------------------------------------------------------
def pass7_q6(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 7 — Q6 log1p(a_in3) lag1 leftover after days_lag1")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    recs = {}
    rows = []
    for name in ("log_in3", "log_in3_lag1", "log_in3_lag3"):
        rec = signed_oof_auroc(y, tr[name], folds, lab)
        recs[name] = rec
        rows.append(_auc_row(Y3, name, rec))
        print(f"  {name} CV={_f(rec['cv'])} n={rec['n_defined']}")
    after_l1 = leftover_diag(y, tr["log_in3_lag1"], (tr["c_n_days_with_tx_lag1"],), folds, lab)
    after_l3 = leftover_diag(y, tr["log_in3_lag3"], (tr["c_n_days_with_tx_lag3"],), folds, lab)
    prose = (
        f"Y3 log_in3_lag1 {_f(recs['log_in3_lag1']['cv'])} leftover after days_lag1 "
        f"rank {_f(after_l1['rank'])} dies={after_l1['honest_dies']}. "
        f"lag3 {_f(recs['log_in3_lag3']['cv'])} leftover after days_lag3 "
        f"rank {_f(after_l3['rank'])} dies={after_l3['honest_dies']}. "
        f"a_growth_12 already CLOSE as Q6 — not rewritten."
    )
    print(prose)
    return {
        "rows": rows,
        "recs": recs,
        "lag1": _cv(recs["log_in3_lag1"]),
        "lag3": _cv(recs["log_in3_lag3"]),
        "l1_rank": after_l1["rank"],
        "l1_dies": after_l1["honest_dies"],
        "l3_rank": after_l3["rank"],
        "l3_dies": after_l3["honest_dies"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 9. Holdout coverage only
# ---------------------------------------------------------------------------
def pass9_hold(hold: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 9 — holdout coverage only (no AUROC)")
    print("=" * 72)
    h = pd.to_numeric(hold["a_in3"], errors="coerce")
    rec = {
        "n_co": int(hold["company_id"].nunique()),
        "n_cm": len(hold),
        "n_nn": int(h.notna().sum()),
        "cov": _pct(int(h.notna().sum()), len(hold)),
        "p50": float(h[h.notna()].median()) if h.notna().any() else float("nan"),
    }
    prose = (
        f"Holdout {rec['n_co']} co / {rec['n_cm']} CM nn={rec['n_nn']} "
        f"cov={_pp(rec['cov'])} p50={_f(rec['p50'], 0)} (no fit, no AUROC)."
    )
    print(prose)
    return {**rec, "prose": prose}


# ---------------------------------------------------------------------------
# 10. Dark 470 vs ERP
# ---------------------------------------------------------------------------
def pass10_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 10 — dark 470 vs ERP size leftover")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    erp = tr["company_id"].isin(book)
    dark = ~erp
    rows = []
    recs = {}
    for name, sl in (("dark", dark), ("ERP", erp)):
        rec = signed_oof_auroc(y, tr["log_in3"], tr["fold"], lab & sl)
        after = leftover_diag(y, tr["log_in3"], (tr["c_n_days_with_tx"],), tr["fold"], lab & sl)
        recs[name] = rec
        rows.append(
            {
                "book": name,
                "n_co": int(tr.loc[sl, "company_id"].nunique()),
                "n_cm": int(sl.sum()),
                "Y3 n": rec["n_defined"],
                "Y3 pos": rec["n_pos"],
                "size CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                "leftover rank": _f(after["rank"]),
                "dies": after["honest_dies"],
            }
        )
        print(f"  {name} size={_f(rec['cv'])} leftover={_f(after['rank'])} dies={after['honest_dies']}")
    prose = (
        f"Dark {int(tr.loc[dark, 'company_id'].nunique())} co size Y3 {_f(recs['dark']['cv'])}; "
        f"ERP {_f(recs['ERP']['cv'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "dark_cv": _cv(recs["dark"]),
        "erp_cv": _cv(recs["ERP"]),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extras
# ---------------------------------------------------------------------------
def extra_twins6_12(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — a_in6 / a_in12 leftover after log_in3 and after days")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rows = []
    leftover = {}
    for name, x in (("log_in6", tr["log_in6"]), ("log_in12", tr["log_in12"])):
        rec = signed_oof_auroc(y, x, tr["fold"], lab)
        after_log = leftover_diag(y, x, (tr["log_in3"],), tr["fold"], lab)
        after_days = leftover_diag(y, x, (tr["c_n_days_with_tx"],), tr["fold"], lab)
        leftover[name] = after_log
        rows.append(
            {
                "feature": name,
                "CV": _f(rec["cv"]),
                "leftover after log_in3": _f(after_log["rank"]),
                "dies vs size": after_log["honest_dies"],
                "leftover after days": _f(after_days["rank"]),
                "dies vs days": after_days["honest_dies"],
            }
        )
        print(f"  {name} CV={_f(rec['cv'])} after size {_f(after_log['rank'])} after days {_f(after_days['rank'])}")
    prose = (
        f"a_in6 leftover after log_in3 {_f(leftover['log_in6']['rank'])} "
        f"dies={leftover['log_in6']['honest_dies']}; "
        f"a_in12 {_f(leftover['log_in12']['rank'])} dies={leftover['log_in12']['honest_dies']}. "
        f"Do not add a_in6/a_in12 besides a_in3."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_salary(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover vs salary (do not overwrite salary_month_qa)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_sal = leftover_diag(y, tr["log_in3"], (tr["c_salary_month"],), tr["fold"], lab)
    after_both = leftover_diag(
        y, tr["log_in3"], (tr["c_n_days_with_tx"], tr["c_salary_month"]), tr["fold"], lab
    )
    sal_after = leftover_diag(y, tr["c_salary_month"], (tr["log_in3"],), tr["fold"], lab)
    prose = (
        f"size leftover after salary rank {_f(after_sal['rank'])} dies={after_sal['honest_dies']}. "
        f"size leftover after days+salary {_f(after_both['rank'])} dies={after_both['honest_dies']}. "
        f"salary leftover after size {_f(sal_after['rank'])} dies={sal_after['honest_dies']} "
        f"(salary_month_qa leftover after days was 0.603 — not overwritten)."
    )
    print(prose)
    return {
        "after_sal": after_sal["rank"],
        "after_both": after_both["rank"],
        "sal_after": sal_after["rank"],
        "prose": prose,
    }


def extra_fold_y2(tr: pd.DataFrame, p3: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — fold-wise leftover; 12-name Y2 drop")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    ids = chronic_12(tr)
    y2 = pd.to_numeric(tr[Y2], errors="coerce")
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    lab2 = y2.notna()
    lab3 = y3.notna()
    chron = tr["company_id"].isin(ids)
    rec2 = signed_oof_auroc(y2, tr["log_in3"], tr["fold"], lab2)
    rec2d = signed_oof_auroc(y2, tr["c_n_days_with_tx"], tr["fold"], lab2)
    rec2_wo = signed_oof_auroc(y2, tr["log_in3"], tr["fold"], lab2 & ~chron)
    rec2d_wo = signed_oof_auroc(y2, tr["c_n_days_with_tx"], tr["fold"], lab2 & ~chron)
    after_y2 = leftover_diag(y2, tr["log_in3"], (tr["c_n_days_with_tx"],), tr["fold"], lab2)
    after_y3_wo = leftover_diag(
        y3, tr["log_in3"], (tr["c_n_days_with_tx"],), tr["fold"], lab3 & ~chron
    )
    fold_rows = []
    for r, rr in zip(p3["after"]["rec"]["folds"], p3["after"]["rrec"]["folds"]):
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
        _auc_row(Y2, "log_in3", rec2),
        _auc_row(Y2, "days", rec2d),
        _auc_row(Y2, "log_in3 wo12", rec2_wo),
        _auc_row(Y2, "days wo12", rec2d_wo),
    ]
    prose = (
        f"Y2 size {_f(rec2['cv'])} days {_f(rec2d['cv'])}. "
        f"Drop {len(ids)} chronic: size {_f(rec2_wo['cv'])} days {_f(rec2d_wo['cv'])}. "
        f"Y2 leftover after days {_f(after_y2['rank'])}. "
        f"Y3 leftover wo12 {_f(after_y3_wo['rank'])} dies={after_y3_wo['honest_dies']}."
    )
    print(prose)
    return {
        "ids": ids,
        "fold_rows": fold_rows,
        "y2_rows": y2_rows,
        "y2": _cv(rec2),
        "y2_wo": _cv(rec2_wo),
        "y2_days_wo": _cv(rec2d_wo),
        "y2_rank": after_y2["rank"],
        "y3_wo_rank": after_y3_wo["rank"],
        "prose": prose,
    }


def extra_bootstrap(tr: pd.DataFrame, n_boot: int = 50) -> dict:
    print("\n" + "=" * 72)
    print(f"EXTRA — company bootstrap leftover after days (n={n_boot})")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    work = tr.loc[lab, ["company_id", "fold", Y3, "log_in3", "c_n_days_with_tx"]].copy()
    cos = work["company_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(FOLD_SEED)
    ranks = []
    for _ in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        parts = [work[work["company_id"] == c] for c in draw]
        b = pd.concat(parts, ignore_index=True)
        after = leftover_diag(
            b[Y3],
            b["log_in3"],
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
    return {"p50": p50, "p10": p10, "p90": p90, "share_die": share_die, "prose": prose}


def extra_permute(tr: pd.DataFrame, n_perm: int = 30) -> dict:
    print("\n" + "=" * 72)
    print(f"EXTRA — permute log_in3 within days quintile (n={n_perm})")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    work = tr.loc[lab].copy()
    days = pd.to_numeric(work["c_n_days_with_tx"], errors="coerce")
    log = work["log_in3"]
    ok = days.notna() & log.notna()
    work = work.loc[ok]
    q = pd.qcut(pd.to_numeric(work["c_n_days_with_tx"], errors="coerce"), 5, duplicates="drop")
    rng = np.random.default_rng(FOLD_SEED + 3)
    ranks = []
    for _ in range(n_perm):
        shuf = work["log_in3"].copy()
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
        f"Permuted-within-days leftover rank p50={_f(p50)} p90={_f(p90)} n={len(ranks)}."
    )
    print(prose)
    return {"p50": p50, "p90": p90, "prose": prose}


def extra_demean_why(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — why demean leftover lives while level leftover dies")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    demean = company_demean(tr["log_in3"], tr["company_id"])
    meanx = company_mean(tr["log_in3"], tr["company_id"])
    rec_d = signed_oof_auroc(y, demean, tr["fold"], lab)
    rec_m = signed_oof_auroc(y, meanx, tr["fold"], lab)
    after_d = leftover_diag(y, demean, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_m = leftover_diag(y, meanx, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_both = leftover_diag(y, demean, (tr["c_n_days_with_tx"], meanx), tr["fold"], lab)
    prose = (
        f"Demean Y3 {_f(rec_d['cv'])} leftover-days {_f(after_d['rank'])} dies={after_d['honest_dies']}. "
        f"Mean Y3 {_f(rec_m['cv'])} leftover-days {_f(after_m['rank'])} dies={after_m['honest_dies']}. "
        f"Demean leftover after days+mean {_f(after_both['rank'])} dies={after_both['honest_dies']}. "
        f"Level leftover 0.521 dies — the contemporaneous size bar is unused after days. "
        f"Demean leftover is a STATE path, not a reason to KEEP a_in3 as engine X."
    )
    print(prose)
    return {
        "demean_cv": _cv(rec_d),
        "mean_cv": _cv(rec_m),
        "demean_rank": after_d["rank"],
        "mean_rank": after_m["rank"],
        "both_rank": after_both["rank"],
        "prose": prose,
    }


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
        after = leftover_diag(y, tr["log_in3"], (tr["c_n_days_with_tx"],), tr["fold"], sl)
        rec = signed_oof_auroc(y, tr["log_in3"], tr["fold"], sl)
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


def extra_opin_clone(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — a_op_in is the SIZE clone (do not swap)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rho = spearman(tr["log_in3"], tr["a_op_in"])
    rec = signed_oof_auroc(y, tr["a_op_in"], tr["fold"], lab)
    after = leftover_diag(y, tr["a_op_in"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_log = leftover_diag(y, tr["a_op_in"], (tr["log_in3"],), tr["fold"], lab)
    acf = median_acf(pd.to_numeric(tr["a_op_in"], errors="coerce"), tr["company_id"], 1)
    prose = (
        f"a_op_in Y3 {_f(rec['cv'])} leftover after days {_f(after['rank'])} dies={after['honest_dies']}. "
        f"leftover after log_in3 {_f(after_log['rank'])} dies={after_log['honest_dies']}. "
        f"ρ vs log_in3 {_f(rho)} acf1={_f(acf)}. "
        f"0.676 single is monthly extremes (acf≈0), not a swap for log1p(a_in3). Do not put a_op_in on the 44."
    )
    print(prose)
    return {
        "cv": _cv(rec),
        "rank": after["rank"],
        "after_log": after_log["rank"],
        "rho": rho,
        "acf": acf,
        "prose": prose,
    }


def extra_wo_fold(p2: dict, p3: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover without weakest / strongest fold")
    print("=" * 72)
    rec = p2["recs"]["log1p(a_in3)"]
    folds = [r["auroc"] for r in rec["folds"] if np.isfinite(r["auroc"])]
    weak_k = int(np.argmin(folds)) if folds else None
    strong_k = int(np.argmax(folds)) if folds else None
    wo_w = float(np.mean([a for i, a in enumerate(folds) if i != weak_k])) if folds else float("nan")
    wo_s = float(np.mean([a for i, a in enumerate(folds) if i != strong_k])) if folds else float("nan")
    rf = [r["auroc"] for r in p3["after"]["rrec"]["folds"] if np.isfinite(r["auroc"])]
    r_wo = float(np.mean(rf[1:])) if len(rf) > 1 else float("nan")
    prose = (
        f"Size without weakest fold {weak_k} = {_f(wo_w)}; without strongest {strong_k} = {_f(wo_s)}. "
        f"Rank leftover without fold 0 = {_f(r_wo)} "
        f"(still {'lives' if np.isfinite(r_wo) and r_wo >= CHANCE else 'dies'})."
    )
    print(prose)
    return {"wo_w": wo_w, "wo_s": wo_s, "r_wo": r_wo, "prose": prose}


def extra_ols_fake(p3: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — OLS fake-days leak")
    print("=" * 72)
    rho = p3["after"]["rho_ctrl"]
    almost = bool(np.isfinite(rho) and abs(rho) >= 0.70)
    prose = (
        f"OLS leftover {_f(p3['ols'])} ρ(resid,days)={_f(rho)} fake={p3['fake']} "
        f"{'almost-fake days leak — rank is honest' if almost else 'not a days leak'}. "
        f"Rank leftover {_f(p3['rank'])} dies={p3['dies']}."
    )
    print(prose)
    return {"rho": rho, "almost": almost, "prose": prose}


def decide(p1, p2, p3) -> dict:
    leftover_lives = bool(np.isfinite(p3["rank"]) and p3["rank"] >= CHANCE and not p3["dies"])
    is_size = True
    twin = bool(p1["twin_days"] or p1["twin_opin"] or p1["twin_in6"] or p1["twin_in12"])
    engine = leftover_lives and bool(p2["beat_size"]) and not is_size and not twin
    if engine:
        role = "KEEP as engine X"
        why = "engine KEEP-as-X passed — unexpected for the size stem"
    elif leftover_lives:
        role = "KEEP as size control only"
        why = (
            f"leftover after days rank {p3['rank']:.3f} lives, but it IS SIZE "
            f"(ρ vs self = 1). Not engine. Not a 15-col card stem. "
            f"Days leftover after size still {_f(p3['inv_rank'])} "
            f"({'dies' if p3['inv_dies'] else 'lives'})."
        )
    else:
        role = "CLOSE as unused leftover"
        why = (
            f"unused leftover after days: honest rank {p3['rank']:.3f} dies "
            f"(OLS {p3['ols']:.3f} fake={p3['fake']}). DROP from the 44 as engine X. "
            f"Size bar 0.617 stays the KEEP-as-X quote, not a card stem."
        )
    return {
        "role": role,
        "why": why,
        "leftover_lives": leftover_lives,
        "engine": engine,
        "is_size": is_size,
        "drop_engine": not leftover_lives or is_size,
        "card": "no — do not put a_in3 on the 15-col card",
    }


def make_plot(tr: pd.DataFrame, p3: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    log = tr["log_in3"]
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    lab = y3.notna() & log.notna()
    q = pd.qcut(log[lab], 5, duplicates="drop")
    rates, xs = [], []
    for i, cat in enumerate(sorted(q.dropna().unique()), start=1):
        sl = lab.copy()
        sl.loc[lab] = q == cat
        rates.append(float(y3[sl].mean()) if sl.any() else float("nan"))
        xs.append(i)
    ax.plot(xs, rates, marker="o", color="#1f4e79", label="log1p(a_in3)")
    q2 = pd.qcut(days[lab], 5, duplicates="drop")
    rates2, xs2 = [], []
    for i, cat in enumerate(sorted(q2.dropna().unique()), start=1):
        sl = lab.copy()
        sl.loc[lab] = q2 == cat
        rates2.append(float(y3[sl].mean()) if sl.any() else float("nan"))
        xs2.append(i)
    ax.plot(xs2, rates2, marker="s", color="#c45c26", label="days")
    ax.set_xlabel("quintile (low → high)")
    ax.set_ylabel("Y3 rate")
    ax.set_title("inverse-size recover vs days")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    rec = p3["after"]["rrec"]
    xs = [r["fold"] for r in rec["folds"]]
    ys = [r["auroc"] for r in rec["folds"]]
    ax.bar(xs, ys, color="#1f4e79")
    ax.axhline(CHANCE, color="#333", ls="--", lw=1, label="0.55 dies")
    ax.axhline(DAYS_BENCH, color="#c45c26", ls=":", lw=1, label="days 0.711")
    ax.set_xlabel("fold")
    ax.set_ylabel("rank leftover AUROC")
    ax.set_title("log1p(a_in3) leftover after days")
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
    p6, p7, p9, p10 = ctx["p6"], ctx["p7"], ctx["p9"], ctx["p10"]
    xt, xs, xf, xb = ctx["xt"], ctx["xs"], ctx["xf"], ctx["xb"]
    xp, xo = ctx["xp"], ctx["xo"]
    xd, xsf, xop, xw = ctx["xd"], ctx["xsf"], ctx["xop"], ctx["xw"]
    lines = [
        "# Unused leftover of `a_in3` after days (size control)",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_in3`. Do not put `a_in3` on the 15-col card. "
        "Do not overwrite `salary_month_qa.*`. Do not grow TURNOVER.",
        "",
        "`a_in3` = trailing-3m operational inflow. Size bar is always `log1p(a_in3)` (quoted 0.617). "
        "This cut asks leftover after `c_n_days_with_tx`.",
        "",
        "## Headline",
        "",
        (
            f"`log1p(a_in3)` on the 44: **{d['role']}** ({d['why']}). "
            f"Y3 leftover after days OLS {_f(p3['ols'])} rank {_f(p3['rank'])} "
            f"({'dies' if p3['dies'] else 'lives'}, fake={p3['fake']}). "
            f"Inverse days after size rank {_f(p3['inv_rank'])}. "
            f"Single size {_f(p2['size3'])} vs days {_f(p2['days'])}. "
            f"15-col card: {d['card']}. "
            f"Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712."
        ),
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        f"| 1 | Who is healthy? | Size is the **control**, not a health Y. {d['role']}. |",
        "| 2 | Who is improving? | Q6 lag leftover after days_lag1. a_growth_12 already CLOSE. |",
        "| 3 | Who is turning? | Not a new Y. Do not invent `y_in3`. |",
        "| 4 | Dip vs fall? | Not a TURNOVER column. |",
        f"| 5 | Why did it change? | Leftover after days rank {_f(p3['rank'])}. Twin: {p1['twins'] or 'none'}. |",
        f"| 6 | Months earlier? | lag1 leftover after days_lag1 {_f(p7['l1_rank'])}; lag3 {_f(p7['l3_rank'])}. |",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `log1p(a_in3)` as 44-col size control | **{d['role']}** | {d['why']} |",
        f"| `a_in3` as engine X / 15-col card | **no** | {d['card']} |",
        f"| `a_op_in` / `a_in6` / `a_in12` besides size | **CLOSE** | twins of the size stem |",
        "",
        "## 1 — Coverage; acf1; SIZE ρ",
        "",
        p1["prose"],
        "",
        _md_table(p1["rows"]),
        "",
        _md_table(p1["rho_rows"]),
        "",
        "## 2 — Single-feature group-fold Y3",
        "",
        p2["prose"],
        "",
        _md_table(p2["rows"]),
        "",
        "## 3 — Honest leftover after days",
        "",
        p3["prose"],
        "",
        f"OLS folds: {p3['after']['folds']}. Rank folds: {p3['after']['rank_folds']}.",
        "",
        "## 4 — Same-n leftover",
        "",
        p4["prose"],
        "",
        _md_table(p4["rows"]),
        "",
        "## 5 — SIZE terciles (T1 inverse-size)",
        "",
        p5["prose"],
        "",
        _md_table(p5["rows"]),
        "",
        "## 6 — ICC / demean",
        "",
        p6["prose"],
        "",
        "## 7 — Q6 lag1 leftover after days_lag1",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 9 — Holdout coverage only",
        "",
        p9["prose"],
        "",
        "## 10 — Dark 470 vs ERP",
        "",
        p10["prose"],
        "",
        _md_table(p10["rows"]),
        "",
        "## Extras",
        "",
        "### a_in6 / a_in12 twins",
        "",
        xt["prose"],
        "",
        _md_table(xt["rows"]),
        "",
        "### leftover vs salary",
        "",
        xs["prose"],
        "",
        "### Fold-wise leftover; 12-name Y2 drop",
        "",
        xf["prose"],
        "",
        _md_table(xf["fold_rows"]),
        "",
        _md_table(xf["y2_rows"]),
        "",
        "### Bootstrap leftover after days",
        "",
        xb["prose"],
        "",
        "### Permute within days quintile",
        "",
        xp["prose"],
        "",
        "### OLS fake-days leak",
        "",
        xo["prose"],
        "",
        "### Demean leftover vs level leftover",
        "",
        xd["prose"],
        "",
        "### Leftover by so-far",
        "",
        xsf["prose"],
        "",
        _md_table(xsf["rows"]),
        "",
        "### a_op_in SIZE clone",
        "",
        xop["prose"],
        "",
        "### Without weakest / strongest fold",
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
        "Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put `a_in3` on the 15-col card.",
        "",
        "## Files written",
        "",
        "- `analysis/evaluate/in3_qa.py`",
        "- `analysis/outputs/in3_qa.md`",
        "- `analysis/outputs/in3_leftover.png`" if ctx.get("png") else "- (no PNG)",
        "- append-only `analysis/experiments/registry.csv`",
        "- `overnight/waves/wave4_in3.md` (end, if WRITE_WAVE)",
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
    p1, p2, p3 = ctx["p1"], ctx["p2"], ctx["p3"]
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
            "metric": "auroc_log1p_a_in3",
            "value": p2["size3"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={p2['days']:.4f} confirm={p2['size_ok']}",
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
            "metric": "auroc_log1p_a_in3_resid_days",
            "value": p3["rank"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"ols={p3['ols']:.4f} dies={p3['dies']} fake={p3['fake']} r2={p3['r2']:.4f}",
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
            "metric": "auroc_days_resid_log_in3",
            "value": p3["inv_rank"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"inverse dies={p3['inv_dies']}",
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
            "metric": "rho_log_in3_vs_days",
            "value": p1["rhos"]["c_n_days_with_tx"],
            "coverage": f"{p1['cov']:.4f}",
            "notes": f"twins={p1['twins']} acf_in3={p1['acf_in3']:.4f} acf_op={p1['acf_op']:.4f}",
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
            "metric": "size_control",
            "value": 1 if d["leftover_lives"] else 0,
            "coverage": f"{p1['cov']:.4f}",
            "notes": d["role"] + " " + d["why"][:160],
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
    p1, p2, p3, p7 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p7"]
    text = (
        f"# Wave 4 — a_in3 leftover after days (size control)\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/in3_qa.py`\n"
        f"- `analysis/outputs/in3_qa.md`\n"
        f"- `analysis/outputs/in3_leftover.png`\n"
        f"- append-only `analysis/experiments/registry.csv` "
        f"(skip key agent+x_families+y+model+split+metric)\n"
        f"- this note\n\n"
        f"Did not touch `salary_month_qa.*`, `ss_qa.*`, `issued_qa.*`, "
        f"`cashflow.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, "
        f"TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, or the parent journal. "
        f"Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. "
        f"Y7 TURNOVER **0.720 / 0.712** unchanged.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| `log1p(a_in3)` as 44-col size control | **{d['role']}** |\n"
        f"| `a_in3` as engine X / 15-col card | **no** |\n\n"
        f"Y3 leftover after days rank {_f(p3['rank'])} (dies={p3['dies']}, fake={p3['fake']}); "
        f"inverse days after size {_f(p3['inv_rank'])}. "
        f"Single size {_f(p2['size3'])} vs days {_f(p2['days'])}. "
        f"ρ vs days {_f(p1['rhos']['c_n_days_with_tx'])}. "
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
    print("in3 leftover QA — unused leftover of a_in3 after days as size control")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, ["log_in3", "c_n_days_with_tx"], (1, 3))
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
    p2 = pass2_singles(tr)
    p3 = pass3_days(tr)
    p4 = pass4_samen(tr)
    p5 = pass5_tercile(tr)
    p6 = pass6_icc(tr)
    p7 = pass7_q6(tr)
    p9 = pass9_hold(hold)
    p10 = pass10_dark(tr, book)
    xt = extra_twins6_12(tr)
    xs = extra_salary(tr)
    xf = extra_fold_y2(tr, p3)
    xo = extra_ols_fake(p3)
    xb = extra_bootstrap(tr, n_boot=50)
    xp = extra_permute(tr, n_perm=30)
    xd = extra_demean_why(tr)
    xsf = extra_sofar(tr)
    xop = extra_opin_clone(tr)
    xw = extra_wo_fold(p2, p3)
    decision = decide(p1, p2, p3)
    print("\n" + "=" * 72)
    print(f"SIZE CONTROL: {decision['role']}")
    print(decision["why"])
    print("=" * 72)
    if not p2["days_ok"]:
        failed.append(f"days replica {p2['days']:.3f} ≠ 0.711")
    if not p2["size_ok"]:
        failed.append(f"size replica {p2['size3']:.3f} ≠ 0.617")
    png = make_plot(tr, p3)
    elapsed = time.time() - t0
    ctx = {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "p6": p6,
        "p7": p7,
        "p9": p9,
        "p10": p10,
        "xt": xt,
        "xs": xs,
        "xf": xf,
        "xb": xb,
        "xp": xp,
        "xo": xo,
        "xd": xd,
        "xsf": xsf,
        "xop": xop,
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
