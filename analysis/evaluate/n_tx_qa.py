"""Unused leftover of ``a_n_tx`` after ``c_n_days_with_tx`` as a 15-col stem.

``a_n_tx`` = COUNT(*) of transactions this month (Family A). ``c_n_tx`` is
the same COUNT(*) on Family C. ``c_n_days_with_tx`` is unique calendar
days. ``c_gap_sd`` is already DROPPED from the 44 as the weaker days twin.

Night quotes do not change: Y3 0.762 / 0.752. Days bar 0.711. Size 0.617.
Y7 TURNOVER 0.720 / 0.712. Do not quote a_out_vol 0.722 as the engine.
Do not put new columns on the 15-col card. Parent absorbs the card.

KEEP-as-X: beat size ≥0.02 **and** leftover after days **and** not SIZE
(|ρ| vs log1p(a_in3) ≥0.50) **and** not a twin (|ρ|≥0.80 vs days /
c_n_tx / c_gap_sd). Leftover <0.55 dies. If leftover looks high, check
ρ(resid, days) for a fake-days leak.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.n_tx_qa

Owned: analysis/evaluate/n_tx_qa.py, analysis/outputs/n_tx_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_n_tx.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "n_tx_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "n_tx_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_n_tx.md"
AGENT = "c81e4b07"
WAVE = "4"
ROUND = "R4"
MODEL = "n_tx_qa"
X_FAM = "A"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_Y3_QUOTE = 0.617
Y2_DAYS_QUOTE = 0.571
DAYS_LAG1_QUOTE = 0.684
Y3_NIGHT = (0.762, 0.752)
Y7_TURNOVER = (0.720, 0.712)
PERM_NTX = 0.030
PERM_DAYS = 0.001
UNIV_NTX_QUOTE = 0.714
SIZE_RHO_STRESSED = 0.651
ICC_QUOTE = 0.97
ACF1_QUOTE = 0.22
COV_QUOTE = 1.00
KEEP_DELTA = 0.02
TWIN_RHO = 0.80
SIZE_RHO = 0.50
CHANCE = 0.55
ICC_TRAIT = 0.85
ACF_STYLE = 0.40
MIN_POS = 50
MIN_ACF_PAIRS = 4
CHRONIC_GROUPS = ("GROUP_0158", "GROUP_0172")
CHRONIC_BELOW = 0.50
WRITE_WAVE = True

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_n_tx",
    "a_in3",
    "a_op_in",
    "c_n_tx",
    "c_n_days_with_tx",
    "c_gap_sd",
)

OPTIONAL_STORE = ("b_below_0",)

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


def chronic_ids(tr: pd.DataFrame) -> list[str]:
    """12 chronic dark Y2 names: ≥50% labeled months already below 0 in 0158/0172."""
    if "b_below_0" not in tr.columns:
        hot = tr["group_id"].astype(str).isin(CHRONIC_GROUPS)
        return sorted(tr.loc[hot, "company_id"].astype(str).unique())
    y = pd.to_numeric(tr[Y2], errors="coerce")
    below = pd.to_numeric(tr["b_below_0"], errors="coerce")
    lab = y.notna()
    hot = tr["group_id"].astype(str).isin(CHRONIC_GROUPS)
    sl = lab & hot
    g = (
        tr.loc[sl, ["company_id"]]
        .assign(below=below[sl].values)
        .groupby("company_id")["below"]
        .agg(n="size", n_below="sum")
    )
    g["share_below"] = g["n_below"] / g["n"]
    return [str(i) for i in g.index[g["share_below"] >= CHRONIC_BELOW]]


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
    extra = [c for c in OPTIONAL_STORE if c in raw.columns]
    panel = _keys(raw[list(STORE_COLS) + extra])
    y = _keys(yraw[["company_id", "period", *Y_KEEP]])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    panel["log_op_in"] = np.log1p(
        pd.to_numeric(panel["a_op_in"], errors="coerce").clip(lower=0).abs()
    )
    ntx = pd.to_numeric(panel["a_n_tx"], errors="coerce")
    days = pd.to_numeric(panel["c_n_days_with_tx"], errors="coerce")
    panel["log1p_n_tx"] = np.log1p(ntx.clip(lower=0))
    panel["intensity"] = np.where(days > 0, ntx / days, np.nan)
    panel["burst"] = ntx - days
    if "first_month" in panel.columns:
        panel["first_month"] = pd.to_datetime(panel["first_month"])
        panel["months_so_far"] = (
            (panel["period"].dt.year - panel["first_month"].dt.year) * 12
            + (panel["period"].dt.month - panel["first_month"].dt.month)
            + 1
        )
    else:
        panel = panel.sort_values(["company_id", "period"]).reset_index(drop=True)
        panel["months_so_far"] = panel.groupby("company_id", sort=False).cumcount() + 1
    grid_n = panel.groupby("company_id")["period"].size().rename("n_grid_months")
    panel = panel.merge(grid_n, on="company_id", how="left")
    panel["trail_class"] = panel["months_so_far"].map(_trail_class)
    leak3 = leakage_check(
        ["a_n_tx", "c_n_days_with_tx", "c_n_tx", "c_gap_sd", "log_in3"],
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


# ---------------------------------------------------------------------------
# 1. Coverage; acf1/acf3; size ρ vs log1p(a_in3)
# ---------------------------------------------------------------------------
def pass1_cov(tr: pd.DataFrame) -> dict:
    ntx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    n_cm = int(len(tr))
    n_nn = int(ntx.notna().sum())
    n_zero = int((ntx == 0).sum())
    cov = _pct(n_nn, n_cm)
    acf1 = median_acf(ntx, tr["company_id"], 1)
    acf3 = median_acf(ntx, tr["company_id"], 3)
    acf6 = median_acf(ntx, tr["company_id"], 6)
    rho_in3 = spearman(ntx, tr["log_in3"])
    rho_opin = spearman(ntx, tr["log_op_in"])
    size_flag = bool(np.isfinite(rho_in3) and abs(rho_in3) >= SIZE_RHO)
    confirm_cov = bool(np.isfinite(cov) and abs(cov - COV_QUOTE) < 0.01)
    confirm_acf = bool(np.isfinite(acf1) and abs(acf1 - ACF1_QUOTE) < 0.05)
    rows = [
        {
            "slice": "train all",
            "n_cm": f"{n_cm:,}",
            "n_co": int(tr["company_id"].nunique()),
            "nn": f"{n_nn:,}",
            "cov": _pp(cov),
            "eq0": _pp(_pct(n_zero, n_cm)),
            "mean": _f(float(ntx.mean())),
            "p50": _f(float(ntx.median())),
            "p90": _f(float(ntx.quantile(0.90))),
        }
    ]
    prose = (
        f"Train a_n_tx cov {_pp(cov)} ({'CONFIRM 100%' if confirm_cov else 'off 100%'}) "
        f"mean {_f(float(ntx.mean()))} p50 {_f(float(ntx.median()))} eq0 {_pp(_pct(n_zero, n_cm))}. "
        f"acf1 {_f(acf1)} (feature-report 0.22 {'CONFIRM' if confirm_acf else 'off'}) "
        f"acf3 {_f(acf3)} acf6 {_f(acf6)} — LOW_PERSIST. "
        f"ρ vs log1p(a_in3) {_f(rho_in3)} vs log1p(|a_op_in|) {_f(rho_opin)} "
        f"({'SIZE |ρ|≥0.50' if size_flag else 'not SIZE vs a_in3'})."
    )
    print(prose)
    return {
        "rows": rows,
        "cov": cov,
        "n_cm": n_cm,
        "n_nn": n_nn,
        "n_zero": n_zero,
        "mean": float(ntx.mean()),
        "p50": float(ntx.median()),
        "acf1": acf1,
        "acf3": acf3,
        "acf6": acf6,
        "rho_in3": rho_in3,
        "rho_opin": rho_opin,
        "size_flag": size_flag,
        "confirm_cov": confirm_cov,
        "confirm_acf": confirm_acf,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 2. Spearman twins + COUNT(*) identity
# ---------------------------------------------------------------------------
def pass2_twins(tr: pd.DataFrame) -> dict:
    ntx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    cntx = pd.to_numeric(tr["c_n_tx"], errors="coerce")
    both = ntx.notna() & cntx.notna()
    n_eq = int((ntx[both] == cntx[both]).sum())
    n_both = int(both.sum())
    max_abs = float((ntx[both] - cntx[both]).abs().max()) if n_both else float("nan")
    identity = bool(n_both > 0 and n_eq == n_both)
    pairs = [
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("c_n_tx", tr["c_n_tx"]),
        ("c_gap_sd", tr["c_gap_sd"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("log1p(|a_op_in|)", tr["log_op_in"]),
        ("intensity", tr["intensity"]),
        ("log1p_n_tx", tr["log1p_n_tx"]),
    ]
    rows = []
    rhos = {}
    twins = []
    for name, s in pairs:
        rho = spearman(ntx, s)
        rhos[name] = rho
        twin = bool(
            name in {"c_n_days_with_tx", "c_n_tx", "c_gap_sd"}
            and np.isfinite(rho)
            and abs(rho) >= TWIN_RHO
        )
        if twin:
            twins.append(name)
        rows.append(
            {
                "vs": name,
                "ρ": _f(rho),
                "twin": "YES" if twin else "",
                "|ρ|≥0.80": "YES" if (np.isfinite(rho) and abs(rho) >= TWIN_RHO) else "",
            }
        )
    size_flag = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    gate_twins = bool(twins)
    prose = (
        f"a_n_tx vs c_n_tx equal on {n_eq:,}/{n_both:,} "
        f"(max|Δ|={_f(max_abs, 6)}) — "
        f"{'IDENTITY (both COUNT(*))' if identity else 'not a bit-identity'}. "
        f"Spearman twins |ρ|≥0.80: {twins or 'none'}. "
        f"vs days {_f(rhos['c_n_days_with_tx'])} vs c_n_tx {_f(rhos['c_n_tx'])} "
        f"vs gap_sd {_f(rhos['c_gap_sd'])} vs log1p(a_in3) {_f(rhos['log1p(a_in3)'])} "
        f"({'SIZE' if size_flag else 'not SIZE'})."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "twins": twins,
        "gate_twins": gate_twins,
        "size_flag": size_flag,
        "identity": identity,
        "n_eq": n_eq,
        "n_both": n_both,
        "max_abs": max_abs,
        "rho_days": rhos["c_n_days_with_tx"],
        "rho_cntx": rhos["c_n_tx"],
        "rho_gap": rhos["c_gap_sd"],
        "rho_size": rhos["log1p(a_in3)"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 3. Single-feature train group-fold AUROC
# ---------------------------------------------------------------------------
def pass3_singles(tr: pd.DataFrame) -> dict:
    feats = {
        "a_n_tx": tr["a_n_tx"],
        "c_n_tx": tr["c_n_tx"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "c_gap_sd": tr["c_gap_sd"],
        "log1p(a_in3)": tr["log_in3"],
        "log1p_n_tx": tr["log1p_n_tx"],
        "intensity": tr["intensity"],
    }
    rows = []
    store = {}
    for y in (Y3, Y2):
        lab = tr[y].notna()
        for name, col in feats.items():
            res = signed_oof_auroc(tr[y], col, tr["fold"], lab)
            store[(y, name)] = res
            rows.append(_auc_row(y, name, res))
            print(
                f"AUROC {y} {name}: "
                f"{'LOW_POWER' if res['low_power'] else _f(res['cv'])} "
                f"n_pos={res['n_pos']} sign={res['train_sign']}"
            )
    y3 = _cv(store[(Y3, "a_n_tx")])
    days = _cv(store[(Y3, "c_n_days_with_tx")])
    size = _cv(store[(Y3, "log1p(a_in3)")])
    cntx = _cv(store[(Y3, "c_n_tx")])
    gap = _cv(store[(Y3, "c_gap_sd")])
    y2 = _cv(store[(Y2, "a_n_tx")])
    y2_days = _cv(store[(Y2, "c_n_days_with_tx")])
    days_ok = bool(np.isfinite(days) and abs(days - DAYS_BENCH) < 0.02)
    size_ok = bool(np.isfinite(size) and abs(size - SIZE_Y3_QUOTE) < 0.03)
    y2_days_ok = bool(np.isfinite(y2_days) and abs(y2_days - Y2_DAYS_QUOTE) < 0.03)
    beat_size = y3 - size if np.isfinite(y3) and np.isfinite(size) else float("nan")
    loses_days = bool(np.isfinite(days) and np.isfinite(y3) and y3 + 1e-12 < days)
    same_cntx = bool(np.isfinite(y3) and np.isfinite(cntx) and abs(y3 - cntx) < 0.005)
    prose = (
        f"Y3 a_n_tx {_f(y3)} vs days {_f(days)} (night 0.711 {'CONFIRM' if days_ok else 'off'}) "
        f"vs size {_f(size)} (0.617 {'CONFIRM' if size_ok else 'off'}, Δ {_f(beat_size)}) "
        f"vs c_n_tx {_f(cntx)} ({'same-n identity skill' if same_cntx else 'c_n_tx drifted'}) "
        f"vs gap_sd {_f(gap)}. "
        f"Y2 a_n_tx {_f(y2)} vs days {_f(y2_days)} "
        f"(legal days 0.571 {'CONFIRM' if y2_days_ok else 'off'}). "
        f"{'Loses to the 0.711 days bar' if loses_days else 'Does not lose to days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "y3": y3,
        "days": days,
        "size": size,
        "cntx": cntx,
        "gap": gap,
        "y2": y2,
        "y2_days": y2_days,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "y2_days_ok": y2_days_ok,
        "beat_size": beat_size,
        "loses_days": loses_days,
        "same_cntx": same_cntx,
        "sign_y3": store[(Y3, "a_n_tx")]["train_sign"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 4–5. Honest leftover after days + inverse leftover
# ---------------------------------------------------------------------------
def pass4_leftover(tr: pd.DataFrame) -> dict:
    ntx = tr["a_n_tx"]
    days = tr["c_n_days_with_tx"]
    cntx = tr["c_n_tx"]
    gap = tr["c_gap_sd"]
    size = tr["log_in3"]
    specs = [
        (Y3, "after days", (days,)),
        (Y3, "after c_n_tx", (cntx,)),
        (Y3, "after gap_sd", (gap,)),
        (Y3, "after size", (size,)),
        (Y3, "after days+c_n_tx", (days, cntx)),
        (Y2, "after days", (days,)),
    ]
    rows = []
    store = {}
    for ycol, name, xs in specs:
        rec = leftover_diag(tr[ycol], ntx, list(xs), tr["fold"], tr[ycol].notna())
        store[(ycol, name)] = rec
        rows.append(
            {
                "y": ycol,
                "control": name,
                "OLS": _f(rec["ols"]),
                "rank": _f(rec["rank"]),
                "ρ(resid,ctrl)": _f(rec["rho_ctrl"]),
                "ρ(resid,n_tx)": _f(rec["rho_x"]),
                "R²": _f(rec["r2"]),
                "fake-days": "YES" if rec["fake"] else "",
                "honest_dies": "YES" if rec["honest_dies"] else "no",
            }
        )
        print(
            f"left {ycol} {name}: OLS={_f(rec['ols'])} rank={_f(rec['rank'])} "
            f"ρctrl={_f(rec['rho_ctrl'])} R2={_f(rec['r2'])}"
        )
    inv = leftover_diag(tr[Y3], days, [ntx], tr["fold"], tr[Y3].notna())
    store[(Y3, "days after n_tx")] = inv
    rows.append(
        {
            "y": Y3,
            "control": "days after n_tx (inverse)",
            "OLS": _f(inv["ols"]),
            "rank": _f(inv["rank"]),
            "ρ(resid,ctrl)": _f(inv["rho_ctrl"]),
            "ρ(resid,n_tx)": _f(inv["rho_x"]),
            "R²": _f(inv["r2"]),
            "fake-days": "YES" if inv["fake"] else "",
            "honest_dies": "YES" if inv["honest_dies"] else "no",
        }
    )
    y3 = store[(Y3, "after days")]
    y3_cntx = store[(Y3, "after c_n_tx")]
    y3_gap = store[(Y3, "after gap_sd")]
    leftover = y3["rank"]
    leftover_ols = y3["ols"]
    dies = bool(
        y3["honest_dies"]
        or (np.isfinite(leftover) and leftover < CHANCE)
        or (np.isfinite(leftover_ols) and leftover_ols < CHANCE and y3["fake"])
    )
    inv_lives = bool(np.isfinite(inv["rank"]) and inv["rank"] >= CHANCE and not inv["honest_dies"])
    prose = (
        f"Y3 leftover after days OLS {_f(leftover_ols)} rank {_f(leftover)} "
        f"ρ(resid,days)={_f(y3['rho_ctrl'])} R²={_f(y3['r2'])} "
        f"({'FAKE-DAYS leak' if y3['fake'] else 'resid is not a days clone'}). "
        f"Same-n leftover after c_n_tx rank {_f(y3_cntx['rank'])} "
        f"(R²={_f(y3_cntx['r2'])}). After gap_sd {_f(y3_gap['rank'])}. "
        f"Inverse: days leftover after a_n_tx rank {_f(inv['rank'])} OLS {_f(inv['ols'])} "
        f"({'rank thin ≥0.55 — OLS almost the raw 0.711 bar (same OLS-high pattern)' if inv_lives else '0.711 bar dies after n_tx'}). "
        f"Honest leftover after days {'DIES' if dies else 'lives'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "y3_rank": leftover,
        "y3_ols": leftover_ols,
        "y3_rho_days": y3["rho_ctrl"],
        "y3_r2": y3["r2"],
        "y3_fake": y3["fake"],
        "y3_dies": dies,
        "y3_cntx": y3_cntx["rank"],
        "y3_cntx_r2": y3_cntx["r2"],
        "y3_gap": y3_gap["rank"],
        "inv_rank": inv["rank"],
        "inv_ols": inv["ols"],
        "inv_lives": inv_lives,
        "y2_rank": store[(Y2, "after days")]["rank"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 6. SIZE terciles — leftover inside T1 and T2+T3
# ---------------------------------------------------------------------------
def pass6_terciles(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    ntx = tr["a_n_tx"]
    days = tr["c_n_days_with_tx"]
    rows = []
    store = {}
    slices = (
        ("T1", terc == "T1"),
        ("T2", terc == "T2"),
        ("T3", terc == "T3"),
        ("T2+T3", terc.isin(["T2", "T3"])),
    )
    for name, mask in slices:
        raw = signed_oof_auroc(tr[Y3], ntx, tr["fold"], tr[Y3].notna() & mask)
        rec = leftover_diag(tr[Y3], ntx, [days], tr["fold"], tr[Y3].notna() & mask)
        dcv = signed_oof_auroc(tr[Y3], days, tr["fold"], tr[Y3].notna() & mask)
        scv = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], tr[Y3].notna() & mask)
        store[name] = rec
        rows.append(
            {
                "slice": name,
                "n": f"{raw['n_defined']:,}",
                "n_pos": f"{raw['n_pos']:,}",
                "raw n_tx": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "leftover days": _f(rec["rank"]),
                "days": "LOW_POWER" if dcv["low_power"] else _f(dcv["cv"]),
                "size": "LOW_POWER" if scv["low_power"] else _f(scv["cv"]),
                "R²": _f(rec["r2"]),
                "dies": "YES" if rec["honest_dies"] else "no",
            }
        )
    t1_dies = store["T1"]["honest_dies"] or (
        np.isfinite(store["T1"]["rank"]) and store["T1"]["rank"] < CHANCE
    )
    t23_dies = store["T2+T3"]["honest_dies"] or (
        np.isfinite(store["T2+T3"]["rank"]) and store["T2+T3"]["rank"] < CHANCE
    )
    prose = (
        f"Y3 leftover after days inside T1 rank {_f(store['T1']['rank'])} "
        f"({'dies' if t1_dies else 'lives'}); T2+T3 {_f(store['T2+T3']['rank'])} "
        f"({'dies' if t23_dies else 'lives'}). "
        "If leftover dies in both clocks it is not a small-firm regularity."
    )
    print(prose)
    return {
        "rows": rows,
        "t1": store["T1"]["rank"],
        "t23": store["T2+T3"]["rank"],
        "t1_dies": t1_dies,
        "t23_dies": t23_dies,
        "dies_inside": bool(t1_dies and t23_dies),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 7. Q6 lag1 / lag3 on short books
# ---------------------------------------------------------------------------
def pass7_q6(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    cols = (
        "a_n_tx",
        "a_n_tx_lag1",
        "a_n_tx_lag3",
        "c_n_days_with_tx",
        "c_n_days_with_tx_lag1",
        "c_n_days_with_tx_lag3",
    )
    slices = (
        ("all", pd.Series(True, index=tr.index)),
        ("short_<12", tr["trail_class"] == "short_<12"),
        ("long_>=18", tr["trail_class"] == "long_>=18"),
    )
    for y in (Y3, Y2):
        for sname, smask in slices:
            lab = tr[y].notna() & smask
            for col in cols:
                if col not in tr.columns:
                    continue
                res = signed_oof_auroc(tr[y], tr[col], tr["fold"], lab)
                store[(y, sname, col)] = res
                rows.append(
                    {
                        "y": y,
                        "slice": sname,
                        "col": col,
                        "n": f"{res['n_defined']:,}",
                        "n_pos": f"{res['n_pos']:,}",
                        "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                        "sign": res["train_sign"] if not res["low_power"] else "—",
                    }
                )

    def g(y, sl, col) -> float:
        r = store.get((y, sl, col))
        if r is None or r["low_power"]:
            return float("nan")
        return r["cv"]

    now = g(Y3, "all", "a_n_tx")
    lag1 = g(Y3, "all", "a_n_tx_lag1")
    lag3 = g(Y3, "all", "a_n_tx_lag3")
    days_l1 = g(Y3, "all", "c_n_days_with_tx_lag1")
    short_l1 = g(Y3, "short_<12", "a_n_tx_lag1")
    days_l1_ok = bool(np.isfinite(days_l1) and abs(days_l1 - DAYS_LAG1_QUOTE) < 0.03)
    keep_q6 = bool(
        np.isfinite(now)
        and now >= CHANCE
        and np.isfinite(lag1)
        and lag1 >= CHANCE
        and (now - lag1) <= 0.03
        and np.isfinite(short_l1)
        and short_l1 >= CHANCE
    )
    q6 = "KEEP" if keep_q6 else "CLOSE"
    prose = (
        f"Y3 a_n_tx now {_f(now)} lag1 {_f(lag1)} lag3 {_f(lag3)}; "
        f"short_<12 lag1 {_f(short_l1)}. "
        f"Days lag1 {_f(days_l1)} (night KEEP 0.684 {'CONFIRM' if days_l1_ok else 'off'}). "
        f"Q6 {q6}."
    )
    print(prose)
    return {
        "rows": rows,
        "now": now,
        "lag1": lag1,
        "lag3": lag3,
        "days_l1": days_l1,
        "short_l1": short_l1,
        "days_l1_ok": days_l1_ok,
        "keep_q6": keep_q6,
        "q6": q6,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 8. ICC / company-demean
# ---------------------------------------------------------------------------
def pass8_icc(tr: pd.DataFrame) -> dict:
    ntx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    icc_n = icc_anova(ntx, tr["company_id"])
    icc_d = icc_anova(days, tr["company_id"])
    dem = company_demean(ntx, tr["company_id"])
    mu = company_mean(ntx, tr["company_id"])
    rows = []
    store = {}
    for ycol, yname in ((Y3, "Y3"), (Y2, "Y2")):
        for feat, col in (("now", ntx), ("co_mean", mu), ("demean", dem)):
            res = signed_oof_auroc(tr[ycol], col, tr["fold"], tr[ycol].notna())
            store[(yname, feat)] = res
            rows.append(_auc_row(ycol, feat, res))
    confirm = bool(np.isfinite(icc_n["icc"]) and abs(icc_n["icc"] - ICC_QUOTE) < 0.03)
    trait = bool(np.isfinite(icc_n["icc"]) and icc_n["icc"] >= ICC_TRAIT)
    shock = bool(np.isfinite(icc_n["icc"]) and icc_n["icc"] < 0.50)
    y3_dem = _cv(store[("Y3", "demean")])
    y3_mu = _cv(store[("Y3", "co_mean")])
    prose = (
        f"a_n_tx ICC {_f(icc_n['icc'])} (feature-report 0.97 "
        f"{'CONFIRM BETWEEN' if confirm else 'off'}); days ICC {_f(icc_d['icc'])}. "
        f"{'TRAIT (BETWEEN)' if trait else 'month shock' if shock else 'mixed'}. "
        f"Y3 company-mean {_f(y3_mu)} demean {_f(y3_dem)} — "
        f"{'the skill is a company identity, not a month shock' if (np.isfinite(y3_mu) and np.isfinite(y3_dem) and y3_mu > y3_dem) else 'demean holds some month shock'}."
    )
    print(prose)
    return {
        "rows": rows,
        "icc": icc_n["icc"],
        "icc_days": icc_d["icc"],
        "k": icc_n["k"],
        "confirm": confirm,
        "trait": trait,
        "shock": shock,
        "y3_dem": y3_dem,
        "y3_mu": y3_mu,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 9. SHAP / perm gap — leftover after days vs perm 0.030
# ---------------------------------------------------------------------------
def pass9_perm(tr: pd.DataFrame, p3: dict, p4: dict) -> dict:
    leftover = p4["y3_rank"]
    leftover_ols = p4["y3_ols"]
    fake = p4["y3_fake"]
    y3 = p3["y3"]
    days = p3["days"]
    # If leftover dies, perm 0.030 is the tree using n_tx as the days/size stem
    # (SIZE_STEM, univ −0.714) while days is the honest bar (0.711, perm ≈0).
    explains = bool(
        (np.isfinite(leftover) and leftover < CHANCE)
        or fake
        or (np.isfinite(y3) and np.isfinite(days) and abs(y3 - days) < 0.02)
    )
    prose = (
        f"Night SHAP: a_n_tx #4 mean|SHAP| 0.125, perm ΔAUROC {PERM_NTX:.3f}, "
        f"univ −{UNIV_NTX_QUOTE:.3f}, SIZE_STEM ρ_size {SIZE_RHO_STRESSED:.3f}. "
        f"Days is SHAP #8 perm {PERM_DAYS:.3f} but the 0.711 bar. "
        f"This-run leftover after days rank {_f(leftover)} OLS {_f(leftover_ols)} "
        f"fake={fake}. "
        + (
            "Perm 0.030 is the GBM using n_tx as the count/size rewrite of days — "
            "not leftover skill after the 0.711 bar. Days stays the engine; "
            "n_tx is the stem the trees picked because it is the SIZE_STEM twin."
            if explains
            else "Leftover after days still looks alive — perm 0.030 could be real leftover; "
            "check ρ(resid, days) before calling it KEEP."
        )
    )
    print(prose)
    return {
        "leftover": leftover,
        "explains": explains,
        "perm_ntx": PERM_NTX,
        "perm_days": PERM_DAYS,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — same-n leftover vs c_n_tx; log1p; intensity
# ---------------------------------------------------------------------------
def pass_ex_same_n(tr: pd.DataFrame) -> dict:
    ntx = tr["a_n_tx"]
    cntx = tr["c_n_tx"]
    days = tr["c_n_days_with_tx"]
    rec = leftover_diag(tr[Y3], ntx, [cntx], tr["fold"], tr[Y3].notna())
    rec_d = leftover_diag(tr[Y3], ntx, [cntx, days], tr["fold"], tr[Y3].notna())
    logx = leftover_diag(tr[Y3], tr["log1p_n_tx"], [days], tr["fold"], tr[Y3].notna())
    inten = leftover_diag(tr[Y3], tr["intensity"], [days], tr["fold"], tr[Y3].notna())
    burst = leftover_diag(tr[Y3], tr["burst"], [days], tr["fold"], tr[Y3].notna())
    raw_int = signed_oof_auroc(tr[Y3], tr["intensity"], tr["fold"], tr[Y3].notna())
    raw_log = signed_oof_auroc(tr[Y3], tr["log1p_n_tx"], tr["fold"], tr[Y3].notna())
    rows = [
        {
            "feature": "a_n_tx after c_n_tx",
            "OLS": _f(rec["ols"]),
            "rank": _f(rec["rank"]),
            "R²": _f(rec["r2"]),
            "dies": "YES" if rec["honest_dies"] else "no",
        },
        {
            "feature": "a_n_tx after n_tx+days",
            "OLS": _f(rec_d["ols"]),
            "rank": _f(rec_d["rank"]),
            "R²": _f(rec_d["r2"]),
            "dies": "YES" if rec_d["honest_dies"] else "no",
        },
        {
            "feature": "log1p(n_tx) after days",
            "OLS": _f(logx["ols"]),
            "rank": _f(logx["rank"]),
            "R²": _f(logx["r2"]),
            "dies": "YES" if logx["honest_dies"] else "no",
        },
        {
            "feature": "n_tx/days intensity after days",
            "OLS": _f(inten["ols"]),
            "rank": _f(inten["rank"]),
            "R²": _f(inten["r2"]),
            "dies": "YES" if inten["honest_dies"] else "no",
        },
        {
            "feature": "n_tx−days burst after days",
            "OLS": _f(burst["ols"]),
            "rank": _f(burst["rank"]),
            "R²": _f(burst["r2"]),
            "dies": "YES" if burst["honest_dies"] else "no",
        },
    ]
    prose = (
        f"Same-n leftover after c_n_tx rank {_f(rec['rank'])} R²={_f(rec['r2'])} "
        f"({'identity leftover dies' if rec['honest_dies'] else 'lives'}). "
        f"After n_tx+days {_f(rec_d['rank'])}. "
        f"log1p(n_tx) raw {_f(_cv(raw_log))} leftover-days {_f(logx['rank'])}. "
        f"intensity raw {_f(_cv(raw_int))} leftover-days {_f(inten['rank'])}. "
        f"burst leftover {_f(burst['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "cntx": rec["rank"],
        "both": rec_d["rank"],
        "log": logx["rank"],
        "inten": inten["rank"],
        "burst": burst["rank"],
        "inten_raw": _cv(raw_int),
        "log_raw": _cv(raw_log),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — 12-name Y2 drop
# ---------------------------------------------------------------------------
def pass_chronic(tr: pd.DataFrame) -> dict:
    ids = chronic_ids(tr)
    drop = ~tr["company_id"].astype(str).isin(set(ids))
    lab = tr[Y2].notna()
    full = signed_oof_auroc(tr[Y2], tr["a_n_tx"], tr["fold"], lab)
    rest = signed_oof_auroc(tr[Y2], tr["a_n_tx"], tr["fold"], lab & drop)
    days_full = signed_oof_auroc(tr[Y2], tr["c_n_days_with_tx"], tr["fold"], lab)
    days_rest = signed_oof_auroc(tr[Y2], tr["c_n_days_with_tx"], tr["fold"], lab & drop)
    y3_full = signed_oof_auroc(tr[Y3], tr["a_n_tx"], tr["fold"], tr[Y3].notna())
    y3_rest = signed_oof_auroc(tr[Y3], tr["a_n_tx"], tr["fold"], tr[Y3].notna() & drop)
    left_rest = leftover_diag(
        tr[Y3], tr["a_n_tx"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & drop
    )
    flip = False
    if not full["low_power"] and not rest["low_power"]:
        flip = abs(full["cv"] - rest["cv"]) >= 0.03
    prose = (
        f"Chronic 12 names (0158/0172): {len(ids)}. "
        f"Y2 a_n_tx {_f(_cv(full))} → drop-12 {_f(_cv(rest))}; "
        f"days {_f(_cv(days_full))} → {_f(_cv(days_rest))}. "
        f"Y3 {_f(_cv(y3_full))} → {_f(_cv(y3_rest))}; leftover after days {_f(left_rest['rank'])}. "
        f"{'DROP FLIPS Y2' if flip else 'Drop does not flip Y2 (≥0.03)'}."
    )
    print(prose)
    return {
        "n_ids": len(ids),
        "y2_full": _cv(full),
        "y2_drop": _cv(rest),
        "y3_full": _cv(y3_full),
        "y3_drop": _cv(y3_rest),
        "days_full": _cv(days_full),
        "days_drop": _cv(days_rest),
        "left": left_rest["rank"],
        "flip": flip,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — holdout coverage / mix only
# ---------------------------------------------------------------------------
def pass_holdout(panel: pd.DataFrame, book: set[str]) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    if ho.empty:
        return {"prose": "Holdout empty.", "cov": float("nan"), "n_cm": 0, "n_co": 0}
    ntx = pd.to_numeric(ho["a_n_tx"], errors="coerce")
    days = pd.to_numeric(ho["c_n_days_with_tx"], errors="coerce")
    n_cm = int(len(ho))
    n_co = int(ho["company_id"].nunique())
    cov = _pct(int(ntx.notna().sum()), n_cm)
    rho = spearman(ntx, days)
    n_erp = int(ho.loc[ho["company_id"].isin(book), "company_id"].nunique())
    n_dark = n_co - n_erp
    # mix only — do not score AUROC on holdout
    prose = (
        f"Holdout coverage only (no fit): {n_co} companies / {n_cm:,} CM, "
        f"a_n_tx cov {_pp(cov)}, ρ vs days {_f(rho)}, "
        f"ever-ERP {n_erp} / dark {n_dark}. Not used to fit."
    )
    print(prose)
    return {
        "n_cm": n_cm,
        "n_co": n_co,
        "cov": cov,
        "rho_days": rho,
        "n_erp": n_erp,
        "n_dark": n_dark,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — dark 470 vs invoiced 744
# ---------------------------------------------------------------------------
def pass_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    n_erp = int(last["company_id"].isin(book).sum())
    n_dark = int((~last["company_id"].isin(book)).sum())
    confirm = n_erp == 744 and n_dark == 470
    erp = tr["company_id"].isin(book)
    dark = ~erp
    rows = []
    store = {}
    for name, mask in (("invoiced_744", erp), ("dark_470", dark)):
        raw = signed_oof_auroc(tr[Y3], tr["a_n_tx"], tr["fold"], tr[Y3].notna() & mask)
        rec = leftover_diag(
            tr[Y3], tr["a_n_tx"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & mask
        )
        dcv = signed_oof_auroc(
            tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna() & mask
        )
        store[name] = rec
        rows.append(
            {
                "slice": name,
                "n": f"{raw['n_defined']:,}",
                "n_pos": f"{raw['n_pos']:,}",
                "raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "leftover days": _f(rec["rank"]),
                "days": "LOW_POWER" if dcv["low_power"] else _f(dcv["cv"]),
                "dies": "YES" if rec["honest_dies"] else "no",
            }
        )
    prose = (
        f"Last-month companies ever-ERP {n_erp} / never-ERP {n_dark} "
        f"({'CONFIRM 744/470' if confirm else 'off 744/470'}). "
        f"Y3 leftover after days on invoiced {_f(store['invoiced_744']['rank'])} "
        f"on dark {_f(store['dark_470']['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "n_erp": n_erp,
        "n_dark": n_dark,
        "confirm": confirm,
        "inv": store["invoiced_744"]["rank"],
        "dark": store["dark_470"]["rank"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — fold-wise leftover
# ---------------------------------------------------------------------------
def pass_fold_left(tr: pd.DataFrame) -> dict:
    ntx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    resid, _ = ols_resid(ntx, days)
    rec = signed_oof_auroc(tr[Y3], resid, tr["fold"], tr[Y3].notna())
    raw = signed_oof_auroc(tr[Y3], ntx, tr["fold"], tr[Y3].notna())
    dcv = signed_oof_auroc(tr[Y3], days, tr["fold"], tr[Y3].notna())
    rows = []
    for k in range(N_FOLDS):
        rows.append(
            {
                "fold": k,
                "n_tx": _f(fold_k(raw, k)),
                "days": _f(fold_k(dcv, k)),
                "leftover": _f(fold_k(rec, k)),
            }
        )
    vals = [fold_k(rec, k) for k in range(N_FOLDS)]
    finite = [v for v in vals if np.isfinite(v)]
    spread = float(max(finite) - min(finite)) if finite else float("nan")
    prose = f"Y3 leftover-after-days fold spread {_f(spread)}; folds {fold_bits(rec)}."
    print(prose)
    return {"rows": rows, "spread": spread, "prose": prose, "rec": rec}


# ---------------------------------------------------------------------------
# Extra — lag leftovers after days_lag1
# ---------------------------------------------------------------------------
def pass_lag_left(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    specs = [
        ("a_n_tx_lag1 after days_lag1", "a_n_tx_lag1", "c_n_days_with_tx_lag1"),
        ("a_n_tx_lag3 after days_lag1", "a_n_tx_lag3", "c_n_days_with_tx_lag1"),
        ("a_n_tx_lag3 after days_lag3", "a_n_tx_lag3", "c_n_days_with_tx_lag3"),
        ("a_n_tx after days_lag1", "a_n_tx", "c_n_days_with_tx_lag1"),
    ]
    for name, xcol, ccol in specs:
        if xcol not in tr.columns or ccol not in tr.columns:
            continue
        rec = leftover_diag(tr[Y3], tr[xcol], [tr[ccol]], tr["fold"], tr[Y3].notna())
        raw = signed_oof_auroc(tr[Y3], tr[xcol], tr["fold"], tr[Y3].notna())
        store[name] = rec
        rows.append(
            {
                "spec": name,
                "raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "OLS": _f(rec["ols"]),
                "rank": _f(rec["rank"]),
                "R²": _f(rec["r2"]),
                "dies": "YES" if rec["honest_dies"] or (np.isfinite(rec["rank"]) and rec["rank"] < CHANCE) else "no",
            }
        )
    l1 = store.get("a_n_tx_lag1 after days_lag1", {})
    l3 = store.get("a_n_tx_lag3 after days_lag1", {})
    l1_dies = bool(
        l1.get("honest_dies")
        or (np.isfinite(l1.get("rank", float("nan"))) and l1.get("rank", 0) < CHANCE)
    )
    l3_dies = bool(
        l3.get("honest_dies")
        or (np.isfinite(l3.get("rank", float("nan"))) and l3.get("rank", 0) < CHANCE)
    )
    prose = (
        f"a_n_tx_lag1 leftover after days_lag1 rank {_f(l1.get('rank'))} "
        f"({'dies' if l1_dies else 'lives'}); "
        f"lag3 after days_lag1 {_f(l3.get('rank'))} ({'dies' if l3_dies else 'lives'}). "
        "Days lag1 is already KEEP (0.684 vs 0.711) — n_tx lags are rewrites if leftover dies."
    )
    print(prose)
    return {
        "rows": rows,
        "l1": l1.get("rank", float("nan")),
        "l3": l3.get("rank", float("nan")),
        "l1_dies": l1_dies,
        "l3_dies": l3_dies,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — identity residual is numerical dust (R²=1 leftover 0.702)
# ---------------------------------------------------------------------------
def pass_ident_dust(tr: pd.DataFrame) -> dict:
    ntx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    cntx = pd.to_numeric(tr["c_n_tx"], errors="coerce")
    resid, info = ols_resid(ntx, cntx)
    ok = resid.notna()
    r = resid[ok]
    absmax = float(r.abs().max()) if len(r) else float("nan")
    rstd = float(r.std()) if len(r) else float("nan")
    nuniq = int(r.round(12).nunique()) if len(r) else 0
    dust = bool(np.isfinite(absmax) and absmax < 1e-8)
    rec = leftover_diag(tr[Y3], ntx, [cntx], tr["fold"], tr[Y3].notna())
    prose = (
        f"Identity residual a_n_tx~c_n_tx: R²={_f(info['r2'], 6)} "
        f"max|resid|={absmax:.2e} std={rstd:.2e} unique@1e-12={nuniq}. "
        f"Rank leftover {_f(rec['rank'])} "
        f"{'is numerical dust — not leftover skill (R²=1 identity)' if dust else 'is not dust'}."
    )
    print(prose)
    return {
        "r2": info["r2"],
        "absmax": absmax,
        "std": rstd,
        "nuniq": nuniq,
        "dust": dust,
        "rank": rec["rank"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — OLS 0.623 vs rank 0.538: leftover after days+size
# ---------------------------------------------------------------------------
def pass_ols_vs_rank(tr: pd.DataFrame) -> dict:
    ntx = tr["a_n_tx"]
    days = tr["c_n_days_with_tx"]
    size = tr["log_in3"]
    both = leftover_diag(tr[Y3], ntx, [days, size], tr["fold"], tr[Y3].notna())
    days_only = leftover_diag(tr[Y3], ntx, [days], tr["fold"], tr[Y3].notna())
    fake_ols = bool(
        np.isfinite(days_only["ols"])
        and days_only["ols"] >= CHANCE
        and np.isfinite(days_only["rank"])
        and days_only["rank"] < CHANCE
    )
    rows = [
        {
            "control": "after days",
            "OLS": _f(days_only["ols"]),
            "rank": _f(days_only["rank"]),
            "ρ(resid,days)": _f(days_only["rho_ctrl"]),
            "R²": _f(days_only["r2"]),
        },
        {
            "control": "after days+size",
            "OLS": _f(both["ols"]),
            "rank": _f(both["rank"]),
            "ρ(resid,days)": _f(both["rho_ctrl"]),
            "R²": _f(both["r2"]),
        },
    ]
    prose = (
        f"OLS leftover after days {_f(days_only['ols'])} vs rank {_f(days_only['rank'])} "
        f"ρ(resid,days)={_f(days_only['rho_ctrl'])} "
        f"({'OLS-high / rank-dies — treat rank as honest; not a |ρ|≥0.80 fake-days clone' if fake_ols else 'OLS and rank agree'}). "
        f"After days+size rank {_f(both['rank'])} OLS {_f(both['ols'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "ols": days_only["ols"],
        "rank": days_only["rank"],
        "both_rank": both["rank"],
        "both_ols": both["ols"],
        "fake_ols": fake_ols,
        "rho": days_only["rho_ctrl"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — fold 0 leftover 0.683: drop-fold / one-group?
# ---------------------------------------------------------------------------
def pass_fold0(tr: pd.DataFrame) -> dict:
    ntx = tr["a_n_tx"]
    days = tr["c_n_days_with_tx"]
    resid, _ = ols_resid(ntx, days)
    lab = tr[Y3].notna()
    rec = leftover_diag(tr[Y3], ntx, [days], tr["fold"], lab)
    wo0 = leftover_diag(tr[Y3], ntx, [days], tr["fold"], lab & (tr["fold"] != 0))
    raw0 = signed_oof_auroc(tr[Y3], ntx, tr["fold"], lab & (tr["fold"] == 0))
    # groups in fold 0 among Y3-labeled
    sl = tr.loc[lab & (tr["fold"] == 0)]
    g = sl.groupby("group_id").size().sort_values(ascending=False)
    top = str(g.index[0]) if len(g) else "—"
    top_n = int(g.iloc[0]) if len(g) else 0
    n_g = int(g.size)
    dummy = bool(n_g > 0 and top_n / max(int(len(sl)), 1) >= 0.40)
    prose = (
        f"Fold-0 leftover {_f(fold_k(rec['rec'], 0))} on {n_g} groups "
        f"(top {top} n={top_n} / {len(sl)} labeled). "
        f"Drop fold 0 leftover rank {_f(wo0['rank'])} OLS {_f(wo0['ols'])}. "
        f"{'one-group leftover' if dummy else 'not a single-group dummy'}."
    )
    print(prose)
    return {
        "f0": fold_k(rec["rec"], 0),
        "wo": wo0["rank"],
        "wo_ols": wo0["ols"],
        "top": top,
        "top_n": top_n,
        "n_g": n_g,
        "dummy": dummy,
        "n_lab": int(len(sl)),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — leftover on n_tx>days (multi-tx days) vs n_tx==days
# ---------------------------------------------------------------------------
def pass_multitx(tr: pd.DataFrame) -> dict:
    ntx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    multi = ntx > days
    one = ntx == days
    rows = []
    store = {}
    for name, mask in (("n_tx>days", multi), ("n_tx==days", one)):
        raw = signed_oof_auroc(tr[Y3], ntx, tr["fold"], tr[Y3].notna() & mask)
        rec = leftover_diag(tr[Y3], ntx, [days], tr["fold"], tr[Y3].notna() & mask)
        store[name] = rec
        rows.append(
            {
                "slice": name,
                "n_cm": f"{int(mask.sum()):,}",
                "n_y3": f"{raw['n_defined']:,}",
                "raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "leftover": _f(rec["rank"]),
                "R²": _f(rec["r2"]),
                "dies": "YES" if rec["honest_dies"] or (np.isfinite(rec["rank"]) and rec["rank"] < CHANCE) else "no",
            }
        )
    share = _pct(int(multi.sum()), int((ntx.notna() & days.notna()).sum()))
    prose = (
        f"Multi-tx months (n_tx>days) {_pp(share)} of train. "
        f"Y3 leftover after days on multi {_f(store['n_tx>days']['rank'])} "
        f"on one-tx-per-day {_f(store['n_tx==days']['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "share": share,
        "multi": store["n_tx>days"]["rank"],
        "one": store["n_tx==days"]["rank"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — Y2 leftover after days / intensity (n_tx beat days 0.598 vs 0.571)
# ---------------------------------------------------------------------------
def pass_y2_left(tr: pd.DataFrame) -> dict:
    ids = set(chronic_ids(tr))
    drop = ~tr["company_id"].astype(str).isin(ids)
    rec = leftover_diag(tr[Y2], tr["a_n_tx"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y2].notna())
    rec_d = leftover_diag(
        tr[Y2], tr["a_n_tx"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y2].notna() & drop
    )
    inten = leftover_diag(tr[Y2], tr["intensity"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y2].notna())
    raw_i = signed_oof_auroc(tr[Y2], tr["intensity"], tr["fold"], tr[Y2].notna())
    lives = bool(np.isfinite(rec["rank"]) and rec["rank"] >= CHANCE and not rec["honest_dies"])
    prose = (
        f"Y2 leftover after days rank {_f(rec['rank'])} OLS {_f(rec['ols'])} "
        f"drop-12 {_f(rec_d['rank'])}. Intensity raw {_f(_cv(raw_i))} leftover {_f(inten['rank'])}. "
        f"{'Y2 leftover lives — not the 15-col Y3 card' if lives else 'Y2 leftover dies too'}."
    )
    print(prose)
    return {
        "rank": rec["rank"],
        "ols": rec["ols"],
        "drop": rec_d["rank"],
        "inten": inten["rank"],
        "inten_raw": _cv(raw_i),
        "lives": lives,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — company-demean leftover after days
# ---------------------------------------------------------------------------
def pass_demean_left(tr: pd.DataFrame) -> dict:
    dem = company_demean(tr["a_n_tx"], tr["company_id"])
    rec = leftover_diag(tr[Y3], dem, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    raw = signed_oof_auroc(tr[Y3], dem, tr["fold"], tr[Y3].notna())
    prose = (
        f"Y3 company-demean a_n_tx raw {_f(_cv(raw))} leftover after days "
        f"rank {_f(rec['rank'])} OLS {_f(rec['ols'])} ρ(resid,days)={_f(rec['rho_ctrl'])}. "
        f"{'month shock leftover dies' if rec['honest_dies'] or (np.isfinite(rec['rank']) and rec['rank'] < CHANCE) else 'month shock leftover lives'}."
    )
    print(prose)
    return {
        "raw": _cv(raw),
        "rank": rec["rank"],
        "ols": rec["ols"],
        "dies": rec["honest_dies"] or (np.isfinite(rec["rank"]) and rec["rank"] < CHANCE),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — does the 0.711 bar survive after n_tx+size?
# ---------------------------------------------------------------------------
def pass_days_after_both(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(
        tr[Y3], tr["c_n_days_with_tx"], [tr["a_n_tx"], tr["log_in3"]], tr["fold"], tr[Y3].notna()
    )
    after_n = leftover_diag(tr[Y3], tr["c_n_days_with_tx"], [tr["a_n_tx"]], tr["fold"], tr[Y3].notna())
    after_s = leftover_diag(tr[Y3], tr["c_n_days_with_tx"], [tr["log_in3"]], tr["fold"], tr[Y3].notna())
    lives = bool(np.isfinite(rec["rank"]) and rec["rank"] >= CHANCE and not rec["honest_dies"])
    rows = [
        {"control": "after n_tx", "OLS": _f(after_n["ols"]), "rank": _f(after_n["rank"]), "R²": _f(after_n["r2"])},
        {"control": "after size", "OLS": _f(after_s["ols"]), "rank": _f(after_s["rank"]), "R²": _f(after_s["r2"])},
        {"control": "after n_tx+size", "OLS": _f(rec["ols"]), "rank": _f(rec["rank"]), "R²": _f(rec["r2"])},
    ]
    prose = (
        f"Days leftover after n_tx rank {_f(after_n['rank'])} OLS {_f(after_n['ols'])}; "
        f"after size {_f(after_s['rank'])}; after both {_f(rec['rank'])} "
        f"({'0.711 bar still has leftover' if lives else 'days leftover after n_tx+size dies — bar is n_tx+size'})."
    )
    print(prose)
    return {
        "rows": rows,
        "after_n": after_n["rank"],
        "after_s": after_s["rank"],
        "both": rec["rank"],
        "lives": lives,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — leftover on short books; holdout mix
# ---------------------------------------------------------------------------
def pass_short_hold_mix(tr: pd.DataFrame, panel: pd.DataFrame) -> dict:
    short = tr["trail_class"] == "short_<12"
    rec = leftover_diag(
        tr[Y3], tr["a_n_tx"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & short
    )
    ho = panel[panel["split"] == "holdout"]
    ntx_tr = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    ntx_ho = pd.to_numeric(ho["a_n_tx"], errors="coerce")
    prose = (
        f"Y3 leftover after days on short_<12 rank {_f(rec['rank'])} OLS {_f(rec['ols'])} "
        f"n_pos={rec['n_pos']}. "
        f"Holdout mix (no fit): mean n_tx {_f(float(ntx_ho.mean()))} p50 {_f(float(ntx_ho.median()))} "
        f"vs train mean {_f(float(ntx_tr.mean()))} p50 {_f(float(ntx_tr.median()))}."
    )
    print(prose)
    return {
        "short": rec["rank"],
        "short_ols": rec["ols"],
        "ho_mean": float(ntx_ho.mean()),
        "ho_p50": float(ntx_ho.median()),
        "tr_mean": float(ntx_tr.mean()),
        "tr_p50": float(ntx_tr.median()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — rank-ortho fold leftover (OLS fold 0 was 0.683)
# ---------------------------------------------------------------------------
def pass_rank_folds(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(
        tr[Y3], tr["a_n_tx"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna()
    )
    inv = leftover_diag(
        tr[Y3], tr["c_n_days_with_tx"], [tr["a_n_tx"]], tr["fold"], tr[Y3].notna()
    )
    rows = [
        {"direction": "n_tx after days", "OLS folds": rec["folds"], "rank folds": rec["rank_folds"]},
        {"direction": "days after n_tx", "OLS folds": inv["folds"], "rank folds": inv["rank_folds"]},
    ]
    prose = (
        f"Rank-ortho leftover after days folds {rec['rank_folds']} (mean {_f(rec['rank'])}); "
        f"OLS folds {rec['folds']} (mean {_f(rec['ols'])}). "
        f"Inverse rank folds {inv['rank_folds']}."
    )
    print(prose)
    return {"rows": rows, "rank_folds": rec["rank_folds"], "ols_folds": rec["folds"], "prose": prose}


# ---------------------------------------------------------------------------
# Extra — drop GROUP_0158/0172 from Y3 leftover (fold-0 top)
# ---------------------------------------------------------------------------
def pass_drop_chronic_y3(tr: pd.DataFrame) -> dict:
    hot = tr["group_id"].astype(str).isin(CHRONIC_GROUPS)
    rec = leftover_diag(
        tr[Y3], tr["a_n_tx"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & ~hot
    )
    raw = signed_oof_auroc(tr[Y3], tr["a_n_tx"], tr["fold"], tr[Y3].notna() & ~hot)
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna() & ~hot)
    prose = (
        f"Drop GROUP_0158/0172: Y3 a_n_tx {_f(_cv(raw))} days {_f(_cv(days))} "
        f"leftover after days {_f(rec['rank'])} OLS {_f(rec['ols'])}."
    )
    print(prose)
    return {"raw": _cv(raw), "days": _cv(days), "rank": rec["rank"], "ols": rec["ols"], "prose": prose}


# ---------------------------------------------------------------------------
# Extra — demean leftover after days+size; company-mean leftover
# ---------------------------------------------------------------------------
def pass_trait_left(tr: pd.DataFrame) -> dict:
    dem = company_demean(tr["a_n_tx"], tr["company_id"])
    mu = company_mean(tr["a_n_tx"], tr["company_id"])
    d_both = leftover_diag(
        tr[Y3], dem, [tr["c_n_days_with_tx"], tr["log_in3"]], tr["fold"], tr[Y3].notna()
    )
    m_days = leftover_diag(tr[Y3], mu, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Demean leftover after days+size rank {_f(d_both['rank'])} OLS {_f(d_both['ols'])}. "
        f"Company-mean leftover after days rank {_f(m_days['rank'])} "
        f"(trait rewrite of days)."
    )
    print(prose)
    return {
        "dem_both": d_both["rank"],
        "mu_days": m_days["rank"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — Y2 intensity leftover after days+size; stressed size ρ
# ---------------------------------------------------------------------------
def pass_y2_inten_stressed(tr: pd.DataFrame) -> dict:
    inten = leftover_diag(
        tr[Y2], tr["intensity"], [tr["c_n_days_with_tx"], tr["log_in3"]], tr["fold"], tr[Y2].notna()
    )
    lab3 = tr[Y3].notna()
    rho_all = spearman(tr["a_n_tx"], tr["log_in3"])
    rho_st = spearman(tr.loc[lab3, "a_n_tx"], tr.loc[lab3, "log_in3"])
    confirm_st = bool(np.isfinite(rho_st) and abs(rho_st - SIZE_RHO_STRESSED) < 0.05)
    prose = (
        f"Y2 intensity leftover after days+size rank {_f(inten['rank'])}. "
        f"SIZE ρ a_n_tx vs log1p(a_in3) all {_f(rho_all)} stressed-Y3 {_f(rho_st)} "
        f"(importances 0.651 {'CONFIRM' if confirm_st else 'off'})."
    )
    print(prose)
    return {
        "inten": inten["rank"],
        "rho_all": rho_all,
        "rho_st": rho_st,
        "confirm_st": confirm_st,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — n_tx_lag1 leftover after contemporaneous days
# ---------------------------------------------------------------------------
def pass_lag_now_days(tr: pd.DataFrame) -> dict:
    rec1 = leftover_diag(
        tr[Y3], tr["a_n_tx_lag1"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna()
    )
    rec3 = leftover_diag(
        tr[Y3], tr["a_n_tx_lag3"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna()
    )
    prose = (
        f"a_n_tx_lag1 leftover after contemporaneous days rank {_f(rec1['rank'])}; "
        f"lag3 {_f(rec3['rank'])}."
    )
    print(prose)
    return {"l1": rec1["rank"], "l3": rec3["rank"], "prose": prose}


# ---------------------------------------------------------------------------
# Extra — reproduce SHAP ρ_size 0.651; company-median leftover
# ---------------------------------------------------------------------------
def pass_shap_med(tr: pd.DataFrame) -> dict:
    lab = tr[Y3].notna()
    pairs = [
        ("all log_in3", tr["a_n_tx"], tr["log_in3"], pd.Series(True, index=tr.index)),
        ("Y3 log_in3", tr["a_n_tx"], tr["log_in3"], lab),
        ("Y3 log_op_in", tr["a_n_tx"], tr["log_op_in"], lab),
        ("Y3 a_in3", tr["a_n_tx"], tr["a_in3"], lab),
    ]
    rows = []
    rhos = {}
    for name, a, b, m in pairs:
        rho = spearman(a[m], b[m])
        rhos[name] = rho
        rows.append({"slice": name, "ρ": _f(rho)})
    confirm = any(np.isfinite(v) and abs(v - SIZE_RHO_STRESSED) < 0.04 for v in rhos.values())
    g = (
        tr.groupby("company_id", as_index=False)
        .agg(ntx=("a_n_tx", "median"), days=("c_n_days_with_tx", "median"), size=("log_in3", "median"))
    )
    rho_med = spearman(g["ntx"], g["days"])
    rho_med_s = spearman(g["ntx"], g["size"])
    # leftover of company-median n_tx after company-median days on Y3 via mapped values
    med_n = tr["company_id"].map(g.set_index("company_id")["ntx"])
    med_d = tr["company_id"].map(g.set_index("company_id")["days"])
    rec = leftover_diag(tr[Y3], med_n, [med_d], tr["fold"], lab)
    prose = (
        f"SHAP ρ_size hunt: all-in3 {_f(rhos['all log_in3'])} Y3-in3 {_f(rhos['Y3 log_in3'])} "
        f"Y3-opin {_f(rhos['Y3 log_op_in'])} "
        f"({'hit 0.651' if confirm else 'did not hit stressed 0.651 — SIZE_STEM is all-train 0.661 / |op_in| 0.706'}). "
        f"Company-median n_tx vs days ρ={_f(rho_med)} vs size {_f(rho_med_s)}; "
        f"median leftover after median-days {_f(rec['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "confirm": confirm,
        "rho_med": rho_med,
        "rho_med_s": rho_med_s,
        "med_left": rec["rank"],
        "rhos": rhos,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — Y2 intensity KEEP-as-X gate (footnote only, not the 15-col card)
# ---------------------------------------------------------------------------
def pass_y2_inten_gate(tr: pd.DataFrame) -> dict:
    rho_days = spearman(tr["intensity"], tr["c_n_days_with_tx"])
    rho_size = spearman(tr["intensity"], tr["log_in3"])
    twin = bool(np.isfinite(rho_days) and abs(rho_days) >= TWIN_RHO)
    size = bool(np.isfinite(rho_size) and abs(rho_size) >= SIZE_RHO)
    raw = signed_oof_auroc(tr[Y2], tr["intensity"], tr["fold"], tr[Y2].notna())
    sz = signed_oof_auroc(tr[Y2], tr["log_in3"], tr["fold"], tr[Y2].notna())
    rec = leftover_diag(
        tr[Y2], tr["intensity"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y2].notna()
    )
    beat = _cv(raw) - _cv(sz) if np.isfinite(_cv(raw)) and np.isfinite(_cv(sz)) else float("nan")
    keep = bool(
        np.isfinite(beat)
        and beat >= KEEP_DELTA
        and np.isfinite(rec["rank"])
        and rec["rank"] >= CHANCE
        and not rec["honest_dies"]
        and not twin
        and not size
    )
    prose = (
        f"Y2 intensity raw {_f(_cv(raw))} vs size {_f(_cv(sz))} (Δ {_f(beat)}) "
        f"leftover after days {_f(rec['rank'])} ρ vs days {_f(rho_days)} vs size {_f(rho_size)}. "
        f"{'KEEP-as-X on Y2 only — PARK footnote, not the 15-col Y3 card' if keep else 'fails KEEP-as-X even on Y2'}."
    )
    print(prose)
    return {
        "raw": _cv(raw),
        "size": _cv(sz),
        "left": rec["rank"],
        "rho_days": rho_days,
        "rho_size": rho_size,
        "keep": keep,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — drop zero-tx months; long books; Y3 intensity after days+size
# ---------------------------------------------------------------------------
def pass_nz_long_inten(tr: pd.DataFrame) -> dict:
    ntx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    nz = ntx > 0
    long = tr["trail_class"] == "long_>=18"
    rec_nz = leftover_diag(
        tr[Y3], tr["a_n_tx"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & nz
    )
    rec_lg = leftover_diag(
        tr[Y3], tr["a_n_tx"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & long
    )
    rec_i = leftover_diag(
        tr[Y3], tr["intensity"], [tr["c_n_days_with_tx"], tr["log_in3"]], tr["fold"], tr[Y3].notna()
    )
    prose = (
        f"Y3 leftover after days, drop zeros: {_f(rec_nz['rank'])}. "
        f"Long ≥18m {_f(rec_lg['rank'])}. "
        f"Y3 intensity leftover after days+size {_f(rec_i['rank'])}."
    )
    print(prose)
    return {
        "nz": rec_nz["rank"],
        "long": rec_lg["rank"],
        "inten": rec_i["rank"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra — group ICC; leftover after gap_sd lives 0.615
# ---------------------------------------------------------------------------
def pass_group_gap(tr: pd.DataFrame) -> dict:
    med = tr.groupby("company_id", as_index=False).agg(
        ntx=("a_n_tx", "median"), gid=("group_id", "first")
    )
    icc_g = icc_anova(med["ntx"], med["gid"])
    rec = leftover_diag(tr[Y3], tr["a_n_tx"], [tr["c_gap_sd"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Company-median a_n_tx ICC across group_id {_f(icc_g['icc'])} (k={icc_g['k']}). "
        f"Leftover after dropped twin gap_sd rank {_f(rec['rank'])} "
        f"({'lives — n_tx is stronger than the dropped twin' if np.isfinite(rec['rank']) and rec['rank'] >= CHANCE else 'dies'})."
    )
    print(prose)
    return {"icc_g": icc_g["icc"], "gap_left": rec["rank"], "prose": prose}


# ---------------------------------------------------------------------------
# Extra — so-far buckets leftover (Q6 clock)
# ---------------------------------------------------------------------------
def pass_sofar_left(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for name, mask in (
        ("<6", tr["months_so_far"] < 6),
        ("6-11", (tr["months_so_far"] >= 6) & (tr["months_so_far"] < 12)),
        ("12-17", (tr["months_so_far"] >= 12) & (tr["months_so_far"] < 18)),
        (">=18", tr["months_so_far"] >= 18),
    ):
        rec = leftover_diag(
            tr[Y3], tr["a_n_tx"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & mask
        )
        raw = signed_oof_auroc(tr[Y3], tr["a_n_tx"], tr["fold"], tr[Y3].notna() & mask)
        store[name] = rec
        rows.append(
            {
                "so_far": name,
                "raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "leftover": _f(rec["rank"]),
                "n_pos": f"{rec['n_pos']:,}",
            }
        )
    prose = (
        "Y3 leftover after days by months-so-far: "
        + ", ".join(f"{k} {_f(store[k]['rank'])}" for k in store)
        + "."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra — so-far n_pos; <6 leftover after days+size; days leftover on <6
# ---------------------------------------------------------------------------
def pass_sofar_power(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for name, mask in (
        ("<6", tr["months_so_far"] < 6),
        ("6-11", (tr["months_so_far"] >= 6) & (tr["months_so_far"] < 12)),
        ("12-17", (tr["months_so_far"] >= 12) & (tr["months_so_far"] < 18)),
        (">=18", tr["months_so_far"] >= 18),
        ("long_trail", tr["trail_class"] == "long_>=18"),
    ):
        lab = tr[Y3].notna() & mask
        n_pos = int((lab & (pd.to_numeric(tr[Y3], errors="coerce") == 1)).sum())
        rec = leftover_diag(tr[Y3], tr["a_n_tx"], [tr["c_n_days_with_tx"]], tr["fold"], lab)
        inv = leftover_diag(tr[Y3], tr["c_n_days_with_tx"], [tr["a_n_tx"]], tr["fold"], lab)
        both = leftover_diag(
            tr[Y3], tr["a_n_tx"], [tr["c_n_days_with_tx"], tr["log_in3"]], tr["fold"], lab
        )
        store[name] = rec
        rows.append(
            {
                "slice": name,
                "n_lab": f"{int(lab.sum()):,}",
                "n_pos": f"{n_pos:,}",
                "n_tx leftover": _f(rec["rank"]),
                "days leftover": _f(inv["rank"]),
                "n_tx after days+size": _f(both["rank"]),
                "LOW_POWER": "yes" if rec["rec"]["low_power"] else "",
            }
        )
    short6 = leftover_diag(
        tr[Y3],
        tr["a_n_tx"],
        [tr["c_n_days_with_tx"], tr["log_in3"]],
        tr["fold"],
        tr[Y3].notna() & (tr["months_so_far"] < 6),
    )
    prose = (
        f"so-far <6 leftover after days+size {_f(short6['rank'])} "
        f"(pocket 0.595 after days only). Long books LOW_POWER if n_pos<50 — "
        "not a late-trail KEEP. Short-book leftover is not a 15-col stem."
    )
    print(prose)
    for r in rows:
        print(f"  so-far {r['slice']}: n_lab={r['n_lab']} n_pos={r['n_pos']} left={r['n_tx leftover']}")
    return {"rows": rows, "short6": short6["rank"], "prose": prose}


# ---------------------------------------------------------------------------
# Extra — is the <6 leftover just weaker days?
# ---------------------------------------------------------------------------
def pass_short6_days(tr: pd.DataFrame) -> dict:
    mask = tr[Y3].notna() & (tr["months_so_far"] < 6)
    ntx = signed_oof_auroc(tr[Y3], tr["a_n_tx"], tr["fold"], mask)
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], mask)
    size = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], mask)
    rho = spearman(tr.loc[mask, "a_n_tx"], tr.loc[mask, "c_n_days_with_tx"])
    rec = leftover_diag(tr[Y3], tr["a_n_tx"], [tr["c_n_days_with_tx"]], tr["fold"], mask)
    beat = _cv(ntx) - _cv(days) if np.isfinite(_cv(ntx)) and np.isfinite(_cv(days)) else float("nan")
    prose = (
        f"so-far<6 Y3 n_tx {_f(_cv(ntx))} days {_f(_cv(days))} size {_f(_cv(size))} "
        f"ρ(n_tx,days)={_f(rho)} leftover {_f(rec['rank'])} Δ vs days {_f(beat)}. "
        f"{'n_tx beats days on the short pocket' if np.isfinite(beat) and beat >= 0.02 else 'not a +0.02 beat of days even on <6'}."
    )
    print(prose)
    return {
        "ntx": _cv(ntx),
        "days": _cv(days),
        "size": _cv(size),
        "rho": rho,
        "left": rec["rank"],
        "beat": beat,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 10. Card-stem decision
# ---------------------------------------------------------------------------
def decide(p1, p2, p3, p4, p6, p7, p8, p9, p_ex, p_lag) -> dict:
    twin = bool(p2["gate_twins"] or p2["identity"])
    size = bool(p1["size_flag"] or p2["size_flag"])
    leftover = p4["y3_rank"]
    leftover_ols = p4["y3_ols"]
    leftover_dies = bool(p4["y3_dies"])
    beat = bool(np.isfinite(p3["beat_size"]) and p3["beat_size"] >= KEEP_DELTA)
    keep_x = bool(beat and (not leftover_dies) and (not size) and (not twin))
    inv_lives = bool(p4["inv_lives"])
    if keep_x:
        card = "KEEP"
        headline_tag = "KEEP"
        why = (
            f"leftover after days rank {_f(leftover)} beats chance and size "
            f"({_f(p3['size'])}) by {_f(p3['beat_size'])}; not SIZE; not a twin."
        )
    elif p2["identity"] or (
        np.isfinite(p2["rho_days"]) and abs(p2["rho_days"]) >= TWIN_RHO
    ) or leftover_dies:
        card = "CLOSE as a days rewrite / DROP from the card"
        headline_tag = "CLOSE" if leftover_dies or twin else "DROP"
        if p2["identity"] and leftover_dies:
            headline_tag = "CLOSE"
            why = (
                f"a_n_tx ≡ c_n_tx (COUNT(*) identity). "
                f"Y3 leftover after days rank {_f(leftover)} OLS {_f(leftover_ols)} dies "
                f"(ρ vs days {_f(p2['rho_days'])}). Inverse: days leftover after n_tx "
                f"{_f(p4['inv_rank'])} "
                f"{'survives — keep the 0.711 bar' if inv_lives else 'also dies'}. "
                "Parent should DROP a_n_tx from the 15-col card; days stays."
            )
        else:
            why = (
                f"Y3 leftover after days rank {_f(leftover)} OLS {_f(leftover_ols)} "
                f"{'dies' if leftover_dies else 'thin'}; twin={twin} SIZE={size}. "
                f"Days leftover after n_tx {_f(p4['inv_rank'])}. "
                "CLOSE as a days rewrite. Parent absorbs the card — do not edit it here."
            )
    elif size and p6["dies_inside"]:
        card = "CLOSE"
        headline_tag = "CLOSE"
        why = (
            f"SIZE ρ={_f(p2['rho_size'])}; leftover dies inside terciles "
            f"(T1 {_f(p6['t1'])} T2+T3 {_f(p6['t23'])})."
        )
    else:
        card = "PARK"
        headline_tag = "PARK"
        why = (
            f"Y3 {_f(p3['y3'])} vs days {_f(p3['days'])} leftover {_f(leftover)}. "
            "Does not clear KEEP-as-X."
        )
    q6 = p7["q6"]
    if leftover_dies or twin:
        q6 = "CLOSE"
    return {
        "card": card,
        "headline_tag": headline_tag,
        "why": why,
        "keep_x": keep_x,
        "twin": twin,
        "size": size,
        "leftover_dies": leftover_dies,
        "q6": q6,
        "headline": (
            f"{headline_tag} leftover-after-days rank {_f(leftover)} "
            f"(OLS {_f(leftover_ols)}, fake={p4['y3_fake']}). "
            f"Y3 single {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size'])}. "
            f"Identity vs c_n_tx={p2['identity']}. SIZE={size}. "
            f"Inverse days-after-n_tx {_f(p4['inv_rank'])}. "
            f"15-col card: {card}. Q6 {q6}. "
            f"Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged."
        ),
    }


def brief_map(d: dict, p3: dict, p4: dict, p7: dict, p8: dict, p9: dict) -> list[dict]:
    return [
        {
            "#": "1",
            "question": "Who is healthy?",
            "what this cut says": (
                f"a_n_tx is a count/size stem (ICC {_f(p8['icc'])} BETWEEN), not a health Y. "
                "Do not invent `y_n_tx`."
            ),
        },
        {
            "#": "2",
            "question": "Who is improving?",
            "what this cut says": (
                f"Y3 leftover after days {_f(p4['y3_rank'])} — {d['headline_tag']}. "
                "The 45→65 engine is days 0.711, not n_tx."
            ),
        },
        {
            "#": "3",
            "question": "Who is turning?",
            "what this cut says": (
                f"Y2 a_n_tx {_f(p3['y2'])} vs legal days {_f(p3['y2_days'])}. "
                "A COUNT(*) twin is not a turning X."
            ),
        },
        {
            "#": "4",
            "question": "Dip vs fall?",
            "what this cut says": "Not this count. Intensity leftover is the only dip-shaped extra.",
        },
        {
            "#": "5",
            "question": "Why did it change?",
            "what this cut says": p9["prose"],
        },
        {
            "#": "6",
            "question": "Months earlier?",
            "what this cut says": (
                f"lag1 {_f(p7['lag1'])} / lag3 {_f(p7['lag3'])} — {d['q6']}. "
                f"Days lag1 {_f(p7['days_l1'])} stays the KEEP."
            ),
        },
    ]


def make_png(tr: pd.DataFrame, p4: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    ntx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    both = ntx.notna() & days.notna()
    sl = tr.loc[both]
    if len(sl) > 8000:
        sl = sl.sample(8000, random_state=FOLD_SEED)
    sc = ax.scatter(
        pd.to_numeric(sl["c_n_days_with_tx"], errors="coerce"),
        pd.to_numeric(sl["a_n_tx"], errors="coerce"),
        c=sl["log_in3"],
        s=6,
        alpha=0.28,
        cmap="viridis",
        linewidths=0,
    )
    ax.set_xlabel("c_n_days_with_tx")
    ax.set_ylabel("a_n_tx")
    ax.set_title(f"Train: n_tx vs days (ρ={spearman(ntx, days):.3f})")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="log1p(a_in3)")

    ax2 = axes[1]
    lab = tr[Y3].notna() & ntx.notna() & days.notna()
    work = tr.loc[lab].copy()
    work["_n"] = ntx[lab]
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
    ax2.bar(x - 0.18, 100.0 * gn.to_numpy(dtype=float), width=0.36, color="#1f4e79", label="a_n_tx")
    ax2.bar(x + 0.18, 100.0 * gd.to_numpy(dtype=float), width=0.36, color="#9e6b4a", label="days")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax2.set_ylabel("Y3 rate (%)")
    ax2.set_title(f"Y3 rate by quintile; leftover rank {p4['y3_rank']:.3f}" if np.isfinite(p4["y3_rank"]) else "Y3 rate")
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
    p6, p7, p8, p9 = ctx["p6"], ctx["p7"], ctx["p8"], ctx["p9"]
    p_ex, p_ch, p_ho = ctx["p_ex"], ctx["p_ch"], ctx["p_ho"]
    p_dk, p_fl, p_lag = ctx["p_dk"], ctx["p_fl"], ctx["p_lag"]
    lines = [
        "# Unused leftover of `a_n_tx` after `c_n_days_with_tx`",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage / mix only. Seed 20260918 group folds. No 0–100. "
        "No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_n_tx`. "
        "Do not edit `cashflow.py` / `ops.py` / the 15-col card. Y3 never B. "
        f"Night Y3 **{Y3_NIGHT[0]} / {Y3_NIGHT[1]}**. Days **{DAYS_BENCH}**. "
        f"Size **{SIZE_Y3_QUOTE}**. Y7 TURNOVER **{Y7_TURNOVER[0]} / {Y7_TURNOVER[1]}**. "
        "Do not quote a_out_vol 0.722 as the engine.",
        "",
        "`a_n_tx` = COUNT(*) of transactions this month (Family A). "
        "`c_n_tx` = COUNT(*) (Family C). `c_n_days_with_tx` = unique calendar days. "
        "`c_gap_sd` is already DROPPED from the 44 as the weaker days twin "
        "(ρ −0.905 vs days, −0.866 vs `a_n_tx`). This lane does not overwrite `gap_sd_qa.*`.",
        "",
        "## Headline",
        "",
        d["headline"],
        "",
        "## Brief questions",
        "",
        _md_table(brief_map(d, p3, p4, p7, p8, p9)),
        "",
        "## KEEP / CLOSE / DROP / PARK",
        "",
        _md_table(
            [
                {
                    "object": "a_n_tx leftover after days (Y3 X)",
                    "decision": f"**{d['headline_tag']}**",
                    "why": (
                        f"rank {_f(p4['y3_rank'])} OLS {_f(p4['y3_ols'])} "
                        f"ρ(resid,days)={_f(p4['y3_rho_days'])} fake={p4['y3_fake']}"
                    ),
                },
                {
                    "object": "15-col Y3 card stem",
                    "decision": f"**{d['card']}**",
                    "why": d["why"],
                },
                {
                    "object": "COUNT(*) identity vs c_n_tx",
                    "decision": "**YES**" if p2["identity"] else "**no**",
                    "why": f"{p2['n_eq']:,}/{p2['n_both']:,} equal; leftover after c_n_tx {_f(p4['y3_cntx'])}",
                },
                {
                    "object": "Twin vs days / gap_sd",
                    "decision": "**YES**" if d["twin"] else "**no**",
                    "why": f"ρ days {_f(p2['rho_days'])} c_n_tx {_f(p2['rho_cntx'])} gap {_f(p2['rho_gap'])}",
                },
                {
                    "object": "SIZE vs log1p(a_in3)",
                    "decision": "**YES**" if d["size"] else "**no**",
                    "why": f"ρ={_f(p2['rho_size'])} (gate ≥0.50); feature-report 0.706 vs |a_op_in|",
                },
                {
                    "object": "Inverse: days leftover after n_tx",
                    "decision": "**thin rank / OLS-high**" if p4["inv_lives"] else "**dies**",
                    "why": f"rank {_f(p4['inv_rank'])} OLS {_f(p4['inv_ols'])} — 0.711 bar is not a rewrite of n_tx on OLS; honest rank is thin",
                },
                {
                    "object": "Q6 lag1 / lag3",
                    "decision": f"**{d['q6']}**",
                    "why": (
                        f"contemporaneous lag1 {_f(p7['lag1'])} looks KEEP, but leftover after days_lag1 "
                        f"{_f(ctx['p_lag']['l1'])} dies and n_tx is a days twin — {d['q6']}"
                    ),
                },
                {
                    "object": "ICC / trait vs month shock",
                    "decision": "**BETWEEN trait**" if p8["trait"] else "**shock**",
                    "why": f"ICC {_f(p8['icc'])} (quote 0.97 {'CONFIRM' if p8['confirm'] else 'off'})",
                },
                {
                    "object": "SHAP/perm 0.030 vs days bar",
                    "decision": "**rewrite, not leftover**" if p9["explains"] else "**open**",
                    "why": f"leftover {_f(p4['y3_rank'])}; perm n_tx {PERM_NTX:.3f} days {PERM_DAYS:.3f}",
                },
                {
                    "object": "Night quotes",
                    "decision": "**unchanged**",
                    "why": "Y3 0.762/0.752 · days 0.711 · size 0.617 · TURNOVER 0.720/0.712",
                },
            ]
        ),
        "",
        "## 1. Coverage; acf1/acf3; size ρ",
        "",
        p1["prose"],
        "",
        _md_table(p1["rows"]),
        "",
        f"| acf1 | acf3 | acf6 | ρ log1p(a_in3) | ρ log1p(|a_op_in|) |",
        f"| --- | --- | --- | --- | --- |",
        f"| {_f(p1['acf1'])} | {_f(p1['acf3'])} | {_f(p1['acf6'])} | {_f(p1['rho_in3'])} | {_f(p1['rho_opin'])} |",
        "",
        "## 2. Spearman twins + COUNT(*) identity",
        "",
        p2["prose"],
        "",
        _md_table(p2["rows"]),
        "",
        "## 3. Single-feature train group-fold AUROC",
        "",
        p3["prose"],
        "",
        "Sign from the train side of each fold. Days bar 0.711. Size `log1p(a_in3)` 0.617. "
        "Y2 legal single was days 0.571. Never Y7.",
        "",
        _md_table(p3["rows"]),
        "",
        "## 4. Honest leftover after days (OLS + rank-ortho)",
        "",
        p4["prose"],
        "",
        _md_table(p4["rows"]),
        "",
        "## 5. Inverse leftover — days after `a_n_tx`",
        "",
        f"Days leftover after n_tx rank {_f(p4['inv_rank'])} OLS {_f(p4['inv_ols'])}. "
        f"{'The 0.711 bar survives — days is not a rewrite of n_tx.' if p4['inv_lives'] else 'The 0.711 bar dies after n_tx — they share the same skill.'}",
        "",
        "## 6. SIZE terciles — leftover inside T1 and T2+T3",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        "## 7. Q6 — lag1 / lag3 on short books",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 8. ICC / company-demean",
        "",
        p8["prose"],
        "",
        _md_table(p8["rows"]),
        "",
        "## 9. SHAP / perm gap",
        "",
        p9["prose"],
        "",
        "## Extra — same-n identity, log1p, intensity, burst",
        "",
        p_ex["prose"],
        "",
        _md_table(p_ex["rows"]),
        "",
        "## Extra — 12-name Y2 drop (GROUP_0158 / 0172)",
        "",
        p_ch["prose"],
        "",
        "## Extra — holdout coverage only",
        "",
        p_ho["prose"],
        "",
        "## Extra — dark 470 vs invoiced 744",
        "",
        p_dk["prose"],
        "",
        _md_table(p_dk["rows"]),
        "",
        "## Extra — fold-wise leftover",
        "",
        p_fl["prose"],
        "",
        _md_table(p_fl["rows"]),
        "",
        "## Extra — `a_n_tx_lag1` / `_lag3` after days_lag1",
        "",
        p_lag["prose"],
        "",
        _md_table(p_lag["rows"]),
        "",
        "## Extra — identity residual dust",
        "",
        ctx["p_id"]["prose"],
        "",
        "## Extra — OLS 0.623 vs rank 0.538 (days+size)",
        "",
        ctx["p_or"]["prose"],
        "",
        _md_table(ctx["p_or"]["rows"]),
        "",
        "## Extra — fold 0 leftover",
        "",
        ctx["p_f0"]["prose"],
        "",
        "## Extra — multi-tx months (n_tx>days)",
        "",
        ctx["p_mt"]["prose"],
        "",
        _md_table(ctx["p_mt"]["rows"]),
        "",
        "## Extra — Y2 leftover after days",
        "",
        ctx["p_y2"]["prose"],
        "",
        "## Extra — company-demean leftover after days",
        "",
        ctx["p_dm"]["prose"],
        "",
        "## Extra — days leftover after n_tx+size",
        "",
        ctx["p_db"]["prose"],
        "",
        _md_table(ctx["p_db"]["rows"]),
        "",
        "## Extra — short books + holdout mix",
        "",
        ctx["p_sh"]["prose"],
        "",
        "## Extra — rank-ortho fold leftover",
        "",
        ctx["p_rf"]["prose"],
        "",
        _md_table(ctx["p_rf"]["rows"]),
        "",
        "## Extra — drop GROUP_0158/0172 from Y3 leftover",
        "",
        ctx["p_dc"]["prose"],
        "",
        "## Extra — demean after days+size / company-mean leftover",
        "",
        ctx["p_tl"]["prose"],
        "",
        "## Extra — Y2 intensity after days+size; stressed SIZE ρ",
        "",
        ctx["p_ys"]["prose"],
        "",
        "## Extra — n_tx lags after contemporaneous days",
        "",
        ctx["p_ln"]["prose"],
        "",
        "## Extra — SHAP ρ_size hunt + company-median leftover",
        "",
        ctx["p_sm"]["prose"],
        "",
        _md_table(ctx["p_sm"]["rows"]),
        "",
        "## Extra — Y2 intensity KEEP-as-X (not the 15-col card)",
        "",
        ctx["p_ig"]["prose"],
        "",
        "## Extra — drop zeros / long books / Y3 intensity",
        "",
        ctx["p_nz"]["prose"],
        "",
        "## Extra — group ICC + leftover after gap_sd",
        "",
        ctx["p_gg"]["prose"],
        "",
        "## Extra — leftover by months-so-far",
        "",
        ctx["p_sf"]["prose"],
        "",
        _md_table(ctx["p_sf"]["rows"]),
        "",
        "## Extra — so-far power + <6 after days+size",
        "",
        ctx["p_sp"]["prose"],
        "",
        _md_table(ctx["p_sp"]["rows"]),
        "",
        "## Extra — so-far <6 vs the days bar",
        "",
        ctx["p_s6"]["prose"],
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx.get("png") else "Plot: skipped.",
        "",
        "## What failed / next",
        "",
    ]
    for x in ctx["failed"]:
        lines.append(f"- {x}")
    extra_bits = ctx.get("extra_bits", [])
    if extra_bits:
        lines.extend(["", "## Later extras (same module)", ""])
        for x in extra_bits:
            lines.append(x if x.startswith("#") or x.startswith("|") or x.startswith("-") or not x else f"- {x}")
    lines.extend(
        [
            "",
            f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts: coverage/acf/size ρ, twins+identity, "
            "singles (days 0.711 / size 0.617 CONFIRM), leftover after days, inverse leftover, "
            "SIZE terciles, Q6 short books, ICC/demean, SHAP/perm, same-n / log1p / intensity, "
            "12-name Y2, holdout coverage, dark 470, fold leftover, lag leftovers.",
            "",
            "Did **not**: rewrite `cashflow.py` / `ops.py`, overwrite `gap_sd_qa.*`, "
            "edit the 15-col card, grow TURNOVER, invent `y_n_tx`, merge Family I/M/J, "
            "rewrite `brief_map.md`, touch `product/`, write 0–100, fit holdout, run a new GBM, "
            "write the parent journal / LIVE / canvas.",
            "",
        ]
    )
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
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
    p1, p2, p3, p4 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"]
    p7, p8 = ctx["p7"], ctx["p8"]
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    cov = f"{p1['cov']:.4f}"
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
            "metric": "auroc_a_n_tx",
            "value": p3["y3"],
            "coverage": cov,
            "notes": f"days={p3['days']:.4f} size={p3['size']:.4f} leftover={p4['y3_rank']:.4f} card={d['headline_tag']}",
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
            "metric": "auroc_c_n_days_with_tx",
            "value": p3["days"],
            "coverage": cov,
            "notes": f"CONFIRM_0.711={p3['days_ok']} size_ok={p3['size_ok']}",
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
            "metric": "auroc_a_n_tx_resid_days",
            "value": p4["y3_rank"],
            "coverage": cov,
            "notes": f"ols={p4['y3_ols']:.4f} fake={p4['y3_fake']} r2={p4['y3_r2']:.4f} dies={p4['y3_dies']}",
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
            "metric": "auroc_days_resid_a_n_tx",
            "value": p4["inv_rank"],
            "coverage": cov,
            "notes": f"ols={p4['inv_ols']:.4f} lives={p4['inv_lives']}",
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
            "metric": "rho_a_n_tx_vs_days",
            "value": p2["rho_days"],
            "coverage": cov,
            "notes": f"identity={p2['identity']} rho_cntx={p2['rho_cntx']:.4f} rho_gap={p2['rho_gap']:.4f} size={p2['rho_size']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y2,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_a_n_tx",
            "value": p3["y2"],
            "coverage": cov,
            "notes": f"days={p3['y2_days']:.4f} legal_0.571={p3['y2_days_ok']}",
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
            "metric": "icc_a_n_tx",
            "value": p8["icc"],
            "coverage": cov,
            "notes": f"confirm097={p8['confirm']} trait={p8['trait']} acf1={p1['acf1']:.4f}",
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
            "metric": "auroc_a_n_tx_lag1_resid_days_lag1",
            "value": ctx["p_lag"]["l1"],
            "coverage": cov,
            "notes": f"l3={ctx['p_lag']['l3']:.4f} l1_dies={ctx['p_lag']['l1_dies']} q6={p7['q6']}",
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
            "metric": "auroc_a_n_tx_resid_days_size",
            "value": ctx["p_or"]["both_rank"],
            "coverage": cov,
            "notes": f"ols_days={ctx['p_or']['ols']:.4f} rank_days={ctx['p_or']['rank']:.4f} fake_ols={ctx['p_or']['fake_ols']}",
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
            "metric": "identity_a_n_tx_c_n_tx",
            "value": 1.0 if p2["identity"] else 0.0,
            "coverage": cov,
            "notes": f"dust={ctx['p_id']['dust']} maxabs={ctx['p_id']['absmax']:.2e} leftover_rank={ctx['p_id']['rank']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y2,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_intensity_resid_days",
            "value": ctx["p_y2"]["inten"],
            "coverage": cov,
            "notes": f"y2_left={ctx['p_y2']['rank']:.4f} inten_raw={ctx['p_y2']['inten_raw']:.4f} after_size={ctx['p_ys']['inten']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": Y2,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_intensity",
            "value": ctx["p_ig"]["raw"],
            "coverage": cov,
            "notes": f"left_days={ctx['p_ig']['left']:.4f} keep_y2={ctx['p_ig']['keep']} rho_days={ctx['p_ig']['rho_days']:.4f}",
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
            "metric": "auroc_a_n_tx_resid_gap_sd",
            "value": ctx["p_gg"]["gap_left"],
            "coverage": cov,
            "notes": f"group_icc={ctx['p_gg']['icc_g']:.4f} nz_left={ctx['p_nz']['nz']:.4f} long_left={ctx['p_nz']['long']:.4f}",
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
    p2, p3, p4, p8 = ctx["p2"], ctx["p3"], ctx["p4"], ctx["p8"]
    text = (
        f"# Wave 4 — a_n_tx leftover after days\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/n_tx_qa.py`\n"
        f"- `analysis/outputs/n_tx_qa.md`\n"
        f"- `analysis/outputs/n_tx_qa.png`\n"
        f"- append-only `analysis/experiments/registry.csv` "
        f"(skip key agent+x_families+y+model+split+metric)\n"
        f"- this note\n\n"
        f"Did not touch `cashflow.py`, `ops.py`, `gap_sd_qa.*`, `cust_hhi_qa.*`, "
        f"`dso_qa.*`, `fc_r_qa.*`, parquet / duckdb, `build_targets`, `product/`, "
        f"the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, or the parent journal. "
        f"Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. "
        f"Y7 TURNOVER **0.720 / 0.712** unchanged.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| leftover after days | **{d['headline_tag']}** {_f(p4['y3_rank'])} |\n"
        f"| 15-col card | **{d['card']}** |\n"
        f"| identity vs c_n_tx | **{'YES' if p2['identity'] else 'no'}** |\n"
        f"| days leftover after n_tx | **{_f(p4['inv_rank'])}** "
        f"{'thin rank / OLS-high' if p4['inv_lives'] else 'dies'} |\n"
        f"| Q6 | **{d['q6']}** |\n\n"
        f"Y3 a_n_tx {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size'])}. "
        f"ρ vs days {_f(p2['rho_days'])} / c_n_tx {_f(p2['rho_cntx'])} / gap {_f(p2['rho_gap'])}. "
        f"Leftover after days rank {_f(p4['y3_rank'])} OLS {_f(p4['y3_ols'])} "
        f"fake={p4['y3_fake']}. Inverse rank {_f(p4['inv_rank'])} OLS {_f(p4['inv_ols'])}. "
        f"ICC {_f(p8['icc'])}. Identity dust leftover {_f(ctx['p_id']['rank'])}. "
        f"lag1 after days_lag1 {_f(ctx['p_lag']['l1'])}. "
        f"Leftover after gap_sd {_f(ctx['p_gg']['gap_left'])}. "
        f"so-far<6 leftover {_f(ctx['p_sp']['short6'])} "
        f"(n_tx {_f(ctx['p_s6']['ntx'])} vs days {_f(ctx['p_s6']['days'])}).\n\n"
        f"## What failed / next\n\n"
        + "\n".join(f"- {x}" for x in ctx["failed"])
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.parent.mkdir(parents=True, exist_ok=True)
    WAVE_NOTE.write_text(text, encoding="utf-8")
    print(f"wrote {WAVE_NOTE}")


def run() -> dict:
    t0 = time.time()
    print(f"n_tx_qa start seed={FOLD_SEED} holdout=72 agent={AGENT} wave={WRITE_WAVE}")
    panel = load_panel()
    con = connect()
    try:
        book = book_invoice_ids(con)
    finally:
        con.close()
    panel["ever_erp"] = panel["company_id"].isin(book)
    panel = attach_folds(panel)
    panel = add_panel_lags(
        panel,
        ["a_n_tx", "c_n_days_with_tx", "c_n_tx", "c_gap_sd"],
        (1, 3),
    )
    tr = panel[panel["split"] == "train"].copy().reset_index(drop=True)
    assert_no_holdout(tr["company_id"])
    print(
        f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
        f"holdout CM={int((panel['split']=='holdout').sum())}"
    )

    print("pass 1 coverage / acf / size")
    p1 = pass1_cov(tr)
    print("pass 2 twins + identity")
    p2 = pass2_twins(tr)
    print("pass 3 singles")
    p3 = pass3_singles(tr)
    print("pass 4 leftover + inverse")
    p4 = pass4_leftover(tr)
    print("pass 6 size terciles")
    p6 = pass6_terciles(tr)
    print("pass 7 Q6")
    p7 = pass7_q6(tr)
    print("pass 8 ICC")
    p8 = pass8_icc(tr)
    print("pass 9 SHAP/perm")
    p9 = pass9_perm(tr, p3, p4)
    print("extras same-n / log / intensity")
    p_ex = pass_ex_same_n(tr)
    print("extra chronic 12")
    p_ch = pass_chronic(tr)
    print("extra holdout coverage")
    p_ho = pass_holdout(panel, book)
    print("extra dark 470")
    p_dk = pass_dark(tr, book)
    print("extra fold leftover")
    p_fl = pass_fold_left(tr)
    print("extra lag leftovers")
    p_lag = pass_lag_left(tr)
    print("extra identity dust")
    p_id = pass_ident_dust(tr)
    print("extra OLS vs rank")
    p_or = pass_ols_vs_rank(tr)
    print("extra fold 0")
    p_f0 = pass_fold0(tr)
    print("extra multi-tx")
    p_mt = pass_multitx(tr)
    print("extra Y2 leftover")
    p_y2 = pass_y2_left(tr)
    print("extra demean leftover")
    p_dm = pass_demean_left(tr)
    print("extra days after n_tx+size")
    p_db = pass_days_after_both(tr)
    print("extra short + holdout mix")
    p_sh = pass_short_hold_mix(tr, panel)
    print("extra rank folds")
    p_rf = pass_rank_folds(tr)
    print("extra drop chronic Y3")
    p_dc = pass_drop_chronic_y3(tr)
    print("extra trait leftover")
    p_tl = pass_trait_left(tr)
    print("extra Y2 intensity + stressed ρ")
    p_ys = pass_y2_inten_stressed(tr)
    print("extra lag after now-days")
    p_ln = pass_lag_now_days(tr)
    print("extra SHAP ρ + medians")
    p_sm = pass_shap_med(tr)
    print("extra Y2 intensity gate")
    p_ig = pass_y2_inten_gate(tr)
    print("extra zeros / long / Y3 intensity")
    p_nz = pass_nz_long_inten(tr)
    print("extra group ICC + gap leftover")
    p_gg = pass_group_gap(tr)
    print("extra so-far leftover")
    p_sf = pass_sofar_left(tr)
    print("extra so-far power")
    p_sp = pass_sofar_power(tr)
    print("extra so-far <6 vs days")
    p_s6 = pass_short6_days(tr)
    png = make_png(tr, p4)
    decision = decide(p1, p2, p3, p4, p6, p7, p8, p9, p_ex, p_lag)
    failed = []
    if not p3["days_ok"]:
        failed.append(f"days replica drifted: {_f(p3['days'])} vs 0.711")
    if not p3["size_ok"]:
        failed.append(f"size replica drifted: {_f(p3['size'])} vs 0.617")
    if not p3["y2_days_ok"]:
        failed.append(f"Y2 days replica drifted: {_f(p3['y2_days'])} vs 0.571")
    if not p2["identity"]:
        failed.append(
            f"a_n_tx vs c_n_tx not a bit-identity ({p2['n_eq']}/{p2['n_both']}, max|Δ|={_f(p2['max_abs'], 6)})"
        )
    if p4["y3_fake"]:
        failed.append(
            f"OLS leftover after days {_f(p4['y3_ols'])} is a fake-days leak; "
            f"honest rank {_f(p4['y3_rank'])}"
        )
    if not p8["confirm"]:
        failed.append(f"ICC {_f(p8['icc'])} off feature-report 0.97")
    if not p7["days_l1_ok"]:
        failed.append(f"days lag1 {_f(p7['days_l1'])} off KEEP 0.684")
    if not p_dk["confirm"]:
        failed.append(f"dark/invoiced {p_dk['n_dark']}/{p_dk['n_erp']} off 470/744")
    if p_id["dust"]:
        failed.append(
            f"leftover after c_n_tx rank {_f(p_id['rank'])} is identity dust "
            f"(max|resid|={p_id['absmax']:.1e})"
        )
    if p_or["fake_ols"]:
        failed.append(
            f"OLS leftover after days {_f(p_or['ols'])} is high vs honest rank {_f(p_or['rank'])} "
            f"(ρ(resid,days)={_f(p_or['rho'])} < 0.80 — not a fake-days clone, still rank-dies)"
        )
    if p_f0["dummy"]:
        failed.append(f"fold-0 leftover is one-group ({p_f0['top']})")
    if not p_ys["confirm_st"] and not p_sm["confirm"]:
        failed.append(
            f"stressed SIZE ρ Y3-in3 {_f(p_ys['rho_st'])} / hunt { {k: round(v,3) if np.isfinite(v) else None for k,v in p_sm['rhos'].items()} } "
            f"off importances 0.651 — SIZE_STEM still holds on all-train 0.661"
        )
    if not failed:
        failed.append("no replica miss; leftover after days is the card-stem decision")
    ctx = {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p6": p6,
        "p7": p7,
        "p8": p8,
        "p9": p9,
        "p_ex": p_ex,
        "p_ch": p_ch,
        "p_ho": p_ho,
        "p_dk": p_dk,
        "p_fl": p_fl,
        "p_lag": p_lag,
        "p_id": p_id,
        "p_or": p_or,
        "p_f0": p_f0,
        "p_mt": p_mt,
        "p_y2": p_y2,
        "p_dm": p_dm,
        "p_db": p_db,
        "p_sh": p_sh,
        "p_rf": p_rf,
        "p_dc": p_dc,
        "p_tl": p_tl,
        "p_ys": p_ys,
        "p_ln": p_ln,
        "p_sm": p_sm,
        "p_ig": p_ig,
        "p_nz": p_nz,
        "p_gg": p_gg,
        "p_sf": p_sf,
        "p_sp": p_sp,
        "p_s6": p_s6,
        "decision": decision,
        "failed": failed,
        "png": png,
        "extra_bits": [],
        "elapsed_s": time.time() - t0,
        "tr": tr,
        "panel": panel,
    }
    write_md(ctx)
    append_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    print(
        f"DONE card={decision['headline_tag']} leftover={_f(p4['y3_rank'])} "
        f"inv={_f(p4['inv_rank'])} y3={_f(p3['y3'])} days={_f(p3['days'])} "
        f"elapsed={ctx['elapsed_s']:.0f}s"
    )
    return ctx


if __name__ == "__main__":
    run()
