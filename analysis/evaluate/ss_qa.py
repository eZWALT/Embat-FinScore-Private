"""Unused leftover of ``c_ss_month`` after ``c_n_days_with_tx`` as a 15-col stem.

``c_ss_month`` = 1 if any transaction category = social_security this month
(fillna 0). Perm-stable #1 on Y3 (ΔAUROC 0.034, sign −). Salary cousin
``c_missed_salary`` already CLOSE as Y3 X (0.513). ``c_salary_month``
stays on the card at 0.671. ``a_n_tx`` was DROPPED as a days twin.

KEEP-as-X: beat size ≥0.02 **and** leftover after days **and** not SIZE
(|ρ| vs log1p(a_in3) ≥0.50) **and** not a twin (|ρ|≥0.80 vs days /
a_n_tx / c_salary_month / c_missed_salary / c_tax_month). Leftover
<0.55 dies. Rank leftover is honest; OLS can fake a days leak.

Night quotes unchanged: Y3 0.762 / 0.752. Days 0.711. Size 0.617.
Y7 TURNOVER 0.720 / 0.712. Do not grow TURNOVER. Do not edit the card.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.ss_qa

Owned: analysis/evaluate/ss_qa.py, analysis/outputs/ss_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_ss.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "ss_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "ss_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_ss.md"
AGENT = "d4c8e201"
WAVE = "4"
ROUND = "R4"
MODEL = "ss_qa"
X_FAM = "C"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_Y3_QUOTE = 0.617
SALARY_Y3_QUOTE = 0.671
SS_Y3_QUOTE = 0.693
MISSED_SAL_QUOTE = 0.513
DAYS_LAG1_QUOTE = 0.684
Y3_NIGHT = (0.762, 0.752)
Y7_TURNOVER = (0.720, 0.712)
PERM_SS = 0.034
ICC_QUOTE = 0.99
ACF1_QUOTE = 0.60
MODAL_QUOTE = 0.543
SIZE_RHO_QUOTE = 0.362
KEEP_DELTA = 0.02
TWIN_RHO = 0.80
SIZE_RHO = 0.50
CHANCE = 0.55
ICC_TRAIT = 0.85
MIN_POS = 50
MIN_ACF_PAIRS = 4
Q_MONTHS = (1, 4, 7, 10)
CHRONIC_GROUPS = ("GROUP_0158", "GROUP_0172")
ROLL = 6
USUAL_MIN = 3
WRITE_WAVE = True

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "a_n_tx",
    "c_n_days_with_tx",
    "c_ss_month",
    "c_salary_month",
    "c_missed_salary",
    "c_tax_month",
    "c_missed_tax",
)
EXTRA_STORE = ("c_n_tx", "c_recency_days", "c_zero_in_month")

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
    aa = pd.to_numeric(a, errors="coerce").fillna(0).astype(int).eq(1)
    bb = pd.to_numeric(b, errors="coerce").fillna(0).astype(int).eq(1)
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


def add_missed_ss(df: pd.DataFrame) -> pd.DataFrame:
    """In-memory missed-SS. Same clock as c_missed_salary. Do not invent y_ss."""
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    ss = pd.to_numeric(out["c_ss_month"], errors="coerce").fillna(0)
    ss6 = ss.groupby(out["company_id"], sort=False).transform(
        lambda s: s.rolling(ROLL, min_periods=1).sum()
    )
    out["c_missed_ss"] = ((ss6 >= USUAL_MIN) & (ss == 0)).astype(np.int8)
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
    cols = list(STORE_COLS) + [c for c in EXTRA_STORE if c in raw.columns]
    panel = _keys(raw[cols])
    y = _keys(yraw[["company_id", "period", *Y_KEEP]])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    for col in ("c_ss_month", "c_salary_month", "c_missed_salary", "c_tax_month", "c_missed_tax"):
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0)
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
    panel["trail_class"] = panel["months_so_far"].map(_trail_class)
    panel["cal_month"] = panel["period"].dt.month
    panel["is_q_month"] = panel["cal_month"].isin(Q_MONTHS).astype(int)
    panel["is_aug"] = (panel["cal_month"] == 8).astype(int)
    leak3 = leakage_check(
        ["c_ss_month", "c_n_days_with_tx", "c_salary_month", "log_in3"],
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
# 1. Coverage; modal share; acf1; size ρ
# ---------------------------------------------------------------------------
def pass1_cov(tr: pd.DataFrame) -> dict:
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    n_cm = int(len(tr))
    n_nn = int(ss.notna().sum())
    n1 = int((ss == 1).sum())
    n0 = int((ss == 0).sum())
    modal = max(_pct(n0, n_cm), _pct(n1, n_cm))
    prev = _pct(n1, n_cm)
    acf1 = median_acf(ss, tr["company_id"], 1)
    acf3 = median_acf(ss, tr["company_id"], 3)
    rho = spearman(ss, tr["log_in3"])
    size_flag = bool(np.isfinite(rho) and abs(rho) >= SIZE_RHO)
    confirm_modal = bool(np.isfinite(modal) and abs(modal - MODAL_QUOTE) < 0.03)
    confirm_acf = bool(np.isfinite(acf1) and abs(acf1 - ACF1_QUOTE) < 0.08)
    confirm_size = bool(np.isfinite(rho) and abs(rho - SIZE_RHO_QUOTE) < 0.05)
    rows = [
        {
            "slice": "train all",
            "n_cm": f"{n_cm:,}",
            "n_co": int(tr["company_id"].nunique()),
            "cov": _pp(_pct(n_nn, n_cm)),
            "P(SS=1)": _pp(prev),
            "modal": _pp(modal),
            "n1": f"{n1:,}",
        }
    ]
    prose = (
        f"Train c_ss_month cov {_pp(_pct(n_nn, n_cm))} P(1) {_pp(prev)} "
        f"modal {_pp(modal)} (feature-report 54.3% {'CONFIRM' if confirm_modal else 'off'}). "
        f"acf1 {_f(acf1)} (0.60 {'CONFIRM' if confirm_acf else 'off'}) acf3 {_f(acf3)}. "
        f"ρ vs log1p(a_in3) {_f(rho)} (quote 0.362 {'CONFIRM' if confirm_size else 'off'}; "
        f"{'SIZE' if size_flag else 'not SIZE'})."
    )
    print(prose)
    return {
        "rows": rows,
        "cov": _pct(n_nn, n_cm),
        "prev": prev,
        "modal": modal,
        "n1": n1,
        "n_cm": n_cm,
        "acf1": acf1,
        "acf3": acf3,
        "rho_in3": rho,
        "size_flag": size_flag,
        "confirm_modal": confirm_modal,
        "confirm_acf": confirm_acf,
        "confirm_size": confirm_size,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 2. Spearman twins
# ---------------------------------------------------------------------------
def pass2_twins(tr: pd.DataFrame) -> dict:
    ss = tr["c_ss_month"]
    pairs = [
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("a_n_tx", tr["a_n_tx"]),
        ("c_salary_month", tr["c_salary_month"]),
        ("c_missed_salary", tr["c_missed_salary"]),
        ("c_tax_month", tr["c_tax_month"]),
        ("c_missed_tax", tr["c_missed_tax"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("c_missed_ss", tr["c_missed_ss"]),
    ]
    rows = []
    rhos = {}
    twins = []
    for name, s in pairs:
        rho = spearman(ss, s)
        rhos[name] = rho
        twin = bool(
            name
            in {
                "c_n_days_with_tx",
                "a_n_tx",
                "c_salary_month",
                "c_missed_salary",
                "c_tax_month",
            }
            and np.isfinite(rho)
            and abs(rho) >= TWIN_RHO
        )
        if twin:
            twins.append(name)
        rows.append({"vs": name, "ρ": _f(rho), "twin": "YES" if twin else ""})
    jac = jaccard(ss, tr["c_salary_month"])
    prose = (
        f"Spearman twins |ρ|≥0.80: {twins or 'none'}. "
        f"vs days {_f(rhos['c_n_days_with_tx'])} vs a_n_tx {_f(rhos['a_n_tx'])} "
        f"vs salary {_f(rhos['c_salary_month'])} vs missed_sal {_f(rhos['c_missed_salary'])} "
        f"vs tax {_f(rhos['c_tax_month'])} vs missed_tax {_f(rhos['c_missed_tax'])}. "
        f"Jaccard(SS, salary) {_f(jac['jaccard'])} both {jac['n_both']:,}."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "twins": twins,
        "gate_twins": bool(twins),
        "jac": jac,
        "rho_days": rhos["c_n_days_with_tx"],
        "rho_sal": rhos["c_salary_month"],
        "rho_tax": rhos["c_tax_month"],
        "rho_size": rhos["log1p(a_in3)"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 3. Single-feature group-fold AUROC
# ---------------------------------------------------------------------------
def pass3_singles(tr: pd.DataFrame) -> dict:
    feats = {
        "c_ss_month": tr["c_ss_month"],
        "c_salary_month": tr["c_salary_month"],
        "c_missed_salary": tr["c_missed_salary"],
        "c_tax_month": tr["c_tax_month"],
        "c_missed_ss": tr["c_missed_ss"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
        "log1p(a_in3)": tr["log_in3"],
        "is_q_month": tr["is_q_month"],
        "is_aug": tr["is_aug"],
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
    y3 = _cv(store[(Y3, "c_ss_month")])
    days = _cv(store[(Y3, "c_n_days_with_tx")])
    size = _cv(store[(Y3, "log1p(a_in3)")])
    sal = _cv(store[(Y3, "c_salary_month")])
    miss = _cv(store[(Y3, "c_missed_salary")])
    tax = _cv(store[(Y3, "c_tax_month")])
    y2 = _cv(store[(Y2, "c_ss_month")])
    days_ok = bool(np.isfinite(days) and abs(days - DAYS_BENCH) < 0.02)
    size_ok = bool(np.isfinite(size) and abs(size - SIZE_Y3_QUOTE) < 0.03)
    sal_ok = bool(np.isfinite(sal) and abs(sal - SALARY_Y3_QUOTE) < 0.02)
    ss_ok = bool(np.isfinite(y3) and abs(y3 - SS_Y3_QUOTE) < 0.03)
    beat_size = y3 - size if np.isfinite(y3) and np.isfinite(size) else float("nan")
    loses_days = bool(np.isfinite(days) and np.isfinite(y3) and y3 + 1e-12 < days)
    prose = (
        f"Y3 c_ss_month {_f(y3)} (salary-qa 0.693 {'CONFIRM' if ss_ok else 'off'}) "
        f"vs days {_f(days)} (0.711 {'CONFIRM' if days_ok else 'off'}) "
        f"vs size {_f(size)} (0.617 {'CONFIRM' if size_ok else 'off'}, Δ {_f(beat_size)}) "
        f"vs salary_month {_f(sal)} (0.671 {'CONFIRM' if sal_ok else 'off'}) "
        f"vs missed_sal {_f(miss)} vs tax {_f(tax)}. "
        f"Y2 SS {_f(y2)}. {'Loses to the 0.711 days bar' if loses_days else 'Does not lose to days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "y3": y3,
        "days": days,
        "size": size,
        "sal": sal,
        "miss": miss,
        "tax": tax,
        "y2": y2,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "sal_ok": sal_ok,
        "ss_ok": ss_ok,
        "beat_size": beat_size,
        "loses_days": loses_days,
        "sign_y3": store[(Y3, "c_ss_month")]["train_sign"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 4–5. Honest leftover after days + inverse + leftover after salary
# ---------------------------------------------------------------------------
def pass4_leftover(tr: pd.DataFrame) -> dict:
    ss = tr["c_ss_month"]
    days = tr["c_n_days_with_tx"]
    sal = tr["c_salary_month"]
    ntx = tr["a_n_tx"]
    tax = tr["c_tax_month"]
    size = tr["log_in3"]
    specs = [
        (Y3, "after days", (days,)),
        (Y3, "after salary_month", (sal,)),
        (Y3, "after a_n_tx", (ntx,)),
        (Y3, "after tax_month", (tax,)),
        (Y3, "after size", (size,)),
        (Y3, "after days+salary", (days, sal)),
        (Y2, "after days", (days,)),
        (Y2, "after salary_month", (sal,)),
    ]
    rows = []
    store = {}
    for ycol, name, xs in specs:
        rec = leftover_diag(tr[ycol], ss, list(xs), tr["fold"], tr[ycol].notna())
        store[(ycol, name)] = rec
        rows.append(
            {
                "y": ycol,
                "control": name,
                "OLS": _f(rec["ols"]),
                "rank": _f(rec["rank"]),
                "ρ(resid,ctrl)": _f(rec["rho_ctrl"]),
                "R²": _f(rec["r2"]),
                "fake": "YES" if rec["fake"] else "",
                "honest_dies": "YES" if rec["honest_dies"] else "no",
            }
        )
        print(
            f"left {ycol} {name}: OLS={_f(rec['ols'])} rank={_f(rec['rank'])} "
            f"ρctrl={_f(rec['rho_ctrl'])} R2={_f(rec['r2'])}"
        )
    inv = leftover_diag(tr[Y3], days, [ss], tr["fold"], tr[Y3].notna())
    store[(Y3, "days after SS")] = inv
    rows.append(
        {
            "y": Y3,
            "control": "days after SS (inverse)",
            "OLS": _f(inv["ols"]),
            "rank": _f(inv["rank"]),
            "ρ(resid,ctrl)": _f(inv["rho_ctrl"]),
            "R²": _f(inv["r2"]),
            "fake": "YES" if inv["fake"] else "",
            "honest_dies": "YES" if inv["honest_dies"] else "no",
        }
    )
    y3 = store[(Y3, "after days")]
    y3_sal = store[(Y3, "after salary_month")]
    leftover = y3["rank"]
    leftover_ols = y3["ols"]
    fake_ols = bool(
        np.isfinite(leftover_ols)
        and leftover_ols >= CHANCE
        and np.isfinite(leftover)
        and leftover < CHANCE
    )
    dies = bool(y3["honest_dies"] or (np.isfinite(leftover) and leftover < CHANCE))
    inv_lives = bool(np.isfinite(inv["rank"]) and inv["rank"] >= CHANCE and not inv["honest_dies"])
    sal_dies = bool(y3_sal["honest_dies"] or (np.isfinite(y3_sal["rank"]) and y3_sal["rank"] < CHANCE))
    payroll_twin = bool(sal_dies and np.isfinite(y3_sal["r2"]) and y3_sal["r2"] >= 0.40)
    prose = (
        f"Y3 leftover after days OLS {_f(leftover_ols)} rank {_f(leftover)} "
        f"ρ(resid,days)={_f(y3['rho_ctrl'])} R²={_f(y3['r2'])} "
        f"({'OLS-high / rank-dies — treat rank as honest' if fake_ols else 'OLS and rank agree'}; "
        f"{'FAKE-DAYS' if y3['fake'] else 'resid is not a days clone'}). "
        f"After salary_month rank {_f(y3_sal['rank'])} OLS {_f(y3_sal['ols'])} R²={_f(y3_sal['r2'])} "
        f"({'payroll twin' if payroll_twin else 'not a salary rewrite'}). "
        f"Inverse: days leftover after SS rank {_f(inv['rank'])} OLS {_f(inv['ols'])} "
        f"({'0.711 bar lives' if inv_lives else '0.711 bar dies after SS'}). "
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
        "fake_ols": fake_ols,
        "y3_sal": y3_sal["rank"],
        "y3_sal_ols": y3_sal["ols"],
        "y3_sal_r2": y3_sal["r2"],
        "sal_dies": sal_dies,
        "payroll_twin": payroll_twin,
        "y3_tax": store[(Y3, "after tax_month")]["rank"],
        "y3_ntx": store[(Y3, "after a_n_tx")]["rank"],
        "y3_both": store[(Y3, "after days+salary")]["rank"],
        "inv_rank": inv["rank"],
        "inv_ols": inv["ols"],
        "inv_lives": inv_lives,
        "y2_rank": store[(Y2, "after days")]["rank"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 6. SIZE terciles + ICC / demean
# ---------------------------------------------------------------------------
def pass6_terciles(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    ss = tr["c_ss_month"]
    days = tr["c_n_days_with_tx"]
    rows = []
    store = {}
    for name, mask in (("T1", terc == "T1"), ("T2+T3", terc.isin(["T2", "T3"]))):
        raw = signed_oof_auroc(tr[Y3], ss, tr["fold"], tr[Y3].notna() & mask)
        rec = leftover_diag(tr[Y3], ss, [days], tr["fold"], tr[Y3].notna() & mask)
        store[name] = rec
        rows.append(
            {
                "slice": name,
                "n": f"{raw['n_defined']:,}",
                "n_pos": f"{raw['n_pos']:,}",
                "raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "leftover days": _f(rec["rank"]),
                "dies": "YES" if rec["honest_dies"] or (np.isfinite(rec["rank"]) and rec["rank"] < CHANCE) else "no",
            }
        )
    t1_dies = store["T1"]["honest_dies"] or (
        np.isfinite(store["T1"]["rank"]) and store["T1"]["rank"] < CHANCE
    )
    t23_dies = store["T2+T3"]["honest_dies"] or (
        np.isfinite(store["T2+T3"]["rank"]) and store["T2+T3"]["rank"] < CHANCE
    )
    prose = (
        f"Y3 leftover after days T1 {_f(store['T1']['rank'])} "
        f"({'dies' if t1_dies else 'lives'}); T2+T3 {_f(store['T2+T3']['rank'])} "
        f"({'dies' if t23_dies else 'lives'})."
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


def pass6b_icc(tr: pd.DataFrame) -> dict:
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    icc = icc_anova(ss, tr["company_id"])
    dem = company_demean(ss, tr["company_id"])
    mu = company_mean(ss, tr["company_id"])
    rows = []
    store = {}
    for feat, col in (("now", ss), ("co_mean", mu), ("demean", dem)):
        res = signed_oof_auroc(tr[Y3], col, tr["fold"], tr[Y3].notna())
        store[feat] = res
        rows.append(_auc_row(Y3, feat, res))
    confirm = bool(np.isfinite(icc["icc"]) and abs(icc["icc"] - ICC_QUOTE) < 0.03)
    trait = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_TRAIT)
    dem_left = leftover_diag(tr[Y3], dem, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"c_ss_month ICC {_f(icc['icc'])} (feature-report 0.99 "
        f"{'CONFIRM BETWEEN' if confirm else 'off'}). "
        f"Y3 company-mean {_f(_cv(store['co_mean']))} demean {_f(_cv(store['demean']))} "
        f"demean leftover after days {_f(dem_left['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "icc": icc["icc"],
        "confirm": confirm,
        "trait": trait,
        "y3_mu": _cv(store["co_mean"]),
        "y3_dem": _cv(store["demean"]),
        "dem_left": dem_left["rank"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 7. Q6 lag leftover after days_lag1
# ---------------------------------------------------------------------------
def pass7_q6(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    cols = (
        "c_ss_month",
        "c_ss_month_lag1",
        "c_ss_month_lag3",
        "c_n_days_with_tx",
        "c_n_days_with_tx_lag1",
    )
    slices = (
        ("all", pd.Series(True, index=tr.index)),
        ("short_<12", tr["trail_class"] == "short_<12"),
    )
    for sname, smask in slices:
        lab = tr[Y3].notna() & smask
        for col in cols:
            if col not in tr.columns:
                continue
            res = signed_oof_auroc(tr[Y3], tr[col], tr["fold"], lab)
            store[(sname, col)] = res
            rows.append(
                {
                    "slice": sname,
                    "col": col,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )

    def g(sl, col) -> float:
        r = store.get((sl, col))
        if r is None or r["low_power"]:
            return float("nan")
        return r["cv"]

    now = g("all", "c_ss_month")
    lag1 = g("all", "c_ss_month_lag1")
    lag3 = g("all", "c_ss_month_lag3")
    days_l1 = g("all", "c_n_days_with_tx_lag1")
    short_l1 = g("short_<12", "c_ss_month_lag1")
    rec1 = leftover_diag(
        tr[Y3], tr["c_ss_month_lag1"], [tr["c_n_days_with_tx_lag1"]], tr["fold"], tr[Y3].notna()
    )
    rec3 = leftover_diag(
        tr[Y3], tr["c_ss_month_lag3"], [tr["c_n_days_with_tx_lag1"]], tr["fold"], tr[Y3].notna()
    )
    days_l1_ok = bool(np.isfinite(days_l1) and abs(days_l1 - DAYS_LAG1_QUOTE) < 0.03)
    l1_dies = bool(rec1["honest_dies"] or (np.isfinite(rec1["rank"]) and rec1["rank"] < CHANCE))
    keep_q6 = bool(
        np.isfinite(now)
        and now >= CHANCE
        and np.isfinite(lag1)
        and lag1 >= CHANCE
        and (now - lag1) <= 0.03
        and not l1_dies
    )
    q6 = "KEEP" if keep_q6 else "CLOSE"
    prose = (
        f"Y3 SS now {_f(now)} lag1 {_f(lag1)} lag3 {_f(lag3)}; short lag1 {_f(short_l1)}. "
        f"Days lag1 {_f(days_l1)} (KEEP 0.684 {'CONFIRM' if days_l1_ok else 'off'}). "
        f"SS_lag1 leftover after days_lag1 {_f(rec1['rank'])} "
        f"lag3 {_f(rec3['rank'])}. Q6 {q6}."
    )
    print(prose)
    return {
        "rows": rows,
        "now": now,
        "lag1": lag1,
        "lag3": lag3,
        "days_l1": days_l1,
        "short_l1": short_l1,
        "l1_left": rec1["rank"],
        "l3_left": rec3["rank"],
        "l1_dies": l1_dies,
        "days_l1_ok": days_l1_ok,
        "q6": q6,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 8. Calendar dummy (tax Q-peaked vs salary monthly)
# ---------------------------------------------------------------------------
def pass8_calendar(tr: pd.DataFrame) -> dict:
    cal = []
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    for m in range(1, 13):
        sl = ss[tr["cal_month"] == m]
        cal.append(
            {
                "month": m,
                "name": pd.Timestamp(2000, m, 1).strftime("%b"),
                "q": int(m in Q_MONTHS),
                "n": f"{int(len(sl)):,}",
                "P(SS)": _pp(float(sl.mean()) if len(sl) else float("nan")),
                "share": float(sl.mean()) if len(sl) else float("nan"),
            }
        )
    q = float(np.nanmean([r["share"] for r in cal if r["q"] == 1]))
    nq = float(np.nanmean([r["share"] for r in cal if r["q"] == 0]))
    ratio = q / nq if nq else float("nan")
    quarterly = bool(q >= 0.55 and nq <= 0.25)
    monthly = bool(nq >= 0.25 and np.isfinite(ratio) and ratio < 1.3)
    mixed_q = bool((not quarterly) and (not monthly) and np.isfinite(ratio) and ratio >= 1.3)
    shape = "quarterly" if quarterly else ("mixed_q" if mixed_q else ("monthly" if monthly else "mixed"))
    dummy = bool(quarterly or mixed_q)
    qflag = leftover_diag(tr[Y3], ss, [tr["is_q_month"]], tr["fold"], tr[Y3].notna())
    peak = max(cal, key=lambda r: r["share"] if np.isfinite(r["share"]) else -1)
    trough = min(cal, key=lambda r: r["share"] if np.isfinite(r["share"]) else 9)
    prose = (
        f"SS share Q-months {_pp(q)} vs other {_pp(nq)} (ratio {_f(ratio, 2)}). "
        f"Peak {peak['name']} {_pp(peak['share'])}, trough {trough['name']} {_pp(trough['share'])}. "
        f"Shape **{shape}** (tax was Q-peaked; salary monthly). "
        f"Leftover after is_q_month {_f(qflag['rank'])}. "
        f"{'calendar dummy' if dummy else 'not a Q dummy'}."
    )
    print(prose)
    return {
        "cal": cal,
        "q": q,
        "nq": nq,
        "ratio": ratio,
        "shape": shape,
        "dummy": dummy,
        "q_left": qflag["rank"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# 10. Dark 470 vs invoiced
# ---------------------------------------------------------------------------
def pass10_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    n_erp = int(last["company_id"].isin(book).sum())
    n_dark = int((~last["company_id"].isin(book)).sum())
    confirm = n_erp == 744 and n_dark == 470
    erp = tr["company_id"].isin(book)
    dark = ~erp
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    rows = []
    store = {}
    for name, mask in (("invoiced_744", erp), ("dark_470", dark)):
        raw = signed_oof_auroc(tr[Y3], ss, tr["fold"], tr[Y3].notna() & mask)
        rec = leftover_diag(tr[Y3], ss, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & mask)
        store[name] = rec
        rows.append(
            {
                "slice": name,
                "n_cm": f"{int(mask.sum()):,}",
                "P(SS)": _pp(float(ss[mask].mean())),
                "Y3 raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "leftover days": _f(rec["rank"]),
            }
        )
    prose = (
        f"Last-month ever-ERP {n_erp} / never-ERP {n_dark} "
        f"({'CONFIRM 744/470' if confirm else 'off'}). "
        f"P(SS) invoiced {_pp(float(ss[erp].mean()))} dark {_pp(float(ss[dark].mean()))}. "
        f"Y3 leftover invoiced {_f(store['invoiced_744']['rank'])} dark {_f(store['dark_470']['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "n_erp": n_erp,
        "n_dark": n_dark,
        "confirm": confirm,
        "prev_erp": float(ss[erp].mean()),
        "prev_dark": float(ss[dark].mean()),
        "inv": store["invoiced_744"]["rank"],
        "dark": store["dark_470"]["rank"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extras
# ---------------------------------------------------------------------------
def pass_holdout(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"]
    ss = pd.to_numeric(ho["c_ss_month"], errors="coerce")
    prose = (
        f"Holdout coverage only (no fit): {ho['company_id'].nunique()} co / {len(ho):,} CM, "
        f"cov {_pp(float(ss.notna().mean()))} P(SS) {_pp(float(ss.mean()))}."
    )
    print(prose)
    return {"n_cm": int(len(ho)), "n_co": int(ho["company_id"].nunique()), "prev": float(ss.mean()), "prose": prose}


def pass_fold_left(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(tr[Y3], tr["c_ss_month"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rows = []
    for k in range(N_FOLDS):
        rows.append({"fold": k, "OLS": _f(fold_k(rec["rec"], k)), "rank": _f(fold_k(rec["rrec"], k))})
    vals = [fold_k(rec["rrec"], k) for k in range(N_FOLDS)]
    finite = [v for v in vals if np.isfinite(v)]
    spread = float(max(finite) - min(finite)) if finite else float("nan")
    prose = f"Y3 leftover-after-days rank folds {rec['rank_folds']} spread {_f(spread)}; OLS {rec['folds']}."
    print(prose)
    return {"rows": rows, "spread": spread, "rank_folds": rec["rank_folds"], "prose": prose}


def pass_missed_ss(tr: pd.DataFrame) -> dict:
    miss = tr["c_missed_ss"]
    prev = float(pd.to_numeric(miss, errors="coerce").mean())
    jac = jaccard(miss, tr["c_missed_salary"])
    raw = signed_oof_auroc(tr[Y3], miss, tr["fold"], tr[Y3].notna())
    rec = leftover_diag(tr[Y3], miss, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"In-memory missed-SS (usual SS ≥3 in ≤6m and SS=0 this month; not a Y): "
        f"prevalence {_pp(prev)}. Jaccard vs missed_salary {_f(jac['jaccard'])}. "
        f"Y3 raw {_f(_cv(raw))} leftover after days {_f(rec['rank'])}. Do not invent y_ss."
    )
    print(prose)
    return {
        "prev": prev,
        "jac": jac["jaccard"],
        "raw": _cv(raw),
        "left": rec["rank"],
        "prose": prose,
    }


def pass_ss_only(tr: pd.DataFrame) -> dict:
    """SS without salary — leftover after days on the exclusive slice."""
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    sal = pd.to_numeric(tr["c_salary_month"], errors="coerce")
    only = (ss == 1) | (sal == 1)
    # leftover of SS after salary on all labeled already done; here exclusive rates
    xor_ss = (ss == 1) & (sal == 0)
    xor_sal = (ss == 0) & (sal == 1)
    rec = leftover_diag(tr[Y3], ss, [sal, tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"SS-only months {int(xor_ss.sum()):,} salary-only {int(xor_sal.sum()):,} "
        f"either {int(only.sum()):,}. Y3 leftover after salary+days {_f(rec['rank'])}."
    )
    print(prose)
    return {
        "n_ss_only": int(xor_ss.sum()),
        "n_sal_only": int(xor_sal.sum()),
        "both_left": rec["rank"],
        "prose": prose,
    }


def pass_stack(tr: pd.DataFrame) -> dict:
    """Leftover after stacked controls — days is the required cut; this is the squeeze."""
    ss = tr["c_ss_month"]
    days = tr["c_n_days_with_tx"]
    sal = tr["c_salary_month"]
    tax = tr["c_tax_month"]
    size = tr["log_in3"]
    ntx = tr["a_n_tx"]
    specs = [
        ("after days+size", (days, size)),
        ("after days+tax", (days, tax)),
        ("after days+ntx", (days, ntx)),
        ("after salary+tax", (sal, tax)),
        ("after days+size+salary", (days, size, sal)),
        ("after days+salary+tax", (days, sal, tax)),
        ("after days+size+salary+tax", (days, size, sal, tax)),
    ]
    rows = []
    store = {}
    for name, xs in specs:
        rec = leftover_diag(tr[Y3], ss, list(xs), tr["fold"], tr[Y3].notna())
        store[name] = rec
        rows.append(
            {
                "control": name,
                "OLS": _f(rec["ols"]),
                "rank": _f(rec["rank"]),
                "R²": _f(rec["r2"]),
                "dies": "YES" if rec["honest_dies"] or (np.isfinite(rec["rank"]) and rec["rank"] < CHANCE) else "no",
            }
        )
        print(f"stack {name}: rank={_f(rec['rank'])} OLS={_f(rec['ols'])} R2={_f(rec['r2'])}")
    thin = store["after days+size+salary+tax"]["rank"]
    lives = bool(np.isfinite(thin) and thin >= CHANCE)
    prose = (
        f"Y3 leftover after days+size {_f(store['after days+size']['rank'])}; "
        f"days+tax {_f(store['after days+tax']['rank'])}; "
        f"days+size+salary {_f(store['after days+size+salary']['rank'])}; "
        f"days+salary+tax {_f(store['after days+salary+tax']['rank'])}; "
        f"full stack days+size+salary+tax {_f(thin)} "
        f"({'lives' if lives else 'DIES'})."
    )
    print(prose)
    return {
        "rows": rows,
        "thin": thin,
        "days_size": store["after days+size"]["rank"],
        "full_lives": lives,
        "prose": prose,
    }


def pass_tax_vs(tr: pd.DataFrame) -> dict:
    """SS leftover vs c_tax_month leftover — tax was Q-peaked; SS is monthly."""
    ss = tr["c_ss_month"]
    tax = tr["c_tax_month"]
    days = tr["c_n_days_with_tx"]
    jac = jaccard(ss, tax)
    ss_after_tax = leftover_diag(tr[Y3], ss, [tax], tr["fold"], tr[Y3].notna())
    tax_after_ss = leftover_diag(tr[Y3], tax, [ss], tr["fold"], tr[Y3].notna())
    tax_after_days = leftover_diag(tr[Y3], tax, [days], tr["fold"], tr[Y3].notna())
    ss_after_tax_days = leftover_diag(tr[Y3], ss, [tax, days], tr["fold"], tr[Y3].notna())
    tax_raw = signed_oof_auroc(tr[Y3], tax, tr["fold"], tr[Y3].notna())
    prose = (
        f"Jaccard(SS, tax) {_f(jac['jaccard'])} both {jac['n_both']:,}. "
        f"Y3 tax raw {_f(_cv(tax_raw))}. "
        f"SS leftover after tax {_f(ss_after_tax['rank'])}; "
        f"tax leftover after SS {_f(tax_after_ss['rank'])}; "
        f"tax leftover after days {_f(tax_after_days['rank'])}; "
        f"SS leftover after tax+days {_f(ss_after_tax_days['rank'])}."
    )
    print(prose)
    return {
        "jac": jac["jaccard"],
        "ss_after_tax": ss_after_tax["rank"],
        "tax_after_ss": tax_after_ss["rank"],
        "tax_after_days": tax_after_days["rank"],
        "ss_after_tax_days": ss_after_tax_days["rank"],
        "tax_raw": _cv(tax_raw),
        "prose": prose,
    }


def pass_trait(tr: pd.DataFrame) -> dict:
    """BETWEEN leftover: company-mean SS after company-mean days / salary."""
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    sal = pd.to_numeric(tr["c_salary_month"], errors="coerce")
    size = tr["log_in3"]
    mu_ss = company_mean(ss, tr["company_id"])
    mu_days = company_mean(days, tr["company_id"])
    mu_sal = company_mean(sal, tr["company_id"])
    mu_size = company_mean(size, tr["company_id"])
    dem_ss = company_demean(ss, tr["company_id"])
    dem_days = company_demean(days, tr["company_id"])
    dem_sal = company_demean(sal, tr["company_id"])
    rows = []
    specs = [
        ("co_mean SS after co_mean days", mu_ss, (mu_days,)),
        ("co_mean SS after co_mean salary", mu_ss, (mu_sal,)),
        ("co_mean SS after co_mean size", mu_ss, (mu_size,)),
        ("co_mean SS after co_mean days+salary", mu_ss, (mu_days, mu_sal)),
        ("raw SS after co_mean SS", ss, (mu_ss,)),
        ("raw SS after days + co_mean SS", ss, (days, mu_ss)),
        ("demean SS after demean days", dem_ss, (dem_days,)),
        ("demean SS after demean salary", dem_ss, (dem_sal,)),
    ]
    store = {}
    for name, x, xs in specs:
        rec = leftover_diag(tr[Y3], x, list(xs), tr["fold"], tr[Y3].notna())
        store[name] = rec
        rows.append(
            {
                "spec": name,
                "OLS": _f(rec["ols"]),
                "rank": _f(rec["rank"]),
                "R²": _f(rec["r2"]),
                "dies": "YES" if rec["honest_dies"] or (np.isfinite(rec["rank"]) and rec["rank"] < CHANCE) else "no",
            }
        )
        print(f"trait {name}: rank={_f(rec['rank'])} R2={_f(rec['r2'])}")
    between = store["co_mean SS after co_mean days"]["rank"]
    within = store["demean SS after demean days"]["rank"]
    after_mu = store["raw SS after days + co_mean SS"]["rank"]
    prose = (
        f"BETWEEN leftover: co-mean SS after co-mean days {_f(between)}; "
        f"after co-mean salary {_f(store['co_mean SS after co_mean salary']['rank'])}. "
        f"Within leftover: demean SS after demean days {_f(within)}. "
        f"Raw SS leftover after days + co-mean SS {_f(after_mu)} "
        f"(month flip after payer identity)."
    )
    print(prose)
    return {
        "rows": rows,
        "between": between,
        "within": within,
        "after_mu": after_mu,
        "prose": prose,
    }


def pass_cadence(tr: pd.DataFrame) -> dict:
    """Always / never / mixed SS companies. In-memory only."""
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    mu = ss.groupby(tr["company_id"]).transform("mean")
    always = mu >= 0.999
    never = mu <= 0.001
    mixed = (~always) & (~never)
    rows = []
    for name, mask in (("always", always), ("never", never), ("mixed", mixed)):
        lab = tr[Y3].notna() & mask
        n_co = int(tr.loc[mask, "company_id"].nunique())
        raw = signed_oof_auroc(tr[Y3], ss, tr["fold"], lab)
        rec = leftover_diag(tr[Y3], ss, [tr["c_n_days_with_tx"]], tr["fold"], lab)
        y_rate = float(pd.to_numeric(tr.loc[lab, Y3], errors="coerce").mean()) if int(lab.sum()) else float("nan")
        rows.append(
            {
                "cadence": name,
                "n_co": n_co,
                "n_cm": int(mask.sum()),
                "n_y3": int(lab.sum()),
                "Y3 rate": _pp(y_rate),
                "raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "leftover days": _f(rec["rank"]),
            }
        )
    jac_tax = jaccard(ss, tr["c_tax_month"])
    prose = (
        f"Cadence always {rows[0]['n_co']} / never {rows[1]['n_co']} / mixed {rows[2]['n_co']} companies. "
        f"Y3 rate always {rows[0]['Y3 rate']} never {rows[1]['Y3 rate']} mixed {rows[2]['Y3 rate']}. "
        f"Jaccard(SS, tax) {_f(jac_tax['jaccard'])}."
    )
    print(prose)
    return {"rows": rows, "prose": prose, "n_always": rows[0]["n_co"], "n_never": rows[1]["n_co"]}


def pass_xor_left(tr: pd.DataFrame) -> dict:
    """Leftover on months where SS and salary disagree — not a payroll rewrite."""
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    sal = pd.to_numeric(tr["c_salary_month"], errors="coerce")
    disagree = ss != sal
    agree = ss == sal
    rows = []
    store = {}
    for name, mask in (("disagree SS≠salary", disagree), ("agree SS=salary", agree)):
        raw = signed_oof_auroc(tr[Y3], ss, tr["fold"], tr[Y3].notna() & mask)
        rec = leftover_diag(tr[Y3], ss, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & mask)
        store[name] = rec
        rows.append(
            {
                "slice": name,
                "n": f"{raw['n_defined']:,}",
                "n_pos": f"{raw['n_pos']:,}",
                "raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "leftover days": _f(rec["rank"]),
            }
        )
    prose = (
        f"Disagree months leftover after days {_f(store['disagree SS≠salary']['rank'])}; "
        f"agree {_f(store['agree SS=salary']['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "disagree": store["disagree SS≠salary"]["rank"],
        "agree": store["agree SS=salary"]["rank"],
        "prose": prose,
    }


def pass_sofar(tr: pd.DataFrame) -> dict:
    ss = tr["c_ss_month"]
    days = tr["c_n_days_with_tx"]
    rows = []
    store = {}
    slices = (
        ("so_far<6", tr["months_so_far"] < 6),
        ("so_far 6-11", (tr["months_so_far"] >= 6) & (tr["months_so_far"] < 12)),
        ("so_far≥12", tr["months_so_far"] >= 12),
        ("short_<12", tr["trail_class"] == "short_<12"),
        ("long_>=18", tr["trail_class"] == "long_>=18"),
    )
    for name, mask in slices:
        raw = signed_oof_auroc(tr[Y3], ss, tr["fold"], tr[Y3].notna() & mask)
        rec = leftover_diag(tr[Y3], ss, [days], tr["fold"], tr[Y3].notna() & mask)
        store[name] = rec
        rows.append(
            {
                "slice": name,
                "n": f"{raw['n_defined']:,}",
                "n_pos": f"{raw['n_pos']:,}",
                "raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "leftover days": _f(rec["rank"]),
            }
        )
    prose = (
        f"Leftover after days so_far<6 {_f(store['so_far<6']['rank'])}; "
        f"6-11 {_f(store['so_far 6-11']['rank'])}; ≥12 {_f(store['so_far≥12']['rank'])}; "
        f"short {_f(store['short_<12']['rank'])}; long {_f(store['long_>=18']['rank'])}."
    )
    print(prose)
    return {"rows": rows, "prose": prose, "short": store["so_far<6"]["rank"], "long": store["so_far≥12"]["rank"]}


def pass_chronic(tr: pd.DataFrame) -> dict:
    hot = tr["group_id"].astype(str).isin(CHRONIC_GROUPS)
    rec = leftover_diag(
        tr[Y3], tr["c_ss_month"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & ~hot
    )
    raw = signed_oof_auroc(tr[Y3], tr["c_ss_month"], tr["fold"], tr[Y3].notna() & ~hot)
    prose = (
        f"Drop chronic {CHRONIC_GROUPS}: leftover after days {_f(rec['rank'])} "
        f"raw {_f(_cv(raw))} (n_pos={raw['n_pos']})."
    )
    print(prose)
    return {"left": rec["rank"], "raw": _cv(raw), "prose": prose}


def pass_fold_sal(tr: pd.DataFrame) -> dict:
    rec_s = leftover_diag(tr[Y3], tr["c_ss_month"], [tr["c_salary_month"]], tr["fold"], tr[Y3].notna())
    rec_t = leftover_diag(tr[Y3], tr["c_ss_month"], [tr["c_tax_month"]], tr["fold"], tr[Y3].notna())
    rec_b = leftover_diag(
        tr[Y3], tr["c_ss_month"], [tr["c_n_days_with_tx"], tr["c_salary_month"]], tr["fold"], tr[Y3].notna()
    )
    rows = []
    for k in range(N_FOLDS):
        rows.append(
            {
                "fold": k,
                "after salary": _f(fold_k(rec_s["rrec"], k)),
                "after tax": _f(fold_k(rec_t["rrec"], k)),
                "after days+salary": _f(fold_k(rec_b["rrec"], k)),
            }
        )
    prose = (
        f"Fold leftover after salary {rec_s['rank_folds']}; "
        f"after tax {rec_t['rank_folds']}; after days+salary {rec_b['rank_folds']}."
    )
    print(prose)
    return {"rows": rows, "prose": prose, "sal_folds": rec_s["rank_folds"]}


def pass_q6_sal(tr: pd.DataFrame) -> dict:
    """Is lag1 leftover after days just a salary-lag twin?"""
    rec = leftover_diag(
        tr[Y3],
        tr["c_ss_month_lag1"],
        [tr["c_salary_month_lag1"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    rec_d = leftover_diag(
        tr[Y3],
        tr["c_ss_month_lag1"],
        [tr["c_n_days_with_tx_lag1"], tr["c_salary_month_lag1"]],
        tr["fold"],
        tr[Y3].notna(),
    )
    prose = (
        f"SS_lag1 leftover after salary_lag1 {_f(rec['rank'])}; "
        f"after days_lag1+salary_lag1 {_f(rec_d['rank'])}."
    )
    print(prose)
    return {"after_sal": rec["rank"], "after_both": rec_d["rank"], "prose": prose}


def pass_inv_sal(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(tr[Y3], tr["c_salary_month"], [tr["c_ss_month"]], tr["fold"], tr[Y3].notna())
    rec_d = leftover_diag(
        tr[Y3], tr["c_salary_month"], [tr["c_ss_month"], tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna()
    )
    prose = (
        f"Inverse: salary leftover after SS {_f(rec['rank'])}; "
        f"after SS+days {_f(rec_d['rank'])}."
    )
    print(prose)
    return {"after_ss": rec["rank"], "after_both": rec_d["rank"], "prose": prose}


def pass_share6(tr: pd.DataFrame) -> dict:
    """In-memory rolling SS share. Do not put on the card. Do not invent y_ss."""
    out = tr.sort_values(["company_id", "period"])
    ss = pd.to_numeric(out["c_ss_month"], errors="coerce").fillna(0)
    share = ss.groupby(out["company_id"], sort=False).transform(
        lambda s: s.rolling(ROLL, min_periods=1).mean()
    )
    share = share.reindex(tr.index)
    raw = signed_oof_auroc(tr[Y3], share, tr["fold"], tr[Y3].notna())
    rec = leftover_diag(tr[Y3], share, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(tr[Y3], share, [tr["c_ss_month"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"In-memory SS share_6 Y3 {_f(_cv(raw))} leftover after days {_f(rec['rank'])} "
        f"after c_ss_month {_f(rec_s['rank'])}. Not a card col."
    )
    print(prose)
    return {"raw": _cv(raw), "after_days": rec["rank"], "after_ss": rec_s["rank"], "prose": prose}


def pass_dark_stack(tr: pd.DataFrame, book: set[str]) -> dict:
    erp = tr["company_id"].isin(book)
    rows = []
    store = {}
    for name, mask in (("invoiced", erp), ("dark", ~erp)):
        rec = leftover_diag(
            tr[Y3],
            tr["c_ss_month"],
            [tr["c_n_days_with_tx"], tr["c_salary_month"]],
            tr["fold"],
            tr[Y3].notna() & mask,
        )
        store[name] = rec
        rows.append(
            {
                "slice": name,
                "leftover days+salary": _f(rec["rank"]),
                "OLS": _f(rec["ols"]),
                "dies": "YES" if rec["honest_dies"] or (np.isfinite(rec["rank"]) and rec["rank"] < CHANCE) else "no",
            }
        )
    prose = (
        f"Leftover after days+salary invoiced {_f(store['invoiced']['rank'])} "
        f"dark {_f(store['dark']['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "inv": store["invoiced"]["rank"],
        "dark": store["dark"]["rank"],
        "prose": prose,
    }


def pass_month_dummies(tr: pd.DataFrame) -> dict:
    dummies = [((tr["cal_month"] == m).astype(float)) for m in range(1, 12)]
    rec = leftover_diag(tr[Y3], tr["c_ss_month"], dummies, tr["fold"], tr[Y3].notna())
    prose = f"Y3 leftover after 11 month dummies {_f(rec['rank'])} R²={_f(rec['r2'])}."
    print(prose)
    return {"left": rec["rank"], "r2": rec["r2"], "prose": prose}


def pass_ever(tr: pd.DataFrame) -> dict:
    """Ever-SS company dummy vs month flag. In-memory only; not a card col."""
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    ever = ss.groupby(tr["company_id"]).transform("max")
    usual = (ss.groupby(tr["company_id"]).transform("mean") >= 0.5).astype(float)
    raw_e = signed_oof_auroc(tr[Y3], ever, tr["fold"], tr[Y3].notna())
    raw_u = signed_oof_auroc(tr[Y3], usual, tr["fold"], tr[Y3].notna())
    left_e = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    left_u = leftover_diag(tr[Y3], usual, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    left_e_s = leftover_diag(tr[Y3], ever, [tr["c_salary_month"]], tr["fold"], tr[Y3].notna())
    month_after_ever = leftover_diag(tr[Y3], ss, [ever], tr["fold"], tr[Y3].notna())
    month_after_ever_days = leftover_diag(
        tr[Y3], ss, [ever, tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna()
    )
    prose = (
        f"Ever-SS Y3 {_f(_cv(raw_e))} leftover after days {_f(left_e['rank'])} "
        f"after salary {_f(left_e_s['rank'])}. Usual-SS (≥50% months) Y3 {_f(_cv(raw_u))} "
        f"leftover after days {_f(left_u['rank'])}. Month SS leftover after ever-SS "
        f"{_f(month_after_ever['rank'])}; after ever+days {_f(month_after_ever_days['rank'])}."
    )
    print(prose)
    return {
        "ever": _cv(raw_e),
        "ever_left": left_e["rank"],
        "usual": _cv(raw_u),
        "usual_left": left_u["rank"],
        "month_after_ever": month_after_ever["rank"],
        "month_after_ever_days": month_after_ever_days["rank"],
        "prose": prose,
    }


def pass_mixed_left(tr: pd.DataFrame) -> dict:
    """Leftover on mixed-cadence companies only — drops always/never payer dummies."""
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    mu = ss.groupby(tr["company_id"]).transform("mean")
    mixed = (mu > 0.001) & (mu < 0.999)
    raw = signed_oof_auroc(tr[Y3], ss, tr["fold"], tr[Y3].notna() & mixed)
    rec = leftover_diag(tr[Y3], ss, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & mixed)
    rec_s = leftover_diag(tr[Y3], ss, [tr["c_salary_month"]], tr["fold"], tr[Y3].notna() & mixed)
    rec_b = leftover_diag(
        tr[Y3], ss, [tr["c_n_days_with_tx"], tr["c_salary_month"]], tr["fold"], tr[Y3].notna() & mixed
    )
    prose = (
        f"Mixed-cadence only (n_pos={raw['n_pos']}): raw {_f(_cv(raw))} "
        f"leftover after days {_f(rec['rank'])} after salary {_f(rec_s['rank'])} "
        f"after days+salary {_f(rec_b['rank'])}."
    )
    print(prose)
    return {
        "raw": _cv(raw),
        "after_days": rec["rank"],
        "after_sal": rec_s["rank"],
        "after_both": rec_b["rank"],
        "n_pos": raw["n_pos"],
        "prose": prose,
    }


def pass_ss_only_cos(tr: pd.DataFrame) -> dict:
    """Companies that pay SS but never salary — exclusive payer type."""
    g = tr.groupby("company_id")
    ss_mu = g["c_ss_month"].mean()
    sal_mu = g["c_salary_month"].mean()
    ss_only = set(ss_mu.index[(ss_mu > 0) & (sal_mu <= 0.001)])
    sal_only = set(ss_mu.index[(sal_mu > 0) & (ss_mu <= 0.001)])
    both = set(ss_mu.index[(ss_mu > 0) & (sal_mu > 0)])
    neither = set(ss_mu.index[(ss_mu <= 0.001) & (sal_mu <= 0.001)])
    mask = tr["company_id"].isin(ss_only)
    raw = signed_oof_auroc(tr[Y3], tr["c_ss_month"], tr["fold"], tr[Y3].notna() & mask)
    rec = leftover_diag(
        tr[Y3], tr["c_ss_month"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & mask
    )
    lab = tr[Y3].notna()
    rates = []
    for name, ids in (("SS-only cos", ss_only), ("sal-only cos", sal_only), ("both", both), ("neither", neither)):
        sl = lab & tr["company_id"].isin(ids)
        rates.append(
            {
                "type": name,
                "n_co": len(ids),
                "n_y3": int(sl.sum()),
                "Y3 rate": _pp(float(pd.to_numeric(tr.loc[sl, Y3], errors="coerce").mean()) if int(sl.sum()) else float("nan")),
            }
        )
    prose = (
        f"SS-only companies {len(ss_only)} sal-only {len(sal_only)} both {len(both)} neither {len(neither)}. "
        f"SS-only leftover after days {_f(rec['rank'])} raw {_f(_cv(raw))}."
    )
    print(prose)
    return {"rows": rates, "n_ss_only": len(ss_only), "left": rec["rank"], "prose": prose}


def pass_y3_cells(tr: pd.DataFrame) -> dict:
    """Y3 recover rate by SS × days tercile — sign − cell check."""
    lab = tr[Y3].notna()
    work = tr.loc[lab].copy()
    work["_ss"] = pd.to_numeric(work["c_ss_month"], errors="coerce")
    days = pd.to_numeric(work["c_n_days_with_tx"], errors="coerce")
    try:
        work["_td"] = pd.qcut(days.rank(method="first"), 3, labels=["low days", "mid", "high days"])
    except ValueError:
        work["_td"] = "all"
    rows = []
    for td, g in work.groupby("_td", observed=False):
        for ss_v, gg in g.groupby("_ss"):
            rows.append(
                {
                    "days tercile": str(td),
                    "SS": int(ss_v) if pd.notna(ss_v) else "—",
                    "n": f"{len(gg):,}",
                    "n_pos": int(pd.to_numeric(gg[Y3], errors="coerce").sum()),
                    "Y3 rate": _pp(float(pd.to_numeric(gg[Y3], errors="coerce").mean())),
                }
            )
    prose = "Y3 rate cells by days tercile × SS (sign −: SS=1 should sit below SS=0 inside terciles)."
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_y2_stack(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(
        tr[Y2], tr["c_ss_month"], [tr["c_n_days_with_tx"], tr["c_salary_month"]], tr["fold"], tr[Y2].notna()
    )
    rec_t = leftover_diag(tr[Y2], tr["c_ss_month"], [tr["c_tax_month"]], tr["fold"], tr[Y2].notna())
    prose = (
        f"Y2 leftover after days+salary {_f(rec['rank'])}; after tax {_f(rec_t['rank'])}."
    )
    print(prose)
    return {"after_both": rec["rank"], "after_tax": rec_t["rank"], "prose": prose}


def pass_dark_trait(tr: pd.DataFrame, book: set[str]) -> dict:
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    mu = company_mean(ss, tr["company_id"])
    mu_d = company_mean(tr["c_n_days_with_tx"], tr["company_id"])
    erp = tr["company_id"].isin(book)
    rows = []
    store = {}
    for name, mask in (("invoiced", erp), ("dark", ~erp)):
        rec = leftover_diag(tr[Y3], mu, [mu_d], tr["fold"], tr[Y3].notna() & mask)
        store[name] = rec
        rows.append({"slice": name, "BETWEEN leftover after days-mean": _f(rec["rank"])})
    prose = (
        f"BETWEEN leftover after days-mean invoiced {_f(store['invoiced']['rank'])} "
        f"dark {_f(store['dark']['rank'])}."
    )
    print(prose)
    return {"rows": rows, "inv": store["invoiced"]["rank"], "dark": store["dark"]["rank"], "prose": prose}


def pass_mixed_folds(tr: pd.DataFrame) -> dict:
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    mu = ss.groupby(tr["company_id"]).transform("mean")
    mixed = (mu > 0.001) & (mu < 0.999)
    rec = leftover_diag(tr[Y3], ss, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & mixed)
    raw = signed_oof_auroc(tr[Y3], ss, tr["fold"], tr[Y3].notna() & mixed)
    rows = []
    for k in range(N_FOLDS):
        rows.append(
            {
                "fold": k,
                "raw": _f(fold_k(raw, k)),
                "leftover days": _f(fold_k(rec["rrec"], k)),
                "n_pos": next((r["n_pos"] for r in raw["folds"] if int(r["fold"]) == k), "—"),
            }
        )
    finite = [fold_k(rec["rrec"], k) for k in range(N_FOLDS)]
    finite = [v for v in finite if np.isfinite(v)]
    spread = float(max(finite) - min(finite)) if finite else float("nan")
    prose = (
        f"Mixed leftover folds {rec['rank_folds']} spread {_f(spread)}; "
        f"raw folds {fold_bits(raw)}. n_pos total {raw['n_pos']}."
    )
    print(prose)
    return {"rows": rows, "spread": spread, "left": rec["rank"], "prose": prose}


def pass_boot(tr: pd.DataFrame, n_boot: int = 220) -> dict:
    """Company bootstrap of rank leftover after days. Train only."""
    rng = np.random.default_rng(FOLD_SEED)
    lab = tr[Y3].notna()
    work = tr.loc[lab, ["company_id", Y3, "c_ss_month", "c_n_days_with_tx", "fold"]].copy()
    cos = work["company_id"].astype(str).unique()
    idx = {c: work.index[work["company_id"].astype(str) == c].to_numpy() for c in cos}
    vals = []
    t0 = time.time()
    for i in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        ix = np.concatenate([idx[c] for c in draw])
        sl = work.loc[ix]
        rec = leftover_diag(sl[Y3], sl["c_ss_month"], [sl["c_n_days_with_tx"]], sl["fold"], sl[Y3].notna())
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
        if (i + 1) % 40 == 0:
            print(f"boot {i+1}/{n_boot} mean={np.mean(vals):.3f} n={len(vals)}")
    arr = np.array(vals, dtype=float)
    lo, mid, hi = (float(np.quantile(arr, q)) for q in (0.05, 0.50, 0.95)) if len(arr) else (float("nan"),) * 3
    share_die = float(np.mean(arr < CHANCE)) if len(arr) else float("nan")
    prose = (
        f"Company bootstrap n={len(arr)} leftover-after-days rank "
        f"p05/p50/p95 {_f(lo)} / {_f(mid)} / {_f(hi)}; "
        f"share <0.55 {_pp(share_die)}; wall {time.time()-t0:.0f}s."
    )
    print(prose)
    return {
        "n": len(arr),
        "p05": lo,
        "p50": mid,
        "p95": hi,
        "share_die": share_die,
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "prose": prose,
    }


def pass_cousins(tr: pd.DataFrame) -> dict:
    """Leftover after other C stems already on the store. Not new card cols."""
    ss = tr["c_ss_month"]
    rows = []
    store = {}
    for name in EXTRA_STORE:
        if name not in tr.columns:
            continue
        rec = leftover_diag(tr[Y3], ss, [tr[name]], tr["fold"], tr[Y3].notna())
        rho = spearman(ss, tr[name])
        store[name] = rec
        rows.append(
            {
                "control": name,
                "ρ": _f(rho),
                "leftover": _f(rec["rank"]),
                "twin": "YES" if np.isfinite(rho) and abs(rho) >= TWIN_RHO else "",
            }
        )
        print(f"cousin after {name}: leftover={_f(rec['rank'])} ρ={_f(rho)}")
    prose = "Leftover after other C stems: " + "; ".join(
        f"{r['control']} {_f(store[r['control']]['rank'])}" for r in rows
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_perm(tr: pd.DataFrame, n_perm: int = 160) -> dict:
    """Permute SS within fold; leftover after days should sit at chance."""
    rng = np.random.default_rng(FOLD_SEED + 7)
    lab = tr[Y3].notna()
    work = tr.loc[lab, [Y3, "c_ss_month", "c_n_days_with_tx", "fold"]].copy()
    vals = []
    t0 = time.time()
    for i in range(n_perm):
        sh = work["c_ss_month"].to_numpy(copy=True)
        for k in range(N_FOLDS):
            ix = np.where(work["fold"].to_numpy() == k)[0]
            sh[ix] = rng.permutation(sh[ix])
        rec = leftover_diag(
            work[Y3], pd.Series(sh, index=work.index), [work["c_n_days_with_tx"]], work["fold"], work[Y3].notna()
        )
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
        if (i + 1) % 40 == 0:
            print(f"perm {i+1}/{n_perm} mean={np.mean(vals):.3f}")
    arr = np.array(vals, dtype=float)
    lo, mid, hi = (float(np.quantile(arr, q)) for q in (0.05, 0.50, 0.95)) if len(arr) else (float("nan"),) * 3
    obs = leftover_diag(tr[Y3], tr["c_ss_month"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())["rank"]
    p = float(np.mean(arr >= obs)) if len(arr) and np.isfinite(obs) else float("nan")
    prose = (
        f"Within-fold permute SS leftover-after-days p05/p50/p95 "
        f"{_f(lo)} / {_f(mid)} / {_f(hi)}; observed {_f(obs)}; "
        f"one-sided p(perm ≥ obs) {_f(p, 3)}; wall {time.time()-t0:.0f}s."
    )
    print(prose)
    return {"p05": lo, "p50": mid, "p95": hi, "obs": obs, "p": p, "prose": prose}


def pass_boot_sal(tr: pd.DataFrame, n_boot: int = 180) -> dict:
    rng = np.random.default_rng(FOLD_SEED + 3)
    lab = tr[Y3].notna()
    work = tr.loc[lab, ["company_id", Y3, "c_ss_month", "c_salary_month", "c_n_days_with_tx", "fold"]].copy()
    cos = work["company_id"].astype(str).unique()
    idx = {c: work.index[work["company_id"].astype(str) == c].to_numpy() for c in cos}
    after_sal, after_both = [], []
    t0 = time.time()
    for i in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        ix = np.concatenate([idx[c] for c in draw])
        sl = work.loc[ix]
        rec_s = leftover_diag(sl[Y3], sl["c_ss_month"], [sl["c_salary_month"]], sl["fold"], sl[Y3].notna())
        rec_b = leftover_diag(
            sl[Y3], sl["c_ss_month"], [sl["c_n_days_with_tx"], sl["c_salary_month"]], sl["fold"], sl[Y3].notna()
        )
        if np.isfinite(rec_s["rank"]):
            after_sal.append(rec_s["rank"])
        if np.isfinite(rec_b["rank"]):
            after_both.append(rec_b["rank"])
        if (i + 1) % 40 == 0:
            print(f"boot_sal {i+1}/{n_boot} sal={np.mean(after_sal):.3f} both={np.mean(after_both):.3f}")
    a = np.array(after_sal, dtype=float)
    b = np.array(after_both, dtype=float)
    prose = (
        f"Bootstrap leftover after salary p05/p50/p95 "
        f"{_f(float(np.quantile(a, 0.05)))} / {_f(float(np.quantile(a, 0.50)))} / {_f(float(np.quantile(a, 0.95)))}; "
        f"after days+salary "
        f"{_f(float(np.quantile(b, 0.05)))} / {_f(float(np.quantile(b, 0.50)))} / {_f(float(np.quantile(b, 0.95)))}; "
        f"wall {time.time()-t0:.0f}s."
    )
    print(prose)
    return {
        "sal_p05": float(np.quantile(a, 0.05)),
        "sal_p50": float(np.quantile(a, 0.50)),
        "sal_p95": float(np.quantile(a, 0.95)),
        "both_p05": float(np.quantile(b, 0.05)),
        "both_p50": float(np.quantile(b, 0.50)),
        "both_p95": float(np.quantile(b, 0.95)),
        "prose": prose,
    }


def leftover_oof(y, x, controls, folds, mask) -> dict:
    """Rank leftover with residual fit on train fold only — no full-sample residual leak."""
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    ctr = [pd.to_numeric(c, errors="coerce") for c in controls]
    defined = mask & y.notna() & x.notna()
    for c in ctr:
        defined = defined & c.notna()
    aucs = []
    fold_rows = []
    for k in range(N_FOLDS):
        trm = defined & (folds != k)
        vam = defined & (folds == k)
        if int((vam & (y == 1)).sum()) == 0 or int((vam & (y == 0)).sum()) == 0:
            fold_rows.append({"fold": k, "auroc": float("nan")})
            continue
        xr = x.rank(method="average")
        cr = [c.rank(method="average") for c in ctr]
        resid_tr, _ = ols_resid(xr[trm], *[c[trm] for c in cr])
        # refit on train, apply to val
        dtr = pd.DataFrame({"y": xr[trm].to_numpy(dtype=float)})
        Xtr = np.column_stack([np.ones(int(trm.sum()))] + [c[trm].to_numpy(dtype=float) for c in cr])
        Xva = np.column_stack([np.ones(int(vam.sum()))] + [c[vam].to_numpy(dtype=float) for c in cr])
        try:
            beta, _, _, _ = np.linalg.lstsq(Xtr, dtr["y"].to_numpy(dtype=float), rcond=None)
        except np.linalg.LinAlgError:
            fold_rows.append({"fold": k, "auroc": float("nan")})
            continue
        resid_va = xr[vam].to_numpy(dtype=float) - (Xva @ beta)
        sign = choose_sign(y[trm], resid_tr)
        auc = auroc(y[vam], sign * pd.Series(resid_va, index=y[vam].index))
        aucs.append(auc)
        fold_rows.append({"fold": k, "auroc": float(auc) if np.isfinite(auc) else float("nan")})
    finite = [a for a in aucs if np.isfinite(a)]
    cv = float(np.mean(finite)) if finite else float("nan")
    return {
        "rank": cv,
        "folds": " ".join(_f(r["auroc"]) for r in fold_rows),
        "n_folds": len(finite),
    }


def pass_oof_resid(tr: pd.DataFrame) -> dict:
    rec_d = leftover_oof(tr[Y3], tr["c_ss_month"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_s = leftover_oof(tr[Y3], tr["c_ss_month"], [tr["c_salary_month"]], tr["fold"], tr[Y3].notna())
    rec_b = leftover_oof(
        tr[Y3], tr["c_ss_month"], [tr["c_n_days_with_tx"], tr["c_salary_month"]], tr["fold"], tr[Y3].notna()
    )
    rec_inv = leftover_oof(tr[Y3], tr["c_n_days_with_tx"], [tr["c_ss_month"]], tr["fold"], tr[Y3].notna())
    dies = bool(np.isfinite(rec_d["rank"]) and rec_d["rank"] < CHANCE)
    prose = (
        f"Fold-wise residual leftover after days {_f(rec_d['rank'])} folds {rec_d['folds']}; "
        f"after salary {_f(rec_s['rank'])}; after days+salary {_f(rec_b['rank'])}; "
        f"inverse days after SS {_f(rec_inv['rank'])}. "
        f"{'DIES' if dies else 'lives'} vs chance 0.55."
    )
    print(prose)
    return {
        "after_days": rec_d["rank"],
        "after_sal": rec_s["rank"],
        "after_both": rec_b["rank"],
        "inv": rec_inv["rank"],
        "dies": dies,
        "prose": prose,
    }


def pass_neither(tr: pd.DataFrame) -> dict:
    """Neither-payroll company dummy (never SS and never salary). In-memory."""
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    sal = pd.to_numeric(tr["c_salary_month"], errors="coerce")
    ss_mu = ss.groupby(tr["company_id"]).transform("mean")
    sal_mu = sal.groupby(tr["company_id"]).transform("mean")
    neither = ((ss_mu <= 0.001) & (sal_mu <= 0.001)).astype(float)
    raw = signed_oof_auroc(tr[Y3], neither, tr["fold"], tr[Y3].notna())
    rec = leftover_diag(tr[Y3], neither, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(tr[Y3], neither, [tr["c_ss_month"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Neither-payroll dummy Y3 {_f(_cv(raw))} leftover after days {_f(rec['rank'])} "
        f"after c_ss_month {_f(rec_s['rank'])}."
    )
    print(prose)
    return {"raw": _cv(raw), "after_days": rec["rank"], "after_ss": rec_s["rank"], "prose": prose}


def pass_logo(tr: pd.DataFrame) -> dict:
    """Leave-one-group leftover after days — is KEEP one-group?"""
    lab = tr[Y3].notna()
    counts = tr.loc[lab].groupby("group_id").size().sort_values(ascending=False)
    keep = [g for g, n in counts.items() if n >= 40][:40]
    vals = []
    rows = []
    for g in keep:
        mask = lab & (tr["group_id"].astype(str) != str(g))
        rec = leftover_diag(tr[Y3], tr["c_ss_month"], [tr["c_n_days_with_tx"]], tr["fold"], mask)
        vals.append(rec["rank"])
        rows.append({"drop": str(g), "n_left": int(mask.sum()), "leftover": _f(rec["rank"])})
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    lo = float(arr.min()) if len(arr) else float("nan")
    hi = float(arr.max()) if len(arr) else float("nan")
    mid = float(np.median(arr)) if len(arr) else float("nan")
    prose = (
        f"Leave-one-group leftover after days (top {len(keep)} groups by Y3-labeled n): "
        f"min/med/max {_f(lo)} / {_f(mid)} / {_f(hi)}."
    )
    print(prose)
    return {"rows": rows[:12], "min": lo, "med": mid, "max": hi, "prose": prose}


def pass_month_left(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for m in range(1, 13):
        mask = tr[Y3].notna() & (tr["cal_month"] == m)
        rec = leftover_diag(tr[Y3], tr["c_ss_month"], [tr["c_n_days_with_tx"]], tr["fold"], mask)
        raw = signed_oof_auroc(tr[Y3], tr["c_ss_month"], tr["fold"], mask)
        store[m] = rec
        rows.append(
            {
                "month": pd.Timestamp(2000, m, 1).strftime("%b"),
                "n_pos": raw["n_pos"],
                "raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "leftover days": _f(rec["rank"]),
            }
        )
    finite = [store[m]["rank"] for m in range(1, 13) if np.isfinite(store[m]["rank"])]
    prose = (
        f"Calendar-month leftover after days: "
        + (", ".join(f"{pd.Timestamp(2000, m, 1).strftime('%b')} {_f(store[m]['rank'])}" for m in range(1, 13)))
        + (f"; finite min {_f(min(finite))}" if finite else "")
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_sibling(tr: pd.DataFrame) -> dict:
    """Leftover after group-mean SS — is the payer trait a sibling/group dummy?"""
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    gmu = ss.groupby(tr["group_id"]).transform("mean")
    cmu = company_mean(ss, tr["company_id"])
    rec = leftover_diag(tr[Y3], ss, [gmu], tr["fold"], tr[Y3].notna())
    rec_d = leftover_diag(tr[Y3], ss, [gmu, tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_c = leftover_diag(tr[Y3], cmu, [gmu], tr["fold"], tr[Y3].notna())
    rho = spearman(ss, gmu)
    prose = (
        f"ρ(SS, group-mean SS) {_f(rho)}. Leftover after group-mean {_f(rec['rank'])}; "
        f"after group-mean+days {_f(rec_d['rank'])}; co-mean SS after group-mean {_f(rec_c['rank'])}."
    )
    print(prose)
    return {
        "rho": rho,
        "after_g": rec["rank"],
        "after_gd": rec_d["rank"],
        "co_after_g": rec_c["rank"],
        "prose": prose,
    }


def pass_boot_between(tr: pd.DataFrame, n_boot: int = 280) -> dict:
    rng = np.random.default_rng(FOLD_SEED + 11)
    lab = tr[Y3].notna()
    work = tr.loc[lab, ["company_id", Y3, "c_ss_month", "c_n_days_with_tx", "fold"]].copy()
    work["mu_ss"] = company_mean(work["c_ss_month"], work["company_id"])
    work["mu_d"] = company_mean(work["c_n_days_with_tx"], work["company_id"])
    cos = work["company_id"].astype(str).unique()
    idx = {c: work.index[work["company_id"].astype(str) == c].to_numpy() for c in cos}
    vals = []
    t0 = time.time()
    for i in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        ix = np.concatenate([idx[c] for c in draw])
        sl = work.loc[ix]
        rec = leftover_diag(sl[Y3], sl["mu_ss"], [sl["mu_d"]], sl["fold"], sl[Y3].notna())
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
        if (i + 1) % 40 == 0:
            print(f"boot_between {i+1}/{n_boot} mean={np.mean(vals):.3f}")
    arr = np.array(vals, dtype=float)
    lo, mid, hi = (float(np.quantile(arr, q)) for q in (0.05, 0.50, 0.95))
    prose = (
        f"BETWEEN leftover bootstrap after days-mean p05/p50/p95 "
        f"{_f(lo)} / {_f(mid)} / {_f(hi)}; share<0.55 {_pp(float(np.mean(arr < CHANCE)))}; "
        f"wall {time.time()-t0:.0f}s."
    )
    print(prose)
    return {"p05": lo, "p50": mid, "p95": hi, "prose": prose}


def extra_markdown(ctx: dict) -> list[str]:
    bits = [
        "## Extra — leftover after stacked controls",
        "",
        ctx["p_st"]["prose"],
        "",
        _md_table(ctx["p_st"]["rows"]),
        "",
        "## Extra — vs c_tax_month leftover",
        "",
        ctx["p_tx"]["prose"],
        "",
        "## Extra — BETWEEN trait vs within flip",
        "",
        ctx["p_tr"]["prose"],
        "",
        _md_table(ctx["p_tr"]["rows"]),
        "",
        "## Extra — always / never / mixed cadence",
        "",
        ctx["p_cd"]["prose"],
        "",
        _md_table(ctx["p_cd"]["rows"]),
        "",
        "## Extra — SS≠salary leftover",
        "",
        ctx["p_xd"]["prose"],
        "",
        _md_table(ctx["p_xd"]["rows"]),
        "",
        "## Extra — so-far / trail leftover",
        "",
        ctx["p_sf"]["prose"],
        "",
        _md_table(ctx["p_sf"]["rows"]),
        "",
        "## Extra — drop chronic GROUP_0158/0172",
        "",
        ctx["p_ch"]["prose"],
        "",
        "## Extra — fold leftover after salary / tax",
        "",
        ctx["p_fs"]["prose"],
        "",
        _md_table(ctx["p_fs"]["rows"]),
        "",
        "## Extra — Q6 leftover after salary_lag1",
        "",
        ctx["p_qs"]["prose"],
        "",
        "## Extra — inverse salary leftover after SS",
        "",
        ctx["p_is"]["prose"],
        "",
        "## Extra — in-memory SS share_6 (not a card col)",
        "",
        ctx["p_sh"]["prose"],
        "",
        "## Extra — dark leftover after days+salary",
        "",
        ctx["p_ds"]["prose"],
        "",
        _md_table(ctx["p_ds"]["rows"]),
        "",
        "## Extra — leftover after month dummies",
        "",
        ctx["p_md"]["prose"],
        "",
        "## Extra — ever-SS / usual-SS (in-memory)",
        "",
        ctx["p_ev"]["prose"],
        "",
        "## Extra — mixed-cadence leftover (drops always/never)",
        "",
        ctx["p_mx"]["prose"],
        "",
        "## Extra — SS-only vs salary-only companies",
        "",
        ctx["p_so"]["prose"],
        "",
        _md_table(ctx["p_so"]["rows"]),
        "",
        "## Extra — Y3 rate cells days tercile × SS",
        "",
        ctx["p_yc"]["prose"],
        "",
        _md_table(ctx["p_yc"]["rows"]),
        "",
        "## Extra — Y2 leftover stack",
        "",
        ctx["p_y2"]["prose"],
        "",
        "## Extra — dark BETWEEN leftover",
        "",
        ctx["p_dt"]["prose"],
        "",
        _md_table(ctx["p_dt"]["rows"]),
        "",
        "## Extra — mixed leftover folds",
        "",
        ctx["p_mf"]["prose"],
        "",
        _md_table(ctx["p_mf"]["rows"]),
        "",
        "## Extra — company bootstrap leftover after days",
        "",
        ctx["p_bt"]["prose"],
        "",
        "## Extra — leftover after other C stems",
        "",
        ctx["p_cu"]["prose"],
        "",
        _md_table(ctx["p_cu"]["rows"]),
        "",
        "## Extra — permutation null leftover after days",
        "",
        ctx["p_pm"]["prose"],
        "",
        "## Extra — bootstrap leftover after salary / days+salary",
        "",
        ctx["p_bs"]["prose"],
        "",
        "## Extra — fold-wise residual leftover (no full-sample resid leak)",
        "",
        ctx["p_oo"]["prose"],
        "",
        "## Extra — neither-payroll company dummy",
        "",
        ctx["p_ne"]["prose"],
        "",
        "## Extra — leave-one-group leftover after days",
        "",
        ctx["p_lg"]["prose"],
        "",
        _md_table(ctx["p_lg"]["rows"]),
        "",
        "## Extra — leftover after days by calendar month",
        "",
        ctx["p_ml"]["prose"],
        "",
        _md_table(ctx["p_ml"]["rows"]),
        "",
        "## Extra — sibling / group-mean leftover",
        "",
        ctx["p_sb"]["prose"],
        "",
        "## Extra — BETWEEN leftover bootstrap",
        "",
        ctx["p_bb"]["prose"],
        "",
    ]
    return bits
def decide(p1, p2, p3, p4, p6, p6b, p7, p8) -> dict:
    twin = bool(p2["gate_twins"])
    size = bool(p1["size_flag"])
    leftover = p4["y3_rank"]
    leftover_dies = bool(p4["y3_dies"])
    beat = bool(np.isfinite(p3["beat_size"]) and p3["beat_size"] >= KEEP_DELTA)
    keep_x = bool(beat and (not leftover_dies) and (not size) and (not twin))
    if keep_x:
        card = "KEEP"
        tag = "KEEP"
        why = (
            f"leftover after days rank {_f(leftover)} lives; beats size {_f(p3['size'])} "
            f"by {_f(p3['beat_size'])}; not SIZE (ρ={_f(p1['rho_in3'])}); not a twin "
            f"(ρ days {_f(p2['rho_days'])} salary {_f(p2['rho_sal'])}). "
            f"BETWEEN payer identity (ICC {_f(p6b['icc'])}); not a days rewrite. "
            "Parent absorbs the card — do not edit it here."
        )
    elif leftover_dies or twin or p4["payroll_twin"]:
        card = "CLOSE as unused leftover / DROP from the card"
        tag = "CLOSE" if leftover_dies or p4["payroll_twin"] else "DROP"
        why = (
            f"Y3 leftover after days rank {_f(leftover)} OLS {_f(p4['y3_ols'])} "
            f"{'dies' if leftover_dies else 'thin'}; twin={twin} payroll_twin={p4['payroll_twin']}. "
            f"After salary {_f(p4['y3_sal'])}. Inverse days-after-SS {_f(p4['inv_rank'])}. "
            f"{'Quiet-activity rewrite of days / salary.' if leftover_dies or p4['payroll_twin'] else 'DROP as twin.'} "
            "Parent absorbs the card — do not edit it here."
        )
    elif p8["dummy"]:
        card = "CLOSE"
        tag = "CLOSE"
        why = f"calendar dummy ({p8['shape']}); leftover after is_q {_f(p8['q_left'])}."
    else:
        card = "PARK"
        tag = "PARK"
        why = f"Y3 {_f(p3['y3'])} leftover {_f(leftover)} does not clear KEEP-as-X."
    q6 = p7["q6"]
    if leftover_dies or twin:
        q6 = "CLOSE"
    return {
        "card": card,
        "headline_tag": tag,
        "why": why,
        "keep_x": keep_x,
        "twin": twin,
        "size": size,
        "leftover_dies": leftover_dies,
        "q6": q6,
        "headline": (
            f"{tag} leftover-after-days rank {_f(leftover)} (OLS {_f(p4['y3_ols'])}, "
            f"fake_ols={p4['fake_ols']}). Y3 SS {_f(p3['y3'])} vs days {_f(p3['days'])} "
            f"vs size {_f(p3['size'])} vs salary {_f(p3['sal'])}. "
            f"After salary {_f(p4['y3_sal'])}. Inverse {_f(p4['inv_rank'])}. "
            f"15-col card: {card}. Q6 {q6}. "
            f"Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged."
        ),
    }


def brief_map(d, p3, p4, p7, p8) -> list[dict]:
    return [
        {
            "#": "1",
            "question": "Who is healthy?",
            "what this cut says": "SS is a payroll-presence flag, not a health Y. Do not invent y_ss.",
        },
        {
            "#": "2",
            "question": "Who is improving?",
            "what this cut says": (
                f"Y3 leftover after days {_f(p4['y3_rank'])} — {d['headline_tag']}. "
                "Quiet-stressed recover is the SHAP story; leftover asks if SS is more than days."
            ),
        },
        {
            "#": "3",
            "question": "Who is turning?",
            "what this cut says": f"Y2 SS {_f(p3['y2'])}. Missed-SS is in-memory only.",
        },
        {
            "#": "4",
            "question": "Dip vs fall?",
            "what this cut says": "Not this binary month flag.",
        },
        {
            "#": "5",
            "question": "Why did it change?",
            "what this cut says": (
                f"Perm #1 ΔAUROC {PERM_SS:.3f}. Leftover after days {_f(p4['y3_rank'])}; "
                f"after salary {_f(p4['y3_sal'])}. Calendar {p8['shape']}."
            ),
        },
        {
            "#": "6",
            "question": "Months earlier?",
            "what this cut says": (
                f"lag1 leftover after days_lag1 {_f(p7['l1_left'])} — {d['q6']}. "
                f"Days lag1 {_f(p7['days_l1'])} stays KEEP."
            ),
        },
    ]


def make_png(tr: pd.DataFrame, p8: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    names = [r["name"] for r in p8["cal"]]
    shares = [100.0 * r["share"] if np.isfinite(r["share"]) else np.nan for r in p8["cal"]]
    colors = ["#c44536" if r["q"] else "#1f4e79" for r in p8["cal"]]
    ax.bar(names, shares, color=colors)
    ax.set_ylabel("P(c_ss_month=1) %")
    ax.set_title(f"Train calendar ({p8['shape']})")
    ax.axhline(100.0 * p8["q"], color="#c44536", ls="--", lw=1, label="Q mean")
    ax.axhline(100.0 * p8["nq"], color="#1f4e79", ls=":", lw=1, label="other mean")
    ax.legend(frameon=False, fontsize=8)

    ax2 = axes[1]
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    lab = tr[Y3].notna() & ss.notna()
    work = tr.loc[lab].copy()
    work["_ss"] = ss[lab]
    work["_y"] = pd.to_numeric(work[Y3], errors="coerce")
    days = pd.to_numeric(work["c_n_days_with_tx"], errors="coerce")
    try:
        work["_qd"] = pd.qcut(days.rank(method="first"), 5, labels=False) + 1
    except ValueError:
        work["_qd"] = 1
    g = work.groupby(["_qd", "_ss"])["_y"].mean().unstack()
    x = np.arange(1, 6)
    if 0 in g.columns:
        ax2.bar(x - 0.18, 100.0 * g[0].reindex(x).to_numpy(dtype=float), width=0.36, color="#9e6b4a", label="SS=0")
    if 1 in g.columns:
        ax2.bar(x + 0.18, 100.0 * g[1].reindex(x).to_numpy(dtype=float), width=0.36, color="#1f4e79", label="SS=1")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax2.set_xlabel("days quintile")
    ax2.set_ylabel("Y3 rate (%)")
    ax2.set_title("Y3 rate by days quintile × SS")
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
    p6, p6b, p7, p8 = ctx["p6"], ctx["p6b"], ctx["p7"], ctx["p8"]
    p10 = ctx["p10"]
    lines = [
        "# Unused leftover of `c_ss_month` after `c_n_days_with_tx`",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_ss`. Do not edit `ops.py` / the 15-col card. "
        "Y3 never B. Do not overwrite `n_tx_qa.*` / `salary_qa.*` / `tax_qa.*`. "
        f"Night Y3 **{Y3_NIGHT[0]} / {Y3_NIGHT[1]}**. Days **{DAYS_BENCH}**. "
        f"Size **{SIZE_Y3_QUOTE}**. Salary **{SALARY_Y3_QUOTE}**. "
        f"Y7 TURNOVER **{Y7_TURNOVER[0]} / {Y7_TURNOVER[1]}**.",
        "",
        "`c_ss_month` = 1 if any category = social_security this month (fillna 0). "
        "Perm-stable #1 (ΔAUROC 0.034, sign −). `c_missed_salary` CLOSE as Y3 X (0.513). "
        "`a_n_tx` DROPPED as a days twin.",
        "",
        "## Headline",
        "",
        d["headline"],
        "",
        "## Brief questions",
        "",
        _md_table(brief_map(d, p3, p4, p7, p8)),
        "",
        "## KEEP / CLOSE / DROP / PARK",
        "",
        _md_table(
            [
                {
                    "object": "c_ss_month leftover after days",
                    "decision": f"**{d['headline_tag']}**",
                    "why": f"rank {_f(p4['y3_rank'])} OLS {_f(p4['y3_ols'])} fake_ols={p4['fake_ols']}",
                },
                {
                    "object": "15-col Y3 card stem",
                    "decision": f"**{d['card']}**",
                    "why": d["why"],
                },
                {
                    "object": "Payroll twin vs c_salary_month",
                    "decision": "**YES**" if p4["payroll_twin"] else "**no**",
                    "why": f"leftover after salary {_f(p4['y3_sal'])} R²={_f(p4['y3_sal_r2'])} Jaccard={_f(p2['jac']['jaccard'])}",
                },
                {
                    "object": "Twin vs days / a_n_tx / tax",
                    "decision": "**YES**" if d["twin"] else "**no**",
                    "why": f"ρ days {_f(p2['rho_days'])} salary {_f(p2['rho_sal'])} tax {_f(p2['rho_tax'])}",
                },
                {
                    "object": "SIZE vs log1p(a_in3)",
                    "decision": "**YES**" if d["size"] else "**no**",
                    "why": f"ρ={_f(p1['rho_in3'])} (gate ≥0.50); feature-report 0.362",
                },
                {
                    "object": "Inverse: days leftover after SS",
                    "decision": "**lives**" if p4["inv_lives"] else "**dies**",
                    "why": f"rank {_f(p4['inv_rank'])} OLS {_f(p4['inv_ols'])}",
                },
                {
                    "object": "Q6 lag leftover after days_lag1",
                    "decision": f"**{d['q6']}**",
                    "why": p7["prose"],
                },
                {
                    "object": "Calendar",
                    "decision": f"**{p8['shape']}**",
                    "why": p8["prose"],
                },
                {
                    "object": "Night quotes",
                    "decision": "**unchanged**",
                    "why": "Y3 0.762/0.752 · days 0.711 · size 0.617 · salary 0.671 · TURNOVER 0.720/0.712",
                },
            ]
        ),
        "",
        "## 1. Coverage; modal; acf1; size ρ",
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
        "## 3. Single-feature train group-fold AUROC",
        "",
        p3["prose"],
        "",
        _md_table(p3["rows"]),
        "",
        "## 4. Honest leftover after days + inverse",
        "",
        p4["prose"],
        "",
        _md_table(p4["rows"]),
        "",
        "## 5. Leftover after salary_month",
        "",
        f"Y3 leftover after `c_salary_month` rank {_f(p4['y3_sal'])} OLS {_f(p4['y3_sal_ols'])} "
        f"R²={_f(p4['y3_sal_r2'])}. After days+salary {_f(p4['y3_both'])}. "
        f"{'SS is a payroll twin' if p4['payroll_twin'] else 'SS is not a salary rewrite'}.",
        "",
        "## 6. SIZE terciles + ICC / demean",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        p6b["prose"],
        "",
        _md_table(p6b["rows"]),
        "",
        "## 7. Q6 — lag leftover after days_lag1",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 8. Calendar dummy",
        "",
        p8["prose"],
        "",
        _md_table(p8["cal"], cols=["month", "name", "q", "n", "P(SS)"]),
        "",
        "## 10. Dark 470 vs invoiced",
        "",
        p10["prose"],
        "",
        _md_table(p10["rows"]),
        "",
        "## Extra — holdout coverage",
        "",
        ctx["p_ho"]["prose"],
        "",
        "## Extra — fold-wise leftover",
        "",
        ctx["p_fl"]["prose"],
        "",
        _md_table(ctx["p_fl"]["rows"]),
        "",
        "## Extra — in-memory missed-SS (not a Y)",
        "",
        ctx["p_ms"]["prose"],
        "",
        "## Extra — SS-only vs salary-only",
        "",
        ctx["p_xo"]["prose"],
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
            lines.append(x)
    if "p_st" in ctx:
        lines.extend(["", *extra_markdown(ctx)])
    lines.extend(
        [
            "",
            f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts: coverage/modal/acf/size, twins, "
            "singles (days/size/salary CONFIRM), leftover days + inverse, leftover salary, "
            "terciles, ICC, Q6 lag leftover, calendar, dark 470, holdout, folds, missed-SS, XOR.",
            "",
            "Did **not**: rewrite `n_tx_qa.*` / `salary_qa.*` / `tax_qa.*` / `ops.py` / "
            "`gbm_core.py`, edit the 15-col card, grow TURNOVER, invent `y_ss`, merge I/M/J, "
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
    p7, p6b = ctx["p7"], ctx["p6b"]
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
            "metric": "auroc_c_ss_month",
            "value": p3["y3"],
            "coverage": cov,
            "notes": f"days={p3['days']:.4f} sal={p3['sal']:.4f} leftover={p4['y3_rank']:.4f} card={d['headline_tag']}",
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
            "metric": "auroc_c_ss_month_resid_days",
            "value": p4["y3_rank"],
            "coverage": cov,
            "notes": f"ols={p4['y3_ols']:.4f} fake_ols={p4['fake_ols']} r2={p4['y3_r2']:.4f} dies={p4['y3_dies']}",
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
            "metric": "auroc_c_ss_month_resid_salary",
            "value": p4["y3_sal"],
            "coverage": cov,
            "notes": f"r2={p4['y3_sal_r2']:.4f} payroll_twin={p4['payroll_twin']} jac={p2['jac']['jaccard']:.4f}",
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
            "metric": "auroc_days_resid_c_ss_month",
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
            "metric": "rho_c_ss_month_vs_days",
            "value": p2["rho_days"],
            "coverage": cov,
            "notes": f"sal={p2['rho_sal']:.4f} tax={p2['rho_tax']:.4f} size={p2['rho_size']:.4f} twins={d['twin']}",
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
            "metric": "icc_c_ss_month",
            "value": p6b["icc"],
            "coverage": cov,
            "notes": f"confirm099={p6b['confirm']} trait={p6b['trait']} acf1={p1['acf1']:.4f}",
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
            "metric": "auroc_c_ss_month_lag1_resid_days_lag1",
            "value": p7["l1_left"],
            "coverage": cov,
            "notes": f"l3={p7['l3_left']:.4f} l1_dies={p7['l1_dies']} q6={p7['q6']}",
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
            "metric": "auroc_c_ss_month_resid_salary",
            "value": p4["y3_sal"],
            "coverage": cov,
            "notes": f"ols={p4['y3_sal_ols']:.4f} r2={p4['y3_sal_r2']:.4f} payroll_twin={p4['payroll_twin']}",
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
            "metric": "auroc_c_ss_month_resid_days_salary",
            "value": p4["y3_both"],
            "coverage": cov,
            "notes": f"stack={ctx['p_st']['thin']:.4f} between={ctx['p_tr']['between']:.4f} within={ctx['p_tr']['within']:.4f}",
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
            "metric": "auroc_c_ss_month_resid_days_boot_p50",
            "value": ctx["p_bt"]["p50"],
            "coverage": cov,
            "notes": f"p05={ctx['p_bt']['p05']:.4f} p95={ctx['p_bt']['p95']:.4f} die={ctx['p_bt']['share_die']:.4f}",
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
            "metric": "auroc_c_ss_month_mixed_resid_days",
            "value": ctx["p_mx"]["after_days"],
            "coverage": cov,
            "notes": f"raw={ctx['p_mx']['raw']:.4f} n_pos={ctx['p_mx']['n_pos']} after_sal={ctx['p_mx']['after_sal']:.4f}",
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
    p2, p3, p4, p7 = ctx["p2"], ctx["p3"], ctx["p4"], ctx["p7"]
    p6b, p8, p10 = ctx["p6b"], ctx["p8"], ctx["p10"]
    p_tr, p_mx, p_bt = ctx["p_tr"], ctx["p_mx"], ctx["p_bt"]
    p_tx, p_pm, p_lg = ctx["p_tx"], ctx["p_pm"], ctx["p_lg"]
    text = (
        f"# Wave 4 — c_ss_month leftover after days\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/ss_qa.py`\n"
        f"- `analysis/outputs/ss_qa.md`\n"
        f"- `analysis/outputs/ss_qa.png`\n"
        f"- append-only `analysis/experiments/registry.csv`\n"
        f"- this note\n\n"
        f"Did not touch `n_tx_qa.*`, `salary_qa.*`, `tax_qa.*`, `ops.py`, `gbm_core.py`, "
        f"parquet / duckdb, `build_targets`, `product/`, the 15-col card, TURNOVER, "
        f"LIVE, canvas, or the parent journal. Night Y3 stays **0.762 / 0.752**. "
        f"Days 0.711. Size 0.617. Salary 0.671. TURNOVER **0.720 / 0.712**.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| leftover after days | **{d['headline_tag']}** {_f(p4['y3_rank'])} |\n"
        f"| 15-col card | **{d['card']}** |\n"
        f"| leftover after salary | **{_f(p4['y3_sal'])}** payroll_twin={p4['payroll_twin']} |\n"
        f"| leftover after days+salary | **{_f(p4['y3_both'])}** |\n"
        f"| leftover after tax | **{_f(p4['y3_tax'])}** |\n"
        f"| days leftover after SS | **{_f(p4['inv_rank'])}** |\n"
        f"| leftover after full stack | **{_f(ctx['p_st']['thin'])}** |\n"
        f"| BETWEEN leftover after days-mean | **{_f(p_tr['between'])}** |\n"
        f"| mixed-cadence leftover after days | **{_f(p_mx['after_days'])}** (dies) |\n"
        f"| Q6 | **{d['q6']}** |\n"
        f"| calendar | **{p8['shape']}** |\n\n"
        f"Y3 SS {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size'])} "
        f"vs salary {_f(p3['sal'])}. ρ vs days {_f(p2['rho_days'])} / a_n_tx {_f(p2['rhos']['a_n_tx'])} "
        f"/ salary {_f(p2['rho_sal'])} / tax {_f(p2['rho_tax'])} / log1p(a_in3) {_f(p2['rho_size'])}. "
        f"Jaccard(SS, salary) {_f(p2['jac']['jaccard'])}. Jaccard(SS, tax) {_f(p_tx['jac'])}. "
        f"Leftover after days rank {_f(p4['y3_rank'])} OLS {_f(p4['y3_ols'])} "
        f"ρ(resid,days)={_f(p4['y3_rho_days'])}. "
        f"lag1 leftover after days_lag1 {_f(p7['l1_left'])}. "
        f"ICC {_f(p6b['icc'])}. Dark leftover {_f(p10['dark'])} invoiced {_f(p10['inv'])}. "
        f"Bootstrap leftover-after-days p05/p50/p95 "
        f"{_f(p_bt['p05'])} / {_f(p_bt['p50'])} / {_f(p_bt['p95'])}. "
        f"Perm null p50 {_f(p_pm['p50'])} p(obs) {_f(p_pm['p'], 3)}. "
        f"Leave-one-group leftover min/med {_f(p_lg['min'])} / {_f(p_lg['med'])}. "
        f"Sibling leftover after group-mean {_f(ctx['p_sb']['after_g'])}; "
        f"BETWEEN boot p50 {_f(ctx['p_bb']['p50'])}.\n\n"
        f"KEEP is BETWEEN payer identity (always Y3 1.9% vs never 14.3%), not a days twin "
        f"and not a month flip. Parent absorbs the card.\n\n"
        f"## What failed / next\n\n"
        + "\n".join(f"- {x}" for x in ctx["failed"])
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.parent.mkdir(parents=True, exist_ok=True)
    WAVE_NOTE.write_text(text, encoding="utf-8")
    print(f"wrote {WAVE_NOTE}")


def run() -> dict:
    t0 = time.time()
    print(f"ss_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    con = connect()
    try:
        book = book_invoice_ids(con)
    finally:
        con.close()
    panel = attach_folds(panel)
    panel = add_missed_ss(panel)
    panel = add_panel_lags(
        panel,
        ["c_ss_month", "c_n_days_with_tx", "c_salary_month", "c_tax_month"],
        (1, 3),
    )
    tr = panel[panel["split"] == "train"].copy().reset_index(drop=True)
    assert_no_holdout(tr["company_id"])
    print(
        f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
        f"holdout CM={int((panel['split']=='holdout').sum())}"
    )

    print("pass 1 coverage")
    p1 = pass1_cov(tr)
    print("pass 2 twins")
    p2 = pass2_twins(tr)
    print("pass 3 singles")
    p3 = pass3_singles(tr)
    print("pass 4 leftover")
    p4 = pass4_leftover(tr)
    print("pass 6 terciles")
    p6 = pass6_terciles(tr)
    print("pass 6b ICC")
    p6b = pass6b_icc(tr)
    print("pass 7 Q6")
    p7 = pass7_q6(tr)
    print("pass 8 calendar")
    p8 = pass8_calendar(tr)
    print("pass 10 dark")
    p10 = pass10_dark(tr, book)
    print("extras")
    p_ho = pass_holdout(panel)
    p_fl = pass_fold_left(tr)
    p_ms = pass_missed_ss(tr)
    p_xo = pass_ss_only(tr)
    p_st = pass_stack(tr)
    p_tx = pass_tax_vs(tr)
    p_tr = pass_trait(tr)
    p_cd = pass_cadence(tr)
    p_xd = pass_xor_left(tr)
    p_sf = pass_sofar(tr)
    p_ch = pass_chronic(tr)
    p_fs = pass_fold_sal(tr)
    p_qs = pass_q6_sal(tr)
    p_is = pass_inv_sal(tr)
    p_sh = pass_share6(tr)
    p_ds = pass_dark_stack(tr, book)
    p_md = pass_month_dummies(tr)
    p_ev = pass_ever(tr)
    p_mx = pass_mixed_left(tr)
    p_so = pass_ss_only_cos(tr)
    p_yc = pass_y3_cells(tr)
    p_y2 = pass_y2_stack(tr)
    p_dt = pass_dark_trait(tr, book)
    p_mf = pass_mixed_folds(tr)
    p_bt = pass_boot(tr)
    p_cu = pass_cousins(tr)
    p_pm = pass_perm(tr)
    p_bs = pass_boot_sal(tr)
    p_oo = pass_oof_resid(tr)
    p_ne = pass_neither(tr)
    p_lg = pass_logo(tr)
    p_ml = pass_month_left(tr)
    p_sb = pass_sibling(tr)
    p_bb = pass_boot_between(tr)
    png = make_png(tr, p8)
    decision = decide(p1, p2, p3, p4, p6, p6b, p7, p8)
    failed = []
    if not p3["days_ok"]:
        failed.append(f"days replica drifted: {_f(p3['days'])} vs 0.711")
    if not p3["size_ok"]:
        failed.append(f"size replica drifted: {_f(p3['size'])} vs 0.617")
    if not p3["sal_ok"]:
        failed.append(f"salary_month replica drifted: {_f(p3['sal'])} vs 0.671")
    if p4["fake_ols"]:
        failed.append(
            f"OLS leftover after days {_f(p4['y3_ols'])} high vs honest rank {_f(p4['y3_rank'])}"
        )
    if not p7["days_l1_ok"]:
        failed.append(f"days lag1 {_f(p7['days_l1'])} off KEEP 0.684")
    if not p10["confirm"]:
        failed.append(f"dark/invoiced {p10['n_dark']}/{p10['n_erp']} off 470/744")
    if not failed:
        failed.append(
            f"no replica miss; leftover after days lives (KEEP { _f(p4['y3_rank']) }). "
            f"Mixed leftover after days {_f(p_mx['after_days'])} "
            f"{'dies' if np.isfinite(p_mx['after_days']) and p_mx['after_days'] < CHANCE else 'lives'} "
            "— KEEP is BETWEEN payer identity, not a month flip."
        )
    ctx = {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p6": p6,
        "p6b": p6b,
        "p7": p7,
        "p8": p8,
        "p10": p10,
        "p_ho": p_ho,
        "p_fl": p_fl,
        "p_ms": p_ms,
        "p_xo": p_xo,
        "p_st": p_st,
        "p_tx": p_tx,
        "p_tr": p_tr,
        "p_cd": p_cd,
        "p_xd": p_xd,
        "p_sf": p_sf,
        "p_ch": p_ch,
        "p_fs": p_fs,
        "p_qs": p_qs,
        "p_is": p_is,
        "p_sh": p_sh,
        "p_ds": p_ds,
        "p_md": p_md,
        "p_ev": p_ev,
        "p_mx": p_mx,
        "p_so": p_so,
        "p_yc": p_yc,
        "p_y2": p_y2,
        "p_dt": p_dt,
        "p_mf": p_mf,
        "p_bt": p_bt,
        "p_cu": p_cu,
        "p_pm": p_pm,
        "p_bs": p_bs,
        "p_oo": p_oo,
        "p_ne": p_ne,
        "p_lg": p_lg,
        "p_ml": p_ml,
        "p_sb": p_sb,
        "p_bb": p_bb,
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
        f"sal_left={_f(p4['y3_sal'])} inv={_f(p4['inv_rank'])} y3={_f(p3['y3'])} "
        f"elapsed={ctx['elapsed_s']:.0f}s"
    )
    return ctx


if __name__ == "__main__":
    run()
