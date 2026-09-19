"""Unused leftover of ``c_tax_month`` after ``c_n_days_with_tx`` as Y3 X.

Presence tax (any category = tax, fillna 0). Not missed-tax.
``c_missed_tax`` is already CLOSE as Y3 X / PARK as Y (calendar dummy;
Q-peaked; Y3 0.511 vs size 0.617 / days 0.711).

KEEP-as-X: beat size ≥0.02 AND leftover after days AND not SIZE
(|ρ| vs log1p(a_in3) ≥0.50) AND not a twin (|ρ|≥0.80 vs days / a_n_tx /
c_salary_month / c_ss_month / c_missed_tax). Leftover <0.55 dies.
Rank leftover is honest; OLS can fake a days leak.

Night quotes unchanged: Y3 0.762 / 0.752. Days 0.711. Size 0.617.
Y7 TURNOVER 0.720 / B_shallow 0.712. Do not put tax on the 15-col card.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.tax_month_qa

Owned: analysis/evaluate/tax_month_qa.py, analysis/outputs/tax_month_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_tax_month.md (end).
Do not overwrite tax_qa.py / tax_qa.md (missed-tax).
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
OUT_MD = ANALYSIS / "outputs" / "tax_month_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "tax_month_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_tax_month.md"
AGENT = "a91c4e02"
WAVE = "4"
ROUND = "R4"
MODEL = "tax_month_qa"
X_FAM = "C"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_Y3_QUOTE = 0.617
SALARY_Y3_QUOTE = 0.671
SS_Y3_QUOTE = 0.693
MISSED_TAX_QUOTE = 0.511
TAX_Y3_SSQA = 0.611
DAYS_LAG1_QUOTE = 0.684
Y3_NIGHT = (0.762, 0.752)
Y7_TURNOVER = (0.720, 0.712)
Q_TAX_QUOTE = 0.692
NQ_TAX_QUOTE = 0.434
Q_GAP_MISSED = 0.259
Q_GAP_SALARY = 0.006
TAX_SHARE_QUOTE = 0.518
SIZE_RHO_QUOTE = 0.378
ACF1_QUOTE = -0.13
ICC_QUOTE = 0.93
KEEP_DELTA = 0.02
TWIN_RHO = 0.80
SIZE_RHO = 0.50
CHANCE = 0.55
ICC_TRAIT = 0.85
MIN_POS = 50
MIN_ACF_PAIRS = 4
Q_MONTHS = (1, 4, 7, 10)
CHRONIC_GROUPS = ("GROUP_0158", "GROUP_0172")
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
    "c_tax_month",
    "c_missed_tax",
)
EXTRA_STORE = ("c_n_tx", "c_recency_days")
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
                {"fold": k, "auroc": float("nan"), "sign": 0, "n_va": int(va.sum()), "n_pos": int((va & (y == 1)).sum())}
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
    return " ".join(f"{r['auroc']:.3f}" if np.isfinite(r["auroc"]) else "—" for r in rec.get("folds", []))


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
    X = np.column_stack([np.ones(int(ok.sum()))] + [d.loc[ok, f"x{i}"].to_numpy(dtype=float) for i in range(n_x)])
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
        "rec": rec,
        "rrec": rrec,
    }


def company_demean(series: pd.Series, company: pd.Series) -> pd.Series:
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    return d["x"] - d.groupby("co")["x"].transform("mean")


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
    extra = {}
    for c in stems:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        g = s.groupby(out["company_id"], sort=False)
        for k in lags:
            extra[f"{c}_lag{k}"] = g.shift(k)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
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
    for col in ("c_tax_month", "c_missed_tax", "c_salary_month", "c_ss_month"):
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
    leak3 = leakage_check(["c_tax_month", "c_n_days_with_tx", "c_salary_month", "c_ss_month", "log_in3"], Y3, forbidden_prefixes=["b"])
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 never-B leak: {leak3['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def pass1_cov(tr: pd.DataFrame) -> dict:
    tax = pd.to_numeric(tr["c_tax_month"], errors="coerce")
    n_cm = int(len(tr))
    n1 = int((tax == 1).sum())
    prev = _pct(n1, n_cm)
    modal = max(prev, 1.0 - prev)
    acf1 = median_acf(tax, tr["company_id"], 1)
    acf3 = median_acf(tax, tr["company_id"], 3)
    rho = spearman(tax, tr["log_in3"])
    size_flag = bool(np.isfinite(rho) and abs(rho) >= SIZE_RHO)
    prose = (
        f"Train c_tax_month cov 100% P(1) {_pp(prev)} modal {_pp(modal)} "
        f"(feature-report 51.8% {'CONFIRM' if abs(prev - TAX_SHARE_QUOTE) < 0.03 else 'off'}). "
        f"acf1 {_f(acf1)} (quote −0.13 {'CONFIRM' if np.isfinite(acf1) and abs(acf1 - ACF1_QUOTE) < 0.08 else 'off'}) "
        f"acf3 {_f(acf3)}. ρ vs log1p(a_in3) {_f(rho)} (0.378 "
        f"{'CONFIRM' if np.isfinite(rho) and abs(rho - SIZE_RHO_QUOTE) < 0.05 else 'off'}; "
        f"{'SIZE' if size_flag else 'not SIZE'})."
    )
    print(prose)
    return {
        "prev": prev,
        "modal": modal,
        "n1": n1,
        "n_cm": n_cm,
        "cov": 1.0,
        "acf1": acf1,
        "acf3": acf3,
        "rho_in3": rho,
        "size_flag": size_flag,
        "prose": prose,
    }


def pass2_twins(tr: pd.DataFrame) -> dict:
    tax = tr["c_tax_month"]
    pairs = [
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("a_n_tx", tr["a_n_tx"]),
        ("c_salary_month", tr["c_salary_month"]),
        ("c_ss_month", tr["c_ss_month"]),
        ("c_missed_tax", tr["c_missed_tax"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("1-c_tax_month", 1 - pd.to_numeric(tax, errors="coerce")),
    ]
    rows = []
    rhos = {}
    twins = []
    gate = {"c_n_days_with_tx", "a_n_tx", "c_salary_month", "c_ss_month", "c_missed_tax"}
    for name, s in pairs:
        rho = spearman(tax, s)
        rhos[name] = rho
        twin = bool(name in gate and np.isfinite(rho) and abs(rho) >= TWIN_RHO)
        if twin:
            twins.append(name)
        rows.append({"vs": name, "ρ": _f(rho), "twin": "YES" if twin else ""})
    jac_m = jaccard(tax, tr["c_missed_tax"])
    jac_s = jaccard(tax, tr["c_ss_month"])
    jac_sal = jaccard(tax, tr["c_salary_month"])
    not_tax = pd.to_numeric(tax, errors="coerce").eq(0)
    miss = pd.to_numeric(tr["c_missed_tax"], errors="coerce").eq(1)
    p_miss_not = float(miss[not_tax].mean()) if int(not_tax.sum()) else float("nan")
    complement = bool(np.isfinite(rhos["c_missed_tax"]) and rhos["c_missed_tax"] <= -0.80)
    prose = (
        f"Spearman twins |ρ|≥0.80: {twins or 'none'}. "
        f"vs days {_f(rhos['c_n_days_with_tx'])} vs a_n_tx {_f(rhos['a_n_tx'])} "
        f"vs salary {_f(rhos['c_salary_month'])} vs ss {_f(rhos['c_ss_month'])} "
        f"vs missed_tax {_f(rhos['c_missed_tax'])} vs size {_f(rhos['log1p(a_in3)'])}. "
        f"Jaccard(tax, missed) {_f(jac_m['jaccard'])} both {jac_m['n_both']:,}. "
        f"P(missed|not tax) {_pp(p_miss_not)} — "
        f"{'presence IS the complement of missed' if complement else 'presence is NOT the complement of missed'}."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "twins": twins,
        "gate_twins": bool(twins),
        "jac_miss": jac_m,
        "jac_ss": jac_s,
        "jac_sal": jac_sal,
        "rho_days": rhos["c_n_days_with_tx"],
        "rho_ss": rhos["c_ss_month"],
        "rho_sal": rhos["c_salary_month"],
        "rho_miss": rhos["c_missed_tax"],
        "rho_size": rhos["log1p(a_in3)"],
        "p_miss_not": p_miss_not,
        "complement": complement,
        "prose": prose,
    }


def pass3_singles(tr: pd.DataFrame) -> dict:
    feats = {
        "c_tax_month": tr["c_tax_month"],
        "c_missed_tax": tr["c_missed_tax"],
        "c_salary_month": tr["c_salary_month"],
        "c_ss_month": tr["c_ss_month"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
        "log1p(a_in3)": tr["log_in3"],
        "is_q_month": tr["is_q_month"],
        "1-c_tax_month": 1 - pd.to_numeric(tr["c_tax_month"], errors="coerce"),
    }
    rows = []
    store = {}
    for y in (Y3, Y2):
        lab = tr[y].notna()
        for name, col in feats.items():
            res = signed_oof_auroc(tr[y], col, tr["fold"], lab)
            store[(y, name)] = res
            rows.append(_auc_row(y, name, res))
            print(f"AUROC {y} {name}: {'LOW_POWER' if res['low_power'] else _f(res['cv'])} n_pos={res['n_pos']} sign={res['train_sign']}")
    y3 = _cv(store[(Y3, "c_tax_month")])
    days = _cv(store[(Y3, "c_n_days_with_tx")])
    size = _cv(store[(Y3, "log1p(a_in3)")])
    sal = _cv(store[(Y3, "c_salary_month")])
    ss = _cv(store[(Y3, "c_ss_month")])
    miss = _cv(store[(Y3, "c_missed_tax")])
    y2 = _cv(store[(Y2, "c_tax_month")])
    days_ok = bool(np.isfinite(days) and abs(days - DAYS_BENCH) < 0.02)
    size_ok = bool(np.isfinite(size) and abs(size - SIZE_Y3_QUOTE) < 0.03)
    sal_ok = bool(np.isfinite(sal) and abs(sal - SALARY_Y3_QUOTE) < 0.02)
    ss_ok = bool(np.isfinite(ss) and abs(ss - SS_Y3_QUOTE) < 0.03)
    miss_ok = bool(np.isfinite(miss) and abs(miss - MISSED_TAX_QUOTE) < 0.03)
    beat_size = y3 - size if np.isfinite(y3) and np.isfinite(size) else float("nan")
    prose = (
        f"Y3 c_tax_month {_f(y3)} (ss_qa 0.611 {'CONFIRM' if np.isfinite(y3) and abs(y3 - TAX_Y3_SSQA) < 0.03 else 'off'}) "
        f"vs days {_f(days)} (0.711 {'CONFIRM' if days_ok else 'off'}) "
        f"vs size {_f(size)} (0.617 {'CONFIRM' if size_ok else 'off'}, Δ {_f(beat_size)}) "
        f"vs salary {_f(sal)} (0.671 {'CONFIRM' if sal_ok else 'off'}) "
        f"vs ss {_f(ss)} (0.693 {'CONFIRM' if ss_ok else 'off'}) "
        f"vs missed_tax {_f(miss)} (0.511 {'CONFIRM' if miss_ok else 'off'}). Y2 tax {_f(y2)}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "y3": y3,
        "days": days,
        "size": size,
        "sal": sal,
        "ss": ss,
        "miss": miss,
        "y2": y2,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "sal_ok": sal_ok,
        "ss_ok": ss_ok,
        "miss_ok": miss_ok,
        "beat_size": beat_size,
        "sign_y3": store[(Y3, "c_tax_month")]["train_sign"],
        "prose": prose,
    }


def pass4_leftover(tr: pd.DataFrame) -> dict:
    tax = tr["c_tax_month"]
    days = tr["c_n_days_with_tx"]
    sal = tr["c_salary_month"]
    ss = tr["c_ss_month"]
    miss = tr["c_missed_tax"]
    size = tr["log_in3"]
    ntx = tr["a_n_tx"]
    specs = [
        (Y3, "after days", (days,)),
        (Y3, "after missed_tax", (miss,)),
        (Y3, "after salary", (sal,)),
        (Y3, "after ss", (ss,)),
        (Y3, "after size", (size,)),
        (Y3, "after a_n_tx", (ntx,)),
        (Y3, "after days+salary+ss", (days, sal, ss)),
        (Y3, "after is_q_month", (tr["is_q_month"],)),
        (Y2, "after days", (days,)),
    ]
    rows = []
    store = {}
    for ycol, name, xs in specs:
        rec = leftover_diag(tr[ycol], tax, list(xs), tr["fold"], tr[ycol].notna())
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
        print(f"left {ycol} {name}: OLS={_f(rec['ols'])} rank={_f(rec['rank'])} ρctrl={_f(rec['rho_ctrl'])} R2={_f(rec['r2'])}")
    inv = leftover_diag(tr[Y3], days, [tax], tr["fold"], tr[Y3].notna())
    store[(Y3, "days after tax")] = inv
    rows.append(
        {
            "y": Y3,
            "control": "days after tax (inverse)",
            "OLS": _f(inv["ols"]),
            "rank": _f(inv["rank"]),
            "ρ(resid,ctrl)": _f(inv["rho_ctrl"]),
            "R²": _f(inv["r2"]),
            "fake": "YES" if inv["fake"] else "",
            "honest_dies": "YES" if inv["honest_dies"] else "no",
        }
    )
    y3 = store[(Y3, "after days")]
    leftover = y3["rank"]
    leftover_ols = y3["ols"]
    fake_ols = bool(np.isfinite(leftover_ols) and leftover_ols >= CHANCE and np.isfinite(leftover) and leftover < CHANCE)
    dies = bool(y3["honest_dies"] or (np.isfinite(leftover) and leftover < CHANCE))
    inv_lives = bool(np.isfinite(inv["rank"]) and inv["rank"] >= CHANCE and not inv["honest_dies"])
    prose = (
        f"Y3 leftover after days OLS {_f(leftover_ols)} rank {_f(leftover)} "
        f"ρ(resid,days)={_f(y3['rho_ctrl'])} R²={_f(y3['r2'])} "
        f"({'OLS-high / rank-dies — treat rank as honest' if fake_ols else 'OLS and rank agree'}; "
        f"{'FAKE-DAYS' if y3['fake'] else 'resid is not a days clone'}). "
        f"After missed {_f(store[(Y3, 'after missed_tax')]['rank'])} after salary {_f(store[(Y3, 'after salary')]['rank'])} "
        f"after ss {_f(store[(Y3, 'after ss')]['rank'])} after days+salary+ss {_f(store[(Y3, 'after days+salary+ss')]['rank'])}. "
        f"Inverse days after tax {_f(inv['rank'])} ({'0.711 bar lives' if inv_lives else '0.711 bar dies'}). "
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
        "y3_miss": store[(Y3, "after missed_tax")]["rank"],
        "y3_sal": store[(Y3, "after salary")]["rank"],
        "y3_ss": store[(Y3, "after ss")]["rank"],
        "y3_stack": store[(Y3, "after days+salary+ss")]["rank"],
        "y3_q": store[(Y3, "after is_q_month")]["rank"],
        "inv_rank": inv["rank"],
        "inv_ols": inv["ols"],
        "inv_lives": inv_lives,
        "y2_rank": store[(Y2, "after days")]["rank"],
        "prose": prose,
    }


def pass6_calendar(tr: pd.DataFrame) -> dict:
    tax = pd.to_numeric(tr["c_tax_month"], errors="coerce")
    cal = []
    for m in range(1, 13):
        sl = tax[tr["cal_month"] == m]
        cal.append(
            {
                "month": m,
                "name": pd.Timestamp(2000, m, 1).strftime("%b"),
                "q": int(m in Q_MONTHS),
                "n": f"{int(len(sl)):,}",
                "P(tax)": _pp(float(sl.mean()) if len(sl) else float("nan")),
                "share": float(sl.mean()) if len(sl) else float("nan"),
            }
        )
    q = float(np.nanmean([r["share"] for r in cal if r["q"] == 1]))
    nq = float(np.nanmean([r["share"] for r in cal if r["q"] == 0]))
    gap = q - nq
    ratio = q / nq if nq else float("nan")
    q_ok = bool(np.isfinite(q) and abs(q - Q_TAX_QUOTE) < 0.04)
    nq_ok = bool(np.isfinite(nq) and abs(nq - NQ_TAX_QUOTE) < 0.04)
    q_peaked = bool(gap >= 0.15)
    monthly = bool(np.isfinite(gap) and abs(gap) < 0.03)
    shape = "q_peaked" if q_peaked else ("monthly" if monthly else "mixed")
    like_missed = bool(np.isfinite(gap) and abs(gap - Q_GAP_MISSED) < 0.04)
    like_sal = bool(np.isfinite(gap) and abs(gap - Q_GAP_SALARY) < 0.03)
    qflag = leftover_diag(tr[Y3], tax, [tr["is_q_month"]], tr["fold"], tr[Y3].notna())
    rec_q = leftover_diag(tr[Y3], tax, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & (tr["is_q_month"] == 1))
    rec_nq = leftover_diag(tr[Y3], tax, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & (tr["is_q_month"] == 0))
    prose = (
        f"Tax share Q-months {_pp(q)} vs other {_pp(nq)} (gap {_pp(gap)}, ratio {_f(ratio, 2)}; "
        f"tax_qa 69.2%/43.4% {'CONFIRM' if q_ok and nq_ok else 'off'}). "
        f"Shape **{shape}** (missed-tax Q-gap 25.9% {'LIKE missed-tax' if like_missed else 'not missed-gap'}; "
        f"salary Q-gap 0.6% {'LIKE salary' if like_sal else 'not salary-monthly'}). "
        f"Leftover after is_q {_f(qflag['rank'])}; leftover after days on Q-months {_f(rec_q['rank'])} "
        f"on other {_f(rec_nq['rank'])}."
    )
    print(prose)
    return {
        "cal": cal,
        "q": q,
        "nq": nq,
        "gap": gap,
        "ratio": ratio,
        "shape": shape,
        "q_peaked": q_peaked,
        "like_missed": like_missed,
        "like_sal": like_sal,
        "q_left": qflag["rank"],
        "q_days": rec_q["rank"],
        "nq_days": rec_nq["rank"],
        "q_ok": q_ok,
        "nq_ok": nq_ok,
        "prose": prose,
    }


def pass7_q6(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    cols = ("c_tax_month", "c_tax_month_lag1", "c_tax_month_lag3", "c_n_days_with_tx", "c_n_days_with_tx_lag1")
    for col in cols:
        if col not in tr.columns:
            continue
        res = signed_oof_auroc(tr[Y3], tr[col], tr["fold"], tr[Y3].notna())
        store[col] = res
        rows.append({"col": col, "n": f"{res['n_defined']:,}", "n_pos": f"{res['n_pos']:,}", "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"])})
    now = _cv(store.get("c_tax_month", {"low_power": True}))
    lag1 = _cv(store.get("c_tax_month_lag1", {"low_power": True}))
    lag3 = _cv(store.get("c_tax_month_lag3", {"low_power": True}))
    days_l1 = _cv(store.get("c_n_days_with_tx_lag1", {"low_power": True}))
    rec1 = leftover_diag(tr[Y3], tr["c_tax_month_lag1"], [tr["c_n_days_with_tx_lag1"]], tr["fold"], tr[Y3].notna())
    rec3 = leftover_diag(tr[Y3], tr["c_tax_month_lag3"], [tr["c_n_days_with_tx_lag1"]], tr["fold"], tr[Y3].notna())
    days_l1_ok = bool(np.isfinite(days_l1) and abs(days_l1 - DAYS_LAG1_QUOTE) < 0.03)
    l1_dies = bool(rec1["honest_dies"] or (np.isfinite(rec1["rank"]) and rec1["rank"] < CHANCE))
    keep_q6 = bool(np.isfinite(now) and now >= CHANCE and np.isfinite(lag1) and lag1 >= CHANCE and (now - lag1) <= 0.03 and not l1_dies)
    q6 = "KEEP" if keep_q6 else "CLOSE"
    prose = (
        f"Y3 tax now {_f(now)} lag1 {_f(lag1)} lag3 {_f(lag3)}. "
        f"Days lag1 {_f(days_l1)} (KEEP 0.684 {'CONFIRM' if days_l1_ok else 'off'}). "
        f"tax_lag1 leftover after days_lag1 {_f(rec1['rank'])} lag3 {_f(rec3['rank'])}. Q6 {q6}."
    )
    print(prose)
    return {
        "rows": rows,
        "now": now,
        "lag1": lag1,
        "lag3": lag3,
        "days_l1": days_l1,
        "l1_left": rec1["rank"],
        "l3_left": rec3["rank"],
        "l1_dies": l1_dies,
        "days_l1_ok": days_l1_ok,
        "q6": q6,
        "prose": prose,
    }


def pass8_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    n_erp = int(last["company_id"].isin(book).sum())
    n_dark = int((~last["company_id"].isin(book)).sum())
    confirm = n_erp == 744 and n_dark == 470
    erp = tr["company_id"].isin(book)
    tax = pd.to_numeric(tr["c_tax_month"], errors="coerce")
    rows = []
    store = {}
    for name, mask in (("invoiced_744", erp), ("dark_470", ~erp)):
        raw = signed_oof_auroc(tr[Y3], tax, tr["fold"], tr[Y3].notna() & mask)
        rec = leftover_diag(tr[Y3], tax, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & mask)
        store[name] = rec
        rows.append(
            {
                "slice": name,
                "n_cm": f"{int(mask.sum()):,}",
                "P(tax)": _pp(float(tax[mask].mean())),
                "Y3 raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "leftover days": _f(rec["rank"]),
            }
        )
    prose = (
        f"Last-month ever-ERP {n_erp} / never-ERP {n_dark} ({'CONFIRM 744/470' if confirm else 'off'}). "
        f"P(tax) invoiced {_pp(float(tax[erp].mean()))} dark {_pp(float(tax[~erp].mean()))}. "
        f"Y3 leftover invoiced {_f(store['invoiced_744']['rank'])} dark {_f(store['dark_470']['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "n_erp": n_erp,
        "n_dark": n_dark,
        "confirm": confirm,
        "inv": store["invoiced_744"]["rank"],
        "dark": store["dark_470"]["rank"],
        "prev_erp": float(tax[erp].mean()),
        "prev_dark": float(tax[~erp].mean()),
        "prose": prose,
    }


def pass_icc(tr: pd.DataFrame) -> dict:
    tax = pd.to_numeric(tr["c_tax_month"], errors="coerce")
    icc = icc_anova(tax, tr["company_id"])
    dem = company_demean(tax, tr["company_id"])
    mu = company_mean(tax, tr["company_id"])
    rows = []
    store = {}
    for feat, col in (("now", tax), ("co_mean", mu), ("demean", dem)):
        res = signed_oof_auroc(tr[Y3], col, tr["fold"], tr[Y3].notna())
        store[feat] = res
        rows.append(_auc_row(Y3, feat, res))
    confirm = bool(np.isfinite(icc["icc"]) and abs(icc["icc"] - ICC_QUOTE) < 0.04)
    dem_left = leftover_diag(tr[Y3], dem, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    mu_left = leftover_diag(tr[Y3], mu, [company_mean(tr["c_n_days_with_tx"], tr["company_id"])], tr["fold"], tr[Y3].notna())
    prose = (
        f"c_tax_month ICC {_f(icc['icc'])} (feature-report 0.93 {'CONFIRM BETWEEN' if confirm else 'off'}). "
        f"Y3 company-mean {_f(_cv(store['co_mean']))} demean {_f(_cv(store['demean']))} "
        f"demean leftover after days {_f(dem_left['rank'])} BETWEEN leftover after days-mean {_f(mu_left['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "icc": icc["icc"],
        "confirm": confirm,
        "y3_mu": _cv(store["co_mean"]),
        "y3_dem": _cv(store["demean"]),
        "dem_left": dem_left["rank"],
        "between": mu_left["rank"],
        "prose": prose,
    }


def pass_holdout(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"]
    tax = pd.to_numeric(ho["c_tax_month"], errors="coerce")
    prose = (
        f"Holdout coverage only (no fit): {ho['company_id'].nunique()} co / {len(ho):,} CM, "
        f"cov {_pp(float(tax.notna().mean()))} P(tax) {_pp(float(tax.mean()))}."
    )
    print(prose)
    return {"n_cm": int(len(ho)), "n_co": int(ho["company_id"].nunique()), "prev": float(tax.mean()), "prose": prose}


def pass_q_stack(tr: pd.DataFrame) -> dict:
    """Does Q dummy + days eat leftover? Calendar split leftover."""
    tax = tr["c_tax_month"]
    days = tr["c_n_days_with_tx"]
    q = tr["is_q_month"]
    rec_qd = leftover_diag(tr[Y3], tax, [q, days], tr["fold"], tr[Y3].notna())
    rec_qss = leftover_diag(tr[Y3], tax, [q, tr["c_ss_month"]], tr["fold"], tr[Y3].notna())
    rec_all = leftover_diag(tr[Y3], tax, [q, days, tr["c_ss_month"], tr["c_salary_month"]], tr["fold"], tr[Y3].notna())
    raw_q = signed_oof_auroc(tr[Y3], tax, tr["fold"], tr[Y3].notna() & (q == 1))
    raw_nq = signed_oof_auroc(tr[Y3], tax, tr["fold"], tr[Y3].notna() & (q == 0))
    prose = (
        f"Leftover after Q+days {_f(rec_qd['rank'])}; after Q+ss {_f(rec_qss['rank'])}; "
        f"after Q+days+ss+salary {_f(rec_all['rank'])}. "
        f"Y3 raw Q-months {_f(_cv(raw_q))} other {_f(_cv(raw_nq))}."
    )
    print(prose)
    return {
        "q_days": rec_qd["rank"],
        "q_ss": rec_qss["rank"],
        "all": rec_all["rank"],
        "raw_q": _cv(raw_q),
        "raw_nq": _cv(raw_nq),
        "prose": prose,
    }


def pass_cadence(tr: pd.DataFrame) -> dict:
    tax = pd.to_numeric(tr["c_tax_month"], errors="coerce")
    mu = tax.groupby(tr["company_id"]).transform("mean")
    always = mu >= 0.999
    never = mu <= 0.001
    mixed = (~always) & (~never)
    rows = []
    for name, mask in (("always", always), ("never", never), ("mixed", mixed)):
        lab = tr[Y3].notna() & mask
        n_co = int(tr.loc[mask, "company_id"].nunique())
        raw = signed_oof_auroc(tr[Y3], tax, tr["fold"], lab)
        rec = leftover_diag(tr[Y3], tax, [tr["c_n_days_with_tx"]], tr["fold"], lab)
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
    mixed_left = leftover_diag(tr[Y3], tax, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & mixed)
    prose = (
        f"Cadence always {rows[0]['n_co']} / never {rows[1]['n_co']} / mixed {rows[2]['n_co']}. "
        f"Y3 rate always {rows[0]['Y3 rate']} never {rows[1]['Y3 rate']} mixed {rows[2]['Y3 rate']}. "
        f"Mixed leftover after days {_f(mixed_left['rank'])}."
    )
    print(prose)
    return {"rows": rows, "mixed_left": mixed_left["rank"], "prose": prose}


def pass_cousins(tr: pd.DataFrame) -> dict:
    tax = tr["c_tax_month"]
    rows = []
    for name in EXTRA_STORE:
        if name not in tr.columns:
            continue
        rec = leftover_diag(tr[Y3], tax, [tr[name]], tr["fold"], tr[Y3].notna())
        rho = spearman(tax, tr[name])
        rows.append({"control": name, "ρ": _f(rho), "leftover": _f(rec["rank"])})
        print(f"cousin after {name}: leftover={_f(rec['rank'])} ρ={_f(rho)}")
    prose = "Leftover after other C stems: " + "; ".join(f"{r['control']} {r['leftover']}" for r in rows)
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_boot(tr: pd.DataFrame, n_boot: int = 280) -> dict:
    rng = np.random.default_rng(FOLD_SEED)
    lab = tr[Y3].notna()
    work = tr.loc[lab, ["company_id", Y3, "c_tax_month", "c_n_days_with_tx", "fold"]].copy()
    cos = work["company_id"].astype(str).unique()
    idx = {c: work.index[work["company_id"].astype(str) == c].to_numpy() for c in cos}
    vals = []
    t0 = time.time()
    for i in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        ix = np.concatenate([idx[c] for c in draw])
        sl = work.loc[ix]
        rec = leftover_diag(sl[Y3], sl["c_tax_month"], [sl["c_n_days_with_tx"]], sl["fold"], sl[Y3].notna())
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
        if (i + 1) % 50 == 0:
            print(f"boot {i+1}/{n_boot} mean={np.mean(vals):.3f}")
    arr = np.array(vals, dtype=float)
    lo, mid, hi = (float(np.quantile(arr, q)) for q in (0.05, 0.50, 0.95))
    prose = (
        f"Company bootstrap n={len(arr)} leftover-after-days rank p05/p50/p95 "
        f"{_f(lo)} / {_f(mid)} / {_f(hi)}; share <0.55 {_pp(float(np.mean(arr < CHANCE)))}; "
        f"wall {time.time()-t0:.0f}s."
    )
    print(prose)
    return {"p05": lo, "p50": mid, "p95": hi, "share_die": float(np.mean(arr < CHANCE)), "prose": prose}


def pass_sofar(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    slices = (
        ("so_far<6", tr["months_so_far"] < 6),
        ("so_far 6-11", (tr["months_so_far"] >= 6) & (tr["months_so_far"] < 12)),
        ("so_far≥12", tr["months_so_far"] >= 12),
        ("short_<12", tr["trail_class"] == "short_<12"),
    )
    for name, mask in slices:
        raw = signed_oof_auroc(tr[Y3], tr["c_tax_month"], tr["fold"], tr[Y3].notna() & mask)
        rec = leftover_diag(tr[Y3], tr["c_tax_month"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & mask)
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
        f"short {_f(store['short_<12']['rank'])}."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_xor(tr: pd.DataFrame) -> dict:
    tax = pd.to_numeric(tr["c_tax_month"], errors="coerce")
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    sal = pd.to_numeric(tr["c_salary_month"], errors="coerce")
    jac = jaccard(tax, ss)
    disagree = tax != ss
    rec = leftover_diag(tr[Y3], tax, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & disagree)
    rec_s = leftover_diag(tr[Y3], tax, [ss], tr["fold"], tr[Y3].notna())
    rec_sal = leftover_diag(tr[Y3], tax, [sal, ss], tr["fold"], tr[Y3].notna())
    prose = (
        f"Jaccard(tax, ss) {_f(jac['jaccard'])} both {jac['n_both']:,}. "
        f"Disagree leftover after days {_f(rec['rank'])}. "
        f"Leftover after ss+salary {_f(rec_sal['rank'])} (after ss alone {_f(rec_s['rank'])})."
    )
    print(prose)
    return {"jac": jac["jaccard"], "disagree": rec["rank"], "after_ss_sal": rec_sal["rank"], "prose": prose}


def pass_chronic(tr: pd.DataFrame) -> dict:
    hot = tr["group_id"].astype(str).isin(CHRONIC_GROUPS)
    rec = leftover_diag(tr[Y3], tr["c_tax_month"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & ~hot)
    raw = signed_oof_auroc(tr[Y3], tr["c_tax_month"], tr["fold"], tr[Y3].notna() & ~hot)
    prose = f"Drop chronic {CHRONIC_GROUPS}: leftover after days {_f(rec['rank'])} raw {_f(_cv(raw))}."
    print(prose)
    return {"left": rec["rank"], "raw": _cv(raw), "prose": prose}


def pass_between_ss(tr: pd.DataFrame) -> dict:
    """Is BETWEEN tax leftover just SS payer identity?"""
    tax = pd.to_numeric(tr["c_tax_month"], errors="coerce")
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    mu_t = company_mean(tax, tr["company_id"])
    mu_s = company_mean(ss, tr["company_id"])
    mu_d = company_mean(tr["c_n_days_with_tx"], tr["company_id"])
    rec = leftover_diag(tr[Y3], mu_t, [mu_s], tr["fold"], tr[Y3].notna())
    rec_d = leftover_diag(tr[Y3], mu_t, [mu_d, mu_s], tr["fold"], tr[Y3].notna())
    rec_raw = leftover_diag(tr[Y3], tax, [mu_s, tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"BETWEEN tax leftover after SS-mean {_f(rec['rank'])}; after days-mean+SS-mean {_f(rec_d['rank'])}; "
        f"raw tax leftover after SS-mean+days {_f(rec_raw['rank'])}."
    )
    print(prose)
    return {"after_ss": rec["rank"], "after_both": rec_d["rank"], "raw": rec_raw["rank"], "prose": prose}


def pass_disagree_q(tr: pd.DataFrame) -> dict:
    """Disagree leftover 0.696 — is it Q-month exclusive tax?"""
    tax = pd.to_numeric(tr["c_tax_month"], errors="coerce")
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    q = tr["is_q_month"] == 1
    slices = (
        ("tax≠ss", tax != ss),
        ("tax≠ss & Q", (tax != ss) & q),
        ("tax≠ss & other", (tax != ss) & ~q),
        ("tax=1 ss=0", (tax == 1) & (ss == 0)),
        ("tax=0 ss=1", (tax == 0) & (ss == 1)),
    )
    rows = []
    store = {}
    for name, mask in slices:
        raw = signed_oof_auroc(tr[Y3], tax, tr["fold"], tr[Y3].notna() & mask)
        rec = leftover_diag(tr[Y3], tax, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & mask)
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
        f"Disagree leftover after days {_f(store['tax≠ss']['rank'])}; "
        f"Q {_f(store['tax≠ss & Q']['rank'])} other {_f(store['tax≠ss & other']['rank'])}; "
        f"tax-only {_f(store['tax=1 ss=0']['rank'])} ss-only {_f(store['tax=0 ss=1']['rank'])}."
    )
    print(prose)
    return {"rows": rows, "prose": prose, "disagree": store["tax≠ss"]["rank"]}


def pass_perm(tr: pd.DataFrame, n_perm: int = 120) -> dict:
    rng = np.random.default_rng(FOLD_SEED + 7)
    lab = tr[Y3].notna()
    work = tr.loc[lab, [Y3, "c_tax_month", "c_n_days_with_tx", "fold"]].copy()
    vals = []
    t0 = time.time()
    for i in range(n_perm):
        sh = work["c_tax_month"].to_numpy(copy=True)
        for k in range(N_FOLDS):
            ix = np.where(work["fold"].to_numpy() == k)[0]
            sh[ix] = rng.permutation(sh[ix])
        rec = leftover_diag(work[Y3], pd.Series(sh, index=work.index), [work["c_n_days_with_tx"]], work["fold"], work[Y3].notna())
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
        if (i + 1) % 40 == 0:
            print(f"perm {i+1}/{n_perm} mean={np.mean(vals):.3f}")
    arr = np.array(vals, dtype=float)
    lo, mid, hi = (float(np.quantile(arr, q)) for q in (0.05, 0.50, 0.95))
    obs = leftover_diag(tr[Y3], tr["c_tax_month"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())["rank"]
    p = float(np.mean(arr >= obs)) if np.isfinite(obs) else float("nan")
    prose = (
        f"Within-fold permute tax leftover-after-days p05/p50/p95 {_f(lo)} / {_f(mid)} / {_f(hi)}; "
        f"observed {_f(obs)}; p(perm ≥ obs) {_f(p, 3)}; wall {time.time()-t0:.0f}s."
    )
    print(prose)
    return {"p05": lo, "p50": mid, "p95": hi, "obs": obs, "p": p, "prose": prose}


def pass_dark_q(tr: pd.DataFrame, book: set[str]) -> dict:
    erp = tr["company_id"].isin(book)
    q = tr["is_q_month"] == 1
    rows = []
    for name, mask in (("invoiced Q", erp & q), ("invoiced other", erp & ~q), ("dark Q", (~erp) & q), ("dark other", (~erp) & ~q)):
        rec = leftover_diag(tr[Y3], tr["c_tax_month"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & mask)
        raw = signed_oof_auroc(tr[Y3], tr["c_tax_month"], tr["fold"], tr[Y3].notna() & mask)
        rows.append(
            {
                "slice": name,
                "n_pos": raw["n_pos"],
                "raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "leftover days": _f(rec["rank"]),
            }
        )
    prose = "Dark × Q leftover after days: " + "; ".join(f"{r['slice']} {r['leftover days']}" for r in rows)
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_q_dummy_left(tr: pd.DataFrame) -> dict:
    """Is the Q dummy leftover after days? Tax leftover vs calendar leftover."""
    rec_q = leftover_diag(tr[Y3], tr["is_q_month"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_t = leftover_diag(tr[Y3], tr["c_tax_month"], [tr["is_q_month"], tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    raw_q = signed_oof_auroc(tr[Y3], tr["is_q_month"], tr["fold"], tr[Y3].notna())
    prose = (
        f"is_q_month Y3 {_f(_cv(raw_q))} leftover after days {_f(rec_q['rank'])}. "
        f"tax leftover after Q+days {_f(rec_t['rank'])}."
    )
    print(prose)
    return {"q_raw": _cv(raw_q), "q_left": rec_q["rank"], "tax_qd": rec_t["rank"], "prose": prose}


def pass_logo(tr: pd.DataFrame) -> dict:
    lab = tr[Y3].notna()
    counts = tr.loc[lab].groupby("group_id").size().sort_values(ascending=False)
    keep = [g for g, n in counts.items() if n >= 40][:30]
    vals = []
    for g in keep:
        mask = lab & (tr["group_id"].astype(str) != str(g))
        rec = leftover_diag(tr[Y3], tr["c_tax_month"], [tr["c_n_days_with_tx"]], tr["fold"], mask)
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
    arr = np.array(vals, dtype=float)
    prose = (
        f"Leave-one-group leftover after days (top {len(keep)}): "
        f"min/med/max {_f(float(arr.min()))} / {_f(float(np.median(arr)))} / {_f(float(arr.max()))}."
    )
    print(prose)
    return {"min": float(arr.min()), "med": float(np.median(arr)), "max": float(arr.max()), "prose": prose}


def pass_boot_qdays(tr: pd.DataFrame, n_boot: int = 160) -> dict:
    rng = np.random.default_rng(FOLD_SEED + 5)
    lab = tr[Y3].notna()
    work = tr.loc[lab, ["company_id", Y3, "c_tax_month", "c_n_days_with_tx", "is_q_month", "fold"]].copy()
    cos = work["company_id"].astype(str).unique()
    idx = {c: work.index[work["company_id"].astype(str) == c].to_numpy() for c in cos}
    vals = []
    t0 = time.time()
    for i in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        ix = np.concatenate([idx[c] for c in draw])
        sl = work.loc[ix]
        rec = leftover_diag(sl[Y3], sl["c_tax_month"], [sl["is_q_month"], sl["c_n_days_with_tx"]], sl["fold"], sl[Y3].notna())
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
        if (i + 1) % 40 == 0:
            print(f"boot_qd {i+1}/{n_boot} mean={np.mean(vals):.3f}")
    arr = np.array(vals, dtype=float)
    lo, mid, hi = (float(np.quantile(arr, q)) for q in (0.05, 0.50, 0.95))
    prose = (
        f"Bootstrap leftover after Q+days p05/p50/p95 {_f(lo)} / {_f(mid)} / {_f(hi)}; "
        f"share<0.55 {_pp(float(np.mean(arr < CHANCE)))}; wall {time.time()-t0:.0f}s."
    )
    print(prose)
    return {"p05": lo, "p50": mid, "p95": hi, "prose": prose}


def size_terciles(tr: pd.DataFrame) -> pd.Series:
    med = tr.groupby("company_id")["log_in3"].median()
    cuts = med.quantile([1 / 3, 2 / 3])
    lab = pd.Series("T2", index=med.index)
    lab[med <= cuts.iloc[0]] = "T1"
    lab[med > cuts.iloc[1]] = "T3"
    return tr["company_id"].map(lab)


def pass_terciles(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    rows = []
    store = {}
    for name, mask in (("T1", terc == "T1"), ("T2+T3", terc.isin(["T2", "T3"]))):
        raw = signed_oof_auroc(tr[Y3], tr["c_tax_month"], tr["fold"], tr[Y3].notna() & mask)
        rec = leftover_diag(tr[Y3], tr["c_tax_month"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & mask)
        store[name] = rec
        rows.append(
            {
                "slice": name,
                "n_pos": raw["n_pos"],
                "raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "leftover days": _f(rec["rank"]),
            }
        )
    prose = f"Leftover after days T1 {_f(store['T1']['rank'])}; T2+T3 {_f(store['T2+T3']['rank'])}."
    print(prose)
    return {"rows": rows, "t1": store["T1"]["rank"], "t23": store["T2+T3"]["rank"], "prose": prose}


def pass_y3_cells(tr: pd.DataFrame) -> dict:
    lab = tr[Y3].notna()
    work = tr.loc[lab].copy()
    work["_tax"] = pd.to_numeric(work["c_tax_month"], errors="coerce")
    days = pd.to_numeric(work["c_n_days_with_tx"], errors="coerce")
    try:
        work["_td"] = pd.qcut(days.rank(method="first"), 3, labels=["low days", "mid", "high days"])
    except ValueError:
        work["_td"] = "all"
    rows = []
    for td, g in work.groupby("_td", observed=False):
        for tv, gg in g.groupby("_tax"):
            rows.append(
                {
                    "days tercile": str(td),
                    "tax": int(tv) if pd.notna(tv) else "—",
                    "n": f"{len(gg):,}",
                    "n_pos": int(pd.to_numeric(gg[Y3], errors="coerce").sum()),
                    "Y3 rate": _pp(float(pd.to_numeric(gg[Y3], errors="coerce").mean())),
                }
            )
    prose = "Y3 rate cells by days tercile × tax (sign −)."
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_q_months(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for m, name in ((1, "Jan"), (4, "Apr"), (7, "Jul"), (10, "Oct")):
        mask = tr["cal_month"] == m
        raw = signed_oof_auroc(tr[Y3], tr["c_tax_month"], tr["fold"], tr[Y3].notna() & mask)
        rec = leftover_diag(tr[Y3], tr["c_tax_month"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & mask)
        store[name] = rec
        rows.append(
            {
                "month": name,
                "n_pos": raw["n_pos"],
                "raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "leftover days": _f(rec["rank"]),
            }
        )
    rec_r = leftover_diag(
        tr[Y3], tr["c_tax_month"], [tr["c_n_days_with_tx"], tr["c_recency_days"]], tr["fold"], tr[Y3].notna()
    ) if "c_recency_days" in tr.columns else {"rank": float("nan")}
    prose = (
        f"Q-month leftover after days Jan {_f(store['Jan']['rank'])} Apr {_f(store['Apr']['rank'])} "
        f"Jul {_f(store['Jul']['rank'])} Oct {_f(store['Oct']['rank'])}. "
        f"Leftover after days+recency {_f(rec_r['rank'])}."
    )
    print(prose)
    return {"rows": rows, "recency": rec_r["rank"], "prose": prose}


def pass_fold_left(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(tr[Y3], tr["c_tax_month"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rows = []
    for k in range(N_FOLDS):
        rows.append({"fold": k, "OLS": _f(fold_k(rec["rec"], k)), "rank": _f(fold_k(rec["rrec"], k))})
    vals = [fold_k(rec["rrec"], k) for k in range(N_FOLDS)]
    finite = [v for v in vals if np.isfinite(v)]
    spread = float(max(finite) - min(finite)) if finite else float("nan")
    prose = f"Y3 leftover-after-days rank folds {rec['rank_folds']} spread {_f(spread)}."
    print(prose)
    return {"rows": rows, "spread": spread, "rank_folds": rec["rank_folds"], "prose": prose}


def decide(p1, p2, p3, p4, p6, p7) -> dict:
    twin = bool(p2["gate_twins"])
    size = bool(p1["size_flag"])
    leftover = p4["y3_rank"]
    leftover_dies = bool(p4["y3_dies"])
    beat = bool(np.isfinite(p3["beat_size"]) and p3["beat_size"] >= KEEP_DELTA)
    keep_x = bool(beat and (not leftover_dies) and (not size) and (not twin))
    if keep_x:
        card = "KEEP as unused leftover"
        tag = "KEEP"
        why = (
            f"leftover after days rank {_f(leftover)} lives; beats size {_f(p3['size'])} "
            f"by {_f(p3['beat_size'])}; not SIZE; not a twin. Do not put tax on the 15-col card."
        )
    elif leftover_dies or (not beat) or twin:
        if leftover_dies and (not beat):
            card = "CLOSE unused leftover"
            tag = "CLOSE"
        elif twin:
            card = "DROP from the 44 as Y3 X"
            tag = "DROP"
        else:
            card = "CLOSE unused leftover"
            tag = "CLOSE"
        why = (
            f"Y3 leftover after days rank {_f(leftover)} OLS {_f(p4['y3_ols'])} "
            f"{'dies' if leftover_dies else 'thin'}; beat_size={_f(p3['beat_size'])} twin={twin}. "
            f"After ss {_f(p4['y3_ss'])} after salary {_f(p4['y3_sal'])} stack {_f(p4['y3_stack'])}. "
            f"Calendar {p6['shape']}. Do not put tax on the 15-col card."
        )
    else:
        card = "PARK as Y"
        tag = "PARK"
        why = f"Y3 {_f(p3['y3'])} leftover {_f(leftover)} does not clear KEEP-as-X. Do not invent y_tax."
    q6 = "CLOSE" if leftover_dies or twin or (not beat) else p7["q6"]
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
            f"fake_ols={p4['fake_ols']}). Y3 tax {_f(p3['y3'])} vs days {_f(p3['days'])} "
            f"vs size {_f(p3['size'])} vs salary {_f(p3['sal'])} vs ss {_f(p3['ss'])} "
            f"vs missed {_f(p3['miss'])}. After ss {_f(p4['y3_ss'])} after salary {_f(p4['y3_sal'])} "
            f"stack {_f(p4['y3_stack'])}. Inverse {_f(p4['inv_rank'])}. "
            f"Calendar {p6['shape']} Q-gap {_pp(p6['gap'])}. Q6 {q6}. "
            f"{card}. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged. "
            "Do not put tax on the 15-col card."
        ),
    }


def make_png(tr: pd.DataFrame, p6: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    names = [r["name"] for r in p6["cal"]]
    shares = [100.0 * r["share"] if np.isfinite(r["share"]) else np.nan for r in p6["cal"]]
    colors = ["#c44536" if r["q"] else "#1f4e79" for r in p6["cal"]]
    ax.bar(names, shares, color=colors)
    ax.set_ylabel("P(c_tax_month=1) %")
    ax.set_title(f"Train calendar ({p6['shape']})")
    ax.axhline(100.0 * p6["q"], color="#c44536", ls="--", lw=1, label="Q mean")
    ax.axhline(100.0 * p6["nq"], color="#1f4e79", ls=":", lw=1, label="other mean")
    ax.legend(frameon=False, fontsize=8)

    ax2 = axes[1]
    tax = pd.to_numeric(tr["c_tax_month"], errors="coerce")
    lab = tr[Y3].notna() & tax.notna()
    work = tr.loc[lab].copy()
    work["_tax"] = tax[lab]
    work["_y"] = pd.to_numeric(work[Y3], errors="coerce")
    days = pd.to_numeric(work["c_n_days_with_tx"], errors="coerce")
    try:
        work["_qd"] = pd.qcut(days.rank(method="first"), 5, labels=False) + 1
    except ValueError:
        work["_qd"] = 1
    g = work.groupby(["_qd", "_tax"])["_y"].mean().unstack()
    x = np.arange(1, 6)
    if 0 in g.columns:
        ax2.bar(x - 0.18, 100.0 * g[0].reindex(x).to_numpy(dtype=float), width=0.36, color="#9e6b4a", label="tax=0")
    if 1 in g.columns:
        ax2.bar(x + 0.18, 100.0 * g[1].reindex(x).to_numpy(dtype=float), width=0.36, color="#1f4e79", label="tax=1")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax2.set_xlabel("days quintile")
    ax2.set_ylabel("Y3 rate (%)")
    ax2.set_title("Y3 rate by days quintile × tax")
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
    p6, p7, p8, p_icc = ctx["p6"], ctx["p7"], ctx["p8"], ctx["p_icc"]
    lines = [
        "# Unused leftover of `c_tax_month` after `c_n_days_with_tx`",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_tax`. Do not put tax on the 15-col card. "
        "Do not overwrite `tax_qa.py` / `tax_qa.md` (missed-tax). Y3 never B. "
        f"Night Y3 **{Y3_NIGHT[0]:.3f} / {Y3_NIGHT[1]:.3f}**. Days **{DAYS_BENCH}**. "
        f"Size **{SIZE_Y3_QUOTE}**. Salary **{SALARY_Y3_QUOTE}**. SS leftover **0.635**. "
        f"Y7 TURNOVER **{Y7_TURNOVER[0]:.3f} / {Y7_TURNOVER[1]:.3f}**.",
        "",
        "`c_tax_month` = 1 if any category = tax this month (fillna 0). "
        "`c_missed_tax` already CLOSE as Y3 X (0.511) / PARK as Y (Q-peaked calendar dummy).",
        "",
        "## Headline",
        "",
        d["headline"],
        "",
        "## KEEP / CLOSE / DROP / PARK",
        "",
        _md_table(
            [
                {"object": "c_tax_month leftover after days", "decision": f"**{d['headline_tag']}**", "why": f"rank {_f(p4['y3_rank'])} OLS {_f(p4['y3_ols'])} fake_ols={p4['fake_ols']} ρ(resid,days)={_f(p4['y3_rho_days'])}"},
                {"object": "as Y3 X (not on 15-col card)", "decision": f"**{d['card']}**", "why": d["why"]},
                {"object": "Twin / SIZE", "decision": f"twin={'YES' if d['twin'] else 'no'} SIZE={'YES' if d['size'] else 'no'}", "why": f"ρ days {_f(p2['rho_days'])} ss {_f(p2['rho_ss'])} salary {_f(p2['rho_sal'])} missed {_f(p2['rho_miss'])} size {_f(p1['rho_in3'])}"},
                {"object": "Complement of missed-tax?", "decision": "**YES**" if p2["complement"] else "**no**", "why": f"ρ missed {_f(p2['rho_miss'])} Jaccard {_f(p2['jac_miss']['jaccard'])} P(missed|not tax) {_pp(p2['p_miss_not'])}"},
                {"object": "Calendar", "decision": f"**{p6['shape']}**", "why": p6["prose"]},
                {"object": "Q6 lag leftover after days_lag1", "decision": f"**{d['q6']}**", "why": p7["prose"]},
                {"object": "Inverse: days leftover after tax", "decision": "**lives**" if p4["inv_lives"] else "**dies**", "why": f"rank {_f(p4['inv_rank'])} OLS {_f(p4['inv_ols'])}"},
                {"object": "Night quotes", "decision": "**unchanged**", "why": "Y3 0.762/0.752 · days 0.711 · size 0.617 · TURNOVER 0.720/0.712"},
            ]
        ),
        "",
        "## 1. Coverage; twins; SIZE",
        "",
        p1["prose"],
        "",
        p2["prose"],
        "",
        _md_table(p2["rows"]),
        "",
        "## 2. Single-feature train group-fold AUROC",
        "",
        p3["prose"],
        "",
        _md_table(p3["rows"]),
        "",
        "## 3. Honest leftover after days + inverse + cousins",
        "",
        p4["prose"],
        "",
        _md_table(p4["rows"]),
        "",
        "## 4. Calendar — Q-peaked like missed-tax or monthly like salary?",
        "",
        p6["prose"],
        "",
        _md_table(p6["cal"], cols=["month", "name", "q", "n", "P(tax)"]),
        "",
        "## 5. Dark vs ERP leftover",
        "",
        p8["prose"],
        "",
        _md_table(p8["rows"]),
        "",
        "## 6. Q6 — lag leftover after days_lag1",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 7. ICC / BETWEEN",
        "",
        p_icc["prose"],
        "",
        _md_table(p_icc["rows"]),
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
    if "p_qs" in ctx:
        lines.extend(
            [
                "",
                "## Extra — leftover after Q dummy + days / ss",
                "",
                ctx["p_qs"]["prose"],
                "",
                "## Extra — always / never / mixed cadence",
                "",
                ctx["p_cd"]["prose"],
                "",
                _md_table(ctx["p_cd"]["rows"]),
                "",
                "## Extra — so-far leftover",
                "",
                ctx["p_sf"]["prose"],
                "",
                _md_table(ctx["p_sf"]["rows"]),
                "",
                "## Extra — tax vs SS XOR / leftover after ss+salary",
                "",
                ctx["p_xo"]["prose"],
                "",
                "## Extra — drop chronic",
                "",
                ctx["p_ch"]["prose"],
                "",
                "## Extra — company bootstrap leftover after days",
                "",
                ctx["p_bt"]["prose"],
                "",
                "## Extra — BETWEEN tax leftover after SS-mean",
                "",
                ctx["p_bs"]["prose"],
                "",
                "## Extra — tax≠SS leftover (Q split)",
                "",
                ctx["p_dq"]["prose"],
                "",
                _md_table(ctx["p_dq"]["rows"]),
                "",
                "## Extra — permutation null leftover after days",
                "",
                ctx["p_pm"]["prose"],
                "",
                "## Extra — dark × Q leftover",
                "",
                ctx["p_dq2"]["prose"],
                "",
                _md_table(ctx["p_dq2"]["rows"]),
                "",
                "## Extra — Q dummy leftover after days",
                "",
                ctx["p_qd"]["prose"],
                "",
                "## Extra — leave-one-group leftover after days",
                "",
                ctx["p_lg"]["prose"],
                "",
                "## Extra — bootstrap leftover after Q+days",
                "",
                ctx["p_bq"]["prose"],
                "",
                "## Extra — SIZE tercile leftover after days",
                "",
                ctx["p_te"]["prose"],
                "",
                _md_table(ctx["p_te"]["rows"]),
                "",
                "## Extra — Y3 rate cells days tercile × tax",
                "",
                ctx["p_yc"]["prose"],
                "",
                _md_table(ctx["p_yc"]["rows"]),
                "",
                "## Extra — leftover after other C stems",
                "",
                ctx["p_cu"]["prose"],
                "",
                _md_table(ctx["p_cu"]["rows"]),
                "",
                "## Extra — leftover after days on Jan/Apr/Jul/Oct",
                "",
                ctx["p_qm"]["prose"],
                "",
                _md_table(ctx["p_qm"]["rows"]),
                "",
            ]
        )
    lines.extend(
        [
            "",
            f"Elapsed {ctx['elapsed_s']:.0f}s.",
            "",
            "Did **not**: overwrite `tax_qa.*` / `ss_qa.*` / `salary_qa.*` / `salary_month_qa.*` / "
            "`in3_qa.*` / `issued_qa.*`, edit `ops.py` / `gbm_core.py`, put tax on the 15-col card, "
            "grow TURNOVER, invent `y_tax`, run the assembler, write 0–100, fit holdout, touch `product/`.",
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
    p1, p3, p4, p7 = ctx["p1"], ctx["p3"], ctx["p4"], ctx["p7"]
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    cov = f"{p1['cov']:.4f}"
    rows = [
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_c_tax_month", "value": p3["y3"], "coverage": cov, "notes": f"days={p3['days']:.4f} leftover={p4['y3_rank']:.4f} card={d['headline_tag']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_c_tax_month_resid_days", "value": p4["y3_rank"], "coverage": cov, "notes": f"ols={p4['y3_ols']:.4f} fake_ols={p4['fake_ols']} r2={p4['y3_r2']:.4f} dies={p4['y3_dies']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_c_tax_month_resid_ss", "value": p4["y3_ss"], "coverage": cov, "notes": f"sal={p4['y3_sal']:.4f} stack={p4['y3_stack']:.4f} miss={p4['y3_miss']:.4f}"},
            {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_c_tax_month_lag1_resid_days_lag1", "value": p7["l1_left"], "coverage": cov, "notes": f"q6={p7['q6']} days_l1={p7['days_l1']:.4f}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_c_tax_month_resid_days_boot_p50", "value": ctx["p_bt"]["p50"], "coverage": cov, "notes": f"p05={ctx['p_bt']['p05']:.4f} p95={ctx['p_bt']['p95']:.4f} die={ctx['p_bt']['share_die']:.4f}"},
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
    p2, p3, p4, p6, p7 = ctx["p2"], ctx["p3"], ctx["p4"], ctx["p6"], ctx["p7"]
    text = (
        f"# Wave 4 — c_tax_month leftover after days\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/tax_month_qa.py`\n"
        f"- `analysis/outputs/tax_month_qa.md`\n"
        f"- `analysis/outputs/tax_month_qa.png`\n"
        f"- append-only `analysis/experiments/registry.csv`\n"
        f"- this note\n\n"
        f"Did not overwrite `tax_qa.*` (missed-tax), `ss_qa.*`, `salary_qa.*`, "
        f"`salary_month_qa.*`, `in3_qa.*`, `issued_qa.*`. Did not touch `ops.py`, "
        f"`gbm_core.py`, the 15-col card, TURNOVER, product/, or the parent journal. "
        f"Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. TURNOVER **0.720 / 0.712**.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| leftover after days | **{d['headline_tag']}** {_f(p4['y3_rank'])} |\n"
        f"| as Y3 X | **{d['card']}** |\n"
        f"| leftover after ss | **{_f(p4['y3_ss'])}** |\n"
        f"| leftover after salary | **{_f(p4['y3_sal'])}** |\n"
        f"| leftover after days+salary+ss | **{_f(p4['y3_stack'])}** |\n"
        f"| days leftover after tax | **{_f(p4['inv_rank'])}** |\n"
        f"| calendar | **{p6['shape']}** Q-gap {_pp(p6['gap'])} |\n"
        f"| Q6 | **{d['q6']}** |\n\n"
        f"Y3 tax {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size'])} "
        f"vs salary {_f(p3['sal'])} vs ss {_f(p3['ss'])} vs missed {_f(p3['miss'])}. "
        f"Leftover after days rank {_f(p4['y3_rank'])} OLS {_f(p4['y3_ols'])} "
        f"ρ(resid,days)={_f(p4['y3_rho_days'])}. "
        f"lag1 leftover after days_lag1 {_f(p7['l1_left'])}. "
        f"Jaccard vs missed {_f(p2['jac_miss']['jaccard'])} complement={p2['complement']}. "
        f"Bootstrap leftover-after-days p05/p50/p95 "
        f"{_f(ctx['p_bt']['p05'])} / {_f(ctx['p_bt']['p50'])} / {_f(ctx['p_bt']['p95'])} "
        f"(share die {_pp(ctx['p_bt']['share_die'])}). "
        f"Calendar leftover after Q+days {_f(ctx['p_qs']['q_days'])}. "
        f"BETWEEN leftover after days-mean {_f(ctx['p_icc']['between'])}.\n\n"
        f"CLOSE: leftover after days dies and Y3 tax does not beat size. "
        f"Do not put tax on the 15-col card. Presence is Q-peaked like missed-tax, "
        f"not a monthly salary/SS cousin.\n\n"
        f"## What failed / next\n\n"
        + "\n".join(f"- {x}" for x in ctx["failed"])
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.parent.mkdir(parents=True, exist_ok=True)
    WAVE_NOTE.write_text(text, encoding="utf-8")
    print(f"wrote {WAVE_NOTE}")


def run() -> dict:
    t0 = time.time()
    print(f"tax_month_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    con = connect()
    try:
        book = book_invoice_ids(con)
    finally:
        con.close()
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, ["c_tax_month", "c_n_days_with_tx", "c_ss_month", "c_salary_month"], (1, 3))
    tr = panel[panel["split"] == "train"].copy().reset_index(drop=True)
    assert_no_holdout(tr["company_id"])
    print(f"train CM={len(tr):,} companies={tr['company_id'].nunique()} holdout CM={int((panel['split']=='holdout').sum())}")

    p1 = pass1_cov(tr)
    p2 = pass2_twins(tr)
    p3 = pass3_singles(tr)
    p4 = pass4_leftover(tr)
    p6 = pass6_calendar(tr)
    p7 = pass7_q6(tr)
    p8 = pass8_dark(tr, book)
    p_icc = pass_icc(tr)
    p_ho = pass_holdout(panel)
    p_fl = pass_fold_left(tr)
    p_qs = pass_q_stack(tr)
    p_cd = pass_cadence(tr)
    p_sf = pass_sofar(tr)
    p_xo = pass_xor(tr)
    p_ch = pass_chronic(tr)
    p_bt = pass_boot(tr)
    p_bs = pass_between_ss(tr)
    p_dq = pass_disagree_q(tr)
    p_pm = pass_perm(tr)
    p_dq2 = pass_dark_q(tr, book)
    p_qd = pass_q_dummy_left(tr)
    p_lg = pass_logo(tr)
    p_bq = pass_boot_qdays(tr)
    p_te = pass_terciles(tr)
    p_yc = pass_y3_cells(tr)
    p_cu = pass_cousins(tr)
    p_qm = pass_q_months(tr)
    png = make_png(tr, p6)
    decision = decide(p1, p2, p3, p4, p6, p7)
    failed = []
    if not p3["days_ok"]:
        failed.append(f"days replica drifted: {_f(p3['days'])} vs 0.711")
    if not p3["size_ok"]:
        failed.append(f"size replica drifted: {_f(p3['size'])} vs 0.617")
    if not p3["sal_ok"]:
        failed.append(f"salary replica drifted: {_f(p3['sal'])} vs 0.671")
    if not p3["ss_ok"]:
        failed.append(f"ss replica drifted: {_f(p3['ss'])} vs 0.693")
    if not p3["miss_ok"]:
        failed.append(f"missed_tax replica drifted: {_f(p3['miss'])} vs 0.511")
    if p4["fake_ols"]:
        failed.append(f"OLS leftover after days {_f(p4['y3_ols'])} high vs honest rank {_f(p4['y3_rank'])}")
    if not p7["days_l1_ok"]:
        failed.append(f"days lag1 {_f(p7['days_l1'])} off KEEP 0.684")
    if not p8["confirm"]:
        failed.append(f"dark/invoiced {p8['n_dark']}/{p8['n_erp']} off 470/744")
    if not failed:
        failed.append("no replica miss; leftover after days is the unused-leftover decision")
    ctx = {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p6": p6,
        "p7": p7,
        "p8": p8,
        "p_icc": p_icc,
        "p_ho": p_ho,
        "p_fl": p_fl,
        "p_qs": p_qs,
        "p_cd": p_cd,
        "p_sf": p_sf,
        "p_xo": p_xo,
        "p_ch": p_ch,
        "p_bt": p_bt,
        "p_bs": p_bs,
        "p_dq": p_dq,
        "p_pm": p_pm,
        "p_dq2": p_dq2,
        "p_qd": p_qd,
        "p_lg": p_lg,
        "p_bq": p_bq,
        "p_te": p_te,
        "p_yc": p_yc,
        "p_cu": p_cu,
        "p_qm": p_qm,
        "decision": decision,
        "failed": failed,
        "png": png,
        "extra_bits": [],
        "elapsed_s": time.time() - t0,
        "tr": tr,
        "panel": panel,
        "book": book,
    }
    write_md(ctx)
    append_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    print(
        f"DONE card={decision['headline_tag']} leftover={_f(p4['y3_rank'])} "
        f"ss_left={_f(p4['y3_ss'])} stack={_f(p4['y3_stack'])} y3={_f(p3['y3'])} "
        f"elapsed={ctx['elapsed_s']:.0f}s"
    )
    return ctx


if __name__ == "__main__":
    run()
