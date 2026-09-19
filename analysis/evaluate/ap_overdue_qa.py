"""Unused leftover of ``e_ap_overdue`` after ``c_n_days_with_tx`` as Y3 X.

NORTH_STAR: ``e_ap_overdue`` = overdue AP |amount| / open (due < period end).
Feature report: 57.5% cov, acf1 0.51, ICC 0.95 BETWEEN, size_ρ −0.202.
``e_ap_overdue_30`` is a twin (|ρ| 0.85). Delay QA already CLOSE
``e_delay_paid`` / ``e_ar_overdue`` leftovers as Y7 and DROP from the 44
as Y3 X (Y3 0.512; leftover after days 0.427). ``e_ap_open`` CLOSE leftover
0.418. DPO DROP locked. TURNOVER 0.720 must not grow. Y5 never E.
Dark 470 stay NaN not 0. Do **not** overwrite delay_qa / dpo_qa / ap_open_qa.

KEEP-as-X: beat size ≥0.02 AND leftover after days AND not SIZE
(|ρ| vs log1p(a_in3) ≥0.50) AND not a twin (|ρ|≥0.80 vs days / a_n_tx /
e_ap_open / e_dpo_proxy / e_delay_paid). Leftover <0.55 dies.
Rank leftover is honest; OLS can fake a days leak.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.ap_overdue_qa

Owned: analysis/evaluate/ap_overdue_qa.py, analysis/outputs/ap_overdue_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_ap_overdue.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "ap_overdue_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "ap_overdue_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_ap_overdue.md"
AGENT = "e8b2c0d4"
WAVE = "4"
ROUND = "R4"
MODEL = "ap_overdue_qa"
X_FAM = "E"

Y3 = "y3_recover_cash_6m"
Y5 = "y5_ap_od30_ownp80"
Y7 = "y7_top1_lost"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
OPEN_Y3_QUOTE = 0.569
OPEN_LEFTOVER_QUOTE = 0.418
DELAY_Y3_QUOTE = 0.512
DELAY_LEFTOVER_QUOTE = 0.427
TURNOVER_QUOTE = 0.720
B_SHALLOW_QUOTE = 0.712
NIGHT_Y3 = 0.762
NIGHT_Y3_CORE = 0.752
COV_QUOTE = 0.575
ICC_QUOTE = 0.95
ACF1_QUOTE = 0.51
SIZE_RHO_QUOTE = -0.202
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
    "e_ap_overdue",
    "e_ap_overdue_30",
    "e_ap_open",
    "e_dpo_proxy",
    "e_delay_paid",
    "e_ar_overdue",
)

Y_KEEP = (Y3, Y5, Y7)
TWIN_GATES = (
    "c_n_days_with_tx",
    "a_n_tx",
    "e_ap_open",
    "e_dpo_proxy",
    "e_delay_paid",
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
    have_30 = "e_ap_overdue_30" in raw.columns
    cols = [c for c in STORE_COLS if c in raw.columns]
    missing = [c for c in STORE_COLS if c not in raw.columns]
    if missing and set(missing) != {"e_ap_overdue_30"}:
        hard = [c for c in missing if c != "e_ap_overdue_30"]
        if hard:
            raise RuntimeError(f"monthly.parquet missing {hard}")
    ymiss = [c for c in Y_KEEP if c not in yraw.columns]
    if ymiss:
        raise RuntimeError(f"targets.parquet missing {ymiss}")
    panel = _keys(raw[cols])
    y = _keys(yraw[["company_id", "period", *Y_KEEP]])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    od = pd.to_numeric(panel["e_ap_overdue"], errors="coerce")
    panel["first_month"] = pd.to_datetime(panel["first_month"])
    panel["months_so_far"] = (
        (panel["period"].dt.year - panel["first_month"].dt.year) * 12
        + (panel["period"].dt.month - panel["first_month"].dt.month)
        + 1
    )
    panel["so_far_class"] = panel["months_so_far"].map(_trail_class)
    panel["early6"] = panel["period"] < DELAY_MASK_BEFORE
    panel["has_od30"] = have_30
    if not have_30:
        panel["e_ap_overdue_30"] = np.nan
    panel = add_panel_lags(
        panel,
        ["e_ap_overdue", "e_ap_open", "e_delay_paid", "c_n_days_with_tx", "e_ap_overdue_30"],
        (1, 3),
    )
    leak7 = leakage_check(["e_ap_overdue", "e_ap_overdue_lag1"], Y7, forbidden_prefixes=["d"])
    leak3 = leakage_check(["e_ap_overdue"], Y3, forbidden_prefixes=["b"])
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
    od = pd.to_numeric(tr["e_ap_overdue"], errors="coerce")
    n_cm = int(len(tr))
    n_nn = int(od.notna().sum())
    dark_nn = int(od[dark].notna().sum())
    dark_zero = int((od[dark] == 0).sum())
    dark_pos = int((od[dark] > 0).sum())
    erp_nn = int(od[erp].notna().sum())
    erp_zero = int((od[erp] == 0).sum())
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    n_erp_co = int(tr.loc[erp, "company_id"].nunique())
    rho_size, n_size = spearman_n(od, tr["log_in3"])
    size_flag = bool(np.isfinite(rho_size) and abs(rho_size) >= SIZE_RHO)
    finite = od[od.notna()]
    p50 = float(finite.median()) if n_nn else float("nan")
    p99 = float(finite.quantile(0.99)) if n_nn else float("nan")
    mx = float(finite.max()) if n_nn else float("nan")
    eq0 = _pct(int((finite == 0).sum()), n_nn)
    acf1 = median_acf(od, tr["company_id"], 1)
    icc = icc_anova(od, tr["company_id"])
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
        f"Train e_ap_overdue nn={n_nn:,} cov={_pp(_pct(n_nn, n_cm))} "
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
    od = tr["e_ap_overdue"]
    pairs = [
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("a_n_tx", tr["a_n_tx"]),
        ("e_ap_open", tr["e_ap_open"]),
        ("e_dpo_proxy", tr["e_dpo_proxy"]),
        ("e_delay_paid", tr["e_delay_paid"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("e_ap_overdue_30", tr["e_ap_overdue_30"]),
        ("e_ar_overdue", tr["e_ar_overdue"]),
        ("e_ap_overdue_lag1", tr["e_ap_overdue_lag1"]),
    ]
    rows = []
    rhos = {}
    twins = []
    for name, s in pairs:
        rho, n = spearman_n(od, s)
        rhos[name] = rho
        twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
        if twin:
            twins.append(name)
        rows.append({"vs": name, "ρ": _f(rho), "n": f"{n:,}", "twin?": "TWIN" if twin else ""})
    size_flag = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    gate = [t for t in twins if t in TWIN_GATES]
    prose = (
        f"ap_overdue vs days ρ={_f(rhos['c_n_days_with_tx'])} "
        f"({'TWIN' if 'c_n_days_with_tx' in twins else 'not a twin'}). "
        f"vs a_n_tx {_f(rhos['a_n_tx'])} vs e_ap_open {_f(rhos['e_ap_open'])} "
        f"vs DPO {_f(rhos['e_dpo_proxy'])} vs delay_paid {_f(rhos['e_delay_paid'])} "
        f"vs log1p(a_in3) {_f(rhos['log1p(a_in3)'])} "
        f"{'SIZE' if size_flag else 'not SIZE'}. "
        f"vs overdue_30 {_f(rhos['e_ap_overdue_30'])} vs AR overdue {_f(rhos['e_ar_overdue'])} "
        f"vs own lag1 {_f(rhos['e_ap_overdue_lag1'])}. "
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
        "rho_open": rhos["e_ap_open"],
        "rho_dpo": rhos["e_dpo_proxy"],
        "rho_delay": rhos["e_delay_paid"],
        "rho_30": rhos["e_ap_overdue_30"],
        "rho_ar": rhos["e_ar_overdue"],
        "rho_lag1": rhos["e_ap_overdue_lag1"],
        "prose": prose,
    }


def pass3_singles(tr: pd.DataFrame) -> dict:
    feats = [
        ("e_ap_overdue", tr["e_ap_overdue"]),
        ("e_ap_overdue_lag1", tr["e_ap_overdue_lag1"]),
        ("e_ap_overdue_30", tr["e_ap_overdue_30"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("log1p_a_in3", tr["log_in3"]),
        ("e_ap_open", tr["e_ap_open"]),
        ("e_delay_paid", tr["e_delay_paid"]),
        ("e_dpo_proxy", tr["e_dpo_proxy"]),
        ("e_ar_overdue", tr["e_ar_overdue"]),
        ("a_n_tx", tr["a_n_tx"]),
    ]
    rows = []
    store = {}
    for yname in (Y3, Y7):
        for feat, s in feats:
            rec = signed_oof_auroc(tr[yname], s, tr["fold"], tr[yname].notna())
            print(f"{yname} {feat}: {_f(_cv(rec))} n={rec['n_defined']:,} pos={rec['n_pos']:,}")
            rows.append(_auc_row(yname, feat, rec))
            store[(yname, feat)] = rec
    y3 = _cv(store[(Y3, "e_ap_overdue")])
    days = _cv(store[(Y3, "c_n_days_with_tx")])
    size = _cv(store[(Y3, "log1p_a_in3")])
    open_auc = _cv(store[(Y3, "e_ap_open")])
    delay = _cv(store[(Y3, "e_delay_paid")])
    dpo = _cv(store[(Y3, "e_dpo_proxy")])
    od30 = _cv(store[(Y3, "e_ap_overdue_30")])
    beat_size = y3 - size if np.isfinite(y3) and np.isfinite(size) else float("nan")
    prose = (
        f"Y3 e_ap_overdue {_f(y3)} vs days {_f(days)} (CONFIRM {DAYS_BENCH:.3f}) "
        f"vs size {_f(size)} (CONFIRM {SIZE_QUOTE:.3f}) vs e_ap_open {_f(open_auc)} "
        f"(CONFIRM {OPEN_Y3_QUOTE:.3f}) vs delay_paid {_f(delay)} vs DPO {_f(dpo)} "
        f"vs overdue_30 {_f(od30)}. Beat size {_f(beat_size)}. "
        f"Y7 overdue {_f(_cv(store[(Y7, 'e_ap_overdue')]))}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3": y3,
        "days": days,
        "size": size,
        "open": open_auc,
        "delay": delay,
        "dpo": dpo,
        "od30": od30,
        "y7": _cv(store[(Y7, "e_ap_overdue")]),
        "lag1": _cv(store[(Y3, "e_ap_overdue_lag1")]),
        "beat_size": beat_size,
        "prose": prose,
    }


def pass4_leftover(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ap_overdue"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    inv = leftover_diag(
        tr[Y3], tr["c_n_days_with_tx"], (tr["e_ap_overdue"],), tr["fold"], tr[Y3].notna()
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
            "bar": "inverse: days leftover after overdue",
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
        f"Inverse: days leftover after overdue rank {_f(inv['rank'])} "
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


def pass5_after_open(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr["e_ap_overdue"], (tr["e_ap_open"],), tr["fold"], tr[Y3].notna()
    )
    both = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue"],
        (tr["e_ap_open"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    inv = leftover_diag(
        tr[Y3], tr["e_ap_open"], (tr["e_ap_overdue"],), tr["fold"], tr[Y3].notna()
    )
    rewrite = bool(d["honest_dies"])
    rows = [
        {
            "bar": "leftover after e_ap_open",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "bar": "leftover after open+days",
            "rank": _f(both["rank"]),
            "OLS": _f(both["ols"]),
            "ρ": _f(both["rho_ctrl"]),
            "fake?": "FALSE clone" if both["fake"] else "",
        },
        {
            "bar": "open leftover after overdue",
            "rank": _f(inv["rank"]),
            "OLS": _f(inv["ols"]),
            "ρ": _f(inv["rho_ctrl"]),
            "fake?": "FALSE clone" if inv["fake"] else "",
        },
    ]
    prose = (
        f"Y3 leftover after e_ap_open rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"ρ={_f(d['rho_ctrl'])} "
        f"{'REWRITE of open stock' if rewrite else 'not just the open stock'}. "
        f"After open+days {_f(both['rank'])} fake={both['fake']}. "
        f"Open leftover after overdue {_f(inv['rank'])} (open CLOSE leftover {OPEN_LEFTOVER_QUOTE:.3f})."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "ols": d["ols"],
        "both": both["rank"],
        "inv": inv["rank"],
        "rewrite": rewrite,
        "prose": prose,
    }


def pass6_after_dpo_delay(tr: pd.DataFrame) -> dict:
    dpo = leftover_diag(
        tr[Y3], tr["e_ap_overdue"], (tr["e_dpo_proxy"],), tr["fold"], tr[Y3].notna()
    )
    dpo_days = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue"],
        (tr["e_dpo_proxy"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    delay = leftover_diag(
        tr[Y3], tr["e_ap_overdue"], (tr["e_delay_paid"],), tr["fold"], tr[Y3].notna()
    )
    delay_days = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue"],
        (tr["e_delay_paid"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "leftover after DPO",
            "rank": _f(dpo["rank"]),
            "OLS": _f(dpo["ols"]),
            "ρ": _f(dpo["rho_ctrl"]),
            "fake?": "FALSE clone" if dpo["fake"] else "",
        },
        {
            "bar": "leftover after DPO+days",
            "rank": _f(dpo_days["rank"]),
            "OLS": _f(dpo_days["ols"]),
            "ρ": _f(dpo_days["rho_ctrl"]),
            "fake?": "FALSE clone" if dpo_days["fake"] else "",
        },
        {
            "bar": "leftover after delay_paid",
            "rank": _f(delay["rank"]),
            "OLS": _f(delay["ols"]),
            "ρ": _f(delay["rho_ctrl"]),
            "fake?": "FALSE clone" if delay["fake"] else "",
        },
        {
            "bar": "leftover after delay+days",
            "rank": _f(delay_days["rank"]),
            "OLS": _f(delay_days["ols"]),
            "ρ": _f(delay_days["rho_ctrl"]),
            "fake?": "FALSE clone" if delay_days["fake"] else "",
        },
    ]
    prose = (
        f"Y3 leftover after DPO rank {_f(dpo['rank'])}; after DPO+days {_f(dpo_days['rank'])}. "
        f"After delay_paid {_f(delay['rank'])}; after delay+days {_f(delay_days['rank'])}. "
        f"DPO DROP locked; delay leftover CLOSE (Y3 {DELAY_Y3_QUOTE:.3f} leftover {DELAY_LEFTOVER_QUOTE:.3f}). "
        "Do not overwrite delay_qa / dpo_qa."
    )
    print(prose)
    return {
        "rows": rows,
        "dpo": dpo["rank"],
        "dpo_days": dpo_days["rank"],
        "delay": delay["rank"],
        "delay_days": delay_days["rank"],
        "prose": prose,
    }


def pass_od30(tr: pd.DataFrame) -> dict:
    if tr["e_ap_overdue_30"].notna().sum() < 50:
        prose = "e_ap_overdue_30 missing — skip."
        print(prose)
        return {"rows": [], "rank": float("nan"), "inv": float("nan"), "both": float("nan"), "prose": prose}
    d = leftover_diag(
        tr[Y3], tr["e_ap_overdue"], (tr["e_ap_overdue_30"],), tr["fold"], tr[Y3].notna()
    )
    inv = leftover_diag(
        tr[Y3], tr["e_ap_overdue_30"], (tr["e_ap_overdue"],), tr["fold"], tr[Y3].notna()
    )
    both = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue"],
        (tr["e_ap_overdue_30"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    days30 = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue_30"],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {
            "bar": "overdue leftover after overdue_30",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "bar": "overdue_30 leftover after overdue",
            "rank": _f(inv["rank"]),
            "OLS": _f(inv["ols"]),
            "ρ": _f(inv["rho_ctrl"]),
            "fake?": "FALSE clone" if inv["fake"] else "",
        },
        {
            "bar": "overdue leftover after 30+days",
            "rank": _f(both["rank"]),
            "OLS": _f(both["ols"]),
            "ρ": _f(both["rho_ctrl"]),
            "fake?": "FALSE clone" if both["fake"] else "",
        },
        {
            "bar": "overdue_30 leftover after days",
            "rank": _f(days30["rank"]),
            "OLS": _f(days30["ols"]),
            "ρ": _f(days30["rho_ctrl"]),
            "fake?": "FALSE clone" if days30["fake"] else "",
        },
    ]
    prose = (
        f"vs overdue_30: leftover {_f(d['rank'])} (twin quote 0.85); "
        f"inverse {_f(inv['rank'])}; after 30+days {_f(both['rank'])}; "
        f"overdue_30 leftover-days {_f(days30['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "rank": d["rank"],
        "inv": inv["rank"],
        "both": both["rank"],
        "days30": days30["rank"],
        "prose": prose,
    }


def pass7_y5(tr: pd.DataFrame) -> dict:
    leak = leakage_check(["e_ap_overdue"], Y5, forbidden_prefixes=["e"])
    d = leftover_diag(
        tr[Y5], tr["e_ap_overdue"], (tr["log_in3"],), tr["fold"], tr[Y5].notna()
    )
    raw = signed_oof_auroc(tr[Y5], tr["e_ap_overdue"], tr["fold"], tr[Y5].notna())
    rows = [
        {
            "bar": "Y5 leftover after size (report only)",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "raw": _f(_cv(raw)),
            "leak_ok": str(leak["ok"]),
        }
    ]
    prose = (
        f"Y5 leftover after size rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"raw {_f(_cv(raw))}. Y5 never E — no AUROC as card X, leak_ok={leak['ok']}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "raw": _cv(raw), "leak_ok": leak["ok"], "prose": prose}


def pass8_q6(tr: pd.DataFrame) -> dict:
    short = tr["so_far_class"] == "short_<12"
    raw = signed_oof_auroc(
        tr[Y7], tr["e_ap_overdue_lag1"], tr["fold"], short & tr[Y7].notna()
    )
    d = leftover_diag(
        tr[Y7],
        tr["e_ap_overdue_lag1"],
        (tr["c_n_days_with_tx_lag1"],),
        tr["fold"],
        short & tr[Y7].notna(),
    )
    now = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue"],
        (tr["c_n_days_with_tx_lag1"],),
        tr["fold"],
        short & tr[Y3].notna(),
    )
    y3lag = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue_lag1"],
        (tr["c_n_days_with_tx_lag1"],),
        tr["fold"],
        short & tr[Y3].notna(),
    )
    early = tr["early6"]
    delay_early = int(pd.to_numeric(tr.loc[early, "e_delay_paid"], errors="coerce").notna().sum())
    rows = [
        {
            "slice": "Q6 Y7 overdue_lag1 short",
            "CV": "LOW_POWER" if raw["low_power"] else _f(_cv(raw)),
            "n": f"{raw['n_defined']:,}",
            "n_pos": f"{raw['n_pos']:,}",
        },
        {
            "slice": "Q6 Y7 lag1 leftover after days_lag1 short",
            "rank": _f(d["rank"]),
            "OLS": _f(d["ols"]),
            "ρ": _f(d["rho_ctrl"]),
            "fake?": "FALSE clone" if d["fake"] else "",
        },
        {
            "slice": "Q6 Y3 now leftover after days_lag1 short",
            "rank": _f(now["rank"]),
            "OLS": _f(now["ols"]),
            "ρ": _f(now["rho_ctrl"]),
            "fake?": "FALSE clone" if now["fake"] else "",
        },
        {
            "slice": "Q6 Y3 lag1 leftover after days_lag1 short",
            "rank": _f(y3lag["rank"]),
            "OLS": _f(y3lag["ols"]),
            "ρ": _f(y3lag["rho_ctrl"]),
            "fake?": "FALSE clone" if y3lag["fake"] else "",
        },
    ]
    prose = (
        f"Q6 Y7 overdue_lag1 on short {_f(_cv(raw))}. "
        f"lag1 leftover after days_lag1 short rank {_f(d['rank'])} fake={d['fake']}. "
        f"Y3 now leftover after days_lag1 short {_f(now['rank'])}. "
        f"Y3 lag1 leftover {_f(y3lag['rank'])}. "
        f"Delay first-6 nn={delay_early} (delay Q6 CLOSE empty until month 7). "
        "Do not claim a TURNOVER seat."
    )
    print(prose)
    return {
        "rows": rows,
        "y7": _cv(raw),
        "lag_rank": d["rank"],
        "now_rank": now["rank"],
        "y3_lag": y3lag["rank"],
        "delay_early": delay_early,
        "prose": prose,
    }


def pass10_boot(tr: pd.DataFrame) -> dict:
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
            sub["e_ap_overdue"],
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
    od = pd.to_numeric(ho["e_ap_overdue"], errors="coerce")
    dark = ~ho["company_id"].isin(book)
    rows = [
        {
            "slice": "holdout 72",
            "n_cm": f"{len(ho):,}",
            "companies": f"{ho['company_id'].nunique()}",
            "cov": _pp(_pct(int(od.notna().sum()), len(ho))),
            "dark_nn": int(od[dark].notna().sum()),
        }
    ]
    prose = (
        f"Holdout 72 coverage only: {len(ho):,} CM / {ho['company_id'].nunique()} companies. "
        f"overdue cov {_pp(_pct(int(od.notna().sum()), len(ho)))}; dark nn={int(od[dark].notna().sum())}. "
        "No AUROC."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_terciles(tr: pd.DataFrame) -> dict:
    terc = size_terciles(tr)
    rows = []
    store = {}
    for lab, mask in (
        ("T1", terc == "T1"),
        ("T2+T3", terc.isin(["T2", "T3"])),
        ("T3", terc == "T3"),
    ):
        d = leftover_diag(
            tr[Y3],
            tr["e_ap_overdue"],
            (tr["c_n_days_with_tx"],),
            tr["fold"],
            mask & tr[Y3].notna(),
        )
        store[lab] = d["rank"]
        rows.append(
            {
                "slice": lab,
                "rank": _f(d["rank"]),
                "OLS": _f(d["ols"]),
                "n": f"{d['n']:,}",
            }
        )
    prose = (
        f"SIZE terciles leftover after days: T1 {_f(store['T1'])} "
        f"T2+T3 {_f(store['T2+T3'])} T3 {_f(store['T3'])}."
    )
    print(prose)
    return {"rows": rows, "t1": store["T1"], "t23": store["T2+T3"], "prose": prose}


def pass_company_med(tr: pd.DataFrame) -> dict:
    med = tr.groupby("company_id").agg(
        od=("e_ap_overdue", "median"),
        days=("c_n_days_with_tx", "median"),
        size=("log_in3", "median"),
        op=("e_ap_open", "median"),
        delay=("e_delay_paid", "median"),
    )
    rho_d, n_d = spearman_n(med["od"], med["days"])
    rho_s, n_s = spearman_n(med["od"], med["size"])
    rho_o, n_o = spearman_n(med["od"], med["op"])
    rho_p, n_p = spearman_n(med["od"], med["delay"])
    size_flag = bool(np.isfinite(rho_s) and abs(rho_s) >= SIZE_RHO)
    rows = [
        {"vs": "days", "ρ": _f(rho_d), "n": f"{n_d:,}"},
        {"vs": "log1p(a_in3)", "ρ": _f(rho_s), "n": f"{n_s:,}", "SIZE?": "SIZE" if size_flag else ""},
        {"vs": "e_ap_open", "ρ": _f(rho_o), "n": f"{n_o:,}"},
        {"vs": "e_delay_paid", "ρ": _f(rho_p), "n": f"{n_p:,}"},
    ]
    prose = (
        f"Company-median ρ overdue vs days {_f(rho_d)} vs log1p(a_in3) {_f(rho_s)} "
        f"{'SIZE' if size_flag else 'not SIZE'} vs open {_f(rho_o)} vs delay {_f(rho_p)}."
    )
    print(prose)
    return {"rows": rows, "rho_size": rho_s, "size_flag": size_flag, "prose": prose}


def pass_so_far(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for lab in ("short_<12", "mid_12_17", "long_>=18"):
        m = tr["so_far_class"] == lab
        d = leftover_diag(
            tr[Y3],
            tr["e_ap_overdue"],
            (tr["c_n_days_with_tx"],),
            tr["fold"],
            m & tr[Y3].notna(),
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
            tr[Y3],
            tr["e_ap_overdue"],
            (tr["c_n_days_with_tx"],),
            tr["fold"],
            m & tr[Y3].notna(),
        )
        bits.append(_f(d["rank"]))
        rows.append({"fold": k, "rank": _f(d["rank"]), "OLS": _f(d["ols"]), "n": f"{d['n']:,}"})
    prose = f"Per-fold rank leftover after days: {' '.join(bits)}."
    print(prose)
    return {"rows": rows, "bits": bits, "prose": prose}


def pass_q6_midlong(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for lab in ("mid_12_17", "long_>=18"):
        m = tr["so_far_class"] == lab
        raw = signed_oof_auroc(tr[Y7], tr["e_ap_overdue_lag1"], tr["fold"], m & tr[Y7].notna())
        d = leftover_diag(
            tr[Y7],
            tr["e_ap_overdue_lag1"],
            (tr["c_n_days_with_tx_lag1"],),
            tr["fold"],
            m & tr[Y7].notna(),
        )
        store[lab] = (_cv(raw), d["rank"])
        rows.append(
            {
                "slice": f"Q6 Y7 {lab}",
                "raw": _f(_cv(raw)),
                "leftover": _f(d["rank"]),
                "n": f"{raw['n_defined']:,}",
            }
        )
    prose = (
        f"Q6 mid Y7 overdue_lag1 {_f(store['mid_12_17'][0])} leftover {_f(store['mid_12_17'][1])}; "
        f"long Y7 {_f(store['long_>=18'][0])}. Delay Q6 CLOSE. Do not claim a TURNOVER seat."
    )
    print(prose)
    return {
        "rows": rows,
        "mid": store["mid_12_17"][1],
        "long": store["long_>=18"][0],
        "prose": prose,
    }


def pass_keep_card(p1, p2, p3, p4) -> dict:
    beat = bool(np.isfinite(p3["beat_size"]) and p3["beat_size"] >= KEEP_DELTA)
    leftover_ok = not bool(p4["y3_dies"])
    size_ok = not bool(p1["size_flag"] or p2["size_flag"])
    twin_ok = not bool(p2["gate_twins"])
    rows = [
        {"gate": "beat size ≥0.02", "ok": "PASS" if beat else "FAIL", "value": _f(p3["beat_size"])},
        {
            "gate": "leftover after days ≥0.55 and not fake",
            "ok": "PASS" if leftover_ok else "FAIL",
            "value": _f(p4["y3_rank"]),
        },
        {"gate": "not SIZE |ρ|≥0.50 vs log1p(a_in3)", "ok": "PASS" if size_ok else "FAIL", "value": _f(p2["rho_size"])},
        {
            "gate": "not twin vs days / n_tx / open / DPO / delay_paid",
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
        "Do not put e_ap_overdue on the 15-col card."
    )
    print(prose)
    return {"rows": rows, "beat": beat, "leftover_ok": leftover_ok, "prose": prose}


def pass_erp_late(tr: pd.DataFrame, book: set[str]) -> dict:
    erp = tr["company_id"].isin(book)
    late = ~tr["early6"]
    d = leftover_diag(
        tr[Y3], tr["e_ap_overdue"], (tr["c_n_days_with_tx"],), tr["fold"], erp & tr[Y3].notna()
    )
    late_d = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue"],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        late & tr[Y3].notna(),
    )
    rows = [
        {"slice": "ever-ERP leftover-days", "rank": _f(d["rank"]), "OLS": _f(d["ols"])},
        {"slice": "late (month≥7) leftover-days", "rank": _f(late_d["rank"]), "OLS": _f(late_d["ols"])},
    ]
    prose = (
        f"ever-ERP leftover-days {_f(d['rank'])} fake={d['fake']}; "
        f"late {_f(late_d['rank'])}."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "late": late_d["rank"], "prose": prose}


def pass_between(tr: pd.DataFrame) -> dict:
    od = pd.to_numeric(tr["e_ap_overdue"], errors="coerce")
    cmean = od.groupby(tr["company_id"]).transform("mean")
    within = od - cmean
    after_mean = leftover_diag(tr[Y3], tr["e_ap_overdue"], (cmean,), tr["fold"], tr[Y3].notna())
    after_lag1 = leftover_diag(
        tr[Y3], tr["e_ap_overdue"], (tr["e_ap_overdue_lag1"],), tr["fold"], tr[Y3].notna()
    )
    within_d = leftover_diag(
        tr[Y3], within, (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    rows = [
        {"bar": "leftover after company-mean", "rank": _f(after_mean["rank"]), "OLS": _f(after_mean["ols"])},
        {"bar": "leftover after own lag1", "rank": _f(after_lag1["rank"]), "OLS": _f(after_lag1["ols"])},
        {"bar": "within leftover-days", "rank": _f(within_d["rank"]), "OLS": _f(within_d["ols"])},
    ]
    prose = (
        f"BETWEEN: leftover after company-mean {_f(after_mean['rank'])}; "
        f"after own lag1 {_f(after_lag1['rank'])}; within leftover-days {_f(within_d['rank'])}. "
        "ICC 0.95 BETWEEN."
    )
    print(prose)
    return {
        "rows": rows,
        "mean": after_mean["rank"],
        "lag1": after_lag1["rank"],
        "within": within_d["rank"],
        "prose": prose,
    }


def pass_days_size_card(tr: pd.DataFrame) -> dict:
    ds = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue"],
        (tr["c_n_days_with_tx"], tr["log_in3"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    raw = pd.read_parquet(STORE, columns=["company_id", "period", "c_ss_month", "c_salary_month"])
    raw = _keys(raw)
    work = tr.merge(raw, on=["company_id", "period"], how="left")
    card = leftover_diag(
        work[Y3],
        work["e_ap_overdue"],
        (work["c_ss_month"], work["c_salary_month"], work["c_n_days_with_tx"]),
        work["fold"],
        work[Y3].notna(),
    )
    ar = leftover_diag(
        tr[Y3], tr["e_ap_overdue"], (tr["e_ar_overdue"],), tr["fold"], tr[Y3].notna()
    )
    ar_days = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue"],
        (tr["e_ar_overdue"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    triple = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue"],
        (tr["e_ap_open"], tr["e_dpo_proxy"], tr["e_delay_paid"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {"bar": "leftover after days+size", "rank": _f(ds["rank"]), "OLS": _f(ds["ols"]), "ρ": _f(ds["rho_ctrl"])},
        {"bar": "leftover after ss+salary+days", "rank": _f(card["rank"]), "OLS": _f(card["ols"])},
        {"bar": "leftover after e_ar_overdue", "rank": _f(ar["rank"]), "OLS": _f(ar["ols"]), "ρ": _f(ar["rho_ctrl"])},
        {"bar": "leftover after AR overdue+days", "rank": _f(ar_days["rank"]), "OLS": _f(ar_days["ols"])},
        {"bar": "leftover after open+DPO+delay", "rank": _f(triple["rank"]), "OLS": _f(triple["ols"])},
    ]
    prose = (
        f"leftover after days+size {_f(ds['rank'])}; ss+salary+days {_f(card['rank'])}; "
        f"after AR overdue {_f(ar['rank'])} +days {_f(ar_days['rank'])}; "
        f"open+DPO+delay {_f(triple['rank'])}. Stay off the 15-col card."
    )
    print(prose)
    return {
        "rows": rows,
        "ds": ds["rank"],
        "card": card["rank"],
        "ar": ar["rank"],
        "ar_days": ar_days["rank"],
        "triple": triple["rank"],
        "prose": prose,
    }


def pass_pos_winsor(tr: pd.DataFrame) -> dict:
    od = pd.to_numeric(tr["e_ap_overdue"], errors="coerce")
    pos = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue"],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        (od > 0) & tr[Y3].notna(),
    )
    hi = float(od.quantile(0.99)) if od.notna().any() else 1.0
    w = od.clip(upper=hi)
    win = leftover_diag(tr[Y3], w, (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna())
    both = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue"],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        od.notna() & tr["e_ap_open"].notna() & tr[Y3].notna(),
    )
    rows = [
        {"bar": "overdue>0 leftover-days", "rank": _f(pos["rank"]), "OLS": _f(pos["ols"]), "n": f"{pos['n']:,}"},
        {"bar": "winsor99 leftover-days", "rank": _f(win["rank"]), "OLS": _f(win["ols"])},
        {"bar": "open-defined leftover-days", "rank": _f(both["rank"]), "OLS": _f(both["ols"])},
    ]
    prose = (
        f"overdue>0 leftover-days {_f(pos['rank'])}; winsor {_f(win['rank'])}; "
        f"open-defined {_f(both['rank'])}."
    )
    print(prose)
    return {"rows": rows, "pos": pos["rank"], "win": win["rank"], "both": both["rank"], "prose": prose}


def pass_samen(tr: pd.DataFrame) -> dict:
    """Days / size / overdue on the overdue-defined Y3 rows — leftover n is smaller."""
    od = pd.to_numeric(tr["e_ap_overdue"], errors="coerce")
    m = od.notna() & tr[Y3].notna()
    days_s = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], m)
    size_s = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], m)
    od_s = signed_oof_auroc(tr[Y3], tr["e_ap_overdue"], tr["fold"], m)
    ntx = leftover_diag(
        tr[Y3], tr["e_ap_overdue"], (tr["a_n_tx"],), tr["fold"], tr[Y3].notna()
    )
    ntx_days = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue"],
        (tr["a_n_tx"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    trail = leftover_diag(
        tr[Y3], tr["e_ap_overdue"], (tr["months_so_far"],), tr["fold"], tr[Y3].notna()
    )
    trail_days = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue"],
        (tr["months_so_far"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {"bar": "Y3 overdue same-n", "CV": _f(_cv(od_s)), "n": f"{od_s['n_defined']:,}", "n_pos": f"{od_s['n_pos']:,}"},
        {"bar": "Y3 days same-n", "CV": _f(_cv(days_s)), "n": f"{days_s['n_defined']:,}", "n_pos": f"{days_s['n_pos']:,}"},
        {"bar": "Y3 size same-n", "CV": _f(_cv(size_s)), "n": f"{size_s['n_defined']:,}", "n_pos": f"{size_s['n_pos']:,}"},
        {"bar": "leftover after a_n_tx", "rank": _f(ntx["rank"]), "OLS": _f(ntx["ols"])},
        {"bar": "leftover after n_tx+days", "rank": _f(ntx_days["rank"]), "OLS": _f(ntx_days["ols"])},
        {"bar": "leftover after months_so_far", "rank": _f(trail["rank"]), "OLS": _f(trail["ols"])},
        {"bar": "leftover after trail+days", "rank": _f(trail_days["rank"]), "OLS": _f(trail_days["ols"])},
    ]
    beat_same = (
        _cv(od_s) - _cv(size_s)
        if np.isfinite(_cv(od_s)) and np.isfinite(_cv(size_s))
        else float("nan")
    )
    prose = (
        f"same-n Y3 overdue {_f(_cv(od_s))} days {_f(_cv(days_s))} size {_f(_cv(size_s))} "
        f"beat-size {_f(beat_same)}; leftover after n_tx {_f(ntx['rank'])} +days {_f(ntx_days['rank'])}; "
        f"trail {_f(trail['rank'])} +days {_f(trail_days['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "od": _cv(od_s),
        "days": _cv(days_s),
        "size": _cv(size_s),
        "beat": beat_same,
        "ntx": ntx["rank"],
        "trail_days": trail_days["rank"],
        "prose": prose,
    }


def pass_delay_lag_dpo_def(tr: pd.DataFrame) -> dict:
    dlag = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue"],
        (tr["e_delay_paid_lag1"],),
        tr["fold"],
        tr[Y3].notna(),
    )
    dlag_days = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue"],
        (tr["e_delay_paid_lag1"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    dpo_ok = pd.to_numeric(tr["e_dpo_proxy"], errors="coerce").notna()
    dpo_def = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue"],
        (tr["c_n_days_with_tx"],),
        tr["fold"],
        dpo_ok & tr[Y3].notna(),
    )
    dpo_both = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue"],
        (tr["e_dpo_proxy"], tr["c_n_days_with_tx"]),
        tr["fold"],
        dpo_ok & tr[Y3].notna(),
    )
    rec = leftover_diag(
        tr[Y3], tr["e_ap_overdue"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    sign = rec["rrec"].get("train_sign", 0)
    rows = [
        {"bar": "leftover after delay_lag1", "rank": _f(dlag["rank"]), "OLS": _f(dlag["ols"])},
        {"bar": "leftover after delay_lag1+days", "rank": _f(dlag_days["rank"]), "OLS": _f(dlag_days["ols"])},
        {"bar": "DPO-defined leftover-days", "rank": _f(dpo_def["rank"]), "OLS": _f(dpo_def["ols"])},
        {"bar": "DPO-defined leftover DPO+days", "rank": _f(dpo_both["rank"]), "OLS": _f(dpo_both["ols"])},
        {"bar": "leftover rank sign", "sign": str(sign), "rank": _f(rec["rank"])},
    ]
    prose = (
        f"leftover after delay_lag1 {_f(dlag['rank'])} +days {_f(dlag_days['rank'])}; "
        f"DPO-defined leftover-days {_f(dpo_def['rank'])} DPO+days {_f(dpo_both['rank'])}; "
        f"rank leftover sign={sign}. Delay Q6 CLOSE — do not grow TURNOVER."
    )
    print(prose)
    return {
        "rows": rows,
        "dlag": dlag["rank"],
        "dlag_days": dlag_days["rank"],
        "dpo_def": dpo_def["rank"],
        "dpo_both": dpo_both["rank"],
        "sign": sign,
        "prose": prose,
    }


def pass_open_lag_joint(tr: pd.DataFrame) -> dict:
    ol = leftover_diag(
        tr[Y3], tr["e_ap_overdue"], (tr["e_ap_open_lag1"],), tr["fold"], tr[Y3].notna()
    )
    ol_days = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue"],
        (tr["e_ap_open_lag1"], tr["c_n_days_with_tx"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    joint = leftover_diag(
        tr[Y3],
        tr["e_ap_overdue"],
        (tr["c_n_days_with_tx"], tr["a_n_tx"], tr["log_in3"]),
        tr["fold"],
        tr[Y3].notna(),
    )
    rows = [
        {"bar": "leftover after open_lag1", "rank": _f(ol["rank"]), "OLS": _f(ol["ols"]), "ρ": _f(ol["rho_ctrl"])},
        {"bar": "leftover after open_lag1+days", "rank": _f(ol_days["rank"]), "OLS": _f(ol_days["ols"])},
        {"bar": "leftover after days+n_tx+size", "rank": _f(joint["rank"]), "OLS": _f(joint["ols"])},
    ]
    prose = (
        f"leftover after open_lag1 {_f(ol['rank'])} +days {_f(ol_days['rank'])}; "
        f"days+n_tx+size {_f(joint['rank'])}. Stay off the 15-col card."
    )
    print(prose)
    return {
        "rows": rows,
        "ol": ol["rank"],
        "ol_days": ol_days["rank"],
        "joint": joint["rank"],
        "prose": prose,
    }


def pass_y7_leftover(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y7], tr["e_ap_overdue"], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y7].notna()
    )
    after_open = leftover_diag(
        tr[Y7], tr["e_ap_overdue"], (tr["e_ap_open"],), tr["fold"], tr[Y7].notna()
    )
    rows = [
        {"bar": "Y7 leftover after days", "rank": _f(d["rank"]), "OLS": _f(d["ols"]), "ρ": _f(d["rho_ctrl"])},
        {"bar": "Y7 leftover after open", "rank": _f(after_open["rank"]), "OLS": _f(after_open["ols"])},
    ]
    prose = (
        f"Y7 leftover-days {_f(d['rank'])}; leftover-open {_f(after_open['rank'])}. "
        "Do not grow TURNOVER 0.720."
    )
    print(prose)
    return {"rows": rows, "rank": d["rank"], "open": after_open["rank"], "prose": prose}


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
            "Do not put e_ap_overdue on the 15-col card."
        )
    elif leftover_dies and y3_loses:
        card = "DROP from the 44 as Y3 X / CLOSE unused leftover"
        tag = "DROP"
        why = (
            f"Y3 {_f(p3['y3'])} loses to days {_f(p3['days'])}; leftover after days "
            f"rank {_f(leftover)} OLS {_f(p4['y3_ols'])} "
            f"{'FALSE clone' if p4['y3_fake'] else 'dies'}. "
            f"Inverse days-after-overdue {_f(p4['inv_rank'])}. "
            "Confirms delay_qa DROP of delay/overdue as Y3 X. Stay off the 15-col card. "
            "Do not grow TURNOVER."
        )
    elif leftover_dies or twin:
        card = "CLOSE unused leftover / KEEP off the 15-col card"
        tag = "CLOSE"
        why = (
            f"Y3 leftover after days rank {_f(leftover)} OLS {_f(p4['y3_ols'])} "
            f"{'dies' if leftover_dies else 'thin'}; twin={twin} SIZE={size}. "
            f"Inverse days-after-overdue {_f(p4['inv_rank'])}. "
            "Stay off the 15-col card. Do not grow TURNOVER."
        )
    elif size:
        card = "CLOSE as SIZE / KEEP off the 15-col card"
        tag = "CLOSE"
        why = f"SIZE ρ={_f(p2['rho_size'])}; leftover after days {_f(leftover)}. Stay off the card."
    else:
        card = "CLOSE unused leftover / KEEP off the 15-col card"
        tag = "CLOSE"
        why = (
            f"Y3 {_f(p3['y3'])} vs days {_f(p3['days'])} leftover {_f(leftover)}. "
            "Does not clear KEEP-as-X. Stay off the card."
        )
    if p5.get("rewrite"):
        why += " Overdue leftover after open dies — rewrite of open stock."
    return {"card": card, "headline_tag": tag, "why": why, "keep_x": keep_x}


def make_png(tr: pd.DataFrame, p4: dict) -> str | None:
    if not HAS_MPL:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.4))
    od = pd.to_numeric(tr["e_ap_overdue"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    m = od.notna() & days.notna()
    ax[0].scatter(days[m], od[m], s=4, alpha=0.15, c="#2c5f6e")
    ax[0].set_xlabel("c_n_days_with_tx")
    ax[0].set_ylabel("e_ap_overdue")
    ax[0].set_title("AP overdue share vs days")
    labels = ["overdue after days", "days after overdue"]
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
        "# Unused leftover of `e_ap_overdue` after `c_n_days_with_tx`",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        f"`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        f"Holdout 72 is coverage only. Seed {FOLD_SEED} group folds. No 0–100. "
        f"No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_ap_overdue`. "
        f"Night Y7 stays **TURNOVER {TURNOVER_QUOTE:.3f} / B_shallow {B_SHALLOW_QUOTE:.3f}**. "
        f"Do **not** grow TURNOVER. Do **not** put e_ap_overdue on the 15-col Y3 card. "
        f"Night Y3 stays **{NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f}**. Days **{DAYS_BENCH:.3f}**. "
        f"Size **{SIZE_QUOTE:.3f}**. Y7 never D. Y5 never E. Y3 never B. Dark 470 stay NaN not 0. "
        f"AP open leftover {OPEN_LEFTOVER_QUOTE:.3f} CLOSED — do not overwrite ap_open_qa. "
        f"Delay leftover {DELAY_LEFTOVER_QUOTE:.3f} CLOSE / DROP as Y3 X — do not overwrite delay_qa. "
        "DPO already DROP — do not overwrite dpo_qa.",
        "",
        "`e_ap_overdue` = overdue AP |amount| / open (due < period end). "
        "Feature report: 57.5% cov, acf1 0.51, ICC 0.95 BETWEEN. "
        "`e_ap_overdue_30` twin |ρ| 0.85.",
        "",
        "## Headline",
        "",
        f"{d['headline_tag']} leftover-after-days rank {_f(ctx['p4']['y3_rank'])} "
        f"(OLS {_f(ctx['p4']['y3_ols'])}, fake={ctx['p4']['y3_fake']}). "
        f"Y3 overdue {_f(ctx['p3']['y3'])} vs days {_f(ctx['p3']['days'])} vs size {_f(ctx['p3']['size'])} "
        f"vs open {_f(ctx['p3']['open'])} vs delay_paid {_f(ctx['p3']['delay'])} vs DPO {_f(ctx['p3']['dpo'])}. "
        f"SIZE={ctx['p2']['size_flag']} twin={bool(ctx['p2']['gate_twins'])}. "
        f"Inverse days-after-overdue {_f(ctx['p4']['inv_rank'])}. "
        f"Leftover after open {_f(ctx['p5']['rank'])}. "
        f"Leftover after DPO {_f(ctx['p6']['dpo'])} / delay {_f(ctx['p6']['delay'])}. "
        f"Card: {d['card']}. "
        f"Night Y3 {NIGHT_Y3:.3f}/{NIGHT_Y3_CORE:.3f}, days {DAYS_BENCH:.3f}, "
        f"size {SIZE_QUOTE:.3f}, TURNOVER {TURNOVER_QUOTE:.3f}/{B_SHALLOW_QUOTE:.3f} unchanged.",
        "",
        "## PARK / CLOSE / KEEP / DROP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| overdue leftover after days (Y3 X) | **{d['headline_tag']}** | {d['why']} |",
        "| 15-col Y3 card stem | **KEEP off the card** | do not put e_ap_overdue on the card |",
        "| TURNOVER add-on | **CLOSE** | do not grow 0.720 |",
        f"| AP open leftover | **CLOSE (locked)** | rank {OPEN_LEFTOVER_QUOTE:.3f} |",
        f"| delay / overdue as Y3 X | **DROP (locked delay_qa)** | Y3 {DELAY_Y3_QUOTE:.3f} leftover {DELAY_LEFTOVER_QUOTE:.3f} |",
        "| DPO | **DROP (locked)** | do not reopen dpo_qa |",
        "| health Y `y_ap_overdue` | **PARK** | do not invent y_ap_overdue |",
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
        "## 5. Leftover after e_ap_open / DPO / delay_paid",
        "",
        ctx["p5"]["prose"],
        "",
        _md_table(ctx["p5"]["rows"]),
        "",
        ctx["p6"]["prose"],
        "",
        _md_table(ctx["p6"]["rows"]),
        "",
        "## 6. vs e_ap_overdue_30",
        "",
        ctx["p30"]["prose"],
        "",
        _md_table(ctx["p30"]["rows"]),
        "",
        "## 7. Dark 470",
        "",
        f"Dark never-ERP {ctx['p1']['n_dark_co']} nn={ctx['p1']['dark_nn']} "
        f"zero={ctx['p1']['dark_zero']} pos={ctx['p1']['dark_pos']}. "
        f"{'CONFIRM NaN not 0' if ctx['p1']['dark_ok'] else 'BOOK stub or check'}.",
        "",
        "## 8. Q6 lag1 leftover after days_lag1",
        "",
        ctx["p8"]["prose"],
        "",
        _md_table(ctx["p8"]["rows"]),
        "",
        "## 9. Y5 leftover after size (report only)",
        "",
        ctx["p7"]["prose"],
        "",
        _md_table(ctx["p7"]["rows"]),
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
        "## Extra — Q6 mid / long",
        "",
        ctx["pq6"]["prose"],
        "",
        _md_table(ctx["pq6"]["rows"]),
        "",
        "## Extra — KEEP-as-X scorecard",
        "",
        ctx["pk"]["prose"],
        "",
        _md_table(ctx["pk"]["rows"]),
        "",
        "## Extra — ever-ERP / late leftover",
        "",
        ctx["perp"]["prose"],
        "",
        _md_table(ctx["perp"]["rows"]),
        "",
        "## Extra — BETWEEN identity",
        "",
        ctx["pbl"]["prose"],
        "",
        _md_table(ctx["pbl"]["rows"]),
        "",
        "## Extra — Y7 leftover (do not grow TURNOVER)",
        "",
        ctx["py7"]["prose"],
        "",
        _md_table(ctx["py7"]["rows"]),
        "",
        "## Extra — leftover after days+size / card KEEP / AR overdue / open+DPO+delay",
        "",
        ctx["pds"]["prose"],
        "",
        _md_table(ctx["pds"]["rows"]),
        "",
        "## Extra — overdue>0 / winsor / open-defined",
        "",
        ctx["ppw"]["prose"],
        "",
        _md_table(ctx["ppw"]["rows"]),
        "",
        "## Extra — same-n days/size / leftover after n_tx / trail",
        "",
        ctx["psn"]["prose"],
        "",
        _md_table(ctx["psn"]["rows"]),
        "",
        "## Extra — leftover after delay_lag1 / DPO-defined",
        "",
        ctx["pdl"]["prose"],
        "",
        _md_table(ctx["pdl"]["rows"]),
        "",
        "## Extra — leftover after open_lag1 / days+n_tx+size",
        "",
        ctx["poj"]["prose"],
        "",
        _md_table(ctx["poj"]["rows"]),
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
    already = existing.count("ap_overdue_qa")
    if already >= 5:
        print("registry: no new rows")
        return
    ts = _now_iso()
    want = [
        ("auroc_e_ap_overdue", ctx["p3"]["y3"], Y3, f"days={ctx['p3']['days']:.4f} size={ctx['p3']['size']:.4f}"),
        (
            "auroc_od_resid_days_rank",
            ctx["p4"]["y3_rank"],
            Y3,
            f"ols={ctx['p4']['y3_ols']:.4f} fake={ctx['p4']['y3_fake']}",
        ),
        (
            "auroc_od_resid_open_rank",
            ctx["p5"]["rank"],
            Y3,
            f"ols={ctx['p5']['ols']:.4f} rewrite={ctx['p5']['rewrite']}",
        ),
        (
            "auroc_od_resid_dpo_rank",
            ctx["p6"]["dpo"],
            Y3,
            f"delay={ctx['p6']['delay']:.4f}",
        ),
        (
            "rho_od_vs_days",
            ctx["p2"]["rho_days"],
            Y3,
            f"rho_open={ctx['p2']['rho_open']:.4f} rho_size={ctx['p2']['rho_size']:.4f}",
        ),
    ]
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
        "# Wave 4 — AP overdue leftover after days",
        "",
        f"Agent `{AGENT}`. Train group-fold seed {FOLD_SEED}. Holdout 72 coverage only.",
        "",
        "## Files written",
        "",
        "- `analysis/evaluate/ap_overdue_qa.py`",
        "- `analysis/outputs/ap_overdue_qa.md`",
        f"- `{OUT_PNG.name}`" if ctx.get("png") else "- (no PNG)",
        "- append-only `analysis/experiments/registry.csv`",
        "- this note",
        "",
        "Did not touch `delay_qa.*`, `dpo_qa.*`, `ap_open_qa.*`, `n_accounts_qa.*`, "
        "`util_snap_qa.*`, `invoices.py`, `product/`, parquet / duckdb, "
        "`build_targets`, parent journal, LIVE, canvas, TURNOVER, or the 15-col card. "
        f"Night Y7 stays **TURNOVER {TURNOVER_QUOTE:.3f} / B_shallow {B_SHALLOW_QUOTE:.3f}**. "
        f"Night Y3 stays **{NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f}**. Days {DAYS_BENCH:.3f}. "
        f"Size {SIZE_QUOTE:.3f}.",
        "",
        "## Locked verdict",
        "",
        "| object | decision |",
        "| --- | --- |",
        f"| overdue leftover after days (Y3) | **{d['headline_tag']}** |",
        "| 15-col Y3 card | **KEEP off the card** |",
        "| TURNOVER add-on | **CLOSE** |",
        "| AP open leftover | **CLOSE (locked)** |",
        "| delay / overdue as Y3 X | **DROP (locked delay_qa)** |",
        "| DPO | **DROP (locked)** |",
        "| y_ap_overdue | **PARK** |",
        "",
        f"{d['headline_tag']} leftover-after-days rank {_f(ctx['p4']['y3_rank'])} "
        f"(OLS {_f(ctx['p4']['y3_ols'])}, fake={ctx['p4']['y3_fake']}). "
        f"Y3 overdue {_f(ctx['p3']['y3'])} vs days {_f(ctx['p3']['days'])} vs size {_f(ctx['p3']['size'])}. "
        f"SIZE={ctx['p2']['size_flag']} twin={bool(ctx['p2']['gate_twins'])}. "
        f"Inverse days-after-overdue {_f(ctx['p4']['inv_rank'])}. "
        f"Leftover after open {_f(ctx['p5']['rank'])} rewrite={ctx['p5']['rewrite']}. "
        f"Leftover after DPO {_f(ctx['p6']['dpo'])} / delay {_f(ctx['p6']['delay'])}. "
        f"Boot leftover-days p50={_f(ctx['p10']['p50'])} p05={_f(ctx['p10']['p05'])}. "
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
    print(f"ap_overdue_qa start seed={FOLD_SEED} holdout=72 agent={AGENT} wave={WRITE_WAVE}")
    panel = load_panel()
    print(f"store has e_ap_overdue_30={bool(panel['has_od30'].iloc[0])}")
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
    print("pass 5 leftover after open")
    p5 = pass5_after_open(tr)
    print("pass 6 leftover after DPO / delay")
    p6 = pass6_after_dpo_delay(tr)
    print("pass vs overdue_30")
    p30 = pass_od30(tr)
    print("pass 7 Y5 report only")
    p7 = pass7_y5(tr)
    print("pass 8 Q6")
    p8 = pass8_q6(tr)
    print("pass 10 bootstrap")
    p10 = pass10_boot(tr)
    print("extra holdout")
    ph = pass_holdout(panel, book)
    print("extra terciles")
    pt = pass_terciles(tr)
    print("extra company-median")
    pcm = pass_company_med(tr)
    print("extra so_far")
    psf = pass_so_far(tr)
    print("extra per-fold leftover")
    pf = pass_folds(tr)
    print("extra Q6 mid/long")
    pq6 = pass_q6_midlong(tr)
    print("extra KEEP-as-X scorecard")
    pk = pass_keep_card(p1, p2, p3, p4)
    print("extra ever-ERP leftover")
    perp = pass_erp_late(tr, book)
    print("extra BETWEEN")
    pbl = pass_between(tr)
    print("extra Y7 leftover")
    py7 = pass_y7_leftover(tr)
    print("extra leftover after days+size / card / AR")
    pds = pass_days_size_card(tr)
    print("extra overdue>0 / winsor")
    ppw = pass_pos_winsor(tr)
    print("extra same-n / n_tx / trail")
    psn = pass_samen(tr)
    print("extra leftover after delay_lag1 / DPO-defined")
    pdl = pass_delay_lag_dpo_def(tr)
    print("extra leftover after open_lag1 / joint")
    poj = pass_open_lag_joint(tr)
    png = make_png(tr, p4)
    decision = decide(p1, p2, p3, p4, p5)
    failed = [
        f"Y3 overdue {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size'])} beat={_f(p3['beat_size'])}",
        f"leftover-after-days rank {_f(p4['y3_rank'])} OLS {_f(p4['y3_ols'])} fake={p4['y3_fake']} dies={p4['y3_dies']}",
        f"inverse days-after-overdue {_f(p4['inv_rank'])}",
        f"leftover after open {_f(p5['rank'])} rewrite={p5['rewrite']} +days {_f(p5['both'])}",
        f"leftover after DPO {_f(p6['dpo'])} +days {_f(p6['dpo_days'])} after delay {_f(p6['delay'])} +days {_f(p6['delay_days'])}",
        f"vs overdue_30 leftover {_f(p30['rank'])} leftover-days-30 {_f(p30.get('days30', float('nan')))}",
        f"Q6 Y7 lag1 leftover {_f(p8['lag_rank'])} Y3 now {_f(p8['now_rank'])} delay_early_nn={p8['delay_early']}",
        f"Y5 leftover-size {_f(p7['rank'])} leak_ok={p7['leak_ok']}",
        f"boot leftover-days p05={_f(p10['p05'])} p50={_f(p10['p50'])} p95={_f(p10['p95'])}",
        f"terciles T1 {_f(pt['t1'])} T2+T3 {_f(pt['t23'])}",
        f"so_far short {_f(psf['short'])} mid {_f(psf['mid'])}",
        f"per-fold {' '.join(pf['bits'])}",
        f"Q6 mid leftover {_f(pq6['mid'])} long raw {_f(pq6['long'])}",
        f"ever-ERP leftover {_f(perp['rank'])} late {_f(perp['late'])}",
        f"BETWEEN cmean {_f(pbl['mean'])} lag1 {_f(pbl['lag1'])} within {_f(pbl['within'])}",
        f"Y7 leftover-days {_f(py7['rank'])} leftover-open {_f(py7['open'])}",
        f"leftover after days+size {_f(pds['ds'])} ss+salary+days {_f(pds['card'])} AR {_f(pds['ar'])} triple {_f(pds['triple'])}",
        f"overdue>0 leftover {_f(ppw['pos'])} winsor {_f(ppw['win'])} open-defined {_f(ppw['both'])}",
        f"same-n overdue {_f(psn['od'])} days {_f(psn['days'])} size {_f(psn['size'])} beat={_f(psn['beat'])} leftover-ntx {_f(psn['ntx'])} trail+days {_f(psn['trail_days'])}",
        f"leftover after delay_lag1 {_f(pdl['dlag'])} +days {_f(pdl['dlag_days'])} DPO-defined {_f(pdl['dpo_def'])} DPO+days {_f(pdl['dpo_both'])} sign={pdl['sign']}",
        f"leftover after open_lag1 {_f(poj['ol'])} +days {_f(poj['ol_days'])} days+n_tx+size {_f(poj['joint'])}",
        f"card: {decision['card']}",
        "do not grow TURNOVER 0.720; do not put e_ap_overdue on the 15-col card",
    ]
    ctx = {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "p6": p6,
        "p30": p30,
        "p7": p7,
        "p8": p8,
        "p10": p10,
        "ph": ph,
        "pt": pt,
        "pcm": pcm,
        "psf": psf,
        "pf": pf,
        "pq6": pq6,
        "pk": pk,
        "perp": perp,
        "pbl": pbl,
        "py7": py7,
        "pds": pds,
        "ppw": ppw,
        "psn": psn,
        "pdl": pdl,
        "poj": poj,
        "png": png,
        "decision": decision,
        "failed": failed,
    }
    write_md(ctx)
    append_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    elapsed = time.time() - t0
    print(
        f"DONE elapsed={elapsed:.0f}s tag={decision['headline_tag']} card={decision['card']}"
    )
    print(
        f"{decision['headline_tag']} leftover-after-days rank {_f(p4['y3_rank'])} "
        f"(OLS {_f(p4['y3_ols'])}, fake={p4['y3_fake']}). "
        f"Y3 overdue {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size'])}. "
        f"SIZE={p2['size_flag']} twin={bool(p2['gate_twins'])}. "
        f"Inverse days-after-overdue {_f(p4['inv_rank'])}. "
        f"Leftover after open {_f(p5['rank'])}. "
        f"Leftover after DPO {_f(p6['dpo'])} / delay {_f(p6['delay'])}. "
        f"Card: {decision['card']}. "
        f"Night Y3 {NIGHT_Y3:.3f}/{NIGHT_Y3_CORE:.3f}, days {DAYS_BENCH:.3f}, "
        f"size {SIZE_QUOTE:.3f}, TURNOVER {TURNOVER_QUOTE:.3f}/{B_SHALLOW_QUOTE:.3f} unchanged."
    )
    return ctx


if __name__ == "__main__":
    run()
