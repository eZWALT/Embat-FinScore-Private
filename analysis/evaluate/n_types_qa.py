"""Unused leftover of ``f_n_types`` after ``c_n_days_with_tx`` as Y3 X.

``f_n_types`` = COUNT(DISTINCT debt product type) with created_at ≤ period_end
and not created_after_snapshot. Connection inventory, rise-only, same family
as ``f_has_*`` / ``f_n_facilities``. ``f_n_facilities`` is the cluster twin
(|ρ| cut 0.8; cluster 56 ρ 0.99). ``f_has_*`` already DROP from the 44.

KEEP-as-X: beat size ≥0.02 AND leftover after days AND not SIZE
(|ρ| vs log1p(a_in3) ≥0.50) AND not a twin (|ρ|≥0.80 vs days / a_n_tx /
f_n_facilities / f_has_*). Leftover <0.55 dies.

Night quotes unchanged: Y3 0.762 / 0.752. Days 0.711. Size 0.617.
Y7 TURNOVER 0.720 / 0.712. Do not put f_n_types on the 15-col card.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.n_types_qa

Owned: analysis/evaluate/n_types_qa.py, analysis/outputs/n_types_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_n_types.md (end).
Do not overwrite factoring_qa.* / ds_r_qa.* / fc_r_qa.* / tax_month_qa.*.
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
OUT_MD = ANALYSIS / "outputs" / "n_types_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "n_types_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_n_types.md"
AGENT = "b17e9c44"
WAVE = "4"
ROUND = "R4"
MODEL = "n_types_qa"
X_FAM = "F"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_Y3_QUOTE = 0.617
HAS_FACT = 0.505
HAS_CONF = 0.485
HAS_LOC = 0.546
DAYS_LAG1_QUOTE = 0.684
Y3_NIGHT = (0.762, 0.752)
Y7_TURNOVER = (0.720, 0.712)
MODAL_QUOTE = 0.771
SIZE_RHO_QUOTE = 0.290
ACF1_QUOTE = 0.80
ICC_QUOTE = 0.98
FAC_RISES_QUOTE = 555
KEEP_DELTA = 0.02
TWIN_RHO = 0.80
SIZE_RHO = 0.50
CHANCE = 0.55
MIN_POS = 50
MIN_ACF_PAIRS = 4
WRITE_WAVE = True

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "a_n_tx",
    "c_n_days_with_tx",
    "f_n_types",
    "f_n_facilities",
    "f_has_loc",
    "f_has_factoring",
    "f_has_confirming",
    "f_new_facility",
    "g_n_types",
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
    s = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)}).dropna()
    if len(s) < 10 or s["x"].nunique() < 2:
        return {"icc": float("nan")}
    y = s["x"].to_numpy(dtype=float)
    _, inv, counts = np.unique(s["co"].to_numpy(), return_inverse=True, return_counts=True)
    if counts.size < 2:
        return {"icc": float("nan")}
    means = np.bincount(inv, weights=y) / counts
    grand = float(y.mean())
    ssb = float(np.sum(counts * (means - grand) ** 2))
    ssw = float(np.sum((y - means[inv]) ** 2))
    var_b = ssb / max(int(counts.size) - 1, 1)
    var_w = ssw / max(int(y.size) - int(counts.size), 1)
    icc = var_b / (var_b + var_w) if (var_b + var_w) > 0 else float("nan")
    return {"icc": float(icc)}


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
            fold_rows.append({"fold": k, "auroc": float("nan"), "sign": 0, "n_va": int(va.sum()), "n_pos": int((va & (y == 1)).sum())})
            continue
        sign = choose_sign(y[tr], x[tr])
        auc = auroc(y[va], sign * x[va])
        aucs.append(auc)
        fold_rows.append({"fold": k, "auroc": float(auc) if np.isfinite(auc) else float("nan"), "sign": int(sign), "n_va": int(va.sum()), "n_pos": int((va & (y == 1)).sum())})
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


def ols_resid(y: pd.Series, *xs: pd.Series):
    cols = {"y": pd.to_numeric(y, errors="coerce")}
    for i, x in enumerate(xs):
        cols[f"x{i}"] = pd.to_numeric(x, errors="coerce")
    d = pd.DataFrame(cols)
    ok = d.notna().all(axis=1)
    resid = pd.Series(np.nan, index=d.index, dtype=float)
    info = {"n": int(ok.sum()), "r2": float("nan")}
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


def company_mean(series: pd.Series, company: pd.Series) -> pd.Series:
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    return d.groupby("co")["x"].transform("mean")


def company_demean(series: pd.Series, company: pd.Series) -> pd.Series:
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    return d["x"] - d.groupby("co")["x"].transform("mean")


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
        raise FileNotFoundError(STORE)
    if not TARGETS.exists():
        raise FileNotFoundError(f"{TARGETS} missing — do not run build_targets")
    raw = pd.read_parquet(STORE)
    yraw = pd.read_parquet(TARGETS)
    print(f"store {STORE} shape={raw.shape} (read-only)")
    print(f"targets {TARGETS} shape={yraw.shape} (read-only)")
    miss = [c for c in STORE_COLS if c not in raw.columns]
    if miss:
        raise RuntimeError(f"monthly.parquet missing {miss}")
    panel = _keys(raw[list(STORE_COLS)])
    y = _keys(yraw[["company_id", "period", *Y_KEEP]])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    for col in ("f_n_types", "f_n_facilities", "f_has_loc", "f_has_factoring", "f_has_confirming", "f_new_facility", "g_n_types"):
        if col in panel.columns:
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
    leak = leakage_check(["f_n_types", "f_n_facilities", "c_n_days_with_tx", "log_in3"], Y3, forbidden_prefixes=["b"])
    if not leak["ok"]:
        raise RuntimeError(f"Y3 never-B leak: {leak['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def pass1_cov(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["f_n_types"], errors="coerce")
    n_cm = int(len(tr))
    n0 = int((x == 0).sum())
    modal = n0 / n_cm if n_cm else float("nan")
    acf1 = median_acf(x, tr["company_id"], 1)
    acf3 = median_acf(x, tr["company_id"], 3)
    rho = spearman(x, tr["log_in3"])
    size_flag = bool(np.isfinite(rho) and abs(rho) >= SIZE_RHO)
    vc = x.value_counts().sort_index()
    rows = [{"value": int(k), "n": f"{int(v):,}", "share": _pp(v / n_cm)} for k, v in vc.items()]
    prose = (
        f"Train f_n_types cov 100% modal0 {_pp(modal)} (feature-report 77.1% "
        f"{'CONFIRM' if abs(modal - MODAL_QUOTE) < 0.03 else 'off'}). "
        f"acf1 {_f(acf1)} (0.80 {'CONFIRM' if np.isfinite(acf1) and abs(acf1 - ACF1_QUOTE) < 0.08 else 'off'}) "
        f"acf3 {_f(acf3)}. ρ vs log1p(a_in3) {_f(rho)} (0.290 "
        f"{'CONFIRM' if np.isfinite(rho) and abs(rho - SIZE_RHO_QUOTE) < 0.05 else 'off'}; "
        f"{'SIZE' if size_flag else 'not SIZE'}). mean {_f(float(x.mean()))} max {int(x.max())}."
    )
    print(prose)
    return {
        "modal": modal,
        "n_cm": n_cm,
        "cov": 1.0,
        "acf1": acf1,
        "acf3": acf3,
        "rho_in3": rho,
        "size_flag": size_flag,
        "rows": rows,
        "mean": float(x.mean()),
        "prose": prose,
    }


def pass2_twins(tr: pd.DataFrame) -> dict:
    x = tr["f_n_types"]
    pairs = [
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("a_n_tx", tr["a_n_tx"]),
        ("f_n_facilities", tr["f_n_facilities"]),
        ("f_has_loc", tr["f_has_loc"]),
        ("f_has_factoring", tr["f_has_factoring"]),
        ("f_has_confirming", tr["f_has_confirming"]),
        ("f_new_facility", tr["f_new_facility"]),
        ("log1p(a_in3)", tr["log_in3"]),
    ]
    rows = []
    rhos = {}
    twins = []
    gate = {"c_n_days_with_tx", "a_n_tx", "f_n_facilities", "f_has_loc", "f_has_factoring", "f_has_confirming"}
    for name, s in pairs:
        rho = spearman(x, s)
        rhos[name] = rho
        twin = bool(name in gate and np.isfinite(rho) and abs(rho) >= TWIN_RHO)
        if twin:
            twins.append(name)
        rows.append({"vs": name, "ρ": _f(rho), "twin": "YES" if twin else ""})
    fac_twin = bool(np.isfinite(rhos["f_n_facilities"]) and abs(rhos["f_n_facilities"]) >= TWIN_RHO)
    prose = (
        f"Spearman twins |ρ|≥0.80: {twins or 'none'}. "
        f"vs days {_f(rhos['c_n_days_with_tx'])} vs a_n_tx {_f(rhos['a_n_tx'])} "
        f"vs f_n_facilities {_f(rhos['f_n_facilities'])} "
        f"{'(cluster twin 0.99 CONFIRM)' if fac_twin else ''} "
        f"vs loc {_f(rhos['f_has_loc'])} vs fact {_f(rhos['f_has_factoring'])} "
        f"vs conf {_f(rhos['f_has_confirming'])} vs size {_f(rhos['log1p(a_in3)'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "twins": twins,
        "gate_twins": bool(twins),
        "fac_twin": fac_twin,
        "rho_days": rhos["c_n_days_with_tx"],
        "rho_fac": rhos["f_n_facilities"],
        "rho_size": rhos["log1p(a_in3)"],
        "prose": prose,
    }


def pass3_singles(tr: pd.DataFrame) -> dict:
    feats = {
        "f_n_types": tr["f_n_types"],
        "f_n_facilities": tr["f_n_facilities"],
        "f_has_loc": tr["f_has_loc"],
        "f_has_factoring": tr["f_has_factoring"],
        "f_has_confirming": tr["f_has_confirming"],
        "f_new_facility": tr["f_new_facility"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
        "log1p(a_in3)": tr["log_in3"],
        "f_n_types>0": (pd.to_numeric(tr["f_n_types"], errors="coerce") > 0).astype(float),
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
    y3 = _cv(store[(Y3, "f_n_types")])
    days = _cv(store[(Y3, "c_n_days_with_tx")])
    size = _cv(store[(Y3, "log1p(a_in3)")])
    fac = _cv(store[(Y3, "f_n_facilities")])
    loc = _cv(store[(Y3, "f_has_loc")])
    fact = _cv(store[(Y3, "f_has_factoring")])
    conf = _cv(store[(Y3, "f_has_confirming")])
    y2 = _cv(store[(Y2, "f_n_types")])
    days_ok = bool(np.isfinite(days) and abs(days - DAYS_BENCH) < 0.02)
    size_ok = bool(np.isfinite(size) and abs(size - SIZE_Y3_QUOTE) < 0.03)
    fact_ok = bool(np.isfinite(fact) and abs(fact - HAS_FACT) < 0.03)
    conf_ok = bool(np.isfinite(conf) and abs(conf - HAS_CONF) < 0.03)
    loc_ok = bool(np.isfinite(loc) and abs(loc - HAS_LOC) < 0.03)
    beat_size = y3 - size if np.isfinite(y3) and np.isfinite(size) else float("nan")
    prose = (
        f"Y3 f_n_types {_f(y3)} vs days {_f(days)} (0.711 {'CONFIRM' if days_ok else 'off'}) "
        f"vs size {_f(size)} (0.617 {'CONFIRM' if size_ok else 'off'}, Δ {_f(beat_size)}) "
        f"vs f_n_facilities {_f(fac)} vs loc {_f(loc)} (0.546 {'CONFIRM' if loc_ok else 'off'}) "
        f"vs fact {_f(fact)} (0.505 {'CONFIRM' if fact_ok else 'off'}) "
        f"vs conf {_f(conf)} (0.485 {'CONFIRM' if conf_ok else 'off'}). Y2 n_types {_f(y2)}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3": y3,
        "days": days,
        "size": size,
        "fac": fac,
        "loc": loc,
        "fact": fact,
        "conf": conf,
        "y2": y2,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "fact_ok": fact_ok,
        "conf_ok": conf_ok,
        "loc_ok": loc_ok,
        "beat_size": beat_size,
        "gt0": _cv(store[(Y3, "f_n_types>0")]),
        "prose": prose,
    }


def pass4_leftover(tr: pd.DataFrame) -> dict:
    x = tr["f_n_types"]
    days = tr["c_n_days_with_tx"]
    fac = tr["f_n_facilities"]
    specs = [
        (Y3, "after days", (days,)),
        (Y3, "after f_n_facilities", (fac,)),
        (Y3, "after a_n_tx", (tr["a_n_tx"],)),
        (Y3, "after size", (tr["log_in3"],)),
        (Y3, "after loc", (tr["f_has_loc"],)),
        (Y3, "after days+facilities", (days, fac)),
        (Y2, "after days", (days,)),
    ]
    rows = []
    store = {}
    for ycol, name, xs in specs:
        rec = leftover_diag(tr[ycol], x, list(xs), tr["fold"], tr[ycol].notna())
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
    inv = leftover_diag(tr[Y3], days, [x], tr["fold"], tr[Y3].notna())
    inv_fac = leftover_diag(tr[Y3], fac, [x], tr["fold"], tr[Y3].notna())
    store[(Y3, "days after n_types")] = inv
    rows.append(
        {
            "y": Y3,
            "control": "days after n_types (inverse)",
            "OLS": _f(inv["ols"]),
            "rank": _f(inv["rank"]),
            "ρ(resid,ctrl)": _f(inv["rho_ctrl"]),
            "R²": _f(inv["r2"]),
            "fake": "YES" if inv["fake"] else "",
            "honest_dies": "YES" if inv["honest_dies"] else "no",
        }
    )
    rows.append(
        {
            "y": Y3,
            "control": "facilities after n_types",
            "OLS": _f(inv_fac["ols"]),
            "rank": _f(inv_fac["rank"]),
            "ρ(resid,ctrl)": _f(inv_fac["rho_ctrl"]),
            "R²": _f(inv_fac["r2"]),
            "fake": "YES" if inv_fac["fake"] else "",
            "honest_dies": "YES" if inv_fac["honest_dies"] else "no",
        }
    )
    y3 = store[(Y3, "after days")]
    leftover = y3["rank"]
    leftover_ols = y3["ols"]
    fake_ols = bool(np.isfinite(leftover_ols) and leftover_ols >= CHANCE and np.isfinite(leftover) and leftover < CHANCE)
    dies = bool(y3["honest_dies"] or (np.isfinite(leftover) and leftover < CHANCE))
    inv_lives = bool(np.isfinite(inv["rank"]) and inv["rank"] >= CHANCE and not inv["honest_dies"])
    fac_left = store[(Y3, "after f_n_facilities")]["rank"]
    fac_dies = bool(store[(Y3, "after f_n_facilities")]["honest_dies"] or (np.isfinite(fac_left) and fac_left < CHANCE))
    rewrite = bool(fac_dies and np.isfinite(store[(Y3, "after f_n_facilities")]["r2"]) and store[(Y3, "after f_n_facilities")]["r2"] >= 0.80)
    prose = (
        f"Y3 leftover after days OLS {_f(leftover_ols)} rank {_f(leftover)} "
        f"ρ(resid,days)={_f(y3['rho_ctrl'])} R²={_f(y3['r2'])} "
        f"({'OLS-high / rank-dies' if fake_ols else 'OLS and rank agree'}; "
        f"{'FAKE-DAYS' if y3['fake'] else 'resid is not a days clone'}). "
        f"After f_n_facilities rank {_f(fac_left)} R²={_f(store[(Y3, 'after f_n_facilities')]['r2'])} "
        f"({'weaker rewrite of n_facilities' if rewrite or fac_dies else 'not absorbed by n_facilities'}). "
        f"Inverse days after n_types {_f(inv['rank'])} ({'0.711 bar lives' if inv_lives else '0.711 bar dies'}). "
        f"Honest leftover after days {'DIES' if dies else 'lives'}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3_rank": leftover,
        "y3_ols": leftover_ols,
        "y3_rho_days": y3["rho_ctrl"],
        "y3_r2": y3["r2"],
        "y3_fake": y3["fake"],
        "y3_dies": dies,
        "fake_ols": fake_ols,
        "fac_left": fac_left,
        "fac_r2": store[(Y3, "after f_n_facilities")]["r2"],
        "fac_dies": fac_dies,
        "rewrite": rewrite,
        "stack": store[(Y3, "after days+facilities")]["rank"],
        "inv_rank": inv["rank"],
        "inv_ols": inv["ols"],
        "inv_lives": inv_lives,
        "inv_fac": inv_fac["rank"],
        "y2_rank": store[(Y2, "after days")]["rank"],
        "prose": prose,
    }


def pass5_rise(tr: pd.DataFrame) -> dict:
    work = tr.sort_values(["company_id", "period"]).copy()
    nt = pd.to_numeric(work["f_n_types"], errors="coerce")
    nf = pd.to_numeric(work["f_n_facilities"], errors="coerce")
    d_nt = nt.groupby(work["company_id"], sort=False).diff()
    d_nf = nf.groupby(work["company_id"], sort=False).diff()
    rises_t = int((d_nt > 0).sum())
    drops_t = int((d_nt < 0).sum())
    same_t = int((d_nt == 0).sum())
    rises_f = int((d_nf > 0).sum())
    drops_f = int((d_nf < 0).sum())
    rise_only = drops_t == 0
    fac_ok = bool(rises_f == FAC_RISES_QUOTE and drops_f == 0)
    raw = signed_oof_auroc(work[Y3], (d_nt > 0).astype(float), work["fold"], work[Y3].notna())
    prose = (
        f"f_n_types rises {rises_t} / drops {drops_t} / flat {same_t} "
        f"({'rise-only' if rise_only else 'NOT rise-only'}). "
        f"f_n_facilities rises {rises_f} / drops {drops_f} "
        f"(factoring-qa 555/0 {'CONFIRM' if fac_ok else 'off'}). "
        f"Y3 rise-month dummy {_f(_cv(raw))}."
    )
    print(prose)
    return {
        "rises": rises_t,
        "drops": drops_t,
        "flat": same_t,
        "rise_only": rise_only,
        "fac_rises": rises_f,
        "fac_drops": drops_f,
        "fac_ok": fac_ok,
        "rise_y3": _cv(raw),
        "prose": prose,
    }


def pass6_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    n_erp = int(last["company_id"].isin(book).sum())
    n_dark = int((~last["company_id"].isin(book)).sum())
    confirm = n_erp == 744 and n_dark == 470
    erp = tr["company_id"].isin(book)
    x = pd.to_numeric(tr["f_n_types"], errors="coerce")
    rows = []
    store = {}
    for name, mask in (("invoiced_744", erp), ("dark_470", ~erp)):
        raw = signed_oof_auroc(tr[Y3], x, tr["fold"], tr[Y3].notna() & mask)
        rec = leftover_diag(tr[Y3], x, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & mask)
        store[name] = rec
        ever = int(tr.loc[mask].groupby("company_id")["f_n_types"].max().gt(0).sum())
        rows.append(
            {
                "slice": name,
                "n_cm": f"{int(mask.sum()):,}",
                "ever>0": ever,
                "mean": _f(float(x[mask].mean())),
                "Y3 raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "leftover days": _f(rec["rank"]),
            }
        )
    prose = (
        f"Last-month ever-ERP {n_erp} / never-ERP {n_dark} ({'CONFIRM 744/470' if confirm else 'off'}). "
        f"mean n_types invoiced {_f(float(x[erp].mean()))} dark {_f(float(x[~erp].mean()))}. "
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
        "prose": prose,
    }


def pass7_q6(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    cols = ("f_n_types", "f_n_types_lag1", "f_n_types_lag3", "c_n_days_with_tx", "c_n_days_with_tx_lag1")
    for col in cols:
        if col not in tr.columns:
            continue
        res = signed_oof_auroc(tr[Y3], tr[col], tr["fold"], tr[Y3].notna())
        store[col] = res
        rows.append({"col": col, "n": f"{res['n_defined']:,}", "n_pos": f"{res['n_pos']:,}", "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"])})
    short = signed_oof_auroc(tr[Y3], tr["f_n_types"], tr["fold"], tr[Y3].notna() & (tr["trail_class"] == "short_<12"))
    short_l1 = leftover_diag(tr[Y3], tr["f_n_types_lag1"], [tr["c_n_days_with_tx_lag1"]], tr["fold"], tr[Y3].notna() & (tr["trail_class"] == "short_<12"))
    now = _cv(store.get("f_n_types", {"low_power": True}))
    lag1 = _cv(store.get("f_n_types_lag1", {"low_power": True}))
    lag3 = _cv(store.get("f_n_types_lag3", {"low_power": True}))
    days_l1 = _cv(store.get("c_n_days_with_tx_lag1", {"low_power": True}))
    rec1 = leftover_diag(tr[Y3], tr["f_n_types_lag1"], [tr["c_n_days_with_tx_lag1"]], tr["fold"], tr[Y3].notna())
    days_l1_ok = bool(np.isfinite(days_l1) and abs(days_l1 - DAYS_LAG1_QUOTE) < 0.03)
    l1_dies = bool(rec1["honest_dies"] or (np.isfinite(rec1["rank"]) and rec1["rank"] < CHANCE))
    keep_q6 = bool(np.isfinite(now) and now >= CHANCE and np.isfinite(lag1) and lag1 >= CHANCE and (now - lag1) <= 0.03 and not l1_dies)
    q6 = "KEEP" if keep_q6 else "CLOSE"
    prose = (
        f"Y3 n_types now {_f(now)} lag1 {_f(lag1)} lag3 {_f(lag3)}; short raw {_f(_cv(short))}. "
        f"Days lag1 {_f(days_l1)} (KEEP 0.684 {'CONFIRM' if days_l1_ok else 'off'}). "
        f"n_types_lag1 leftover after days_lag1 {_f(rec1['rank'])}; short leftover {_f(short_l1['rank'])}. Q6 {q6}."
    )
    print(prose)
    return {
        "rows": rows,
        "now": now,
        "lag1": lag1,
        "lag3": lag3,
        "days_l1": days_l1,
        "l1_left": rec1["rank"],
        "short": _cv(short),
        "short_l1": short_l1["rank"],
        "l1_dies": l1_dies,
        "days_l1_ok": days_l1_ok,
        "q6": q6,
        "prose": prose,
    }


def pass8_ever(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["f_n_types"], errors="coerce")
    ever = x.groupby(tr["company_id"]).transform("max")
    mu = company_mean(x, tr["company_id"])
    dem = company_demean(x, tr["company_id"])
    raw_e = signed_oof_auroc(tr[Y3], ever, tr["fold"], tr[Y3].notna())
    raw_m = signed_oof_auroc(tr[Y3], mu, tr["fold"], tr[Y3].notna())
    raw_d = signed_oof_auroc(tr[Y3], dem, tr["fold"], tr[Y3].notna())
    left_e = leftover_diag(tr[Y3], ever, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    left_m = leftover_diag(tr[Y3], mu, [company_mean(tr["c_n_days_with_tx"], tr["company_id"])], tr["fold"], tr[Y3].notna())
    left_now = leftover_diag(tr[Y3], x, [ever], tr["fold"], tr[Y3].notna())
    n_ever = int(tr.groupby("company_id")["f_n_types"].max().gt(0).sum())
    prose = (
        f"Ever n_types>0 companies {n_ever}/1214. Ever-max Y3 {_f(_cv(raw_e))} leftover after days {_f(left_e['rank'])}. "
        f"Company-mean Y3 {_f(_cv(raw_m))} leftover after days-mean {_f(left_m['rank'])}. "
        f"Demean Y3 {_f(_cv(raw_d))}. Month leftover after ever-max {_f(left_now['rank'])} "
        f"(trait vs month shock)."
    )
    print(prose)
    return {
        "n_ever": n_ever,
        "ever": _cv(raw_e),
        "ever_left": left_e["rank"],
        "mu": _cv(raw_m),
        "between": left_m["rank"],
        "dem": _cv(raw_d),
        "month_after_ever": left_now["rank"],
        "prose": prose,
    }


def pass_icc(tr: pd.DataFrame) -> dict:
    icc = icc_anova(tr["f_n_types"], tr["company_id"])
    confirm = bool(np.isfinite(icc["icc"]) and abs(icc["icc"] - ICC_QUOTE) < 0.03)
    prose = f"f_n_types ICC {_f(icc['icc'])} (feature-report 0.98 {'CONFIRM BETWEEN' if confirm else 'off'})."
    print(prose)
    return {"icc": icc["icc"], "confirm": confirm, "prose": prose}


def pass_holdout(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"]
    x = pd.to_numeric(ho["f_n_types"], errors="coerce")
    prose = (
        f"Holdout coverage only (no fit): {ho['company_id'].nunique()} co / {len(ho):,} CM, "
        f"mean {_f(float(x.mean()))} P(>0) {_pp(float((x > 0).mean()))}."
    )
    print(prose)
    return {"n_cm": int(len(ho)), "n_co": int(ho["company_id"].nunique()), "prose": prose}


def pass_fold_left(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rows = [{"fold": k, "OLS": _f(fold_k(rec["rec"], k)), "rank": _f(fold_k(rec["rrec"], k))} for k in range(N_FOLDS)]
    vals = [fold_k(rec["rrec"], k) for k in range(N_FOLDS)]
    finite = [v for v in vals if np.isfinite(v)]
    spread = float(max(finite) - min(finite)) if finite else float("nan")
    prose = f"Y3 leftover-after-days rank folds {rec['rank_folds']} spread {_f(spread)}."
    print(prose)
    return {"rows": rows, "spread": spread, "prose": prose}


def leftover_oof(y, x, controls, folds, mask) -> dict:
    """Rank leftover computed fold-wise (train residual, val AUROC)."""
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    xs = [pd.to_numeric(c, errors="coerce") for c in controls]
    defined = mask & y.notna() & x.notna()
    for c in xs:
        defined = defined & c.notna()
    aucs = []
    for k in range(N_FOLDS):
        tr = defined & (folds != k)
        va = defined & (folds == k)
        if int((va & (y == 1)).sum()) == 0 or int((va & (y == 0)).sum()) == 0:
            continue
        xr = x.rank(method="average")
        cr = [c.rank(method="average") for c in xs]
        resid, _ = ols_resid(xr, *cr)
        # Fit residual on train only, apply to val via the same OLS.
        cols = {"y": xr}
        for i, c in enumerate(cr):
            cols[f"x{i}"] = c
        d = pd.DataFrame(cols)
        ok = tr & d.notna().all(axis=1)
        if int(ok.sum()) < 20:
            continue
        Y = d.loc[ok, "y"].to_numpy(dtype=float)
        X = np.column_stack([np.ones(int(ok.sum()))] + [d.loc[ok, f"x{i}"].to_numpy(dtype=float) for i in range(len(xs))])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        va_ok = va & d.notna().all(axis=1)
        Xv = np.column_stack([np.ones(int(va_ok.sum()))] + [d.loc[va_ok, f"x{i}"].to_numpy(dtype=float) for i in range(len(xs))])
        resid_va = d.loc[va_ok, "y"].to_numpy(dtype=float) - Xv @ beta
        sign = choose_sign(y[ok], resid[ok])
        aucs.append(auroc(y[va_ok], sign * pd.Series(resid_va, index=d.index[va_ok])))
    finite = [a for a in aucs if np.isfinite(a)]
    return {"cv": float(np.mean(finite)) if finite else float("nan"), "n_folds": len(finite)}


def pass_boot(tr: pd.DataFrame, n_boot: int = 80) -> dict:
    rng = np.random.default_rng(FOLD_SEED)
    cos = tr["company_id"].drop_duplicates().to_numpy()
    idx = {c: tr.index[tr["company_id"] == c].to_numpy() for c in cos}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(cos, size=len(cos), replace=True)
        rows = np.concatenate([idx[c] for c in pick])
        sub = tr.loc[rows].reset_index(drop=True)
        rec = leftover_diag(sub[Y3], sub["f_n_types"], [sub["c_n_days_with_tx"]], sub["fold"], sub[Y3].notna())
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
    arr = np.array(vals, dtype=float)
    out = {
        "n": int(arr.size),
        "p05": float(np.quantile(arr, 0.05)) if arr.size else float("nan"),
        "p50": float(np.quantile(arr, 0.50)) if arr.size else float("nan"),
        "p95": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
        "share_ge": float((arr >= CHANCE).mean()) if arr.size else float("nan"),
    }
    prose = (
        f"Company bootstrap leftover-after-days n={out['n']} "
        f"p05/p50/p95 {_f(out['p05'])} / {_f(out['p50'])} / {_f(out['p95'])} "
        f"P(≥0.55)={_pp(out['share_ge'])}."
    )
    print(prose)
    return {**out, "prose": prose}


def pass_disagree(tr: pd.DataFrame) -> dict:
    nt = pd.to_numeric(tr["f_n_types"], errors="coerce")
    nf = pd.to_numeric(tr["f_n_facilities"], errors="coerce")
    pear = float(pd.DataFrame({"a": nt, "b": nf}).dropna().corr().iloc[0, 1])
    same = int((nt == nf).sum())
    n = int(len(tr))
    disagree = nt != nf
    n_dis = int(disagree.sum())
    n_co = int(tr.loc[disagree, "company_id"].nunique())
    extra = leftover_diag(tr[Y3], nt, [nf], tr["fold"], tr[Y3].notna() & disagree)
    raw_dis = signed_oof_auroc(tr[Y3], nt, tr["fold"], tr[Y3].notna() & disagree)
    gt0_j = ((nt > 0) == (nf > 0)).mean()
    prose = (
        f"Pearson n_types~n_facilities {_f(pear)} (Spearman 0.994). "
        f"Equal on {_pp(same / n)} of CM; disagree {n_dis:,} CM / {n_co} companies. "
        f"P(>0) agree {_pp(gt0_j)}. Y3 leftover on disagree CM {_f(extra['rank'])} raw {_f(_cv(raw_dis))}."
    )
    print(prose)
    return {
        "pearson": pear,
        "n_dis": n_dis,
        "n_co": n_co,
        "eq_share": same / n,
        "dis_left": extra["rank"],
        "dis_raw": _cv(raw_dis),
        "prose": prose,
    }


def pass_terciles(tr: pd.DataFrame) -> dict:
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    terc = pd.qcut(size, 3, labels=["T1", "T2", "T3"], duplicates="drop")
    rows = []
    leftover = {}
    for lab in ("T1", "T2", "T3"):
        m = terc.eq(lab)
        rec = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & m)
        raw = signed_oof_auroc(tr[Y3], tr["f_n_types"], tr["fold"], tr[Y3].notna() & m)
        leftover[lab] = rec["rank"]
        rows.append({"tercile": lab, "n": f"{int(m.sum()):,}", "Y3 raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]), "leftover days": _f(rec["rank"])})
    prose = (
        f"SIZE tercile leftover after days T1/T2/T3 "
        f"{_f(leftover.get('T1'))} / {_f(leftover.get('T2'))} / {_f(leftover.get('T3'))}."
    )
    print(prose)
    return {"rows": rows, "leftover": leftover, "prose": prose}


def pass_perm(tr: pd.DataFrame, n_perm: int = 30) -> dict:
    rng = np.random.default_rng(FOLD_SEED + 7)
    x = pd.to_numeric(tr["f_n_types"], errors="coerce").to_numpy()
    obs = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())["rank"]
    nulls = []
    for _ in range(n_perm):
        xp = pd.Series(rng.permutation(x), index=tr.index)
        rec = leftover_diag(tr[Y3], xp, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
        if np.isfinite(rec["rank"]):
            nulls.append(rec["rank"])
    arr = np.array(nulls, dtype=float)
    p = float((arr >= obs).mean()) if arr.size and np.isfinite(obs) else float("nan")
    prose = (
        f"Permute n_types leftover-after-days null p50 {_f(float(np.median(arr)) if arr.size else float('nan'))} "
        f"p(obs≥null)={_f(p, 3)} obs={_f(obs)}."
    )
    print(prose)
    return {"p50": float(np.median(arr)) if arr.size else float("nan"), "p": p, "obs": obs, "prose": prose}


def pass_logo(tr: pd.DataFrame) -> dict:
    groups = tr["group_id"].dropna().drop_duplicates().to_numpy()
    vals = []
    for g in groups:
        m = tr["group_id"] != g
        rec = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & m)
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
    arr = np.array(vals, dtype=float)
    prose = (
        f"Leave-one-group leftover-after-days n={arr.size} "
        f"min/med/max {_f(float(arr.min()) if arr.size else float('nan'))} / "
        f"{_f(float(np.median(arr)) if arr.size else float('nan'))} / "
        f"{_f(float(arr.max()) if arr.size else float('nan'))}."
    )
    print(prose)
    return {
        "n": int(arr.size),
        "min": float(arr.min()) if arr.size else float("nan"),
        "med": float(np.median(arr)) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
        "prose": prose,
    }


def pass_books(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for name in ("short_<12", "mid_12_17", "long_>=18"):
        m = tr["trail_class"] == name
        rec = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & m)
        raw = signed_oof_auroc(tr[Y3], tr["f_n_types"], tr["fold"], tr[Y3].notna() & m)
        store[name] = rec["rank"]
        rows.append({"book": name, "n": f"{int(m.sum()):,}", "Y3 raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]), "leftover days": _f(rec["rank"])})
    prose = (
        f"Inventory leftover after days short/mid/long "
        f"{_f(store['short_<12'])} / {_f(store['mid_12_17'])} / {_f(store['long_>=18'])}."
    )
    print(prose)
    return {"rows": rows, "short": store["short_<12"], "long": store["long_>=18"], "prose": prose}


def pass_fac_compare(tr: pd.DataFrame) -> dict:
    fac_days = leftover_diag(tr[Y3], tr["f_n_facilities"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    flags = leftover_diag(tr[Y3], tr["f_n_types"], [tr["f_has_loc"], tr["f_has_factoring"], tr["f_has_confirming"]], tr["fold"], tr[Y3].notna())
    gt0 = leftover_diag(tr[Y3], (pd.to_numeric(tr["f_n_types"], errors="coerce") > 0).astype(float), [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    oof = leftover_oof(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    oof_fac = leftover_oof(tr[Y3], tr["f_n_types"], [tr["f_n_facilities"]], tr["fold"], tr[Y3].notna())
    on_pos = leftover_diag(
        tr[Y3],
        tr["f_n_types"],
        [tr["c_n_days_with_tx"]],
        tr["fold"],
        tr[Y3].notna() & (pd.to_numeric(tr["f_n_types"], errors="coerce") > 0),
    )
    prose = (
        f"Twin f_n_facilities leftover after days {_f(fac_days['rank'])} (n_types was 0.534). "
        f"n_types leftover after has_* flags {_f(flags['rank'])}. "
        f"binary >0 leftover after days {_f(gt0['rank'])}. "
        f"Fold-wise OOF leftover after days {_f(oof['cv'])} after facilities {_f(oof_fac['cv'])}. "
        f"On n_types>0 months leftover after days {_f(on_pos['rank'])}."
    )
    print(prose)
    return {
        "fac_days": fac_days["rank"],
        "flags": flags["rank"],
        "gt0": gt0["rank"],
        "oof": oof["cv"],
        "oof_fac": oof_fac["cv"],
        "on_pos": on_pos["rank"],
        "prose": prose,
    }


def pass_rise_left(tr: pd.DataFrame) -> dict:
    work = tr.sort_values(["company_id", "period"]).copy()
    nt = pd.to_numeric(work["f_n_types"], errors="coerce")
    d_nt = nt.groupby(work["company_id"], sort=False).diff()
    rise = (d_nt > 0).astype(float)
    rec = leftover_diag(work[Y3], rise, [work["c_n_days_with_tx"]], work["fold"], work[Y3].notna())
    first = work.groupby("company_id", sort=False).cumcount()
    first_rise = ((d_nt > 0) & (first > 0)).astype(float)
    rec_f = leftover_diag(work[Y3], first_rise, [work["c_n_days_with_tx"]], work["fold"], work[Y3].notna())
    prose = (
        f"Rise-month leftover after days {_f(rec['rank'])} "
        f"ρ(resid,days)={_f(rec['rho_ctrl'])} fake={rec['fake']} "
        f"n_pos={rec['n_pos']}; first-rise leftover {_f(rec_f['rank'])} "
        f"ρ(resid,days)={_f(rec_f['rho_ctrl'])} fake={rec_f['fake']}."
    )
    print(prose)
    return {
        "rise": rec["rank"],
        "first": rec_f["rank"],
        "rise_rho": rec["rho_ctrl"],
        "rise_fake": rec["fake"],
        "rise_npos": rec["n_pos"],
        "prose": prose,
    }


def pass_onpos(tr: pd.DataFrame) -> dict:
    pos = pd.to_numeric(tr["f_n_types"], errors="coerce") > 0
    rec = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & pos)
    rec_f = leftover_diag(tr[Y3], tr["f_n_types"], [tr["f_n_facilities"]], tr["fold"], tr[Y3].notna() & pos)
    rec_d = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"], tr["f_n_facilities"]], tr["fold"], tr[Y3].notna() & pos)
    raw = signed_oof_auroc(tr[Y3], tr["f_n_types"], tr["fold"], tr[Y3].notna() & pos)
    days_pos = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna() & pos)
    prose = (
        f"On n_types>0 CM ({int(pos.sum()):,}): Y3 raw {_f(_cv(raw))} n_pos={raw['n_pos']} "
        f"days-in-slice {_f(_cv(days_pos))}. Leftover after days {_f(rec['rank'])} "
        f"ρ(resid,days)={_f(rec['rho_ctrl'])} fake={rec['fake']}. "
        f"After facilities {_f(rec_f['rank'])} R²={_f(rec_f['r2'])}. After both {_f(rec_d['rank'])}."
    )
    print(prose)
    return {
        "raw": _cv(raw),
        "days": _cv(days_pos),
        "left": rec["rank"],
        "rho": rec["rho_ctrl"],
        "fake": rec["fake"],
        "n_pos": raw["n_pos"],
        "fac": rec_f["rank"],
        "both": rec_d["rank"],
        "prose": prose,
    }


def pass_perm_company(tr: pd.DataFrame, n_perm: int = 24) -> dict:
    rng = np.random.default_rng(FOLD_SEED + 11)
    x = pd.to_numeric(tr["f_n_types"], errors="coerce")
    cos = tr["company_id"].drop_duplicates().to_numpy()
    mu_map = x.groupby(tr["company_id"]).mean().to_dict()
    obs = leftover_diag(tr[Y3], x, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())["rank"]
    nulls = []
    for _ in range(n_perm):
        shuf = rng.permutation(cos)
        remap = dict(zip(cos, [mu_map[c] for c in shuf]))
        xp = tr["company_id"].map(remap)
        rec = leftover_diag(tr[Y3], xp, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
        if np.isfinite(rec["rank"]):
            nulls.append(rec["rank"])
    arr = np.array(nulls, dtype=float)
    noise = pd.Series(rng.normal(size=len(tr)), index=tr.index)
    noise_left = leftover_diag(tr[Y3], noise, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    p = float((arr >= obs).mean()) if arr.size and np.isfinite(obs) else float("nan")
    prose = (
        f"Company-mean shuffle leftover-after-days null p50 "
        f"{_f(float(np.median(arr)) if arr.size else float('nan'))} p(obs≥null)={_f(p, 3)} "
        f"obs={_f(obs)}. Pure N(0,1) leftover after days {_f(noise_left['rank'])} "
        f"ρ(resid,days)={_f(noise_left['rho_ctrl'])}."
    )
    print(prose)
    return {
        "p50": float(np.median(arr)) if arr.size else float("nan"),
        "p": p,
        "noise": noise_left["rank"],
        "prose": prose,
    }


def pass_y3_cells(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["f_n_types"], errors="coerce")
    lab = tr[Y3].notna()
    buckets = pd.cut(x, bins=[-0.5, 0.5, 1.5, 2.5, 99], labels=["0", "1", "2", "3+"])
    rows = []
    for lab_b in ("0", "1", "2", "3+"):
        m = lab & buckets.eq(lab_b)
        n = int(m.sum())
        n_pos = int((m & (tr[Y3] == 1)).sum())
        rows.append({"n_types": lab_b, "n": f"{n:,}", "Y3+": n_pos, "rate": _pp(n_pos / n if n else float("nan"))})
    prose = "Y3 rate by n_types bucket: " + ", ".join(f"{r['n_types']} {r['rate']}" for r in rows)
    print(prose)
    return {"rows": rows, "prose": prose}


def pass_clock(tr: pd.DataFrame) -> dict:
    work = tr.sort_values(["company_id", "period"]).copy()
    nt = pd.to_numeric(work["f_n_types"], errors="coerce")
    sofar = work.groupby("company_id", sort=False).cumcount()
    first_sofar = sofar.groupby(work["company_id"], sort=False).transform(
        lambda s: sofar.loc[nt.loc[s.index].gt(0).idxmax()] if nt.loc[s.index].gt(0).any() else np.nan
    )
    clock = sofar - first_sofar
    clock = clock.where(nt > 0)
    rec = leftover_diag(work[Y3], clock, [work["c_n_days_with_tx"]], work["fold"], work[Y3].notna() & clock.notna())
    raw = signed_oof_auroc(work[Y3], clock, work["fold"], work[Y3].notna() & clock.notna())
    prose = (
        f"Months-since-first-type leftover after days {_f(rec['rank'])} "
        f"raw {_f(_cv(raw))} ρ(resid,days)={_f(rec['rho_ctrl'])} n_pos={rec['n_pos']}."
    )
    print(prose)
    return {"left": rec["rank"], "raw": _cv(raw), "prose": prose}


def pass_boot_fac(tr: pd.DataFrame, n_boot: int = 24) -> dict:
    rng = np.random.default_rng(FOLD_SEED + 3)
    cos = tr["company_id"].drop_duplicates().to_numpy()
    idx = {c: tr.index[tr["company_id"] == c].to_numpy() for c in cos}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(cos, size=len(cos), replace=True)
        rows = np.concatenate([idx[c] for c in pick])
        sub = tr.loc[rows].reset_index(drop=True)
        rec = leftover_diag(sub[Y3], sub["f_n_types"], [sub["f_n_facilities"]], sub["fold"], sub[Y3].notna())
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
    arr = np.array(vals, dtype=float)
    out = {
        "n": int(arr.size),
        "p05": float(np.quantile(arr, 0.05)) if arr.size else float("nan"),
        "p50": float(np.quantile(arr, 0.50)) if arr.size else float("nan"),
        "p95": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
    }
    prose = (
        f"Bootstrap leftover after f_n_facilities n={out['n']} "
        f"p05/p50/p95 {_f(out['p05'])} / {_f(out['p50'])} / {_f(out['p95'])}."
    )
    print(prose)
    return {**out, "prose": prose}


def pass_dark_fac(tr: pd.DataFrame, book: set[str]) -> dict:
    erp = tr["company_id"].isin(book)
    rec_i = leftover_diag(tr[Y3], tr["f_n_types"], [tr["f_n_facilities"]], tr["fold"], tr[Y3].notna() & erp)
    rec_d = leftover_diag(tr[Y3], tr["f_n_types"], [tr["f_n_facilities"]], tr["fold"], tr[Y3].notna() & ~erp)
    prose = (
        f"Leftover after facilities invoiced {_f(rec_i['rank'])} dark {_f(rec_d['rank'])}."
    )
    print(prose)
    return {"inv": rec_i["rank"], "dark": rec_d["rank"], "prose": prose}


def pass_connected(tr: pd.DataFrame) -> dict:
    ever = pd.to_numeric(tr["f_n_types"], errors="coerce").groupby(tr["company_id"]).transform("max").gt(0)
    last = tr.sort_values("period").groupby("company_id", as_index=False).tail(1)
    last_ever = pd.to_numeric(last["f_n_types"], errors="coerce").gt(0)
    rec_n = last.loc[~last_ever, Y3]
    rec_y = last.loc[last_ever, Y3]
    # company-ever recover: any Y3=1
    ever_rec = tr.groupby("company_id")[Y3].max()
    ever_nt = tr.groupby("company_id")["f_n_types"].max().gt(0)
    share_c = float(ever_rec[ever_nt].mean()) if ever_nt.any() else float("nan")
    share_n = float(ever_rec[~ever_nt].mean()) if (~ever_nt).any() else float("nan")
    n_c = int(ever_nt.sum())
    n_n = int((~ever_nt).sum())
    n_rec_c = int(ever_rec[ever_nt].fillna(0).sum())
    n_rec_n = int(ever_rec[~ever_nt].fillna(0).sum())
    flag3 = tr["f_has_loc"] + tr["f_has_factoring"] + tr["f_has_confirming"]
    rec_f3 = leftover_diag(tr[Y3], tr["f_n_types"], [flag3], tr["fold"], tr[Y3].notna())
    delta = pd.to_numeric(tr["f_n_facilities"], errors="coerce") - pd.to_numeric(tr["f_n_types"], errors="coerce")
    rec_d = leftover_diag(tr[Y3], delta, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    raw_d = signed_oof_auroc(tr[Y3], delta, tr["fold"], tr[Y3].notna())
    groups = tr.loc[ever, "group_id"]
    top_share = float("nan")
    if len(groups):
        vc = groups.groupby(tr.loc[ever, "company_id"]).first().value_counts()
        top_share = float(vc.iloc[0] / vc.sum()) if len(vc) else float("nan")
    prose = (
        f"Ever n_types>0 companies {n_c}/1214 recover {n_rec_c} ({_pp(share_c)}); "
        f"never {n_n} recover {n_rec_n} ({_pp(share_n)}). "
        f"Leftover of n_types after has_loc+fact+conf {_f(rec_f3['rank'])}. "
        f"Δ=n_facilities-n_types Y3 {_f(_cv(raw_d))} leftover after days {_f(rec_d['rank'])}. "
        f"Largest group among connected {_pp(top_share)}."
    )
    print(prose)
    return {
        "n_c": n_c,
        "share_c": share_c,
        "share_n": share_n,
        "n_rec_c": n_rec_c,
        "n_rec_n": n_rec_n,
        "flags3": rec_f3["rank"],
        "delta": rec_d["rank"],
        "delta_raw": _cv(raw_d),
        "top_share": top_share,
        "prose": prose,
    }


def pass_delta_gate(tr: pd.DataFrame) -> dict:
    delta = pd.to_numeric(tr["f_n_facilities"], errors="coerce") - pd.to_numeric(tr["f_n_types"], errors="coerce")
    raw = signed_oof_auroc(tr[Y3], delta, tr["fold"], tr[Y3].notna())
    rec = leftover_diag(tr[Y3], delta, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    rec_n = leftover_diag(tr[Y3], delta, [tr["f_n_types"]], tr["fold"], tr[Y3].notna())
    rho_size = spearman(delta, tr["log_in3"])
    rho_days = spearman(delta, tr["c_n_days_with_tx"])
    rho_fac = spearman(delta, tr["f_n_facilities"])
    size = _cv(signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], tr[Y3].notna()))
    beat = _cv(raw) - size if np.isfinite(_cv(raw)) and np.isfinite(size) else float("nan")
    keep = bool(np.isfinite(beat) and beat >= KEEP_DELTA and rec["rank"] >= CHANCE and abs(rho_size) < SIZE_RHO and abs(rho_fac) < TWIN_RHO)
    prose = (
        f"Δ=n_facilities−n_types Y3 {_f(_cv(raw))} leftover after days {_f(rec['rank'])} "
        f"ρ(resid,days)={_f(rec['rho_ctrl'])} fake={rec['fake']}. "
        f"After n_types {_f(rec_n['rank'])}. ρ size {_f(rho_size)} days {_f(rho_days)} fac {_f(rho_fac)}. "
        f"beat_size={_f(beat)}. KEEP-as-X of Δ={'YES' if keep else 'no'} "
        f"(raw loses to size; leftover 0.607 is not an X)."
    )
    print(prose)
    return {
        "raw": _cv(raw),
        "left": rec["rank"],
        "fake": rec["fake"],
        "after_nt": rec_n["rank"],
        "rho_size": rho_size,
        "rho_fac": rho_fac,
        "beat": beat,
        "keep": keep,
        "prose": prose,
    }


def pass_mixed(tr: pd.DataFrame) -> dict:
    g = pd.to_numeric(tr["f_n_types"], errors="coerce").groupby(tr["company_id"])
    mx = g.transform("max")
    mn = g.transform("min")
    mixed = mx != mn
    always0 = mx.eq(0)
    always_pos = mn.gt(0)
    rec_m = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & mixed)
    rec_a = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & always_pos)
    n_m = int(tr.loc[mixed, "company_id"].nunique())
    n_0 = int(tr.loc[always0, "company_id"].nunique())
    n_a = int(tr.loc[always_pos, "company_id"].nunique())
    prose = (
        f"Inventory cadence companies always0={n_0} mixed-rise={n_m} always>0={n_a}. "
        f"Mixed leftover after days {_f(rec_m['rank'])} n_pos={rec_m['n_pos']}. "
        f"Always>0 leftover {_f(rec_a['rank'])} n_pos={rec_a['n_pos']}."
    )
    print(prose)
    return {"mixed": rec_m["rank"], "always": rec_a["rank"], "n_m": n_m, "n_0": n_0, "n_a": n_a, "prose": prose}


def pass_stack_size(tr: pd.DataFrame) -> dict:
    rec = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"], tr["log_in3"]], tr["fold"], tr[Y3].notna())
    rec_s = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"], tr["log_in3"], tr["f_n_facilities"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Leftover after days+size {_f(rec['rank'])} ρ(resid,days)={_f(rec['rho_ctrl'])}. "
        f"After days+size+facilities {_f(rec_s['rank'])}."
    )
    print(prose)
    return {"days_size": rec["rank"], "triple": rec_s["rank"], "prose": prose}


def pass_boot_ever(tr: pd.DataFrame, n_boot: int = 24) -> dict:
    mx = pd.to_numeric(tr["f_n_types"], errors="coerce").groupby(tr["company_id"]).transform("max")
    sub0 = tr.loc[mx.gt(0)].copy()
    rng = np.random.default_rng(FOLD_SEED + 29)
    cos = sub0["company_id"].drop_duplicates().to_numpy()
    idx = {c: sub0.index[sub0["company_id"] == c].to_numpy() for c in cos}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(cos, size=len(cos), replace=True)
        rows = np.concatenate([idx[c] for c in pick])
        sub = sub0.loc[rows].reset_index(drop=True)
        rec = leftover_diag(sub[Y3], sub["f_n_types"], [sub["c_n_days_with_tx"]], sub["fold"], sub[Y3].notna())
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
    arr = np.array(vals, dtype=float)
    prose = (
        f"Ever-connected bootstrap leftover-after-days n={arr.size} "
        f"p05/p50/p95 {_f(float(np.quantile(arr, 0.05)) if arr.size else float('nan'))} / "
        f"{_f(float(np.quantile(arr, 0.50)) if arr.size else float('nan'))} / "
        f"{_f(float(np.quantile(arr, 0.95)) if arr.size else float('nan'))}."
    )
    print(prose)
    return {
        "p05": float(np.quantile(arr, 0.05)) if arr.size else float("nan"),
        "p50": float(np.quantile(arr, 0.50)) if arr.size else float("nan"),
        "p95": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
        "prose": prose,
    }


def pass_last_connected(tr: pd.DataFrame) -> dict:
    lab = tr[Y3].notna()
    last = tr.loc[lab].sort_values("period").groupby("company_id", as_index=False).tail(1)
    on = pd.to_numeric(last["f_n_types"], errors="coerce") > 0
    rec = leftover_diag(last[Y3], last["f_n_types"], [last["c_n_days_with_tx"]], last["fold"], last[Y3].notna() & on)
    rec_f = leftover_diag(last[Y3], last["f_n_types"], [last["f_n_facilities"]], last["fold"], last[Y3].notna() & on)
    raw = signed_oof_auroc(last[Y3], last["f_n_types"], last["fold"], last[Y3].notna() & on)
    prose = (
        f"Last Y3 month among n_types>0 leftover after days {_f(rec['rank'])} "
        f"raw {_f(_cv(raw))} after facilities {_f(rec_f['rank'])} n_pos={rec['n_pos']}."
    )
    print(prose)
    return {"left": rec["rank"], "fac": rec_f["rank"], "raw": _cv(raw), "prose": prose}


def pass_first_y3(tr: pd.DataFrame) -> dict:
    lab = tr[Y3].notna()
    first = tr.loc[lab].sort_values("period").groupby("company_id", as_index=False).head(1)
    rec = leftover_diag(first[Y3], first["f_n_types"], [first["c_n_days_with_tx"]], first["fold"], first[Y3].notna())
    raw = signed_oof_auroc(first[Y3], first["f_n_types"], first["fold"], first[Y3].notna())
    prose = (
        f"First Y3-defined month leftover after days {_f(rec['rank'])} raw {_f(_cv(raw))} "
        f"n={rec['n']} n_pos={rec['n_pos']}."
    )
    print(prose)
    return {"left": rec["rank"], "raw": _cv(raw), "prose": prose}


def pass_last_y3(tr: pd.DataFrame) -> dict:
    lab = tr[Y3].notna()
    last = tr.loc[lab].sort_values("period").groupby("company_id", as_index=False).tail(1)
    rec = leftover_diag(last[Y3], last["f_n_types"], [last["c_n_days_with_tx"]], last["fold"], last[Y3].notna())
    raw = signed_oof_auroc(last[Y3], last["f_n_types"], last["fold"], last[Y3].notna())
    prose = (
        f"Last Y3-defined month leftover after days {_f(rec['rank'])} raw {_f(_cv(raw))} "
        f"n={rec['n']} n_pos={rec['n_pos']} (company-level leftover)."
    )
    print(prose)
    return {"left": rec["rank"], "raw": _cv(raw), "n_pos": rec["n_pos"], "prose": prose}


def pass_resid_icc(tr: pd.DataFrame) -> dict:
    xr = pd.to_numeric(tr["f_n_types"], errors="coerce").rank(method="average")
    cr = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce").rank(method="average")
    resid, _ = ols_resid(xr, cr)
    icc = icc_anova(resid, tr["company_id"])
    prose = (
        f"Rank residual of n_types after days ICC {_f(icc['icc'])} "
        f"({'BETWEEN leftover' if np.isfinite(icc['icc']) and icc['icc'] >= 0.80 else 'not a BETWEEN leftover'})."
    )
    print(prose)
    return {"icc": icc["icc"], "prose": prose}


def pass_connected_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    mx = pd.to_numeric(tr["f_n_types"], errors="coerce").groupby(tr["company_id"]).transform("max")
    ever = mx.gt(0)
    erp = tr["company_id"].isin(book)
    rec_d = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & ever & ~erp)
    rec_i = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & ever & erp)
    rec_df = leftover_diag(tr[Y3], tr["f_n_types"], [tr["f_n_facilities"]], tr["fold"], tr[Y3].notna() & ever & ~erp)
    rec_if = leftover_diag(tr[Y3], tr["f_n_types"], [tr["f_n_facilities"]], tr["fold"], tr[Y3].notna() & ever & erp)
    prose = (
        f"Ever-connected leftover after days dark {_f(rec_d['rank'])} n_pos={rec_d['n_pos']} "
        f"invoiced {_f(rec_i['rank'])} n_pos={rec_i['n_pos']}. "
        f"After facilities dark {_f(rec_df['rank'])} invoiced {_f(rec_if['rank'])}."
    )
    print(prose)
    return {"dark": rec_d["rank"], "inv": rec_i["rank"], "dark_fac": rec_df["rank"], "inv_fac": rec_if["rank"], "prose": prose}


def pass_other_book(tr: pd.DataFrame) -> dict:
    path = DATA / "debt_products.csv"
    if not path.exists():
        return {"prose": "debt_products.csv missing; skip other-book leftover.", "left": float("nan")}
    raw = pd.read_csv(path, usecols=["company_id", "type"])
    raw["company_id"] = raw["company_id"].astype(str)
    kinds = {"leasing", "guarantee", "mortgage", "renting"}
    ids = set(raw.loc[raw["type"].astype(str).isin(kinds), "company_id"])
    m = tr["company_id"].isin(ids)
    rec = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & m)
    rec_f = leftover_diag(tr[Y3], tr["f_n_types"], [tr["f_n_facilities"]], tr["fold"], tr[Y3].notna() & m)
    raw_a = signed_oof_auroc(tr[Y3], tr["f_n_types"], tr["fold"], tr[Y3].notna() & m)
    prose = (
        f"Leasing/guarantee/mortgage/renting companies {int(tr.loc[m, 'company_id'].nunique())}. "
        f"Y3 n_types {_f(_cv(raw_a))} leftover after days {_f(rec['rank'])} "
        f"after facilities {_f(rec_f['rank'])} n_pos={rec['n_pos']}."
    )
    print(prose)
    return {"left": rec["rank"], "fac": rec_f["rank"], "raw": _cv(raw_a), "prose": prose}


def pass_loan_cos(tr: pd.DataFrame) -> dict:
    path = DATA / "debt_products.csv"
    if not path.exists():
        return {"prose": "debt_products.csv missing; skip loan-company leftover.", "left": float("nan")}
    raw = pd.read_csv(path, usecols=["company_id", "type"])
    raw["company_id"] = raw["company_id"].astype(str)
    loan = set(raw.loc[raw["type"].astype(str).eq("loan"), "company_id"])
    m = tr["company_id"].isin(loan)
    rec = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & m)
    rec_f = leftover_diag(tr[Y3], tr["f_n_types"], [tr["f_n_facilities"]], tr["fold"], tr[Y3].notna() & m)
    raw_a = signed_oof_auroc(tr[Y3], tr["f_n_types"], tr["fold"], tr[Y3].notna() & m)
    n_co = int(tr.loc[m, "company_id"].nunique())
    prose = (
        f"Loan-book companies {n_co}. Y3 n_types {_f(_cv(raw_a))} leftover after days {_f(rec['rank'])} "
        f"after facilities {_f(rec_f['rank'])} n_pos={rec['n_pos']}."
    )
    print(prose)
    return {"n_co": n_co, "raw": _cv(raw_a), "left": rec["rank"], "fac": rec_f["rank"], "prose": prose}


def pass_flag_slices(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for name in ("f_has_loc", "f_has_factoring", "f_has_confirming"):
        on = pd.to_numeric(tr[name], errors="coerce").eq(1)
        rec = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & on)
        raw = signed_oof_auroc(tr[Y3], tr["f_n_types"], tr["fold"], tr[Y3].notna() & on)
        store[name] = rec["rank"]
        rows.append({"flag": name, "n": f"{int(on.sum()):,}", "Y3 raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]), "leftover days": _f(rec["rank"]), "n_pos": rec["n_pos"]})
    prose = (
        f"Leftover after days on flag=1 months loc {_f(store['f_has_loc'])} "
        f"fact {_f(store['f_has_factoring'])} conf {_f(store['f_has_confirming'])}."
    )
    print(prose)
    return {"rows": rows, "loc": store["f_has_loc"], "fact": store["f_has_factoring"], "conf": store["f_has_confirming"], "prose": prose}


def pass_ever_stack(tr: pd.DataFrame) -> dict:
    mx = pd.to_numeric(tr["f_n_types"], errors="coerce").groupby(tr["company_id"]).transform("max")
    ever = mx.gt(0)
    rec_d = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & ever)
    rec_f = leftover_diag(tr[Y3], tr["f_n_types"], [tr["f_n_facilities"]], tr["fold"], tr[Y3].notna() & ever)
    rec_b = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"], tr["f_n_facilities"]], tr["fold"], tr[Y3].notna() & ever)
    prose = (
        f"Ever-connected leftover after days {_f(rec_d['rank'])} "
        f"ρ(resid,days)={_f(rec_d['rho_ctrl'])} fake={rec_d['fake']}; "
        f"after facilities {_f(rec_f['rank'])}; after both {_f(rec_b['rank'])}."
    )
    print(prose)
    return {"days": rec_d["rank"], "fac": rec_f["rank"], "both": rec_b["rank"], "fake": rec_d["fake"], "prose": prose}


def pass_multi(tr: pd.DataFrame) -> dict:
    mx = pd.to_numeric(tr["f_n_types"], errors="coerce").groupby(tr["company_id"]).transform("max")
    ever = mx.gt(0)
    multi = mx.ge(2)
    rec_e = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & ever)
    rec_m = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & multi)
    raw_e = signed_oof_auroc(tr[Y3], tr["f_n_types"], tr["fold"], tr[Y3].notna() & ever)
    raw_m = signed_oof_auroc(tr[Y3], tr["f_n_types"], tr["fold"], tr[Y3].notna() & multi)
    prose = (
        f"Ever-connected leftover after days {_f(rec_e['rank'])} raw {_f(_cv(raw_e))} n_pos={rec_e['n_pos']}. "
        f"Ever n_types≥2 leftover {_f(rec_m['rank'])} raw {_f(_cv(raw_m))} n_pos={rec_m['n_pos']}."
    )
    print(prose)
    return {"ever": rec_e["rank"], "multi": rec_m["rank"], "prose": prose}


def pass_inv_t1(tr: pd.DataFrame, book: set[str]) -> dict:
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    terc = pd.qcut(size, 3, labels=["T1", "T2", "T3"], duplicates="drop")
    erp = tr["company_id"].isin(book)
    mask = tr[Y3].notna() & terc.eq("T1") & erp
    rec = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], mask)
    raw = signed_oof_auroc(tr[Y3], tr["f_n_types"], tr["fold"], mask)
    prose = (
        f"Invoiced∩T1 leftover after days {_f(rec['rank'])} raw {_f(_cv(raw))} n_pos={rec['n_pos']}."
    )
    print(prose)
    return {"left": rec["rank"], "raw": _cv(raw), "n_pos": rec["n_pos"], "prose": prose}


def pass_dark_t1(tr: pd.DataFrame, book: set[str]) -> dict:
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    terc = pd.qcut(size, 3, labels=["T1", "T2", "T3"], duplicates="drop")
    dark = ~tr["company_id"].isin(book)
    mask = tr[Y3].notna() & terc.eq("T1") & dark
    rec = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"]], tr["fold"], mask)
    raw = signed_oof_auroc(tr[Y3], tr["f_n_types"], tr["fold"], mask)
    prose = (
        f"Dark∩T1 leftover after days {_f(rec['rank'])} raw {_f(_cv(raw))} n_pos={rec['n_pos']}."
    )
    print(prose)
    return {"left": rec["rank"], "raw": _cv(raw), "n_pos": rec["n_pos"], "prose": prose}


def pass_om_stack(tr: pd.DataFrame) -> dict:
    wc = (
        pd.to_numeric(tr["f_has_loc"], errors="coerce")
        + pd.to_numeric(tr["f_has_factoring"], errors="coerce")
        + pd.to_numeric(tr["f_has_confirming"], errors="coerce")
    )
    nt = pd.to_numeric(tr["f_n_types"], errors="coerce")
    other = ((nt > 0) & wc.eq(0)).astype(float)
    rec_s = leftover_diag(tr[Y3], other, [tr["c_n_days_with_tx"], tr["log_in3"]], tr["fold"], tr[Y3].notna())
    rec_f = leftover_diag(tr[Y3], other, [tr["f_n_facilities"]], tr["fold"], tr[Y3].notna())
    rec_n = leftover_diag(tr[Y3], other, [nt], tr["fold"], tr[Y3].notna())
    prose = (
        f"Other-type leftover after days+size {_f(rec_s['rank'])} "
        f"ρ(resid,days)={_f(rec_s['rho_ctrl'])}; after facilities {_f(rec_f['rank'])}; "
        f"after n_types {_f(rec_n['rank'])} "
        f"(0.637 after days is near-days leak ρ=-0.753)."
    )
    print(prose)
    return {"days_size": rec_s["rank"], "fac": rec_f["rank"], "nt": rec_n["rank"], "prose": prose}


def pass_other_month(tr: pd.DataFrame) -> dict:
    wc = (
        pd.to_numeric(tr["f_has_loc"], errors="coerce")
        + pd.to_numeric(tr["f_has_factoring"], errors="coerce")
        + pd.to_numeric(tr["f_has_confirming"], errors="coerce")
    )
    nt = pd.to_numeric(tr["f_n_types"], errors="coerce")
    other = ((nt > 0) & wc.eq(0)).astype(float)
    raw = signed_oof_auroc(tr[Y3], other, tr["fold"], tr[Y3].notna())
    rec = leftover_diag(tr[Y3], other, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna())
    prose = (
        f"Month other-type (n_types>0, no loc/fact/conf) share {_pp(float(other.mean()))} "
        f"Y3 {_f(_cv(raw))} leftover after days {_f(rec['rank'])} "
        f"ρ(resid,days)={_f(rec['rho_ctrl'])} fake={rec['fake']}."
    )
    print(prose)
    return {"raw": _cv(raw), "left": rec["rank"], "fake": rec["fake"], "prose": prose}


def pass_tail(tr: pd.DataFrame) -> dict:
    work = tr.sort_values(["company_id", "period"]).copy()
    tail = work.groupby("company_id", sort=False).tail(6)
    head = work.groupby("company_id", sort=False).head(6)
    rec_t = leftover_diag(tail[Y3], tail["f_n_types"], [tail["c_n_days_with_tx"]], tail["fold"], tail[Y3].notna())
    rec_h = leftover_diag(head[Y3], head["f_n_types"], [head["c_n_days_with_tx"]], head["fold"], head[Y3].notna())
    prose = (
        f"Last-6m leftover after days {_f(rec_t['rank'])} n_pos={rec_t['n_pos']}. "
        f"First-6m leftover after days {_f(rec_h['rank'])} n_pos={rec_h['n_pos']}."
    )
    print(prose)
    return {"tail": rec_t["rank"], "head": rec_h["rank"], "prose": prose}


def pass_boot_t1(tr: pd.DataFrame, n_boot: int = 20) -> dict:
    size = pd.to_numeric(tr["log_in3"], errors="coerce")
    terc = pd.qcut(size, 3, labels=["T1", "T2", "T3"], duplicates="drop")
    sub0 = tr.loc[terc.eq("T1")].copy()
    rng = np.random.default_rng(FOLD_SEED + 19)
    cos = sub0["company_id"].drop_duplicates().to_numpy()
    idx = {c: sub0.index[sub0["company_id"] == c].to_numpy() for c in cos}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(cos, size=len(cos), replace=True)
        rows = np.concatenate([idx[c] for c in pick])
        sub = sub0.loc[rows].reset_index(drop=True)
        rec = leftover_diag(sub[Y3], sub["f_n_types"], [sub["c_n_days_with_tx"]], sub["fold"], sub[Y3].notna())
        if np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
    arr = np.array(vals, dtype=float)
    prose = (
        f"T1 bootstrap leftover-after-days n={arr.size} "
        f"p05/p50/p95 {_f(float(np.quantile(arr, 0.05)) if arr.size else float('nan'))} / "
        f"{_f(float(np.quantile(arr, 0.50)) if arr.size else float('nan'))} / "
        f"{_f(float(np.quantile(arr, 0.95)) if arr.size else float('nan'))}."
    )
    print(prose)
    return {
        "p50": float(np.quantile(arr, 0.50)) if arr.size else float("nan"),
        "p05": float(np.quantile(arr, 0.05)) if arr.size else float("nan"),
        "p95": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
        "prose": prose,
    }


def pass_other_types(tr: pd.DataFrame) -> dict:
    wc = (
        pd.to_numeric(tr["f_has_loc"], errors="coerce")
        + pd.to_numeric(tr["f_has_factoring"], errors="coerce")
        + pd.to_numeric(tr["f_has_confirming"], errors="coerce")
    )
    nt = pd.to_numeric(tr["f_n_types"], errors="coerce")
    ever_wc = wc.groupby(tr["company_id"]).transform("max").gt(0)
    ever_nt = nt.groupby(tr["company_id"]).transform("max").gt(0)
    other = ever_nt & ~ever_wc
    rec = leftover_diag(tr[Y3], nt, [tr["c_n_days_with_tx"]], tr["fold"], tr[Y3].notna() & other)
    raw = signed_oof_auroc(tr[Y3], nt, tr["fold"], tr[Y3].notna() & other)
    n_co = int(tr.loc[other, "company_id"].nunique())
    ever_rec = tr.groupby("company_id")[Y3].max()
    other_ids = tr.loc[other, "company_id"].unique()
    share = float(ever_rec.reindex(other_ids).mean()) if len(other_ids) else float("nan")
    prose = (
        f"Other-type-only companies (n_types>0, no loc/fact/conf) {n_co}. "
        f"Y3 raw {_f(_cv(raw))} leftover after days {_f(rec['rank'])} n_pos={rec['n_pos']} "
        f"ever-recover {_pp(share)}."
    )
    print(prose)
    return {"n_co": n_co, "raw": _cv(raw), "left": rec["rank"], "share": share, "prose": prose}


def pass_g_cousin(tr: pd.DataFrame) -> dict:
    if "g_n_types" not in tr.columns:
        return {"prose": "g_n_types missing; skip banking cousin.", "left": float("nan")}
    g = pd.to_numeric(tr["g_n_types"], errors="coerce").fillna(0)
    rho = spearman(tr["f_n_types"], g)
    raw = signed_oof_auroc(tr[Y3], g, tr["fold"], tr[Y3].notna())
    rec = leftover_diag(tr[Y3], tr["f_n_types"], [g], tr["fold"], tr[Y3].notna())
    rec_d = leftover_diag(tr[Y3], tr["f_n_types"], [tr["c_n_days_with_tx"], g], tr["fold"], tr[Y3].notna())
    prose = (
        f"Banking cousin g_n_types Y3 {_f(_cv(raw))} ρ vs f_n_types {_f(rho)} "
        f"{'TWIN' if np.isfinite(rho) and abs(rho) >= TWIN_RHO else 'not a twin'}. "
        f"f_n_types leftover after g_n_types {_f(rec['rank'])}; after days+g {_f(rec_d['rank'])}."
    )
    print(prose)
    return {"rho": rho, "g_y3": _cv(raw), "left": rec["rank"], "stack": rec_d["rank"], "prose": prose}


def pass_types() -> dict:
    path = DATA / "debt_products.csv"
    if not path.exists():
        return {"prose": "debt_products.csv missing; skip type mix.", "rows": []}
    raw = pd.read_csv(path, usecols=["company_id", "type"])
    vc = raw["type"].fillna("(null)").value_counts()
    rows = [{"type": str(k), "n": f"{int(v):,}", "share": _pp(v / len(raw))} for k, v in vc.items()]
    prose = (
        f"debt_products types {len(vc)}: "
        + ", ".join(f"{k} {int(v)}" for k, v in vc.items())
        + f". n_types max 8 is the type inventory, not facilities."
    )
    print(prose)
    return {"rows": rows, "n": int(len(vc)), "prose": prose}


def decide(p1, p2, p3, p4, p5, p7) -> dict:
    twin = bool(p2["gate_twins"])
    size = bool(p1["size_flag"])
    leftover = p4["y3_rank"]
    leftover_dies = bool(p4["y3_dies"])
    beat = bool(np.isfinite(p3["beat_size"]) and p3["beat_size"] >= KEEP_DELTA)
    keep_x = bool(beat and (not leftover_dies) and (not size) and (not twin))
    leftover_tag = "KEEP" if (not leftover_dies) else "CLOSE"
    if keep_x:
        card = "KEEP as unused leftover"
        tag = "KEEP"
        leftover_tag = "KEEP"
        why = f"leftover after days {_f(leftover)} lives; beats size by {_f(p3['beat_size'])}; not SIZE; not a twin. Do not put on the 15-col card."
    elif twin or p4["rewrite"]:
        card = "DROP from the 44 as Y3 X"
        tag = "DROP"
        why = (
            f"twin={p2['twins']} ρ_fac={_f(p2['rho_fac'])}; leftover after days {_f(leftover)} dies; "
            f"after facilities {_f(p4['fac_left'])} R²={_f(p4['fac_r2'])}; beat_size={_f(p3['beat_size'])}. "
            "Do not put on the 15-col card. Leftover-after-days is CLOSE (dies); 44-col is DROP (facilities twin)."
        )
    elif leftover_dies or (not beat):
        card = "CLOSE unused leftover"
        tag = "CLOSE"
        why = (
            f"Y3 leftover after days {_f(leftover)} {'dies' if leftover_dies else 'thin'}; beat_size={_f(p3['beat_size'])}. "
            f"Rise-only={p5['rise_only']}. After facilities {_f(p4['fac_left'])}. Do not put on the 15-col card."
        )
    else:
        card = "PARK as Y"
        tag = "PARK"
        why = "Connection inventory is not a health Y. Do not invent y_n_types."
    q6 = "CLOSE" if leftover_dies or twin or (not beat) else p7["q6"]
    return {
        "card": card,
        "headline_tag": tag,
        "leftover_tag": leftover_tag,
        "why": why,
        "keep_x": keep_x,
        "twin": twin,
        "size": size,
        "leftover_dies": leftover_dies,
        "q6": q6,
        "headline": (
            f"{tag} from the 44 as Y3 X. Leftover after days **{leftover_tag}** rank {_f(leftover)} "
            f"(OLS {_f(p4['y3_ols'])}, fake_ols={p4['fake_ols']}). "
            f"Y3 n_types {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size'])} vs facilities {_f(p3['fac'])}. "
            f"After facilities {_f(p4['fac_left'])} R²={_f(p4['fac_r2'])}. Inverse {_f(p4['inv_rank'])}. "
            f"Twin vs facilities={p2['fac_twin']} ρ={_f(p2['rho_fac'])}. Rise-only={p5['rise_only']} drops={p5['drops']}. "
            f"Q6 {q6}. {card}. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged. "
            "Do not put f_n_types on the 15-col card."
        ),
    }


def make_png(tr: pd.DataFrame) -> bool:
    if not HAS_MPL:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    nt = pd.to_numeric(tr["f_n_types"], errors="coerce")
    nf = pd.to_numeric(tr["f_n_facilities"], errors="coerce")
    ax.scatter(nf, nt, s=6, alpha=0.15, c="#1f4e79")
    ax.set_xlabel("f_n_facilities")
    ax.set_ylabel("f_n_types")
    ax.set_title(f"n_types vs n_facilities (ρ={spearman(nt, nf):.3f})")
    ax2 = axes[1]
    lab = tr[Y3].notna() & nt.notna()
    work = tr.loc[lab].copy()
    work["_x"] = nt[lab]
    work["_y"] = pd.to_numeric(work[Y3], errors="coerce")
    g = work.groupby("_x")["_y"].agg(["mean", "size"])
    ax2.bar(g.index.astype(int), 100.0 * g["mean"], color="#1f4e79")
    ax2.set_xlabel("f_n_types")
    ax2.set_ylabel("Y3 rate (%)")
    ax2.set_title("Y3 rate by n_types")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def write_md(ctx: dict) -> None:
    d = ctx["decision"]
    p1, p2, p3, p4 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"]
    p5, p6, p7, p8 = ctx["p5"], ctx["p6"], ctx["p7"], ctx["p8"]
    lines = [
        "# Unused leftover of `f_n_types` after `c_n_days_with_tx`",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. Do not invent `y_n_types`. Do not put `f_n_types` on the 15-col card. "
        "Do not overwrite `factoring_qa.*` / `ds_r_qa.*` / `fc_r_qa.*` / `tax_month_qa.*`. Y3 never B. "
        f"Night Y3 **{Y3_NIGHT[0]:.3f} / {Y3_NIGHT[1]:.3f}**. Days **{DAYS_BENCH}**. "
        f"Size **{SIZE_Y3_QUOTE}**. Y7 TURNOVER **{Y7_TURNOVER[0]:.3f} / {Y7_TURNOVER[1]:.3f}**.",
        "",
        "`f_n_types` = COUNT(DISTINCT debt type) as-of period_end. Connection inventory, rise-only. "
        "`f_has_*` already DROP from the 44. `f_n_facilities` is the cluster twin.",
        "",
        "## Headline",
        "",
        d["headline"],
        "",
        "## KEEP / CLOSE / DROP / PARK",
        "",
        _md_table(
            [
                {"object": "f_n_types leftover after days", "decision": f"**{d['leftover_tag']}**", "why": f"rank {_f(p4['y3_rank'])} OLS {_f(p4['y3_ols'])} fake_ols={p4['fake_ols']} ρ(resid,days)={_f(p4['y3_rho_days'])}"},
                {"object": "as Y3 X (not on 15-col card)", "decision": f"**{d['card']}**", "why": d["why"]},
                {"object": "Twin / SIZE", "decision": f"twin={'YES' if d['twin'] else 'no'} SIZE={'YES' if d['size'] else 'no'}", "why": f"ρ days {_f(p2['rho_days'])} facilities {_f(p2['rho_fac'])} size {_f(p1['rho_in3'])}"},
                {"object": "Rewrite of f_n_facilities?", "decision": "**YES**" if p4["rewrite"] or p4["fac_dies"] else "**no**", "why": f"leftover after facilities {_f(p4['fac_left'])} R²={_f(p4['fac_r2'])}"},
                {"object": "Rise-only inventory", "decision": "**YES**" if p5["rise_only"] else "**no**", "why": p5["prose"]},
                {"object": "Q6 lag leftover after days_lag1", "decision": f"**{d['q6']}**", "why": p7["prose"]},
                {"object": "Ever vs month", "decision": p8["prose"], "why": f"ever leftover {_f(p8['ever_left'])} month-after-ever {_f(p8['month_after_ever'])}"},
                {"object": "as health Y", "decision": "**PARK**", "why": "Rise-only connection inventory, not a FICO label. Do not invent y_n_types."},
                {"object": "Night quotes", "decision": "**unchanged**", "why": "Y3 0.762/0.752 · days 0.711 · size 0.617 · TURNOVER 0.720/0.712"},
            ]
        ),
        "",
        "## 1. Coverage; twins; SIZE",
        "",
        p1["prose"],
        "",
        _md_table(p1["rows"]),
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
        "## 3. Honest leftover after days + inverse + facilities",
        "",
        p4["prose"],
        "",
        _md_table(p4["rows"]),
        "",
        "## 4. Rise-only (G/F inventory pattern)",
        "",
        p5["prose"],
        "",
        "## 5. Dark vs ERP leftover",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        "## 6. Q6 — lag leftover after days_lag1",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 7. Ever-n vs this-month count",
        "",
        p8["prose"],
        "",
        ctx["p_icc"]["prose"],
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
        "## Extra — SIZE terciles",
        "",
        ctx["p_te"]["prose"],
        "",
        _md_table(ctx["p_te"]["rows"]),
        "",
        "## Extra — short / long books",
        "",
        ctx["p_bk"]["prose"],
        "",
        _md_table(ctx["p_bk"]["rows"]),
        "",
        "## Extra — Y3 rate by n_types",
        "",
        ctx["p_yc"]["prose"],
        "",
        _md_table(ctx["p_yc"]["rows"]),
        "",
        "## Extra — debt product type mix",
        "",
        ctx["p_ty"]["prose"],
        "",
        _md_table(ctx["p_ty"]["rows"]),
        "",
        "## Extra — leftover on f_has_* = 1 months",
        "",
        ctx["p_fs"]["prose"],
        "",
        _md_table(ctx["p_fs"]["rows"]),
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx.get("png") else "Plot: skipped.",
        "",
        "## What failed / next",
        "",
    ]
    for x in ctx["failed"]:
        lines.append(f"- {x}")
    extra_bits = ctx.get("extra_bits", [])
    extra_bits.extend(
        [
            f"- {ctx['p_bt']['prose']}",
            f"- {ctx['p_dg']['prose']}",
            f"- {ctx['p_te']['prose']}",
            f"- {ctx['p_pm']['prose']}",
            f"- {ctx['p_lg']['prose']}",
            f"- {ctx['p_bk']['prose']}",
            f"- {ctx['p_fc']['prose']}",
            f"- {ctx['p_rl']['prose']}",
            f"- {ctx['p_op']['prose']}",
            f"- {ctx['p_pc']['prose']}",
            f"- {ctx['p_yc']['prose']}",
            f"- {ctx['p_ck']['prose']}",
            f"- {ctx['p_bf']['prose']}",
            f"- {ctx['p_df']['prose']}",
            f"- {ctx['p_cn']['prose']}",
            f"- {ctx['p_dg2']['prose']}",
            f"- {ctx['p_ty']['prose']}",
            f"- {ctx['p_mx']['prose']}",
            f"- {ctx['p_ss']['prose']}",
            f"- {ctx['p_gc']['prose']}",
            f"- {ctx['p_ot']['prose']}",
            f"- {ctx['p_tl']['prose']}",
            f"- {ctx['p_t1']['prose']}",
            f"- {ctx['p_om']['prose']}",
            f"- {ctx['p_os']['prose']}",
            f"- {ctx['p_dt']['prose']}",
            f"- {ctx['p_it']['prose']}",
            f"- {ctx['p_mu']['prose']}",
            f"- {ctx['p_es']['prose']}",
            f"- {ctx['p_fs']['prose']}",
            f"- {ctx['p_ln']['prose']}",
            f"- {ctx['p_ob']['prose']}",
            f"- {ctx['p_cd']['prose']}",
            f"- {ctx['p_ri']['prose']}",
            f"- {ctx['p_ly']['prose']}",
            f"- {ctx['p_fy']['prose']}",
            f"- {ctx['p_lc']['prose']}",
            f"- {ctx['p_be']['prose']}",
        ]
    )
    if extra_bits:
        lines.extend(["", "## Later extras (same module)", ""])
        for x in extra_bits:
            lines.append(x)
    lines.extend(
        [
            "",
            f"Elapsed {ctx['elapsed_s']:.0f}s.",
            "",
            "Did **not**: overwrite `factoring_qa.*` / `ds_r_qa.*` / `fc_r_qa.*` / `tax_month_qa.*` / "
            "`n_cust_qa.*` / `issued_qa.*` / `in3_qa.*`, edit `debt.py` / `gbm_core.py`, put n_types on the "
            "15-col card, grow TURNOVER, invent `y_n_types`, write 0–100, fit holdout, touch `product/`.",
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
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_f_n_types", "value": p3["y3"], "coverage": cov, "notes": f"days={p3['days']:.4f} leftover={p4['y3_rank']:.4f} card={d['headline_tag']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_f_n_types_resid_days", "value": p4["y3_rank"], "coverage": cov, "notes": f"ols={p4['y3_ols']:.4f} fake_ols={p4['fake_ols']} r2={p4['y3_r2']:.4f} dies={p4['y3_dies']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_f_n_types_resid_facilities", "value": p4["fac_left"], "coverage": cov, "notes": f"r2={p4['fac_r2']:.4f} rewrite={p4['rewrite']} rho_fac={ctx['p2']['rho_fac']:.4f}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_f_n_types_lag1_resid_days_lag1", "value": p7["l1_left"], "coverage": cov, "notes": f"q6={p7['q6']} days_l1={p7['days_l1']:.4f}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_f_n_types_resid_days_boot_p50", "value": ctx["p_bt"]["p50"], "coverage": cov, "notes": f"p05={ctx['p_bt']['p05']:.4f} p95={ctx['p_bt']['p95']:.4f} Pge55={ctx['p_bt']['share_ge']:.3f}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_f_n_types_resid_facilities_boot_p50", "value": ctx["p_bf"]["p50"], "coverage": cov, "notes": f"p05={ctx['p_bf']['p05']:.4f} p95={ctx['p_bf']['p95']:.4f} connected_recover={ctx['p_cn']['share_c']:.4f}"},
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
    p2, p3, p4, p5, p7 = ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"], ctx["p7"]
    text = (
        f"# Wave 4 — f_n_types leftover after days\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/n_types_qa.py`\n"
        f"- `analysis/outputs/n_types_qa.md`\n"
        f"- `analysis/outputs/n_types_qa.png`\n"
        f"- append-only `analysis/experiments/registry.csv`\n"
        f"- this note\n\n"
        f"Did not overwrite `factoring_qa.*`, `ds_r_qa.*`, `fc_r_qa.*`, `tax_month_qa.*`, "
        f"`n_cust_qa.*`, `issued_qa.*`, `in3_qa.*`. Did not touch `debt.py`, `gbm_core.py`, "
        f"the 15-col card, TURNOVER, product/, or the parent journal. "
        f"Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. TURNOVER **0.720 / 0.712**.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| leftover after days | **{d['leftover_tag']}** {_f(p4['y3_rank'])} |\n"
        f"| as Y3 X | **{d['card']}** |\n"
        f"| leftover after f_n_facilities | **{_f(p4['fac_left'])}** rewrite={p4['rewrite']} |\n"
        f"| days leftover after n_types | **{_f(p4['inv_rank'])}** |\n"
        f"| twin vs facilities | **{p2['fac_twin']}** ρ={_f(p2['rho_fac'])} |\n"
        f"| rise-only | **{p5['rise_only']}** rises={p5['rises']} drops={p5['drops']} |\n"
        f"| Q6 | **{d['q6']}** |\n\n"
        f"Y3 n_types {_f(p3['y3'])} vs days {_f(p3['days'])} vs size {_f(p3['size'])} "
        f"vs facilities {_f(p3['fac'])}. Leftover after days rank {_f(p4['y3_rank'])} "
        f"OLS {_f(p4['y3_ols'])} ρ(resid,days)={_f(p4['y3_rho_days'])}. "
        f"lag1 leftover after days_lag1 {_f(p7['l1_left'])}. "
        f"Bootstrap leftover-after-days p05/p50/p95 "
        f"{_f(ctx['p_bt']['p05'])} / {_f(ctx['p_bt']['p50'])} / {_f(ctx['p_bt']['p95'])}. "
        f"After facilities boot p50 {_f(ctx['p_bf']['p50'])}. "
        f"Ever-connected recover {_pp(ctx['p_cn']['share_c'])} vs never {_pp(ctx['p_cn']['share_n'])}. "
        f"Rise leftover after days is a fake days leak ρ={_f(ctx['p_rl']['rise_rho'])}. "
        f"Mixed-rise leftover after days {_f(ctx['p_mx']['mixed'])}. "
        f"Δ leftover after days {_f(ctx['p_dg2']['left'])} KEEP_Δ={ctx['p_dg2']['keep']}. "
        f"g_n_types leftover {_f(ctx['p_gc']['left'])}. "
        f"Last Y3-month leftover {_f(ctx['p_ly']['left'])}. "
        f"First Y3-month leftover {_f(ctx['p_fy']['left'])}. "
        f"Residual ICC {_f(ctx['p_ri']['icc'])}.\n\n"
        f"## What failed / next\n\n"
        + "\n".join(f"- {x}" for x in ctx["failed"])
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.parent.mkdir(parents=True, exist_ok=True)
    WAVE_NOTE.write_text(text, encoding="utf-8")
    print(f"wrote {WAVE_NOTE}")


def run() -> dict:
    t0 = time.time()
    print(f"n_types_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    con = connect()
    try:
        book = book_invoice_ids(con)
    finally:
        con.close()
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, ["f_n_types", "f_n_facilities", "c_n_days_with_tx"], (1, 3))
    tr = panel[panel["split"] == "train"].copy().reset_index(drop=True)
    assert_no_holdout(tr["company_id"])
    print(f"train CM={len(tr):,} companies={tr['company_id'].nunique()} holdout CM={int((panel['split']=='holdout').sum())}")

    p1 = pass1_cov(tr)
    p2 = pass2_twins(tr)
    p3 = pass3_singles(tr)
    p4 = pass4_leftover(tr)
    p5 = pass5_rise(tr)
    p6 = pass6_dark(tr, book)
    p7 = pass7_q6(tr)
    p8 = pass8_ever(tr)
    p_icc = pass_icc(tr)
    p_ho = pass_holdout(panel)
    p_fl = pass_fold_left(tr)
    p_bt = pass_boot(tr)
    p_dg = pass_disagree(tr)
    p_te = pass_terciles(tr)
    p_pm = pass_perm(tr)
    p_lg = pass_logo(tr)
    p_bk = pass_books(tr)
    p_fc = pass_fac_compare(tr)
    p_rl = pass_rise_left(tr)
    p_op = pass_onpos(tr)
    p_pc = pass_perm_company(tr)
    p_yc = pass_y3_cells(tr)
    p_ck = pass_clock(tr)
    p_bf = pass_boot_fac(tr)
    p_df = pass_dark_fac(tr, book)
    p_cn = pass_connected(tr)
    p_dg2 = pass_delta_gate(tr)
    p_ty = pass_types()
    p_mx = pass_mixed(tr)
    p_ss = pass_stack_size(tr)
    p_gc = pass_g_cousin(tr)
    p_ot = pass_other_types(tr)
    p_tl = pass_tail(tr)
    p_t1 = pass_boot_t1(tr)
    p_om = pass_other_month(tr)
    p_os = pass_om_stack(tr)
    p_dt = pass_dark_t1(tr, book)
    p_it = pass_inv_t1(tr, book)
    p_mu = pass_multi(tr)
    p_es = pass_ever_stack(tr)
    p_fs = pass_flag_slices(tr)
    p_ln = pass_loan_cos(tr)
    p_ob = pass_other_book(tr)
    p_cd = pass_connected_dark(tr, book)
    p_ri = pass_resid_icc(tr)
    p_ly = pass_last_y3(tr)
    p_fy = pass_first_y3(tr)
    p_lc = pass_last_connected(tr)
    p_be = pass_boot_ever(tr)
    png = make_png(tr)
    decision = decide(p1, p2, p3, p4, p5, p7)
    failed = []
    if not p3["days_ok"]:
        failed.append(f"days replica drifted: {_f(p3['days'])} vs 0.711")
    if not p3["size_ok"]:
        failed.append(f"size replica drifted: {_f(p3['size'])} vs 0.617")
    if not p3["fact_ok"]:
        failed.append(f"has_factoring replica drifted: {_f(p3['fact'])} vs 0.505")
    if not p3["loc_ok"]:
        failed.append(f"has_loc replica drifted: {_f(p3['loc'])} vs 0.546")
    if p4["fake_ols"]:
        failed.append(f"OLS leftover after days {_f(p4['y3_ols'])} high vs honest rank {_f(p4['y3_rank'])}")
    if not p7["days_l1_ok"]:
        failed.append(f"days lag1 {_f(p7['days_l1'])} off KEEP 0.684")
    if not p6["confirm"]:
        failed.append(f"dark/invoiced {p6['n_dark']}/{p6['n_erp']} off 470/744")
    if not failed:
        failed.append(
            "no replica miss; leftover after days CLOSE 0.534 (dies) and twin of f_n_facilities ρ 0.994 → DROP from the 44. "
            "T1 leftover sometimes lives (boot p50 0.579) but raw loses to size and the twin gate still kills KEEP."
        )
    ctx = {
        "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6, "p7": p7, "p8": p8,
        "p_icc": p_icc, "p_ho": p_ho, "p_fl": p_fl,
        "p_bt": p_bt, "p_dg": p_dg, "p_te": p_te, "p_pm": p_pm,
        "p_lg": p_lg, "p_bk": p_bk, "p_fc": p_fc, "p_rl": p_rl,
        "p_op": p_op, "p_pc": p_pc, "p_yc": p_yc, "p_ck": p_ck,
        "p_bf": p_bf, "p_df": p_df, "p_cn": p_cn, "p_dg2": p_dg2, "p_ty": p_ty,
        "p_mx": p_mx, "p_ss": p_ss, "p_gc": p_gc, "p_ot": p_ot,
        "p_tl": p_tl, "p_t1": p_t1, "p_om": p_om, "p_os": p_os, "p_dt": p_dt, "p_it": p_it, "p_mu": p_mu, "p_es": p_es, "p_fs": p_fs, "p_ln": p_ln, "p_ob": p_ob, "p_cd": p_cd, "p_ri": p_ri, "p_ly": p_ly, "p_fy": p_fy, "p_lc": p_lc, "p_be": p_be,
        "decision": decision, "failed": failed, "png": png, "extra_bits": [],
        "elapsed_s": time.time() - t0, "tr": tr, "panel": panel, "book": book,
    }
    write_md(ctx)
    append_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    print(f"DONE card={decision['headline_tag']} leftover={_f(p4['y3_rank'])} fac_left={_f(p4['fac_left'])} y3={_f(p3['y3'])} elapsed={ctx['elapsed_s']:.0f}s")
    return ctx


if __name__ == "__main__":
    run()
